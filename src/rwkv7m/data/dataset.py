from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from .binidx import MMapIndexedDataset


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_magic_prime(data_size, ctx_len):
    dataset_slot = (int(data_size) - 1) // int(ctx_len)
    for candidate in range(dataset_slot, 1, -1):
        if candidate % 3 == 2 and is_prime(candidate):
            return candidate
    raise ValueError("could not find a 3n+2 prime for this data_size and ctx_len")


@dataclass
class BinIdxConfig:
    data_file: str
    ctx_len: int
    batch_size: int = 1
    magic_prime: int | None = None
    epoch_steps: int | None = None
    rank: int = 0
    world_size: int = 1
    sampling_mode: str = "magic"
    loss_mask_after_token: int | None = None


class BinIdxBatchDataset:
    """RWKV-LM-V7 style sampler that returns rwkv7m train_step batches."""

    def __init__(self, config: BinIdxConfig):
        if config.ctx_len <= 0:
            raise ValueError("ctx_len must be positive")
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.world_size <= 0:
            raise ValueError("world_size must be positive")
        if config.rank < 0 or config.rank >= config.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if config.sampling_mode not in {
            "magic",
            "sequential",
            "document",
            "document_sequential",
        }:
            raise ValueError(
                "sampling_mode must be 'magic', 'sequential', 'document', "
                "or 'document_sequential'"
            )
        self.config = config
        self.data = MMapIndexedDataset(config.data_file)
        self.data_size = self.data.data_size
        self.dataset_slot = (self.data_size - 1) // config.ctx_len
        self._document_lane_chunks = None
        self._document_examples = None
        if config.sampling_mode in {"document", "document_sequential"}:
            self.magic_prime = config.magic_prime
            if self.data_size < config.ctx_len + 1:
                raise ValueError("binidx dataset is smaller than ctx_len + 1")
            if config.sampling_mode == "document":
                self._document_examples = self._build_document_examples()
            else:
                self._document_lane_chunks = self._build_document_lane_chunks()
        else:
            self.magic_prime = config.magic_prime or find_magic_prime(
                self.data_size,
                config.ctx_len,
            )
            self._validate_magic_prime()

    def close(self):
        self.data.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _validate_magic_prime(self):
        if not is_prime(self.magic_prime):
            raise ValueError("magic_prime must be prime")
        if self.magic_prime % 3 != 2:
            raise ValueError("magic_prime must satisfy magic_prime % 3 == 2")
        ratio = self.magic_prime / max(self.dataset_slot, 1)
        if ratio <= 0.9 or ratio > 1.0:
            raise ValueError(
                "magic_prime must be close to data_size // ctx_len "
                f"(got ratio {ratio:.4f})"
            )
        if self.data_size < self.config.ctx_len + 1:
            raise ValueError("binidx dataset is smaller than ctx_len + 1")
        if self.magic_prime * self.config.ctx_len + 1 > self.data_size:
            raise ValueError("magic_prime can sample beyond the end of the dataset")
        if self.config.sampling_mode == "sequential":
            lane_count = self.config.batch_size * self.config.world_size
            if self.dataset_slot < lane_count:
                raise ValueError(
                    "sequential sampling requires at least one ctx_len slot per "
                    f"stream lane (dataset_slot={self.dataset_slot}, lanes={lane_count})"
                )

    def _build_document_lane_chunks(self):
        lane_count = self.config.batch_size * self.config.world_size
        lane_chunks = [[] for _ in range(lane_count)]
        item_sizes = np.asarray(self.data.sizes, dtype=np.int64)
        item_offsets = np.concatenate(
            [np.zeros((1,), dtype=np.int64), np.cumsum(item_sizes)]
        )
        document_bounds = np.asarray(self.data.doc_idx, dtype=np.int64)
        eligible_document_index = 0
        for item_start, item_stop in (
            zip(document_bounds[:-1], document_bounds[1:], strict=True)
        ):
            token_start = int(item_offsets[int(item_start)])
            token_stop = int(item_offsets[int(item_stop)])
            chunk_count = (token_stop - token_start - 1) // self.config.ctx_len
            if chunk_count <= 0:
                continue
            lane = eligible_document_index % lane_count
            eligible_document_index += 1
            for chunk_index in range(chunk_count):
                lane_chunks[lane].append(
                    (
                        token_start + chunk_index * self.config.ctx_len,
                        chunk_index == 0,
                    )
                )
        empty_lanes = [index for index, chunks in enumerate(lane_chunks) if not chunks]
        if empty_lanes:
            raise ValueError(
                "document_sequential sampling requires at least one full "
                f"document chunk per lane; empty lanes={empty_lanes}"
            )
        return tuple(tuple(chunks) for chunks in lane_chunks)

    def _build_document_examples(self):
        item_sizes = np.asarray(self.data.sizes, dtype=np.int64)
        item_offsets = np.concatenate(
            [np.zeros((1,), dtype=np.int64), np.cumsum(item_sizes)]
        )
        document_bounds = np.asarray(self.data.doc_idx, dtype=np.int64)
        examples = []
        for item_start, item_stop in zip(
            document_bounds[:-1],
            document_bounds[1:],
            strict=True,
        ):
            token_start = int(item_offsets[int(item_start)])
            token_stop = int(item_offsets[int(item_stop)])
            if token_stop - token_start >= self.config.ctx_len + 1:
                examples.append(token_start)
        if len(examples) < self.config.batch_size * self.config.world_size:
            raise ValueError(
                "document sampling requires at least one sufficiently long "
                "document per global batch lane"
            )
        return tuple(examples)

    @property
    def samples_per_epoch(self):
        if self._document_examples is not None:
            return len(self._document_examples)
        if self._document_lane_chunks is not None:
            return (
                min(len(chunks) for chunks in self._document_lane_chunks)
                * self.config.batch_size
                * self.config.world_size
            )
        if self.config.epoch_steps is not None:
            return self.config.epoch_steps * self.config.batch_size * self.config.world_size
        return self.dataset_slot

    @property
    def sequential_lane_length(self):
        if self.config.sampling_mode != "sequential":
            return None
        lane_count = self.config.batch_size * self.config.world_size
        return self.dataset_slot // lane_count

    def should_reset_state_before_step(self, step):
        lane_length = self.sequential_lane_length
        if lane_length is None:
            return False
        step = int(step)
        return step > 0 and step % lane_length == 0

    def sample_offset(self, sample_index, *, epoch=0):
        if self.config.sampling_mode == "document":
            index = (
                int(epoch) * len(self._document_examples)
                + int(sample_index) * self.config.world_size
                + self.config.rank
            ) % len(self._document_examples)
            return self._document_examples[index]
        if self.config.sampling_mode == "document_sequential":
            batch_idx = int(sample_index) % self.config.batch_size
            step_in_lane = int(sample_index) // self.config.batch_size
            global_lane = self.config.rank * self.config.batch_size + batch_idx
            lane_chunks = self._document_lane_chunks[global_lane]
            position = (
                int(epoch) * len(lane_chunks) + step_in_lane
            ) % len(lane_chunks)
            return lane_chunks[position][0]
        if self.config.sampling_mode == "sequential":
            batch_idx = int(sample_index) % self.config.batch_size
            step_in_lane = int(sample_index) // self.config.batch_size
            global_lane = self.config.rank * self.config.batch_size + batch_idx
            lane_count = self.config.world_size * self.config.batch_size
            lane_length = self.dataset_slot // lane_count
            chunk_index = global_lane * lane_length + (
                (int(epoch) * lane_length + step_in_lane) % lane_length
            )
            return chunk_index * self.config.ctx_len

        ii = (
            1
            + int(epoch) * self.samples_per_epoch
            + int(sample_index) * self.config.world_size
            + self.config.rank
        )
        factor = int(self.magic_prime * ((math.sqrt(5) - 1) / 2))
        return ((factor * ii * ii * ii) % self.magic_prime) * self.config.ctx_len

    def sample_resets_state(self, sample_index, *, epoch=0):
        if self.config.sampling_mode == "document":
            return True
        if self.config.sampling_mode != "document_sequential":
            return False
        batch_idx = int(sample_index) % self.config.batch_size
        step_in_lane = int(sample_index) // self.config.batch_size
        global_lane = self.config.rank * self.config.batch_size + batch_idx
        lane_chunks = self._document_lane_chunks[global_lane]
        position = (
            int(epoch) * len(lane_chunks) + step_in_lane
        ) % len(lane_chunks)
        return bool(lane_chunks[position][1])

    def get_sequence(self, sample_index, *, epoch=0):
        offset = self.sample_offset(sample_index, epoch=epoch)
        req_len = self.config.ctx_len + 1
        return self.data.get_global(offset=offset, length=req_len).astype(np.int32)

    def get_example(self, sample_index, *, epoch=0):
        tokens = self.get_sequence(sample_index, epoch=epoch)
        return tokens[:-1], tokens[1:]

    def get_batch(self, step, *, epoch=0):
        input_ids = []
        target_ids = []
        state_reset_mask = []
        start = int(step) * self.config.batch_size
        for batch_idx in range(self.config.batch_size):
            x, y = self.get_example(start + batch_idx, epoch=epoch)
            input_ids.append(x)
            target_ids.append(y)
            state_reset_mask.append(
                self.sample_resets_state(start + batch_idx, epoch=epoch)
            )
        input_ids = jnp.asarray(np.stack(input_ids), dtype=jnp.int32)
        target_ids = jnp.asarray(np.stack(target_ids), dtype=jnp.int32)
        if self.config.loss_mask_after_token is None:
            mask = jnp.ones(input_ids.shape, dtype=jnp.float32)
        else:
            mask = (
                input_ids == int(self.config.loss_mask_after_token)
            ).astype(jnp.float32)
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "mask": mask,
            "state_reset_mask": jnp.asarray(
                state_reset_mask,
                dtype=jnp.bool_,
            ),
        }

    def iter_batches(self, *, epoch=0, steps=None):
        if steps is None:
            sample_divisor = self.config.batch_size
            if self.config.sampling_mode in {
                "document",
                "document_sequential",
            }:
                sample_divisor *= self.config.world_size
            steps = self.samples_per_epoch // sample_divisor
        for step in range(int(steps)):
            yield self.get_batch(step, epoch=epoch)


def create_binidx_dataset(
    data_file,
    *,
    ctx_len,
    batch_size=1,
    magic_prime=None,
    epoch_steps=None,
    rank=0,
    world_size=1,
    sampling_mode="magic",
    loss_mask_after_token=None,
):
    return BinIdxBatchDataset(
        BinIdxConfig(
            data_file=data_file,
            ctx_len=ctx_len,
            batch_size=batch_size,
            magic_prime=magic_prime,
            epoch_steps=epoch_steps,
            rank=rank,
            world_size=world_size,
            sampling_mode=sampling_mode,
            loss_mask_after_token=loss_mask_after_token,
        )
    )
