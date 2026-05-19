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
        self.config = config
        self.data = MMapIndexedDataset(config.data_file)
        self.data_size = self.data.data_size
        self.dataset_slot = (self.data_size - 1) // config.ctx_len
        self.magic_prime = config.magic_prime or find_magic_prime(self.data_size, config.ctx_len)
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

    @property
    def samples_per_epoch(self):
        if self.config.epoch_steps is not None:
            return self.config.epoch_steps * self.config.batch_size * self.config.world_size
        return self.dataset_slot

    def sample_offset(self, sample_index, *, epoch=0):
        ii = (
            1
            + int(epoch) * self.samples_per_epoch
            + int(sample_index) * self.config.world_size
            + self.config.rank
        )
        factor = int(self.magic_prime * ((math.sqrt(5) - 1) / 2))
        return ((factor * ii * ii * ii) % self.magic_prime) * self.config.ctx_len

    def get_sequence(self, sample_index, *, epoch=0):
        offset = self.sample_offset(sample_index, epoch=epoch)
        req_len = self.config.ctx_len + 1
        return self.data.get(idx=0, offset=offset, length=req_len).astype(np.int32)

    def get_example(self, sample_index, *, epoch=0):
        tokens = self.get_sequence(sample_index, epoch=epoch)
        return tokens[:-1], tokens[1:]

    def get_batch(self, step, *, epoch=0):
        input_ids = []
        target_ids = []
        start = int(step) * self.config.batch_size
        for batch_idx in range(self.config.batch_size):
            x, y = self.get_example(start + batch_idx, epoch=epoch)
            input_ids.append(x)
            target_ids.append(y)
        input_ids = jnp.asarray(np.stack(input_ids), dtype=jnp.int32)
        target_ids = jnp.asarray(np.stack(target_ids), dtype=jnp.int32)
        mask = jnp.ones(input_ids.shape, dtype=jnp.float32)
        return {"input_ids": input_ids, "target_ids": target_ids, "mask": mask}

    def iter_batches(self, *, epoch=0, steps=None):
        if steps is None:
            steps = self.samples_per_epoch // self.config.batch_size
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
        )
    )
