from dataclasses import dataclass
from collections import deque

import jax
import jax.numpy as jnp

from ..data import create_binidx_dataset
from .sharding import host_batch_to_global_arrays


@dataclass(frozen=True)
class BatchLayout:
    global_batch_size: int
    process_batch_size: int
    per_device_batch_size: int
    process_count: int
    local_device_count: int
    local_data_shard_count: int
    data_axis_size: int
    data_shard_indices: tuple[int, ...]


@dataclass
class HostBinIdxDataset:
    dataset: object
    layout: BatchLayout
    process_index: int
    process_count: int

    @property
    def data_size(self):
        return self.dataset.data_size

    @property
    def magic_prime(self):
        return self.dataset.magic_prime

    def get_batch(self, step, *, epoch=0):
        return self.dataset.get_batch(step, epoch=epoch)

    def should_reset_state_before_step(self, step):
        return self.dataset.should_reset_state_before_step(step)

    def iter_batches(self, *, epoch=0, steps=None):
        return self.dataset.iter_batches(epoch=epoch, steps=steps)

    def close(self):
        self.dataset.close()


class _DataShardBatchDataset:
    """Combine deterministic logical data-shard streams for one process."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("datasets must not be empty")
        self.datasets = tuple(datasets)
        self.data_size = self.datasets[0].data_size
        self.magic_prime = self.datasets[0].magic_prime

    def close(self):
        for dataset in self.datasets:
            dataset.close()

    def should_reset_state_before_step(self, step):
        return any(
            dataset.should_reset_state_before_step(step)
            for dataset in self.datasets
        )

    def get_batch(self, step, *, epoch=0):
        batches = [
            dataset.get_batch(step, epoch=epoch) for dataset in self.datasets
        ]
        return {
            key: jnp.concatenate([batch[key] for batch in batches], axis=0)
            for key in batches[0]
        }

    def iter_batches(self, *, epoch=0, steps=None):
        if steps is None:
            steps = min(
                dataset.samples_per_epoch // dataset.config.batch_size
                for dataset in self.datasets
            )
        for step in range(int(steps)):
            yield self.get_batch(step, epoch=epoch)


def compute_batch_layout(
    global_batch_size,
    *,
    process_count,
    local_device_count,
    local_data_shard_count=None,
    data_axis_size=None,
    data_shard_indices=None,
):
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if process_count <= 0:
        raise ValueError("process_count must be positive")
    if local_device_count <= 0:
        raise ValueError("local_device_count must be positive")
    local_data_shard_count = (
        local_device_count
        if local_data_shard_count is None
        else int(local_data_shard_count)
    )
    if local_data_shard_count <= 0:
        raise ValueError("local_data_shard_count must be positive")
    if local_device_count % local_data_shard_count != 0:
        raise ValueError(
            "local_device_count must be divisible by local_data_shard_count"
        )
    uses_mesh_data_layout = (
        data_axis_size is not None or data_shard_indices is not None
    )
    if uses_mesh_data_layout and (
        data_axis_size is None or data_shard_indices is None
    ):
        raise ValueError(
            "data_axis_size and data_shard_indices must be provided together"
        )

    if not uses_mesh_data_layout:
        if global_batch_size % process_count != 0:
            raise ValueError("global_batch_size must be divisible by process_count")
        process_batch_size = global_batch_size // process_count
        if process_batch_size % local_data_shard_count != 0:
            raise ValueError(
                "process-local batch must be divisible by local_data_shard_count"
            )
        per_device_batch_size = process_batch_size // local_data_shard_count
        data_axis_size = process_count * local_data_shard_count
        normalized_data_shard_indices = ()
    else:
        data_axis_size = int(data_axis_size)
        if data_axis_size <= 0:
            raise ValueError("data_axis_size must be positive")
        normalized_data_shard_indices = tuple(
            sorted(map(int, data_shard_indices))
        )
        if not normalized_data_shard_indices:
            raise ValueError("data_shard_indices must not be empty")
        if len(set(normalized_data_shard_indices)) != len(
            normalized_data_shard_indices
        ):
            raise ValueError("data_shard_indices must be unique")
        if any(
            index < 0 or index >= data_axis_size
            for index in normalized_data_shard_indices
        ):
            raise ValueError("data_shard_indices must be within the data axis")
        if len(normalized_data_shard_indices) != local_data_shard_count:
            raise ValueError(
                "data_shard_indices must match local_data_shard_count"
            )
        if global_batch_size % data_axis_size != 0:
            raise ValueError("global_batch_size must be divisible by data_axis_size")
        per_device_batch_size = global_batch_size // data_axis_size
        process_batch_size = per_device_batch_size * local_data_shard_count

    return BatchLayout(
        global_batch_size=global_batch_size,
        process_batch_size=process_batch_size,
        per_device_batch_size=per_device_batch_size,
        process_count=process_count,
        local_device_count=local_device_count,
        local_data_shard_count=local_data_shard_count,
        data_axis_size=data_axis_size,
        data_shard_indices=normalized_data_shard_indices,
    )


def create_host_binidx_dataset(
    data_file,
    *,
    ctx_len,
    global_batch_size,
    magic_prime=None,
    epoch_steps=None,
    process_index=None,
    process_count=None,
    local_device_count=None,
    local_data_shard_count=None,
    data_axis_size=None,
    data_shard_indices=None,
    sampling_mode="magic",
    loss_mask_after_token=None,
):
    process_index = jax.process_index() if process_index is None else int(process_index)
    process_count = jax.process_count() if process_count is None else int(process_count)
    local_device_count = (
        jax.local_device_count() if local_device_count is None else int(local_device_count)
    )
    if process_index < 0 or process_index >= process_count:
        raise ValueError("process_index must satisfy 0 <= process_index < process_count")

    layout = compute_batch_layout(
        global_batch_size,
        process_count=process_count,
        local_device_count=local_device_count,
        local_data_shard_count=local_data_shard_count,
        data_axis_size=data_axis_size,
        data_shard_indices=data_shard_indices,
    )
    if layout.data_shard_indices:
        datasets = []
        try:
            for data_shard_index in layout.data_shard_indices:
                datasets.append(
                    create_binidx_dataset(
                        data_file,
                        ctx_len=ctx_len,
                        batch_size=layout.per_device_batch_size,
                        magic_prime=magic_prime,
                        epoch_steps=epoch_steps,
                        rank=data_shard_index,
                        world_size=layout.data_axis_size,
                        sampling_mode=sampling_mode,
                        loss_mask_after_token=loss_mask_after_token,
                    )
                )
        except Exception:
            for dataset in datasets:
                dataset.close()
            raise
        dataset = _DataShardBatchDataset(datasets)
    else:
        dataset = create_binidx_dataset(
            data_file,
            ctx_len=ctx_len,
            batch_size=layout.process_batch_size,
            magic_prime=magic_prime,
            epoch_steps=epoch_steps,
            rank=process_index,
            world_size=process_count,
            sampling_mode=sampling_mode,
            loss_mask_after_token=loss_mask_after_token,
        )
    return HostBinIdxDataset(
        dataset=dataset,
        layout=layout,
        process_index=process_index,
        process_count=process_count,
    )


def iter_prefetched_global_batches(
    host_dataset,
    sharding,
    *,
    start_step,
    steps,
    prefetch_size=2,
    epoch=0,
):
    prefetch_size = max(1, int(prefetch_size))
    queue = deque()
    next_step = int(start_step)
    stop_step = int(start_step) + int(steps)

    def enqueue(step):
        host_batch = host_dataset.get_batch(step, epoch=epoch)
        queue.append(
            (
                step,
                host_batch_to_global_arrays(
                    host_batch,
                    sharding,
                    host_dataset.layout,
                ),
            )
        )

    while next_step < stop_step and len(queue) < prefetch_size:
        enqueue(next_step)
        next_step += 1

    while queue:
        step, batch = queue.popleft()
        while next_step < stop_step and len(queue) < prefetch_size:
            enqueue(next_step)
            next_step += 1
        yield step, batch
