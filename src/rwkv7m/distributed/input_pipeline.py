from dataclasses import dataclass
from collections import deque

import jax

from ..data import create_binidx_dataset
from .sharding import host_batch_to_global_arrays


@dataclass(frozen=True)
class BatchLayout:
    global_batch_size: int
    process_batch_size: int
    per_device_batch_size: int
    process_count: int
    local_device_count: int


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


def compute_batch_layout(global_batch_size, *, process_count, local_device_count):
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")
    if process_count <= 0:
        raise ValueError("process_count must be positive")
    if local_device_count <= 0:
        raise ValueError("local_device_count must be positive")
    if global_batch_size % process_count != 0:
        raise ValueError("global_batch_size must be divisible by process_count")

    process_batch_size = global_batch_size // process_count
    if process_batch_size % local_device_count != 0:
        raise ValueError("process-local batch must be divisible by local_device_count")

    return BatchLayout(
        global_batch_size=global_batch_size,
        process_batch_size=process_batch_size,
        per_device_batch_size=process_batch_size // local_device_count,
        process_count=process_count,
        local_device_count=local_device_count,
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
    sampling_mode="magic",
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
    )
    dataset = create_binidx_dataset(
        data_file,
        ctx_len=ctx_len,
        batch_size=layout.process_batch_size,
        magic_prime=magic_prime,
        epoch_steps=epoch_steps,
        rank=process_index,
        world_size=process_count,
        sampling_mode=sampling_mode,
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
