from dataclasses import dataclass


@dataclass(frozen=True)
class BatchLayout:
    global_batch_size: int
    process_batch_size: int
    per_device_batch_size: int
    process_count: int
    local_device_count: int


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
