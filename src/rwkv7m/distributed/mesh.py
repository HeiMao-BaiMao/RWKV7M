import os
import math

import jax
from jax.experimental import mesh_utils


def process_info():
    return {
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "device_count": jax.device_count(),
        "devices": [str(device) for device in jax.devices()],
    }


def initialize_jax_distributed(
    *,
    coordinator_address=None,
    num_processes=None,
    process_id=None,
    local_device_ids=None,
):
    coordinator_address = coordinator_address or os.environ.get("JAX_COORDINATOR_ADDRESS")
    num_processes = num_processes or os.environ.get("JAX_NUM_PROCESSES")
    process_id = process_id if process_id is not None else os.environ.get("JAX_PROCESS_ID")

    if coordinator_address is None and num_processes is None and process_id is None:
        return process_info()

    kwargs = {}
    if coordinator_address is not None:
        kwargs["coordinator_address"] = coordinator_address
    if num_processes is not None:
        kwargs["num_processes"] = int(num_processes)
    if process_id is not None:
        kwargs["process_id"] = int(process_id)
    if local_device_ids is not None:
        kwargs["local_device_ids"] = local_device_ids
    jax.distributed.initialize(**kwargs)
    return process_info()


def make_1d_mesh(axis_name="data", devices=None):
    return make_mesh((axis_name,), devices=devices)


def _split_process_mesh_shape(axis_sizes, *, local_device_count, process_count):
    """Split global axes into fast per-process and outer process meshes."""
    local_shape = [1] * len(axis_sizes)
    remaining = int(local_device_count)
    for axis in range(len(axis_sizes) - 1, -1, -1):
        factor = math.gcd(axis_sizes[axis], remaining)
        local_shape[axis] = factor
        remaining //= factor
    if remaining != 1:
        raise ValueError("mesh axes cannot form a complete process-local submesh")

    process_shape = tuple(
        axis_size // local_size
        for axis_size, local_size in zip(axis_sizes, local_shape, strict=True)
    )
    if math.prod(process_shape) != process_count:
        raise ValueError("mesh axes do not match the distributed process layout")
    return tuple(local_shape), process_shape


def make_mesh(axis_names=("data",), *, axis_sizes=None, devices=None, axis_types=None):
    if isinstance(axis_names, str):
        axis_names = (axis_names,)
    axis_names = tuple(axis_names)
    if not axis_names:
        raise ValueError("axis_names must not be empty")

    devices = list(jax.devices() if devices is None else devices)
    if not devices:
        raise ValueError("cannot create a mesh without devices")

    if axis_sizes is None:
        if len(axis_names) != 1:
            raise ValueError("axis_sizes is required for multi-axis meshes")
        axis_sizes = (len(devices),)
    axis_sizes = tuple(int(size) for size in axis_sizes)
    if len(axis_sizes) != len(axis_names):
        raise ValueError("axis_sizes must have the same length as axis_names")
    if any(size <= 0 for size in axis_sizes):
        raise ValueError("axis_sizes must be positive")
    if math.prod(axis_sizes) != len(devices):
        raise ValueError("product of axis_sizes must equal number of devices")

    # jax.make_mesh maps the logical axes onto the physical accelerator
    # topology. A plain reshape preserves enumeration order and can produce a
    # needlessly expensive collective layout on TPU pods.
    if axis_types is None:
        # Keep the data-only compatibility path automatic. The NNX scale CLI
        # opts into Explicit axes when model parallelism is requested, making
        # that parameter/activation contract enforceable for collective audit.
        axis_types = (jax.sharding.AxisType.Auto,) * len(axis_names)
    else:
        axis_types = tuple(axis_types)
    if len(axis_types) != len(axis_names):
        raise ValueError("axis_types must have the same length as axis_names")
    process_count = jax.process_count()
    if process_count > 1:
        if len(devices) % process_count:
            raise ValueError("devices must be evenly distributed across processes")
        local_shape, process_shape = _split_process_mesh_shape(
            axis_sizes,
            local_device_count=len(devices) // process_count,
            process_count=process_count,
        )
        # Treat each process as an outer-network granule so its devices form a
        # rectangular local submesh. This is required for process-local input
        # assembly and is not guaranteed by physical TPU enumeration alone.
        device_mesh = mesh_utils.create_hybrid_device_mesh(
            local_shape,
            process_shape,
            devices,
            process_is_granule=True,
        )
        return jax.sharding.Mesh(device_mesh, axis_names, axis_types=axis_types)

    kwargs = {"devices": devices, "axis_types": axis_types}
    return jax.make_mesh(axis_sizes, axis_names, **kwargs)
