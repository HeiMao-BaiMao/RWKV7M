import os
import math

import jax
import numpy as np
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


def _split_hybrid_mesh_shape(
    axis_sizes,
    *,
    inner_device_count,
    outer_count,
    inner_axis_order=None,
):
    """Split global axes into fast inner and slower outer mesh shapes."""
    local_shape = [1] * len(axis_sizes)
    remaining = int(inner_device_count)
    if inner_axis_order is None:
        inner_axis_order = tuple(range(len(axis_sizes) - 1, -1, -1))
    else:
        inner_axis_order = tuple(map(int, inner_axis_order))
        if sorted(inner_axis_order) != list(range(len(axis_sizes))):
            raise ValueError("inner_axis_order must be an axis permutation")
    for axis in inner_axis_order:
        factor = math.gcd(axis_sizes[axis], remaining)
        local_shape[axis] = factor
        remaining //= factor
    if remaining != 1:
        raise ValueError("mesh axes cannot form a complete inner submesh")

    process_shape = tuple(
        axis_size // local_size
        for axis_size, local_size in zip(axis_sizes, local_shape, strict=True)
    )
    if math.prod(process_shape) != outer_count:
        raise ValueError("mesh axes do not match the outer topology")
    return tuple(local_shape), process_shape


def process_data_shard_indices(mesh, axis_name="data", *, process_index=None):
    """Return sorted logical data-axis shards addressable by one process."""
    process_index = (
        jax.process_index() if process_index is None else int(process_index)
    )
    if axis_name not in mesh.axis_names:
        raise ValueError(f"mesh does not contain data axis {axis_name!r}")
    axis = tuple(mesh.axis_names).index(axis_name)
    indices = {
        logical_index[axis]
        for logical_index, device in np.ndenumerate(mesh.devices)
        if device.process_index == process_index
    }
    if not indices:
        raise ValueError(f"process {process_index} has no devices in the mesh")
    return tuple(sorted(indices))


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
    model_axes = [
        index for index, name in enumerate(axis_names) if name == "model"
    ]
    inner_axis_order = tuple(
        model_axes
        + [
            index
            for index in range(len(axis_names) - 1, -1, -1)
            if index not in model_axes
        ]
    )
    slice_indices = {
        device.slice_index
        for device in devices
        if getattr(device, "slice_index", None) is not None
    }
    if len(slice_indices) > 1:
        if len(devices) % len(slice_indices):
            raise ValueError("devices must be evenly distributed across TPU slices")
        inner_shape, outer_shape = _split_hybrid_mesh_shape(
            axis_sizes,
            inner_device_count=len(devices) // len(slice_indices),
            outer_count=len(slice_indices),
            inner_axis_order=inner_axis_order,
        )
        device_mesh = mesh_utils.create_hybrid_device_mesh(
            inner_shape,
            outer_shape,
            devices,
        )
        return jax.sharding.Mesh(device_mesh, axis_names, axis_types=axis_types)

    process_indices = {device.process_index for device in devices}
    if not slice_indices and all(
        getattr(device, "platform", None) == "tpu" for device in devices
    ):
        physical_coords = []
        for device in devices:
            if not hasattr(device, "coords"):
                physical_coords = []
                break
            physical_coords.append(
                (
                    tuple(device.coords),
                    int(getattr(device, "core_on_chip", 0)),
                )
            )
        if physical_coords and len(set(physical_coords)) != len(physical_coords):
            raise ValueError(
                "TPU physical coordinates repeat but devices do not expose "
                "slice_index; refusing to build a DCN-unaware process mesh"
            )
    if len(process_indices) > 1 and not slice_indices:
        if len(devices) % len(process_indices):
            raise ValueError("devices must be evenly distributed across processes")
        inner_shape, outer_shape = _split_hybrid_mesh_shape(
            axis_sizes,
            inner_device_count=len(devices) // len(process_indices),
            outer_count=len(process_indices),
            inner_axis_order=inner_axis_order,
        )
        device_mesh = mesh_utils.create_hybrid_device_mesh(
            inner_shape,
            outer_shape,
            devices,
            process_is_granule=True,
        )
        return jax.sharding.Mesh(device_mesh, axis_names, axis_types=axis_types)

    kwargs = {"devices": devices, "axis_types": axis_types}
    return jax.make_mesh(axis_sizes, axis_names, **kwargs)
