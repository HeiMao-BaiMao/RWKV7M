import os
import math

import jax


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
        # The existing Linen distributed path relies on automatic GSPMD
        # propagation. JAX 0.10 defaults jax.make_mesh to Explicit axes, which
        # turns otherwise valid constraints into assertions and makes gathers
        # such as the token embedding ambiguous.
        axis_types = (jax.sharding.AxisType.Auto,) * len(axis_names)
    else:
        axis_types = tuple(axis_types)
    if len(axis_types) != len(axis_names):
        raise ValueError("axis_types must have the same length as axis_names")
    kwargs = {"devices": devices, "axis_types": axis_types}
    return jax.make_mesh(axis_sizes, axis_names, **kwargs)
