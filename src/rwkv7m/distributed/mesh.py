import os

import jax
import numpy as np
from jax.sharding import Mesh


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
    devices = list(jax.devices() if devices is None else devices)
    if not devices:
        raise ValueError("cannot create a mesh without devices")
    return Mesh(np.asarray(devices), (axis_name,))
