"""Audit multi-host mesh locality and process-local array assembly."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec

from rwkv7m.distributed import (
    initialize_jax_distributed,
    make_mesh,
    process_data_shard_indices,
)


def _device_summary(device):
    summary = {
        "id": int(device.id),
        "process_index": int(device.process_index),
        "platform": device.platform,
    }
    for name in ("coords", "core_on_chip", "slice_index"):
        if hasattr(device, name):
            value = getattr(device, name)
            summary[name] = (
                [int(item) for item in value]
                if isinstance(value, (list, tuple))
                else int(value)
            )
    return summary


def _process_map(mesh):
    return np.vectorize(
        lambda device: device.process_index,
        otypes=[int],
    )(mesh.devices).tolist()


def _audit_layout(axis_names, axis_sizes, *, axis_types):
    mesh = make_mesh(
        axis_names,
        axis_sizes=axis_sizes,
        axis_types=axis_types,
    )
    data_shard_indices = process_data_shard_indices(mesh)
    local_data_shards = len(data_shard_indices)
    rows_per_data_shard = 2
    local_rows = rows_per_data_shard * local_data_shards
    global_rows = rows_per_data_shard * int(mesh.shape["data"])

    expected = np.arange(global_rows * 3, dtype=np.int32).reshape(
        global_rows,
        3,
    )
    local_data = np.concatenate(
        [
            expected[
                data_shard * rows_per_data_shard :
                (data_shard + 1) * rows_per_data_shard
            ]
            for data_shard in data_shard_indices
        ],
        axis=0,
    )
    if local_data.shape != (local_rows, 3):
        raise AssertionError(f"{axis_names}: invalid process-local batch shape")
    sharding = NamedSharding(mesh, PartitionSpec("data", None))
    global_data = jax.make_array_from_process_local_data(
        sharding,
        local_data,
        global_shape=(global_rows, 3),
    )

    with jax.set_mesh(mesh):
        transformed = jax.jit(
            lambda value: value * 2 + 1,
            in_shardings=sharding,
            out_shardings=sharding,
        )(global_data)
        checksum = jax.jit(
            lambda value: jnp.sum(value),
            in_shardings=sharding,
            out_shardings=NamedSharding(mesh, PartitionSpec()),
        )(global_data)
        jax.block_until_ready((transformed, checksum))

    gathered = multihost_utils.process_allgather(global_data, tiled=True)
    gathered_transformed = multihost_utils.process_allgather(
        transformed,
        tiled=True,
    )
    expected_checksum = int(expected.astype(np.int64).sum())

    if not np.array_equal(gathered, expected):
        raise AssertionError(f"{axis_names}: process-local assembly mismatch")
    if not np.array_equal(gathered_transformed, expected * 2 + 1):
        raise AssertionError(f"{axis_names}: elementwise sharded result mismatch")
    if int(checksum) != expected_checksum:
        raise AssertionError(f"{axis_names}: global reduction mismatch")

    data_shard_mask = np.zeros(int(mesh.shape["data"]), dtype=np.int32)
    data_shard_mask[list(data_shard_indices)] = 1
    data_shard_masks = multihost_utils.process_allgather(data_shard_mask)
    try:
        local_mesh = mesh.local_mesh
        local_mesh_contiguous = True
        local_mesh_shape = dict(local_mesh.shape)
        local_mesh_error = None
    except ValueError as exc:
        local_mesh_contiguous = False
        local_mesh_shape = None
        local_mesh_error = str(exc)
    contiguous_by_process = multihost_utils.process_allgather(
        np.asarray(local_mesh_contiguous, dtype=np.int32)
    )

    local_report = {
        "process_index": jax.process_index(),
        "data_shard_indices": list(data_shard_indices),
        "local_mesh_contiguous": local_mesh_contiguous,
        "local_mesh_shape": local_mesh_shape,
        "local_mesh_error": local_mesh_error,
        "addressable_shards": [
            {
                "device": str(shard.device),
                "index": str(shard.index),
                "shape": list(shard.data.shape),
            }
            for shard in global_data.addressable_shards
        ],
    }
    return {
        "axis_names": list(axis_names),
        "axis_sizes": list(axis_sizes),
        "axis_types": [str(axis_type) for axis_type in mesh.axis_types],
        "process_map": _process_map(mesh),
        "data_shard_masks_by_process": np.asarray(data_shard_masks).tolist(),
        "contiguous_local_mesh_by_process": np.asarray(
            contiguous_by_process
        ).tolist(),
        "local_report_process_0": local_report,
        "global_shape": list(global_data.shape),
        "checksum": int(checksum),
        "expected_checksum": expected_checksum,
        "assembly_matches": True,
        "elementwise_matches": True,
        "reduction_matches": True,
    }


def main() -> None:
    info = initialize_jax_distributed()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    if process_count < 2:
        raise RuntimeError("audit requires at least two JAX processes")
    if jax.device_count() != process_count * local_device_count:
        raise RuntimeError("audit requires a uniform number of devices per process")

    auto = jax.sharding.AxisType.Auto
    explicit = jax.sharding.AxisType.Explicit
    layout_specs = [
        (("data",), (jax.device_count(),), (auto,)),
        (
            ("data", "model"),
            (process_count, local_device_count),
            (explicit, explicit),
        ),
        (
            ("data", "model"),
            (1, jax.device_count()),
            (explicit, explicit),
        ),
    ]
    if jax.device_count() % 2 == 0:
        layout_specs.insert(
            2,
            (
                ("data", "model"),
                (2, jax.device_count() // 2),
                (explicit, explicit),
            ),
        )
    layouts = [
        _audit_layout(
            axis_names,
            axis_sizes,
            axis_types=axis_types,
        )
        for axis_names, axis_sizes, axis_types in layout_specs
    ]
    multihost_utils.sync_global_devices("rwkv7m-multihost-mesh-audit")

    if jax.process_index() == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "runtime": info,
                    "devices": [_device_summary(device) for device in jax.devices()],
                    "layouts": layouts,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
