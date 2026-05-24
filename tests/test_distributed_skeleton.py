import jax
import jax.numpy as jnp
import pytest

from rwkv7m.distributed import (
    compute_batch_layout,
    data_parallel_sharding,
    make_1d_mesh,
    process_info,
    put_to_devices,
    replicated_sharding,
)


def test_process_info_reports_local_runtime():
    info = process_info()
    assert info["process_count"] >= 1
    assert info["local_device_count"] >= 1
    assert info["device_count"] >= 1


def test_make_mesh_and_shard_batch_on_available_devices():
    mesh = make_1d_mesh()
    assert mesh.axis_names == ("data",)
    assert mesh.devices.size == jax.device_count()

    batch = {"x": jnp.arange(jax.device_count())}
    sharding = data_parallel_sharding(mesh)
    placed = put_to_devices(batch, sharding)
    assert placed["x"].shape == (jax.device_count(),)

    replicated = replicated_sharding(mesh)
    scalar = put_to_devices(jnp.asarray(1), replicated)
    assert scalar.shape == ()


def test_compute_batch_layout_validates_divisibility():
    layout = compute_batch_layout(8, process_count=2, local_device_count=2)
    assert layout.process_batch_size == 4
    assert layout.per_device_batch_size == 2

    with pytest.raises(ValueError):
        compute_batch_layout(7, process_count=2, local_device_count=1)
