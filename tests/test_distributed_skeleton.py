import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path
from rwkv7m.distributed import (
    compute_batch_layout,
    create_host_binidx_dataset,
    data_parallel_sharding,
    make_1d_mesh,
    process_info,
    put_to_devices,
    replicated_sharding,
)


def write_distributed_binidx(tmp_path):
    prefix = str(tmp_path / "dist")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.uint16)
    builder.add_item(np.arange(257, dtype=np.uint16))
    builder.end_document()
    builder.finalize(index_file_path(prefix))
    return prefix


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


def test_create_host_binidx_dataset_uses_process_rank_for_sampling(tmp_path):
    prefix = write_distributed_binidx(tmp_path)
    host0 = create_host_binidx_dataset(
        prefix,
        ctx_len=4,
        global_batch_size=4,
        process_index=0,
        process_count=2,
        local_device_count=1,
    )
    host1 = create_host_binidx_dataset(
        prefix,
        ctx_len=4,
        global_batch_size=4,
        process_index=1,
        process_count=2,
        local_device_count=1,
    )
    try:
        assert host0.layout.process_batch_size == 2
        assert host1.layout.process_batch_size == 2
        batch0 = host0.get_batch(0)
        batch1 = host1.get_batch(0)
        assert batch0["input_ids"].shape == (2, 4)
        assert batch1["input_ids"].shape == (2, 4)
        assert not jnp.array_equal(batch0["input_ids"], batch1["input_ids"])
    finally:
        host0.close()
        host1.close()
