from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rwkv7m.distributed.mesh as mesh_module

from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path
from rwkv7m.distributed import (
    compute_batch_layout,
    create_host_binidx_dataset,
    data_parallel_sharding,
    host_batch_to_global_arrays,
    iter_prefetched_global_batches,
    make_1d_mesh,
    make_mesh,
    model_parallel_sharding,
    process_data_shard_indices,
    process_info,
    put_to_devices,
    replicated_sharding,
)
from rwkv7m.distributed.mesh import _split_hybrid_mesh_shape


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
    model_sharding = model_parallel_sharding(mesh, axis_name="data")
    assert model_sharding.mesh is mesh


def test_make_mesh_validates_axis_sizes():
    mesh = make_mesh(("data",), axis_sizes=(jax.device_count(),))
    assert mesh.axis_names == ("data",)
    with pytest.raises(ValueError):
        make_mesh(("data", "model"))


def test_make_mesh_delegates_device_ordering_to_jax(monkeypatch):
    sentinel = object()
    calls = []

    def fake_make_mesh(axis_sizes, axis_names, *, devices, axis_types):
        calls.append((axis_sizes, axis_names, devices, axis_types))
        return sentinel

    monkeypatch.setattr(jax, "make_mesh", fake_make_mesh)
    fake_devices = [SimpleNamespace(process_index=0, slice_index=0)]
    result = make_mesh(
        ("data", "model"),
        axis_sizes=(1, 1),
        devices=fake_devices,
    )
    assert result is sentinel
    assert calls == [
        (
            (1, 1),
            ("data", "model"),
            fake_devices,
            (jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
        )
    ]


def test_make_mesh_keeps_single_slice_topology_in_jax(monkeypatch):
    sentinel = object()
    calls = []

    def fake_make_mesh(axis_sizes, axis_names, *, devices, axis_types):
        calls.append((axis_sizes, axis_names, devices, axis_types))
        return sentinel

    monkeypatch.setattr(jax, "make_mesh", fake_make_mesh)
    fake_devices = [
        SimpleNamespace(process_index=index // 2, slice_index=0)
        for index in range(4)
    ]

    result = make_mesh(("data",), axis_sizes=(4,), devices=fake_devices)

    assert result is sentinel
    assert calls == [
        (
            (4,),
            ("data",),
            fake_devices,
            (jax.sharding.AxisType.Auto,),
        )
    ]


def test_split_hybrid_mesh_shape_prefers_inner_model_axis():
    assert _split_hybrid_mesh_shape(
        (16,), inner_device_count=4, outer_count=4
    ) == ((4,), (4,))
    assert _split_hybrid_mesh_shape(
        (4, 4), inner_device_count=4, outer_count=4
    ) == ((1, 4), (4, 1))
    assert _split_hybrid_mesh_shape(
        (4, 4),
        inner_device_count=8,
        outer_count=2,
        inner_axis_order=(0, 1),
    ) == ((4, 2), (1, 2))

    with pytest.raises(ValueError, match="inner submesh"):
        _split_hybrid_mesh_shape((3, 2), inner_device_count=4, outer_count=2)
    with pytest.raises(ValueError, match="axis permutation"):
        _split_hybrid_mesh_shape(
            (4, 4),
            inner_device_count=4,
            outer_count=4,
            inner_axis_order=(0, 0),
        )


def test_make_mesh_uses_slice_granules_for_multi_slice(monkeypatch):
    calls = []
    sentinel = object()

    def fake_create_hybrid_device_mesh(
        local_shape,
        process_shape,
        devices,
        *,
        process_is_granule=False,
    ):
        calls.append(
            (local_shape, process_shape, devices, process_is_granule)
        )
        return "device-mesh"

    def fake_mesh(device_mesh, axis_names, *, axis_types):
        assert device_mesh == "device-mesh"
        assert axis_names == ("data", "model")
        assert axis_types == (
            jax.sharding.AxisType.Auto,
            jax.sharding.AxisType.Auto,
        )
        return sentinel

    monkeypatch.setattr(
        mesh_module.mesh_utils,
        "create_hybrid_device_mesh",
        fake_create_hybrid_device_mesh,
    )
    monkeypatch.setattr(jax.sharding, "Mesh", fake_mesh)
    fake_devices = [
        SimpleNamespace(process_index=index // 4, slice_index=index // 8)
        for index in range(16)
    ]

    result = make_mesh(
        ("data", "model"),
        axis_sizes=(4, 4),
        devices=fake_devices,
    )

    assert result is sentinel
    assert calls == [((2, 4), (2, 1), fake_devices, False)]


def test_make_mesh_uses_process_granules_without_slice_metadata(monkeypatch):
    calls = []
    sentinel = object()

    def fake_create_hybrid_device_mesh(
        inner_shape,
        outer_shape,
        devices,
        *,
        process_is_granule=False,
    ):
        calls.append(
            (inner_shape, outer_shape, devices, process_is_granule)
        )
        return "device-mesh"

    monkeypatch.setattr(
        mesh_module.mesh_utils,
        "create_hybrid_device_mesh",
        fake_create_hybrid_device_mesh,
    )
    monkeypatch.setattr(
        jax.sharding,
        "Mesh",
        lambda *args, **kwargs: sentinel,
    )
    fake_devices = [
        SimpleNamespace(process_index=index // 2) for index in range(4)
    ]

    result = make_mesh(
        ("data", "model"),
        axis_sizes=(2, 2),
        devices=fake_devices,
    )

    assert result is sentinel
    assert calls == [((1, 2), (2, 1), fake_devices, True)]


def test_make_mesh_rejects_ambiguous_multi_slice_tpu_without_slice_metadata():
    fake_devices = [
        SimpleNamespace(
            process_index=index,
            platform="tpu",
            coords=[0, 0, 0],
            core_on_chip=0,
        )
        for index in range(2)
    ]

    with pytest.raises(ValueError, match="DCN-unaware"):
        make_mesh(
            ("data",),
            axis_sizes=(2,),
            devices=fake_devices,
        )


def test_process_data_shard_indices_handles_replica_processes():
    devices = np.asarray(
        [
            [SimpleNamespace(process_index=0), SimpleNamespace(process_index=1)],
            [SimpleNamespace(process_index=0), SimpleNamespace(process_index=1)],
        ]
    )
    mesh = SimpleNamespace(axis_names=("data", "model"), devices=devices)

    assert process_data_shard_indices(mesh, process_index=0) == (0, 1)
    assert process_data_shard_indices(mesh, process_index=1) == (0, 1)


def test_compute_batch_layout_validates_divisibility():
    layout = compute_batch_layout(8, process_count=2, local_device_count=2)
    assert layout.process_batch_size == 4
    assert layout.per_device_batch_size == 2

    model_parallel_layout = compute_batch_layout(
        1,
        process_count=1,
        local_device_count=4,
        local_data_shard_count=1,
    )
    assert model_parallel_layout.process_batch_size == 1
    assert model_parallel_layout.per_device_batch_size == 1
    assert model_parallel_layout.local_data_shard_count == 1

    cross_process_model_layout = compute_batch_layout(
        1,
        process_count=4,
        local_device_count=4,
        local_data_shard_count=1,
        data_axis_size=1,
        data_shard_indices=(0,),
    )
    assert cross_process_model_layout.process_batch_size == 1
    assert cross_process_model_layout.per_device_batch_size == 1
    assert cross_process_model_layout.data_shard_indices == (0,)

    reordered = compute_batch_layout(
        8,
        process_count=2,
        local_device_count=4,
        local_data_shard_count=2,
        data_axis_size=4,
        data_shard_indices=(3, 1),
    )
    assert reordered.data_shard_indices == (1, 3)

    with pytest.raises(ValueError):
        compute_batch_layout(7, process_count=2, local_device_count=1)
    with pytest.raises(ValueError, match="provided together"):
        compute_batch_layout(
            4,
            process_count=2,
            local_device_count=2,
            data_axis_size=4,
        )
    with pytest.raises(ValueError, match="must be unique"):
        compute_batch_layout(
            4,
            process_count=2,
            local_device_count=2,
            local_data_shard_count=2,
            data_axis_size=4,
            data_shard_indices=(1, 1),
        )


def test_mesh_data_shard_streams_replicate_across_processes(tmp_path):
    prefix = write_distributed_binidx(tmp_path)
    hosts = [
        create_host_binidx_dataset(
            prefix,
            ctx_len=4,
            global_batch_size=8,
            process_index=process_index,
            process_count=3,
            local_device_count=2,
            local_data_shard_count=2,
            data_axis_size=4,
            data_shard_indices=data_shards,
        )
        for process_index, data_shards in (
            (0, (0, 2)),
            (1, (0, 2)),
            (2, (1, 3)),
        )
    ]
    try:
        replica_0 = hosts[0].get_batch(0)
        replica_1 = hosts[1].get_batch(0)
        other = hosts[2].get_batch(0)

        assert replica_0["input_ids"].shape == (4, 4)
        np.testing.assert_array_equal(
            replica_0["input_ids"],
            replica_1["input_ids"],
        )
        assert not np.array_equal(replica_0["input_ids"], other["input_ids"])
        assert hosts[0].layout.process_batch_size == 4
        assert hosts[0].layout.per_device_batch_size == 2
    finally:
        for host in hosts:
            host.close()


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


def test_host_batch_to_global_arrays_on_single_process(tmp_path):
    prefix = write_distributed_binidx(tmp_path)
    host = create_host_binidx_dataset(
        prefix,
        ctx_len=4,
        global_batch_size=2,
        process_index=0,
        process_count=1,
        local_device_count=1,
    )
    try:
        mesh = make_1d_mesh()
        sharding = data_parallel_sharding(mesh)
        batch = host.get_batch(0)
        global_batch = host_batch_to_global_arrays(batch, sharding, host.layout)
        assert global_batch["input_ids"].shape == (2, 4)
        assert global_batch["target_ids"].shape == (2, 4)
        assert global_batch["mask"].shape == (2, 4)
    finally:
        host.close()


def test_iter_prefetched_global_batches_yields_sharded_arrays(tmp_path):
    prefix = write_distributed_binidx(tmp_path)
    host = create_host_binidx_dataset(
        prefix,
        ctx_len=4,
        global_batch_size=2,
        process_index=0,
        process_count=1,
        local_device_count=1,
    )
    try:
        mesh = make_1d_mesh()
        batches = list(
            iter_prefetched_global_batches(
                host,
                data_parallel_sharding(mesh),
                start_step=0,
                steps=2,
                prefetch_size=2,
            )
        )
        assert [step for step, _ in batches] == [0, 1]
        assert batches[0][1]["input_ids"].shape == (2, 4)
    finally:
        host.close()
