import jax
import jax.numpy as jnp
import numpy as np

from rwkv7m import create_train_runtime, tiny_config
from rwkv7m.data import MMapIndexedDatasetBuilder, data_file_path, index_file_path
from rwkv7m.distributed import (
    create_host_binidx_dataset,
    evaluate_batch_data_parallel,
    make_1d_mesh,
    metrics_to_host_dict,
    replicate_train_objects,
    train_batch_data_parallel,
)


def write_train_data(tmp_path):
    prefix = str(tmp_path / "dist-train")
    builder = MMapIndexedDatasetBuilder(data_file_path(prefix), dtype=np.uint16)
    builder.add_item((np.arange(257, dtype=np.uint16) % 32).astype(np.uint16))
    builder.end_document()
    builder.finalize(index_file_path(prefix))
    return prefix


def test_data_parallel_train_batch_runs_on_local_mesh(tmp_path):
    prefix = write_train_data(tmp_path)
    config = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(0),
        config,
        batch_size=1,
        total_steps=2,
    )
    host_dataset = create_host_binidx_dataset(
        prefix,
        ctx_len=4,
        global_batch_size=1,
        process_index=0,
        process_count=1,
        local_device_count=1,
    )
    try:
        dist = replicate_train_objects(runtime, state, mesh=make_1d_mesh())
        dist, metrics = train_batch_data_parallel(
            dist,
            host_dataset.get_batch(0),
            host_dataset.layout,
        )
        assert int(dist.train_state.step) == 1
        assert jnp.isfinite(metrics["loss"])
        host_metrics = metrics_to_host_dict(metrics)
        assert isinstance(host_metrics["loss"], float)

        eval_metrics = evaluate_batch_data_parallel(
            dist,
            host_dataset.get_batch(1),
            host_dataset.layout,
        )
        assert jnp.isfinite(eval_metrics["loss"])
    finally:
        host_dataset.close()
