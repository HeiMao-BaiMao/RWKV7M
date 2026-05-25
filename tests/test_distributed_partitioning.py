import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from rwkv7m import create_train_runtime, tiny_config
from rwkv7m.distributed import (
    make_1d_mesh,
    mesh_axis_size,
    parameter_partition_spec,
    parameter_sharding,
    place_parameter_tree,
)


def test_parameter_partition_rules_for_known_weights():
    assert parameter_partition_spec(
        ("token_embedding", "embedding"),
        jnp.zeros((32, 16)),
        axis_name="model",
        axis_size=1,
    ) == PartitionSpec("model", None)

    assert parameter_partition_spec(
        ("lm_head", "kernel"),
        jnp.zeros((16, 32)),
        axis_name="model",
        axis_size=1,
    ) == PartitionSpec(None, "model")

    assert parameter_partition_spec(
        ("layer_0", "dense", "kernel"),
        jnp.zeros((16, 64)),
        axis_name="model",
        axis_size=1,
    ) == PartitionSpec(None, "model")

    assert parameter_partition_spec(
        ("layer_0", "dense", "bias"),
        jnp.zeros((64,)),
        axis_name="model",
        axis_size=1,
    ) == PartitionSpec()


def test_parameter_partition_falls_back_to_replicated_when_not_divisible():
    assert parameter_partition_spec(
        ("x", "kernel"),
        jnp.zeros((5, 7)),
        axis_name="model",
        axis_size=3,
    ) == PartitionSpec()


def test_place_parameter_tree_uses_named_shardings():
    config = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(0),
        config,
        batch_size=1,
        total_steps=2,
    )
    mesh = make_1d_mesh("data")
    placed = place_parameter_tree(state.params, mesh, axis_name="data")
    assert mesh_axis_size(mesh, "data") == jax.device_count()
    assert placed["token_embedding"]["embedding"].sharding.mesh is mesh

    sharding = parameter_sharding(
        ("lm_head", "kernel"),
        state.params["lm_head"]["kernel"],
        mesh,
        axis_name="data",
    )
    assert sharding.mesh is mesh
