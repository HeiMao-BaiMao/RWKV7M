import jax

from rwkv7m import create_train_runtime, tiny_config
import pytest

from rwkv7m.distributed import make_1d_mesh, place_train_objects, replicate_train_objects


def test_replicate_train_objects_on_local_mesh():
    config = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(0),
        config,
        batch_size=1,
        total_steps=2,
    )
    mesh = make_1d_mesh()
    dist = replicate_train_objects(runtime, state, mesh=mesh)
    assert dist.mesh is mesh
    assert dist.train_state.step.shape == ()
    assert dist.initial_rwkv_state is dist.rwkv_state
    assert dist.initial_screen_state is dist.screen_state
    assert len(dist.rwkv_state) == config.n_layers
    assert len(dist.screen_state.layers) == len(config.screening.screened_layers)


def test_place_train_objects_can_use_model_axis_sharding():
    config = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(0),
        config,
        batch_size=1,
        total_steps=2,
    )
    mesh = make_1d_mesh("data")
    dist = place_train_objects(runtime, state, mesh=mesh, param_axis_name="data")
    assert int(dist.train_state.step) == 0

    with pytest.raises(ValueError):
        place_train_objects(runtime, state, mesh=mesh, param_axis_name="model")
