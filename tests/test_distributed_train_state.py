import jax

from rwkv7m import create_train_runtime, tiny_config
from rwkv7m.distributed import make_1d_mesh, replicate_train_objects


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
    assert len(dist.rwkv_state) == config.n_layers
    assert len(dist.screen_state.layers) == len(config.screening.screened_layers)
