import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

from rwkv7m import (
    create_train_runtime,
    load_model_safetensors,
    load_train_checkpoint,
    load_train_checkpoint_metadata,
    load_train_runtime_state,
    save_train_checkpoint,
    tiny_config,
    train_batch,
)
from rwkv7m.train import generate_toy_batch


def test_train_checkpoint_roundtrip(tmp_path):
    config = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(0),
        config,
        batch_size=1,
        total_steps=2,
    )
    batch = generate_toy_batch(jax.random.PRNGKey(1), 1, 4, config.vocab_size)
    state, _ = train_batch(state, batch, runtime)

    checkpoint_dir = tmp_path / "ckpt"
    save_train_checkpoint(
        checkpoint_dir,
        state,
        config,
        rng_key=jax.random.PRNGKey(2),
        dataset_position={"step": 1},
        metadata={"run": "test"},
        runtime_state={
            "rwkv_state": runtime.rwkv_state,
            "screen_state": runtime.screen_state,
        },
    )

    metadata_config, payload = load_train_checkpoint_metadata(checkpoint_dir)
    assert metadata_config == config
    assert payload["step"] == int(state.step)
    assert payload["dataset_position"] == {"step": 1}
    assert payload["rng_key"] == [0, 2]

    _, template = create_train_runtime(
        jax.random.PRNGKey(3),
        config,
        batch_size=1,
        total_steps=2,
    )
    restored, restored_config, restored_payload = load_train_checkpoint(
        checkpoint_dir,
        template,
    )
    assert restored_config == config
    assert restored_payload["metadata"] == {"run": "test"}
    assert int(restored.step) == int(state.step)

    restored_runtime_state = load_train_runtime_state(
        checkpoint_dir,
        {
            "rwkv_state": runtime.initial_rwkv_state,
            "screen_state": runtime.initial_screen_state,
        },
    )
    assert restored_runtime_state is not None
    assert jnp.allclose(
        restored_runtime_state["rwkv_state"][0].time_mix_x,
        runtime.rwkv_state[0].time_mix_x,
    )

    original_flat = flatten_dict(state.params, sep="/")
    restored_flat = flatten_dict(restored.params, sep="/")
    assert original_flat.keys() == restored_flat.keys()
    for key in original_flat:
        assert jnp.allclose(original_flat[key], restored_flat[key])

    export_params, export_config, export_metadata = load_model_safetensors(
        checkpoint_dir / "model.safetensors"
    )
    assert export_config == config
    assert export_metadata["checkpoint_step"] == str(int(state.step))
    assert flatten_dict(export_params, sep="/").keys() == original_flat.keys()
