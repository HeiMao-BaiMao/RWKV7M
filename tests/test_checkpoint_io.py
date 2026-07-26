import jax
import jax.numpy as jnp
from flax import nnx
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
from rwkv7m.model.screening import ScreeningConfig
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


def test_v5_admission_controller_checkpoint_roundtrip(tmp_path):
    config = tiny_config(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        head_size=8,
    )
    config.screening = ScreeningConfig(
        d_model=32,
        d_slot=16,
        d_k=8,
        d_v=8,
        n_slots=4,
        screened_layers=(0,),
        bank_ids=(0, 0, 1, 2),
        use_write_screening=True,
        write_mode="competitive_novel",
        semantics_version="screening-v5-core",
        gate_space="value",
        candidate_rank=4,
        n_read_tiles=2,
        capacity_calibration="analytic",
        admission_init=0.05,
        admission_controller_enabled=True,
        admission_controller_target=0.05,
        admission_controller_rate_ema_decay=0.5,
        admission_controller_kp=0.1,
        admission_controller_ki=0.02,
        admission_controller_max_step=0.1,
        admission_controller_bias_limit=6.0,
        edit_mode="tied",
        allocation_redundancy_weight=0.0,
    )
    config.lm_head_init = "variance_scaled"
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(10),
        config,
        batch_size=1,
        total_steps=2,
    )
    controller = state.model.layer_0.screening_0
    controller.update_admission_controller(
        jnp.asarray(0.8, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    expected_bias = jnp.array(controller.admission_controller_bias[...])
    expected_rate_ema = jnp.array(
        controller.admission_controller_rate_ema[...]
    )
    expected_error = jnp.array(
        controller.admission_controller_previous_error[...]
    )

    checkpoint_dir = tmp_path / "v5-controller"
    save_train_checkpoint(checkpoint_dir, state, config)
    _, template = create_train_runtime(
        jax.random.PRNGKey(11),
        config,
        batch_size=1,
        total_steps=2,
    )
    restored, restored_config, _ = load_train_checkpoint(
        checkpoint_dir,
        template,
    )
    restored_controller = restored.model.layer_0.screening_0
    assert restored_config == config
    assert jnp.allclose(
        restored_controller.admission_controller_bias[...],
        expected_bias,
    )
    assert jnp.allclose(
        restored_controller.admission_controller_rate_ema[...],
        expected_rate_ema,
    )
    assert jnp.allclose(
        restored_controller.admission_controller_previous_error[...],
        expected_error,
    )

    original_params = {
        "/".join(str(part) for part in path): variable[...]
        for path, variable in nnx.to_flat_state(
            nnx.state(state.model, nnx.Param)
        )
    }
    export_params, _, _ = load_model_safetensors(
        checkpoint_dir / "model.safetensors"
    )
    export_keys = flatten_dict(export_params, sep="/").keys()
    assert any("admission_controller_bias" in key for key in original_params)
    assert any("admission_controller_bias" in key for key in export_keys)
    assert not any("controller_rate_ema" in key for key in export_keys)
    assert not any("controller_previous_error" in key for key in export_keys)
