import jax
import jax.numpy as jnp
import pytest

from rwkv7m import create_runtime, create_train_runtime, tiny_config, train_batch
from rwkv7m.train import generate_toy_batch
from rwkv7m.model.screened_rwkv import (
    ModelConfig,
    create_model_variables,
    init_rwkv_state,
)
from rwkv7m.model.screening import ScreeningConfig, normalize_phase
from rwkv7m.model.state import init_screen_state


def write_config():
    screening = ScreeningConfig(
        d_model=32,
        d_slot=16,
        d_k=8,
        d_v=8,
        n_slots=4,
        screened_layers=(0,),
        bank_ids=(0, 0, 1, 2),
        use_write_screening=True,
    )
    return ModelConfig(
        d_model=32,
        d_ffn=64,
        n_layers=2,
        n_heads=2,
        head_size=16,
        vocab_size=64,
        max_seq_len=16,
        dtype="float32",
        use_screening=True,
        screening=screening,
    )


def test_phase_alias_and_invalid_phase():
    assert normalize_phase("read_only") == "read_screening_only"
    assert normalize_phase("read_screening_only") == "read_screening_only"
    assert normalize_phase("read_write") == "read_write"
    with pytest.raises(ValueError, match="Unknown phase"):
        normalize_phase("inference")


def test_unknown_phase_rejected_by_model_apply():
    cfg = write_config()
    key = jax.random.PRNGKey(0)
    variables, model = create_model_variables(key, cfg, batch_size=1)
    ids = jnp.zeros((1, 2), dtype=jnp.int32)
    with pytest.raises(ValueError, match="Unknown phase"):
        model.apply(
            variables,
            ids,
            init_rwkv_state(1, cfg),
            init_screen_state(1, cfg.screening),
            phase="inference",
            deterministic=True,
        )


def test_read_screening_only_init_then_read_write_apply_works():
    cfg = write_config()
    key = jax.random.PRNGKey(1)
    variables, model = create_model_variables(key, cfg, batch_size=1)
    ids = jnp.arange(4, dtype=jnp.int32)[None, :]
    logits, _, screen_state, stats = model.apply(
        variables,
        ids,
        init_rwkv_state(1, cfg),
        init_screen_state(1, cfg.screening),
        phase="read_write",
        deterministic=True,
    )
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert "rel_write_mean" in stats
    assert screen_state.layers[0].ages.shape == (1, cfg.screening.n_slots)


def test_write_phase_ages_accumulate_across_steps():
    cfg = write_config()
    key = jax.random.PRNGKey(2)
    variables, model = create_model_variables(key, cfg, batch_size=1)
    ids = jnp.zeros((1, 3), dtype=jnp.int32)
    _, _, screen_state, _ = model.apply(
        variables,
        ids,
        init_rwkv_state(1, cfg),
        init_screen_state(1, cfg.screening),
        phase="read_write",
        deterministic=True,
    )
    assert jnp.all(screen_state.layers[0].ages >= 0.0)
    assert jnp.any(screen_state.layers[0].ages > 0.0)


def test_read_screening_only_keeps_ages_unchanged():
    cfg = write_config()
    key = jax.random.PRNGKey(3)
    variables, model = create_model_variables(key, cfg, batch_size=1)
    screen_state = init_screen_state(1, cfg.screening)
    ids = jnp.zeros((1, 3), dtype=jnp.int32)
    _, _, new_screen_state, _ = model.apply(
        variables,
        ids,
        init_rwkv_state(1, cfg),
        screen_state,
        phase="read_screening_only",
        deterministic=True,
    )
    assert jnp.allclose(new_screen_state.layers[0].ages, screen_state.layers[0].ages)


def test_invalid_config_validation():
    with pytest.raises(ValueError, match="bank_ids"):
        ScreeningConfig(n_slots=4, bank_ids=(0, 1))
    with pytest.raises(ValueError, match="bank_ids values"):
        ScreeningConfig(n_slots=4, bank_ids=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="half-life"):
        ScreeningConfig(long_half_life_tokens=0)
    with pytest.raises(ValueError, match="usage_ema_decay"):
        ScreeningConfig(usage_ema_decay=1.0)
    with pytest.raises(ValueError, match="d_model must equal"):
        ModelConfig(d_model=33, n_heads=2, head_size=16, use_screening=False)
    with pytest.raises(ValueError, match="head_chunk_size"):
        ModelConfig(head_chunk_size=0, use_screening=False)
    with pytest.raises(ValueError, match="training_vocab_tile_size"):
        ModelConfig(training_vocab_tile_size=0, use_screening=False)
    with pytest.raises(ValueError, match="screening.d_model"):
        ModelConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            head_size=16,
            use_screening=True,
            screening=ScreeningConfig(
                d_model=64,
                n_slots=4,
                screened_layers=(0,),
                bank_ids=(0, 0, 1, 2),
            ),
        )
    with pytest.raises(ValueError, match="screened_layers"):
        ModelConfig(
            d_model=32,
            n_layers=2,
            n_heads=2,
            head_size=16,
            use_screening=True,
            screening=ScreeningConfig(
                d_model=32,
                n_slots=4,
                screened_layers=(2,),
                bank_ids=(0, 0, 1, 2),
            ),
        )


def test_top_level_runtime_api_imports_and_runs():
    cfg = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime = create_runtime(jax.random.PRNGKey(4), cfg, batch_size=1)
    logits, _, _, stats = runtime.model.apply(
        runtime.variables,
        jnp.zeros((1, 2), dtype=jnp.int32),
        runtime.rwkv_state,
        runtime.screen_state,
        phase="read_screening_only",
        deterministic=True,
    )
    assert logits.shape == (1, 2, cfg.vocab_size)
    assert isinstance(stats, dict)


def test_top_level_train_api_runs_with_short_schedule():
    cfg = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(5),
        cfg,
        batch_size=1,
        total_steps=2,
    )
    batch = generate_toy_batch(jax.random.PRNGKey(6), 1, 4, cfg.vocab_size)
    state, metrics = train_batch(state, batch, runtime)
    assert jnp.isfinite(metrics["loss"])


def test_read_write_training_uses_effective_write_updates():
    cfg = write_config()
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(13),
        cfg,
        batch_size=1,
        total_steps=2,
    )
    batch = generate_toy_batch(jax.random.PRNGKey(14), 1, 4, cfg.vocab_size)
    state, metrics = train_batch(state, batch, runtime, phase="read_write")
    assert jnp.isfinite(metrics["loss"])
    assert metrics["rel_write_effective_mean"] > 0.0
    assert metrics["slot_update_norm_mean"] >= 0.0
    assert metrics["slot_usage_ema_mean"] >= 0.0


def test_train_batch_resets_recurrent_state_by_default():
    cfg = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(7),
        cfg,
        batch_size=1,
        total_steps=2,
    )
    batch = generate_toy_batch(jax.random.PRNGKey(8), 1, 4, cfg.vocab_size)
    original_wkv = runtime.rwkv_state[0].wkv
    state, metrics = train_batch(state, batch, runtime)
    assert jnp.isfinite(metrics["loss"])
    assert jnp.allclose(runtime.rwkv_state[0].wkv, original_wkv)


def test_train_batch_can_carry_state_explicitly():
    cfg = tiny_config(vocab_size=32, d_model=32, n_layers=2, n_heads=2, head_size=16)
    runtime, state = create_train_runtime(
        jax.random.PRNGKey(9),
        cfg,
        batch_size=1,
        total_steps=2,
    )
    batch = generate_toy_batch(jax.random.PRNGKey(10), 1, 4, cfg.vocab_size)
    original_wkv = runtime.rwkv_state[0].wkv
    state, metrics = train_batch(state, batch, runtime, carry_state=True)
    assert jnp.isfinite(metrics["loss"])
    assert not jnp.allclose(runtime.rwkv_state[0].wkv, original_wkv)
