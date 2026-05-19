"""Test script for bfloat16 (bf16) compatibility.

Run this on a machine with bf16 support (most modern CPUs and all GPUs).
Expected: all assertions pass and loss remains finite.
"""

import jax
import jax.numpy as jnp
from rwkv7m.model.screening import ScreeningConfig
from rwkv7m.model.screened_rwkv import ModelConfig, ScreenedRWKVModel, init_rwkv_state, create_model_variables
from rwkv7m.model.state import init_screen_state
from rwkv7m.train.train_loop import generate_toy_batch
from rwkv7m.train.train_step import train_step
from rwkv7m.train.train_state import TrainState, create_optimizer


def test_forward_bf16():
    print("=== Testing bf16 forward pass ===")
    key = jax.random.PRNGKey(42)
    cfg = ModelConfig(
        d_model=128,
        d_ffn=256,
        n_layers=4,
        n_heads=4,
        head_size=32,
        vocab_size=256,
        max_seq_len=64,
        dtype="bfloat16",
        use_screening=True,
        screening=ScreeningConfig(
            d_model=128,
            d_slot=64,
            d_k=32,
            d_v=32,
            n_slots=4,
            screened_layers=(1,),
            bank_ids=(0, 0, 1, 2),
        ),
    )
    model = ScreenedRWKVModel(config=cfg)
    rwkv_state = init_rwkv_state(2, cfg)
    screen_state = init_screen_state(2, cfg.screening)
    key, subkey = jax.random.split(key)
    variables, _ = create_model_variables(subkey, cfg, 2)

    input_ids = jax.random.randint(subkey, (2, 16), 0, cfg.vocab_size)
    logits, _, _, stats = model.apply(
        variables,
        input_ids,
        rwkv_state,
        screen_state,
        phase="read_screening_only",
        deterministic=True,
    )

    # Note: logits are typically float32 even with bf16 model for stability
    assert jnp.all(jnp.isfinite(logits)), "Logits contain NaN/Inf"
    print(f"  logits dtype: {logits.dtype}, shape: {logits.shape}")
    print(f"  stats: {list(stats.keys())}")
    print("  PASS")


def test_backward_bf16():
    print("=== Testing bf16 backward pass ===")
    key = jax.random.PRNGKey(43)
    cfg = ModelConfig(
        d_model=128,
        d_ffn=256,
        n_layers=4,
        n_heads=4,
        head_size=32,
        vocab_size=256,
        max_seq_len=64,
        dtype="bfloat16",
        use_screening=True,
        screening=ScreeningConfig(
            d_model=128,
            d_slot=64,
            d_k=32,
            d_v=32,
            n_slots=4,
            screened_layers=(1,),
            bank_ids=(0, 0, 1, 2),
        ),
    )
    model = ScreenedRWKVModel(config=cfg)
    rwkv_state = init_rwkv_state(2, cfg)
    screen_state = init_screen_state(2, cfg.screening)
    key, subkey = jax.random.split(key)
    variables, _ = create_model_variables(subkey, cfg, 2)

    opt_config = {
        "lr_init": 1e-3,
        "lr_final": 1e-5,
        "warmup_steps": 10,
        "max_grad_norm": 1.0,
        "weight_decay": 0.001,
    }
    tx = create_optimizer(opt_config, total_steps=20)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    batch = generate_toy_batch(subkey, 2, 8, cfg.vocab_size)
    rwkv_state = init_rwkv_state(2, cfg)
    screen_state = init_screen_state(2, cfg.screening)

    train_state, rwkv_state, screen_state, metrics = train_step(
        train_state, batch, rwkv_state, screen_state, phase="read_screening_only"
    )

    assert jnp.isfinite(metrics["loss"]), f"Loss is not finite: {metrics['loss']}"
    print(f"  loss: {float(metrics['loss']):.4f}")
    print("  PASS")


def test_mixed_precision_state():
    print("=== Testing bf16 model with float32 state ===")
    key = jax.random.PRNGKey(44)
    cfg = ModelConfig(
        d_model=64,
        d_ffn=128,
        n_layers=2,
        n_heads=2,
        head_size=32,
        vocab_size=256,
        max_seq_len=32,
        dtype="bfloat16",
        use_screening=True,
        screening=ScreeningConfig(
            d_model=64,
            d_slot=32,
            d_k=16,
            d_v=16,
            n_slots=4,
            screened_layers=(0,),
            bank_ids=(0, 0, 1, 2),
        ),
    )
    key, subkey = jax.random.split(key)
    variables, model = create_model_variables(subkey, cfg, 2)

    # Verify slot state is float32 (important for numerical stability)
    screen_state = init_screen_state(2, cfg.screening)
    assert screen_state.layers[0].slots.dtype == jnp.float32
    print(f"  slot dtype: {screen_state.layers[0].slots.dtype}")
    print("  PASS")


def test_scan_consistency_bf16():
    print("=== Testing scan consistency in bf16 ===")
    key = jax.random.PRNGKey(45)
    cfg = ModelConfig(
        d_model=64,
        d_ffn=128,
        n_layers=2,
        n_heads=2,
        head_size=32,
        vocab_size=256,
        max_seq_len=16,
        dtype="bfloat16",
        use_screening=False,
        screening=ScreeningConfig(),
    )
    key, subkey = jax.random.split(key)
    variables, model = create_model_variables(subkey, cfg, 2)

    T = 4
    input_ids = jax.random.randint(subkey, (2, T), 0, cfg.vocab_size)
    rwkv_state = init_rwkv_state(2, cfg)
    screen_state = init_screen_state(2, cfg.screening)

    # Prefill all T tokens
    logits_prefill, _, _, _ = model.apply(
        variables, input_ids, rwkv_state, screen_state,
        phase="read_screening_only", deterministic=True,
    )

    # Decode one step at a time
    rwkv_state = init_rwkv_state(2, cfg)
    screen_state = init_screen_state(2, cfg.screening)
    logits_steps = []
    for t in range(T):
        tok = input_ids[:, t:t + 1]
        logits_t, rwkv_state, screen_state, _ = model.apply(
            variables, tok, rwkv_state, screen_state,
            phase="read_screening_only", deterministic=True,
        )
        logits_steps.append(logits_t)

    logits_stepwise = jnp.concatenate(logits_steps, axis=1)
    max_diff = jnp.max(jnp.abs(logits_prefill.astype(jnp.float32) - logits_stepwise.astype(jnp.float32)))
    print(f"  max abs diff: {float(max_diff):.6f}")
    assert max_diff < 1e-2, f"Scan consistency failed: {max_diff}"
    print("  PASS")


if __name__ == "__main__":
    test_forward_bf16()
    test_backward_bf16()
    test_mixed_precision_state()
    test_scan_consistency_bf16()
    print("\n=== ALL BF16 TESTS PASSED ===")
