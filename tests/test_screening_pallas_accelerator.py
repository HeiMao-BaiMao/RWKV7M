"""Real-accelerator screening tests skipped by the CPU suite."""

import jax
import jax.numpy as jnp
import pytest

from rwkv7m.kernels.screening_backend import resolve_screening_backend
from rwkv7m.model.screening import unit_norm
from rwkv7m.model.screening_recurrence import (
    ScreeningRecurrenceConfig,
    screening_recurrence,
    screening_recurrence_reference,
)


pytestmark = pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason="requires a real TPU or NVIDIA GPU Pallas lowering",
)


def _config():
    return ScreeningRecurrenceConfig(
        write_enabled=True,
        use_value_unit_norm=True,
        use_leaky_warmup=True,
        leaky_alpha=0.1,
        leaky_gamma=8.0,
        use_age_mask=True,
        age_ref=16.0,
        age_sigma=8.0,
        write_rel_floor=0.01,
        usage_ema_decay=0.99,
        tanh_norm_cap=1.0,
        eps=1e-6,
    )


def _inputs():
    time, batch, slots = 4, 1, 4
    key_size, slot_size, value_size = 8, 8, 8
    keys = jax.random.split(jax.random.key(31), 15)

    def normal(index, shape, scale=0.1, dtype=jnp.float32):
        return (jax.random.normal(keys[index], shape) * scale).astype(dtype)

    q_read = unit_norm(normal(0, (time, batch, key_size)))
    q_write = unit_norm(normal(1, (time, batch, key_size)))
    return (
        q_read,
        q_write,
        normal(2, (time, batch, slots, slot_size), dtype=jnp.bfloat16),
        normal(3, (time, batch, slots, key_size), dtype=jnp.bfloat16),
        normal(4, (time, batch, slots, value_size), dtype=jnp.bfloat16),
        normal(5, (time, batch, slots, key_size), dtype=jnp.bfloat16),
        normal(6, (batch, slots, slot_size)),
        normal(7, (batch, slots, key_size), dtype=jnp.bfloat16),
        normal(8, (batch, slots, value_size), dtype=jnp.bfloat16),
        normal(9, (batch, slots, key_size), dtype=jnp.bfloat16),
        jnp.arange(slots, dtype=jnp.float32)[None, :],
        jnp.linspace(0.0, 0.3, slots, dtype=jnp.float32)[None, :],
        jnp.linspace(0.05, 0.2, slots, dtype=jnp.float32),
        jnp.asarray(-0.4, dtype=jnp.float32),
        jnp.asarray(-0.35, dtype=jnp.float32),
    )


def _loss(function, config, *inputs):
    u, slots, ages, usage, _, _ = function(*inputs, config)
    return (
        jnp.sum(u.astype(jnp.float32) ** 2)
        + 0.01 * jnp.sum(slots**2)
        + 0.001 * jnp.sum(ages**2)
        + 0.01 * jnp.sum(usage**2)
    )


def test_real_accelerator_screening_forward_and_backward_match_reference():
    backend = resolve_screening_backend()
    assert backend.startswith("pallas_")
    inputs = _inputs()
    config = _config()

    expected_outputs = jax.jit(
        lambda *values: screening_recurrence_reference(*values, config)
    )(*inputs)
    actual_outputs = jax.jit(
        lambda *values: screening_recurrence(*values, config)
    )(*inputs)
    jax.block_until_ready((expected_outputs, actual_outputs))

    assert actual_outputs[0].dtype == jnp.bfloat16
    output_tolerances = (
        (3e-2, 3e-2),
        (3e-4, 3e-4),
        (1e-5, 1e-5),
        (3e-4, 3e-4),
        (3e-4, 3e-4),
        (3e-4, 3e-4),
    )
    for actual, expected, (rtol, atol) in zip(
        actual_outputs, expected_outputs, output_tolerances, strict=True
    ):
        assert jnp.allclose(actual, expected, rtol=rtol, atol=atol)

    argnums = tuple(range(15))
    expected_gradients = jax.jit(
        jax.grad(
            lambda *values: _loss(
                screening_recurrence_reference, config, *values
            ),
            argnums=argnums,
        )
    )(*inputs)
    actual_gradients = jax.jit(
        jax.grad(
            lambda *values: _loss(screening_recurrence, config, *values),
            argnums=argnums,
        )
    )(*inputs)
    jax.block_until_ready((expected_gradients, actual_gradients))

    for actual, expected in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert jnp.allclose(
            actual.astype(jnp.float32),
            expected.astype(jnp.float32),
            rtol=5e-2,
            atol=5e-2,
        )
