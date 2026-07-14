"""Real-accelerator smoke tests; skipped by the normal CPU test suite."""

import jax
import jax.numpy as jnp
import pytest

from rwkv7m.kernels import resolve_wkv_backend
from rwkv7m.model.wkv import wkv7, wkv7_reference


pytestmark = pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason="requires a real TPU or NVIDIA GPU Pallas lowering",
)


def _accelerator_inputs():
    keys = jax.random.split(jax.random.key(23), 7)
    vectors = tuple(
        (jax.random.normal(key, (3, 1, 1, 128)) * 0.03).astype(
            jnp.bfloat16
        )
        for key in keys[:6]
    )
    state = (
        jax.random.normal(keys[6], (1, 1, 128, 128)) * 0.03
    ).astype(jnp.float32)
    return (*vectors, state)


def test_real_accelerator_pallas_forward_and_backward_match_reference():
    backend = resolve_wkv_backend()
    assert backend.startswith("pallas_")
    inputs = _accelerator_inputs()

    expected_outputs = jax.jit(wkv7_reference)(*inputs)
    actual_outputs = jax.jit(wkv7)(*inputs)
    jax.block_until_ready(actual_outputs)
    assert jnp.allclose(
        actual_outputs[0], expected_outputs[0], rtol=2e-2, atol=2e-2
    )
    assert jnp.allclose(
        actual_outputs[1], expected_outputs[1], rtol=2e-4, atol=2e-4
    )

    def loss(fn, *values):
        y, final_state = fn(*values)
        return jnp.sum(y.astype(jnp.float32) ** 2) + 0.01 * jnp.sum(
            final_state**2
        )

    argnums = tuple(range(7))
    expected_gradients = jax.jit(
        jax.grad(
            lambda *values: loss(wkv7_reference, *values),
            argnums=argnums,
        )
    )(*inputs)
    actual_gradients = jax.jit(
        jax.grad(
            lambda *values: loss(wkv7, *values),
            argnums=argnums,
        )
    )(*inputs)
    jax.block_until_ready(actual_gradients)
    assert all(
        jnp.allclose(left, right, rtol=3e-2, atol=3e-2)
        for left, right in zip(
            actual_gradients, expected_gradients, strict=True
        )
    )
