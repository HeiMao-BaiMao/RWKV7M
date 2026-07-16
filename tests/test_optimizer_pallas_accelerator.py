"""Real-NVIDIA-GPU fused-optimizer lowering test skipped by the CPU suite."""

import jax
import jax.numpy as jnp
import pytest

from rwkv7m.kernels.optimizer_pallas_gpu import (
    fused_adamw_update_gpu,
    fused_adamw_update_reference,
)


pytestmark = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="requires a real NVIDIA GPU Pallas lowering",
)


def test_real_gpu_fused_adamw_matches_reference():
    kind = str(jax.devices()[0].device_kind).lower()
    lowering = "mosaic" if any(
        marker in kind for marker in ("h100", "h200", "b100", "b200", "blackwell")
    ) else "triton"
    parameter = jnp.linspace(-0.5, 0.5, 1025, dtype=jnp.bfloat16)
    gradient = jnp.sin(jnp.arange(1025, dtype=jnp.float32)) * 0.1
    mu = jnp.zeros((1025,), dtype=jnp.float32)
    nu = jnp.zeros((1025,), dtype=jnp.float32)
    arguments = (parameter, gradient, mu, nu, 0.8, 1e-3, 0.1, 0.001)
    options = dict(
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        apply_weight_decay=True,
        learning_rate_multiplier=1.0,
    )
    expected = jax.jit(
        lambda *values: fused_adamw_update_reference(*values, **options)
    )(*arguments)
    actual = jax.jit(
        lambda *values: fused_adamw_update_gpu(
            *values,
            **options,
            lowering=lowering,
        )
    )(*arguments)
    jax.block_until_ready((expected, actual))
    for left, right in zip(actual, expected, strict=True):
        assert jnp.allclose(left, right, rtol=2e-5, atol=2e-6)
