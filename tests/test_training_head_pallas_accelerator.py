"""Real-accelerator training-head test skipped by the CPU suite."""

import jax
import jax.numpy as jnp
import pytest

from rwkv7m.kernels import resolve_training_loss_backend
from rwkv7m.model.losses import cross_entropy_components, l2wrap_components
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.screened_rwkv import ModelConfig
from rwkv7m.model.training_head import tiled_training_loss_components


pytestmark = pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason="requires a real TPU or NVIDIA GPU Pallas lowering",
)


def test_real_accelerator_tiled_head_forward_and_backward_match_reference():
    backend = resolve_training_loss_backend()
    assert backend.startswith("pallas_")
    hidden_key, kernel_key = jax.random.split(jax.random.key(107))
    hidden = (
        jax.random.normal(hidden_key, (2, 4, 16)) * 0.03
    ).astype(jnp.bfloat16)
    kernel = (
        jax.random.normal(kernel_key, (16, 128)) * 0.03
    ).astype(jnp.bfloat16)
    targets = jnp.arange(8, dtype=jnp.int32).reshape(2, 4) * 7
    mask = jnp.asarray(
        [[1.0, 1.0, 0.5, 1.0], [0.0, 1.0, 1.0, 1.0]],
        dtype=jnp.float32,
    )

    def full_components(active_hidden, active_kernel):
        logits = active_hidden @ active_kernel
        ce_total, ce_count = cross_entropy_components(logits, targets, mask)
        l2_total, l2_count = l2wrap_components(logits)
        return ce_total, ce_count, l2_total, l2_count

    def tiled_components(active_hidden, active_kernel):
        return tiled_training_loss_components(
            active_hidden,
            active_kernel,
            targets,
            mask,
            64,
            backend,
        )

    def objective(function, active_hidden, active_kernel):
        ce_total, ce_count, l2_total, l2_count = function(
            active_hidden, active_kernel
        )
        return ce_total / ce_count + l2_total / l2_count

    expected_outputs = jax.jit(full_components)(hidden, kernel)
    actual_outputs = jax.jit(tiled_components)(hidden, kernel)
    expected_gradients = jax.jit(
        jax.grad(
            lambda active_hidden, active_kernel: objective(
                full_components, active_hidden, active_kernel
            ),
            argnums=(0, 1),
        )
    )(hidden, kernel)
    actual_gradients = jax.jit(
        jax.grad(
            lambda active_hidden, active_kernel: objective(
                tiled_components, active_hidden, active_kernel
            ),
            argnums=(0, 1),
        )
    )(hidden, kernel)
    jax.block_until_ready(
        (expected_outputs, actual_outputs, expected_gradients, actual_gradients)
    )

    for actual, expected in zip(
        actual_outputs, expected_outputs, strict=True
    ):
        assert jnp.allclose(actual, expected, rtol=2e-3, atol=2e-3)
    for actual, expected in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert jnp.allclose(
            actual.astype(jnp.float32),
            expected.astype(jnp.float32),
            rtol=3e-2,
            atol=3e-2,
        )


def test_real_accelerator_model_training_head_uses_tiled_contract():
    config = ModelConfig(
        d_model=16,
        d_ffn=32,
        n_layers=1,
        n_heads=1,
        head_size=16,
        vocab_size=128,
        max_seq_len=4,
        dtype="bfloat16",
        param_dtype="bfloat16",
        training_vocab_tile_size=64,
        use_screening=False,
    )
    model = initialize_nnx_model(jax.random.key(113), config)
    hidden = jnp.ones((1, 4, 16), dtype=jnp.bfloat16) * 0.02
    targets = jnp.asarray([[1, 17, 63, 127]], dtype=jnp.int32)

    def loss(active_hidden):
        components = model.compute_training_loss(active_hidden, targets)
        return (
            components["ce_total"] / components["ce_count"]
            + components["l2_total"] / components["l2_count"]
        )

    value, gradient = jax.jit(jax.value_and_grad(loss))(hidden)
    jax.block_until_ready((value, gradient))
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
