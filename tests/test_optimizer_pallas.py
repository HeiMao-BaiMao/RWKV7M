import copy

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from rwkv7m.kernels import resolve_optimizer_backend
from rwkv7m.kernels.optimizer_pallas_gpu import (
    fused_adamw_optimizer,
    fused_adamw_update_gpu,
    fused_adamw_update_reference,
)
from rwkv7m.train.train_state import (
    decay_mask_fn,
    rwkv_w0_mask_fn,
    screening_mask_fn,
)
from rwkv7m.model.nnx_model import initialize_nnx_model
from rwkv7m.model.screened_rwkv import ModelConfig


@pytest.mark.parametrize("size", (1, 17, 257))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_fused_adamw_gpu_interpret_matches_reference(size, dtype):
    parameter = (jnp.arange(size, dtype=jnp.float32) * 0.01).astype(dtype)
    gradient = jnp.linspace(-0.2, 0.3, size, dtype=jnp.float32)
    mu = jnp.linspace(0.01, 0.02, size, dtype=jnp.float32)
    nu = jnp.linspace(0.02, 0.03, size, dtype=jnp.float32)
    arguments = (
        parameter,
        gradient,
        mu,
        nu,
        jnp.asarray(0.7, dtype=jnp.float32),
        jnp.asarray(1e-3, dtype=jnp.float32),
        jnp.asarray(0.19, dtype=jnp.float32),
        jnp.asarray(0.001999, dtype=jnp.float32),
    )
    options = dict(
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
        apply_weight_decay=True,
        learning_rate_multiplier=2.0,
    )
    expected = fused_adamw_update_reference(*arguments, **options)
    actual = fused_adamw_update_gpu(
        *arguments,
        **options,
        lowering="triton",
        interpret=True,
    )
    for left, right in zip(actual, expected, strict=True):
        assert jnp.allclose(left, right, rtol=2e-6, atol=2e-7)


def test_fused_adamw_transform_matches_optax_contract():
    params = {
        "att": {
            "w0": jnp.linspace(-0.1, 0.1, 17, dtype=jnp.float32),
            "a0": jnp.linspace(0.1, 0.2, 5, dtype=jnp.float32),
        },
        "dense": {
            "kernel": jnp.arange(33, dtype=jnp.float32).reshape(3, 11) / 50.0,
        },
    }
    gradients = jax.tree.map(
        lambda value: jnp.cos(value * 3.0) * 0.2,
        params,
    )
    learning_rate = 1e-3
    common = dict(
        max_grad_norm=0.5,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )
    fused = fused_adamw_optimizer(
        learning_rate=learning_rate,
        moment_dtype=jnp.float32,
        decay_mask_fn=decay_mask_fn,
        w0_mask_fn=rwkv_w0_mask_fn,
        lowering="triton",
        interpret=True,
        **common,
    )
    reference = optax.chain(
        optax.clip_by_global_norm(common["max_grad_norm"]),
        optax.adamw(
            learning_rate,
            weight_decay=common["weight_decay"],
            b1=common["beta1"],
            b2=common["beta2"],
            eps=common["epsilon"],
            mu_dtype=jnp.float32,
            mask=decay_mask_fn,
        ),
        optax.masked(optax.scale(2.0), rwkv_w0_mask_fn),
    )
    fused_state = fused.init(params)
    reference_state = reference.init(params)
    fused_params = copy.deepcopy(params)
    reference_params = copy.deepcopy(params)
    for _ in range(2):
        fused_updates, fused_state = fused.update(
            gradients, fused_state, fused_params
        )
        reference_updates, reference_state = reference.update(
            gradients, reference_state, reference_params
        )
        for left, right in zip(
            jax.tree.leaves(fused_updates),
            jax.tree.leaves(reference_updates),
            strict=True,
        ):
            assert jnp.allclose(left, right, rtol=2e-5, atol=2e-7)
        fused_params = optax.apply_updates(fused_params, fused_updates)
        reference_params = optax.apply_updates(
            reference_params, reference_updates
        )


def test_optimizer_backend_remains_optax_by_default():
    assert resolve_optimizer_backend() == "optax"
    assert resolve_optimizer_backend("pallas_gpu_triton") == "pallas_gpu_triton"
    with pytest.raises(ValueError, match="unknown optimizer backend"):
        resolve_optimizer_backend("cuda")


def test_fused_optimizer_accepts_nnx_parameter_state():
    config = ModelConfig(
        d_model=8,
        d_ffn=16,
        n_layers=1,
        n_heads=1,
        head_size=8,
        vocab_size=16,
        max_seq_len=4,
        dtype="float32",
        use_screening=False,
    )
    model = initialize_nnx_model(jax.random.key(9), config)
    params = nnx.state(model, nnx.Param)
    gradients = jax.tree.map(jnp.ones_like, params)
    optimizer = fused_adamw_optimizer(
        learning_rate=1e-3,
        max_grad_norm=1.0,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        moment_dtype=jnp.float32,
        decay_mask_fn=decay_mask_fn,
        w0_mask_fn=rwkv_w0_mask_fn,
        lowering="triton",
        interpret=True,
    )
    updates, state = optimizer.update(
        gradients,
        optimizer.init(params),
        params,
    )
    assert int(state.count) == 1
    assert all(jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(updates))


def test_fused_optimizer_freezes_screening_before_activation():
    params = {
        "layer_0": {
            "screening_0": {"kernel": jnp.ones((3,), dtype=jnp.float32)},
            "rwkv_block_0": {"kernel": jnp.ones((3,), dtype=jnp.float32)},
        }
    }
    gradients = jax.tree.map(jnp.ones_like, params)
    optimizer = fused_adamw_optimizer(
        learning_rate=1e-3,
        max_grad_norm=100.0,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        moment_dtype=jnp.float32,
        decay_mask_fn=decay_mask_fn,
        w0_mask_fn=rwkv_w0_mask_fn,
        screening_mask_fn=screening_mask_fn,
        screening_learning_rate_multiplier=0.1,
        screening_activation_step=2,
        lowering="triton",
        interpret=True,
    )
    updates, _ = optimizer.update(
        gradients,
        optimizer.init(params),
        params,
    )
    assert jnp.all(updates["layer_0"]["screening_0"]["kernel"] == 0.0)
    assert jnp.any(updates["layer_0"]["rwkv_block_0"]["kernel"] != 0.0)
