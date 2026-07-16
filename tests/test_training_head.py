import jax
import jax.numpy as jnp
import pytest

from rwkv7m.kernels import resolve_training_loss_backend
from rwkv7m.kernels.training_loss_pallas_gpu import (
    _compiler_params as training_loss_gpu_compiler_params,
    training_loss_tile_statistics_gpu,
)
from rwkv7m.kernels.screening_pallas_gpu import (
    ScreeningGPUConfig,
    _compiler_params as screening_gpu_compiler_params,
)
from rwkv7m.kernels.wkv_pallas_gpu import (
    WKVGPUConfig,
    _compiler_params as wkv_gpu_compiler_params,
)
from rwkv7m.kernels.training_loss_pallas_tpu import (
    training_loss_tile_statistics_tpu,
)
from rwkv7m.model.losses import (
    cross_entropy_components,
    l2wrap_components,
)
from rwkv7m.model.screened_rwkv import ModelConfig
from rwkv7m.model.training_head import (
    _reference_tile_statistics,
    tiled_training_loss_components,
)


def _inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(83), 2)
    hidden = (jax.random.normal(keys[0], (2, 3, 4)) * 0.2).astype(dtype)
    kernel = (jax.random.normal(keys[1], (4, 8)) * 0.2).astype(dtype)
    targets = jnp.asarray([[0, 3, 7], [6, 2, 5]], dtype=jnp.int32)
    mask = jnp.asarray([[1.0, 0.5, 1.0], [0.0, 1.0, 1.0]], dtype=jnp.float32)
    return hidden, kernel, targets, mask


def _full_components(hidden, kernel, targets, mask):
    logits = hidden @ kernel
    ce_total, ce_count = cross_entropy_components(logits, targets, mask)
    l2_total, l2_count = l2wrap_components(logits)
    return ce_total, ce_count, l2_total, l2_count


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_tiled_training_loss_matches_full_logits(dtype):
    inputs = _inputs(dtype)
    expected = _full_components(*inputs)
    actual = tiled_training_loss_components(
        *inputs,
        4,
        "reference",
    )
    for left, right in zip(actual, expected, strict=True):
        assert jnp.allclose(left, right, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.bfloat16))
def test_tiled_training_loss_custom_vjp_matches_full_logits(dtype):
    hidden, kernel, targets, mask = _inputs(dtype)

    def objective(function, active_hidden, active_kernel, active_mask):
        ce_total, ce_count, l2_total, l2_count = function(
            active_hidden,
            active_kernel,
            targets,
            active_mask,
        )
        return ce_total / ce_count + l2_total / l2_count

    expected = jax.grad(
        lambda active_hidden, active_kernel, active_mask: objective(
            _full_components,
            active_hidden,
            active_kernel,
            active_mask,
        ),
        argnums=(0, 1, 2),
    )(hidden, kernel, mask)
    actual = jax.grad(
        lambda active_hidden, active_kernel, active_mask: objective(
            lambda *values: tiled_training_loss_components(
                *values,
                4,
                "reference",
            ),
            active_hidden,
            active_kernel,
            active_mask,
        ),
        argnums=(0, 1, 2),
    )(hidden, kernel, mask)
    for left, right in zip(actual, expected, strict=True):
        assert jnp.allclose(
            left.astype(jnp.float32),
            right.astype(jnp.float32),
            rtol=2e-2,
            atol=2e-2,
        )


def test_backend_specific_pallas_tile_statistics_match_reference_interpret():
    _, _, targets, _ = _inputs()
    logits = jnp.arange(48, dtype=jnp.float32).reshape(6, 8) / 17.0
    flat_targets = targets.reshape(-1)
    expected = _reference_tile_statistics(logits, flat_targets, tile_start=0)
    actuals = (
        training_loss_tile_statistics_tpu(
            logits, flat_targets, tile_start=0, interpret=True
        ),
        training_loss_tile_statistics_gpu(
            logits,
            flat_targets,
            tile_start=0,
            lowering="triton",
            interpret=True,
        ),
        training_loss_tile_statistics_gpu(
            logits,
            flat_targets,
            tile_start=0,
            lowering="mosaic",
            interpret=True,
        ),
    )
    for actual in actuals:
        for left, right in zip(actual, expected, strict=True):
            assert jnp.allclose(left, right, rtol=1e-6, atol=1e-6)


def test_training_loss_backend_policy_and_config_validation():
    assert resolve_training_loss_backend(platform="cpu") == "reference"
    assert resolve_training_loss_backend(platform="tpu") == "pallas_tpu"
    assert (
        resolve_training_loss_backend(platform="gpu", device_kind="L40S")
        == "pallas_gpu_triton"
    )
    assert (
        resolve_training_loss_backend(platform="gpu", device_kind="B200")
        == "pallas_gpu_mosaic"
    )
    with pytest.raises(ValueError, match="training_vocab_tile_size"):
        ModelConfig(
            d_model=32,
            n_heads=1,
            head_size=32,
            training_vocab_tile_size=0,
            use_screening=False,
        )
    with pytest.raises(ValueError, match="must be divisible"):
        ModelConfig(
            d_model=32,
            n_heads=1,
            head_size=32,
            vocab_size=130,
            training_vocab_tile_size=64,
            use_screening=False,
        )


def test_mosaic_lowering_uses_mosaic_compiler_parameters():
    from jax.experimental.pallas import mosaic_gpu as plgpu

    assert isinstance(
        training_loss_gpu_compiler_params("mosaic", interpret=False),
        plgpu.CompilerParams,
    )
    assert isinstance(
        screening_gpu_compiler_params(
            "mosaic", ScreeningGPUConfig(), interpret=False
        ),
        plgpu.CompilerParams,
    )
    assert isinstance(
        wkv_gpu_compiler_params("mosaic", WKVGPUConfig(), interpret=False),
        plgpu.CompilerParams,
    )
