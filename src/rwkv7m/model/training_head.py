"""Vocabulary-tiled training head with backend-specific Pallas reductions."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from rwkv7m.kernels.training_loss_backend import (
    resolve_training_loss_backend,
)

from .losses import L2WRAP_FACTOR


def _validate_inputs(hidden, kernel, targets, mask, tile_size):
    if hidden.ndim < 2:
        raise ValueError("hidden must have at least one position and one feature axis")
    if kernel.ndim != 2 or kernel.shape[0] != hidden.shape[-1]:
        raise ValueError(
            "kernel must have shape [hidden_size, vocab_size], got "
            f"{kernel.shape} for hidden size {hidden.shape[-1]}"
        )
    if targets.shape != hidden.shape[:-1]:
        raise ValueError(
            f"targets must have shape {hidden.shape[:-1]}, got {targets.shape}"
        )
    if mask.shape != targets.shape:
        raise ValueError(f"mask must have shape {targets.shape}, got {mask.shape}")
    if not jnp.issubdtype(targets.dtype, jnp.integer):
        raise TypeError("targets must use an integer dtype")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    vocab_size = int(kernel.shape[1])
    if vocab_size <= 0:
        raise ValueError("kernel vocabulary axis must be non-empty")
    if vocab_size > tile_size and vocab_size % tile_size != 0:
        raise ValueError(
            "vocab_size must be divisible by tile_size when multiple tiles "
            f"are required, got vocab_size={vocab_size}, tile_size={tile_size}"
        )


def _reference_tile_statistics(logits, targets, *, tile_start):
    logits_f32 = logits.astype(jnp.float32)
    tile_maximum = jnp.max(logits_f32, axis=-1)
    exponential_sum = jnp.sum(
        jnp.exp(logits_f32 - tile_maximum[:, None]), axis=-1
    )
    indices = jnp.arange(logits.shape[-1], dtype=targets.dtype)[None, :]
    target_mask = indices + tile_start == targets[:, None]
    target_logit = jnp.sum(jnp.where(target_mask, logits_f32, 0.0), axis=-1)
    maximum_count = jnp.sum(
        logits_f32 == tile_maximum[:, None], axis=-1
    ).astype(jnp.float32)
    return tile_maximum, exponential_sum, target_logit, maximum_count


def _tile_statistics(
    logits,
    targets,
    *,
    tile_start,
    backend,
    interpret,
):
    selected = resolve_training_loss_backend(backend)
    if selected == "reference":
        return _reference_tile_statistics(
            logits, targets, tile_start=tile_start
        )
    if selected == "pallas_tpu":
        from rwkv7m.kernels.training_loss_pallas_tpu import (
            training_loss_tile_statistics_tpu,
        )

        return training_loss_tile_statistics_tpu(
            logits,
            targets,
            tile_start=tile_start,
            interpret=interpret,
        )
    from rwkv7m.kernels.training_loss_pallas_gpu import (
        training_loss_tile_statistics_gpu,
    )

    lowering = "mosaic" if selected == "pallas_gpu_mosaic" else "triton"
    return training_loss_tile_statistics_gpu(
        logits,
        targets,
        tile_start=tile_start,
        lowering=lowering,
        interpret=interpret,
    )


def _forward_impl(
    hidden,
    kernel,
    targets,
    mask,
    *,
    tile_size,
    backend,
    interpret,
    l2_factor,
):
    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_targets = targets.reshape((-1,))
    flat_mask = mask.reshape((-1,)).astype(jnp.float32)
    vocab_size = int(kernel.shape[1])
    effective_tile_size = min(tile_size, vocab_size)
    global_maximum = None
    global_exponential_sum = None
    global_target_logit = jnp.zeros(flat_targets.shape, dtype=jnp.float32)
    global_maximum_count = None

    for tile_start in range(0, vocab_size, effective_tile_size):
        tile_kernel = kernel[:, tile_start : tile_start + effective_tile_size]
        tile_logits = flat_hidden @ tile_kernel
        (
            tile_maximum,
            tile_exponential_sum,
            tile_target_logit,
            tile_maximum_count,
        ) = _tile_statistics(
            tile_logits,
            flat_targets,
            tile_start=tile_start,
            backend=backend,
            interpret=interpret,
        )
        global_target_logit += tile_target_logit
        if global_maximum is None:
            global_maximum = tile_maximum
            global_exponential_sum = tile_exponential_sum
            global_maximum_count = tile_maximum_count
            continue
        new_maximum = jnp.maximum(global_maximum, tile_maximum)
        global_exponential_sum = (
            global_exponential_sum
            * jnp.exp(global_maximum - new_maximum)
            + tile_exponential_sum * jnp.exp(tile_maximum - new_maximum)
        )
        greater = tile_maximum > global_maximum
        equal = tile_maximum == global_maximum
        global_maximum_count = jnp.where(
            greater,
            tile_maximum_count,
            jnp.where(
                equal,
                global_maximum_count + tile_maximum_count,
                global_maximum_count,
            ),
        )
        global_maximum = new_maximum

    negative_log_likelihood = (
        jnp.log(global_exponential_sum)
        + global_maximum
        - global_target_logit
    )
    outputs = (
        jnp.sum(negative_log_likelihood * flat_mask),
        jnp.sum(flat_mask),
        0.5 * l2_factor * jnp.sum(jnp.square(global_maximum)),
        jnp.asarray(flat_targets.size, dtype=jnp.float32),
    )
    residuals = (
        hidden,
        kernel,
        targets,
        mask,
        global_maximum,
        global_exponential_sum,
        global_target_logit,
        global_maximum_count,
    )
    return outputs, residuals


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def tiled_training_loss_components(
    hidden,
    kernel,
    targets,
    mask,
    tile_size: int,
    backend: str | None = None,
    interpret: bool = False,
    l2_factor: float = L2WRAP_FACTOR,
):
    """Return CE/L2 numerators and counts without a full vocabulary tensor."""

    _validate_inputs(hidden, kernel, targets, mask, tile_size)
    outputs, _ = _forward_impl(
        hidden,
        kernel,
        targets,
        mask,
        tile_size=tile_size,
        backend=backend,
        interpret=interpret,
        l2_factor=l2_factor,
    )
    return outputs


def _tiled_training_loss_fwd(
    hidden,
    kernel,
    targets,
    mask,
    tile_size,
    backend,
    interpret,
    l2_factor,
):
    _validate_inputs(hidden, kernel, targets, mask, tile_size)
    return _forward_impl(
        hidden,
        kernel,
        targets,
        mask,
        tile_size=tile_size,
        backend=backend,
        interpret=interpret,
        l2_factor=l2_factor,
    )


def _tiled_training_loss_bwd(
    tile_size,
    backend,
    interpret,
    l2_factor,
    residuals,
    cotangents,
):
    del backend, interpret
    (
        hidden,
        kernel,
        targets,
        mask,
        global_maximum,
        global_exponential_sum,
        global_target_logit,
        global_maximum_count,
    ) = residuals
    ce_total_cotangent, ce_count_cotangent, l2_total_cotangent, _ = cotangents
    flat_hidden = hidden.reshape((-1, hidden.shape[-1]))
    flat_targets = targets.reshape((-1,))
    flat_mask = mask.reshape((-1,)).astype(jnp.float32)
    vocab_size = int(kernel.shape[1])
    effective_tile_size = min(tile_size, vocab_size)
    hidden_gradient = jnp.zeros_like(flat_hidden, dtype=jnp.float32)
    kernel_gradients = []

    for tile_start in range(0, vocab_size, effective_tile_size):
        tile_kernel = kernel[:, tile_start : tile_start + effective_tile_size]
        tile_logits = flat_hidden @ tile_kernel
        tile_logits_f32 = tile_logits.astype(jnp.float32)
        probabilities = jnp.exp(
            tile_logits_f32 - global_maximum[:, None]
        ) / global_exponential_sum[:, None]
        indices = jnp.arange(
            tile_logits.shape[-1], dtype=flat_targets.dtype
        )[None, :]
        target_mask = indices + tile_start == flat_targets[:, None]
        logits_gradient = ce_total_cotangent * flat_mask[:, None] * (
            probabilities - target_mask.astype(jnp.float32)
        )
        maximum_mask = tile_logits_f32 == global_maximum[:, None]
        logits_gradient += (
            l2_total_cotangent
            * l2_factor
            * global_maximum[:, None]
            * maximum_mask.astype(jnp.float32)
            / global_maximum_count[:, None]
        )
        compute_gradient = logits_gradient.astype(tile_logits.dtype)
        hidden_gradient += (
            compute_gradient @ tile_kernel.T
        ).astype(jnp.float32)
        kernel_gradients.append(
            (flat_hidden.T @ compute_gradient).astype(kernel.dtype)
        )

    negative_log_likelihood = (
        jnp.log(global_exponential_sum)
        + global_maximum
        - global_target_logit
    )
    mask_gradient = (
        ce_total_cotangent * negative_log_likelihood
        + ce_count_cotangent
    ).reshape(mask.shape).astype(mask.dtype)
    return (
        hidden_gradient.reshape(hidden.shape).astype(hidden.dtype),
        jnp.concatenate(kernel_gradients, axis=-1),
        None,
        mask_gradient,
    )


tiled_training_loss_components.defvjp(
    _tiled_training_loss_fwd,
    _tiled_training_loss_bwd,
)


__all__ = ["tiled_training_loss_components"]
