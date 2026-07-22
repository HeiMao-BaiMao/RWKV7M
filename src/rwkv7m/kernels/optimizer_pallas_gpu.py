"""Fused AdamW parameter-shard update for NVIDIA GPUs.

Global-norm reduction remains a single JAX tree reduction.  For each parameter
shard this kernel fuses gradient clipping, both Adam moments, bias correction,
decoupled weight decay, and the RWKV ``w0`` learning-rate multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import optax


GPULowering = Literal["mosaic", "triton"]


@dataclass(frozen=True)
class FusedAdamWGPUConfig:
    block_size: int = 256
    num_warps: int = 4
    num_stages: int = 1


class FusedAdamWState(NamedTuple):
    count: jax.Array
    mu: object
    nu: object


def _validate_config(config: FusedAdamWGPUConfig) -> None:
    if config.block_size <= 0 or config.block_size & (config.block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if config.num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a supported power of two")
    if config.num_stages <= 0:
        raise ValueError("num_stages must be positive")


def _fused_adamw_kernel(
    parameter_ref,
    gradient_ref,
    mu_ref,
    nu_ref,
    clip_scale_ref,
    learning_rate_ref,
    mu_correction_ref,
    nu_correction_ref,
    update_ref,
    next_mu_ref,
    next_nu_ref,
    *,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    apply_weight_decay: bool,
    learning_rate_multiplier: float,
):
    parameter = parameter_ref[:].astype(jnp.float32)
    gradient = gradient_ref[:].astype(jnp.float32) * clip_scale_ref[0]
    mu = mu_ref[:].astype(jnp.float32)
    nu = nu_ref[:].astype(jnp.float32)
    next_mu = beta1 * mu + (1.0 - beta1) * gradient
    next_nu = beta2 * nu + (1.0 - beta2) * jnp.square(gradient)
    corrected_mu = next_mu / mu_correction_ref[0]
    corrected_nu = next_nu / nu_correction_ref[0]
    direction = corrected_mu / (jnp.sqrt(corrected_nu) + epsilon)
    if apply_weight_decay:
        direction += weight_decay * parameter
    update = (
        -learning_rate_multiplier * learning_rate_ref[0] * direction
    )
    update_ref[:] = update.astype(update_ref.dtype)
    next_mu_ref[:] = next_mu.astype(next_mu_ref.dtype)
    next_nu_ref[:] = next_nu.astype(next_nu_ref.dtype)


def _compiler_params(
    lowering: GPULowering,
    config: FusedAdamWGPUConfig,
    *,
    interpret: bool,
):
    if lowering == "mosaic" and not interpret:
        from jax.experimental.pallas import mosaic_gpu as plgpu

        return plgpu.CompilerParams(dimension_semantics=("parallel",))
    if lowering == "triton" and not interpret:
        from jax.experimental.pallas import triton as pltriton

        return pltriton.CompilerParams(
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    return None


def fused_adamw_update_reference(
    parameter,
    gradient,
    mu,
    nu,
    clip_scale,
    learning_rate,
    mu_correction,
    nu_correction,
    *,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    apply_weight_decay: bool,
    learning_rate_multiplier: float,
):
    parameter_f32 = parameter.astype(jnp.float32)
    gradient_f32 = gradient.astype(jnp.float32) * clip_scale
    next_mu = beta1 * mu.astype(jnp.float32) + (1.0 - beta1) * gradient_f32
    next_nu = beta2 * nu.astype(jnp.float32) + (1.0 - beta2) * jnp.square(
        gradient_f32
    )
    direction = (next_mu / mu_correction) / (
        jnp.sqrt(next_nu / nu_correction) + epsilon
    )
    if apply_weight_decay:
        direction += weight_decay * parameter_f32
    update = -learning_rate_multiplier * learning_rate * direction
    return update.astype(jnp.float32), next_mu.astype(mu.dtype), next_nu.astype(
        nu.dtype
    )


def fused_adamw_update_gpu(
    parameter,
    gradient,
    mu,
    nu,
    clip_scale,
    learning_rate,
    mu_correction,
    nu_correction,
    *,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    apply_weight_decay: bool,
    learning_rate_multiplier: float,
    lowering: GPULowering,
    config: FusedAdamWGPUConfig = FusedAdamWGPUConfig(),
    interpret: bool = False,
):
    _validate_config(config)
    original_shape = parameter.shape
    parameter = jnp.reshape(parameter, (-1,))
    gradient = jnp.reshape(gradient, (-1,))
    mu = jnp.reshape(mu, (-1,))
    nu = jnp.reshape(nu, (-1,))
    element_count = int(parameter.size)
    # Keep a 16-byte-aligned block even for scalar parameter leaves. Pallas
    # masks the padded tail, and Mosaic GPU requires aligned minor dimensions.
    block_size = config.block_size
    grid_size = (element_count + block_size - 1) // block_size
    vector_spec = pl.BlockSpec((block_size,), lambda block: (block,))
    scalar_spec = pl.BlockSpec((1,), lambda block: (0,))
    scalar_inputs = tuple(
        jnp.reshape(jnp.asarray(value, dtype=jnp.float32), (1,))
        for value in (
            clip_scale,
            learning_rate,
            mu_correction,
            nu_correction,
        )
    )
    output_shapes = (
        jax.ShapeDtypeStruct((element_count,), jnp.float32),
        jax.ShapeDtypeStruct((element_count,), mu.dtype),
        jax.ShapeDtypeStruct((element_count,), nu.dtype),
    )
    outputs = pl.pallas_call(
        partial(
            _fused_adamw_kernel,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            apply_weight_decay=apply_weight_decay,
            learning_rate_multiplier=learning_rate_multiplier,
        ),
        out_shape=output_shapes,
        grid=(grid_size,),
        in_specs=(
            vector_spec,
            vector_spec,
            vector_spec,
            vector_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
            scalar_spec,
        ),
        out_specs=(vector_spec, vector_spec, vector_spec),
        interpret=interpret,
        compiler_params=_compiler_params(lowering, config, interpret=interpret),
        name=f"rwkv7_fused_adamw_gpu_{lowering}",
    )(parameter, gradient, mu, nu, *scalar_inputs)
    return tuple(jnp.reshape(value, original_shape) for value in outputs)


def fused_adamw_optimizer(
    *,
    learning_rate,
    max_grad_norm: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    moment_dtype,
    decay_mask_fn,
    w0_mask_fn,
    screening_mask_fn=None,
    screening_learning_rate_multiplier: float = 1.0,
    screening_activation_step: int = 0,
    screening_activation_warmup_steps: int = 0,
    lowering: GPULowering,
    config: FusedAdamWGPUConfig = FusedAdamWGPUConfig(),
    interpret: bool = False,
):
    """Build an Optax-compatible transformation backed by the fused kernel."""

    schedule = learning_rate if callable(learning_rate) else lambda _: learning_rate
    moment_dtype = jnp.dtype(moment_dtype)

    def init_fn(params):
        mu = jax.tree.map(lambda value: jnp.zeros_like(value, dtype=moment_dtype), params)
        nu = jax.tree.map(lambda value: jnp.zeros_like(value, dtype=moment_dtype), params)
        return FusedAdamWState(jnp.zeros([], dtype=jnp.int32), mu, nu)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("fused AdamW requires current parameters")
        count = optax.safe_increment(state.count)
        count_f32 = count.astype(jnp.float32)
        learning_rate_value = jnp.asarray(schedule(state.count), dtype=jnp.float32)
        screening_progress = state.count.astype(jnp.float32) - float(
            screening_activation_step
        )
        if screening_activation_warmup_steps > 0:
            screening_activation = jnp.clip(
                screening_progress
                / float(screening_activation_warmup_steps),
                0.0,
                1.0,
            )
        else:
            screening_activation = (
                screening_progress >= 0.0
            ).astype(jnp.float32)
        mu_correction = 1.0 - jnp.power(beta1, count_f32)
        nu_correction = 1.0 - jnp.power(beta2, count_f32)
        global_norm = optax.tree.norm(updates).astype(jnp.float32)
        clip_scale = jnp.where(
            global_norm > max_grad_norm,
            jnp.asarray(max_grad_norm, dtype=jnp.float32) / global_norm,
            jnp.ones((), dtype=jnp.float32),
        )
        tree_def = jax.tree.structure(params)
        parameter_leaves = jax.tree.leaves(params)
        gradient_leaves = jax.tree.leaves(updates)
        mu_leaves = jax.tree.leaves(state.mu)
        nu_leaves = jax.tree.leaves(state.nu)
        decay_leaves = jax.tree.leaves(decay_mask_fn(params))
        w0_leaves = jax.tree.leaves(w0_mask_fn(params))
        screening_leaves = jax.tree.leaves(
            jax.tree.map(lambda _: False, params)
            if screening_mask_fn is None
            else screening_mask_fn(params)
        )
        results = [
            fused_adamw_update_gpu(
                parameter,
                gradient,
                mu,
                nu,
                clip_scale,
                (
                    learning_rate_value
                    * screening_learning_rate_multiplier
                    * screening_activation
                    if bool(is_screening)
                    else learning_rate_value
                ),
                mu_correction,
                nu_correction,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                weight_decay=weight_decay,
                apply_weight_decay=bool(apply_decay),
                learning_rate_multiplier=(
                    (2.0 if bool(is_w0) else 1.0)
                ),
                lowering=lowering,
                config=config,
                interpret=interpret,
            )
            for (
                parameter,
                gradient,
                mu,
                nu,
                apply_decay,
                is_w0,
                is_screening,
            ) in zip(
                parameter_leaves,
                gradient_leaves,
                mu_leaves,
                nu_leaves,
                decay_leaves,
                w0_leaves,
                screening_leaves,
                strict=True,
            )
        ]
        transformed = tree_def.unflatten([result[0] for result in results])
        next_mu = tree_def.unflatten([result[1] for result in results])
        next_nu = tree_def.unflatten([result[2] for result in results])
        return transformed, FusedAdamWState(count, next_mu, next_nu)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


__all__ = [
    "FusedAdamWGPUConfig",
    "FusedAdamWState",
    "fused_adamw_optimizer",
    "fused_adamw_update_gpu",
    "fused_adamw_update_reference",
]
