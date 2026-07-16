"""GPU Pallas reduction for one vocabulary-logit tile."""

from __future__ import annotations

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


GPULowering = Literal["mosaic", "triton"]


def _tile_statistics_kernel(
    logits_ref,
    targets_ref,
    maximum_ref,
    exponential_sum_ref,
    target_logit_ref,
    maximum_count_ref,
    *,
    tile_start: int,
    tile_size: int,
):
    logits = logits_ref[:][0, :].astype(jnp.float32)
    target = targets_ref[:][0]
    tile_maximum = jnp.max(logits, keepdims=True)
    exponential_sum = jnp.sum(
        jnp.exp(logits - tile_maximum), keepdims=True
    )
    vocab_indices = jnp.arange(tile_size, dtype=target.dtype)
    target_mask = vocab_indices + tile_start == target
    target_logit = jnp.sum(jnp.where(target_mask, logits, 0.0), keepdims=True)
    maximum_count = jnp.sum(logits == tile_maximum, keepdims=True).astype(
        jnp.float32
    )
    maximum_ref[:] = tile_maximum
    exponential_sum_ref[:] = exponential_sum
    target_logit_ref[:] = target_logit
    maximum_count_ref[:] = maximum_count


def _compiler_params(lowering: GPULowering, *, interpret: bool):
    if lowering == "mosaic" and not interpret:
        from jax.experimental.pallas import mosaic_gpu as plgpu

        return plgpu.CompilerParams(
            dimension_semantics=("parallel",),
            reduction_scratch_bytes=6144,
        )
    if lowering == "triton" and not interpret:
        from jax.experimental.pallas import triton as pltriton

        return pltriton.CompilerParams(num_warps=4, num_stages=2)
    return None


def training_loss_tile_statistics_gpu(
    logits,
    targets,
    *,
    tile_start: int,
    lowering: GPULowering,
    interpret: bool = False,
):
    positions, tile_size = logits.shape
    logits_spec = pl.BlockSpec(
        (1, tile_size), lambda position: (position, 0)
    )
    scalar_spec = pl.BlockSpec((1,), lambda position: (position,))
    output_shape = jax.ShapeDtypeStruct((positions,), jnp.float32)
    return pl.pallas_call(
        partial(
            _tile_statistics_kernel,
            tile_start=tile_start,
            tile_size=tile_size,
        ),
        out_shape=(output_shape, output_shape, output_shape, output_shape),
        grid=(positions,),
        in_specs=(logits_spec, scalar_spec),
        out_specs=(scalar_spec, scalar_spec, scalar_spec, scalar_spec),
        interpret=interpret,
        compiler_params=_compiler_params(lowering, interpret=interpret),
        name=f"rwkv7_training_loss_gpu_{lowering}_tile_statistics",
    )(logits, targets)


__all__ = ["training_loss_tile_statistics_gpu"]
