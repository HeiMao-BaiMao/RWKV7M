"""TPU Pallas reduction for one vocabulary-logit tile."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


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
    logits = logits_ref[:][0, :, :].astype(jnp.float32)
    target = targets_ref[:][0, :, :]
    tile_maximum = jnp.max(logits, axis=-1, keepdims=True)
    exponential_sum = jnp.sum(
        jnp.exp(logits - tile_maximum), axis=-1, keepdims=True
    )
    vocab_indices = jnp.arange(tile_size, dtype=target.dtype)[None, :]
    target_mask = vocab_indices + tile_start == target
    target_logit = jnp.sum(
        jnp.where(target_mask, logits, 0.0), axis=-1, keepdims=True
    )
    maximum_count = jnp.sum(
        logits == tile_maximum, axis=-1, keepdims=True
    ).astype(jnp.float32)
    maximum_ref[:] = tile_maximum[None, :, :]
    exponential_sum_ref[:] = exponential_sum[None, :, :]
    target_logit_ref[:] = target_logit[None, :, :]
    maximum_count_ref[:] = maximum_count[None, :, :]


def training_loss_tile_statistics_tpu(
    logits,
    targets,
    *,
    tile_start: int,
    interpret: bool = False,
):
    """Reduce ``[positions, vocab_tile]`` logits without rank-one TPU values."""

    positions, tile_size = logits.shape
    physical_logits = logits[:, None, :]
    physical_targets = targets[:, None, None]
    logits_spec = pl.BlockSpec(
        (1, 1, tile_size), lambda position: (position, 0, 0)
    )
    scalar_spec = pl.BlockSpec(
        (1, 1, 1), lambda position: (position, 0, 0)
    )
    output_shape = jax.ShapeDtypeStruct(
        (positions, 1, 1), jnp.float32
    )
    results = pl.pallas_call(
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
        name="rwkv7_training_loss_tpu_tile_statistics",
    )(physical_logits, physical_targets)
    return tuple(value[:, 0, 0] for value in results)


__all__ = ["training_loss_tile_statistics_tpu"]
