"""Numerically stable loss components shared by reference and NNX paths."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


L2WRAP_FACTOR = 1e-4


def cross_entropy_loss(logits, targets, mask=None, *, target_sharding=None):
    total, count = cross_entropy_components(
        logits,
        targets,
        mask,
        target_sharding=target_sharding,
    )
    return total / jnp.maximum(count, 1.0)


def cross_entropy_components(logits, targets, mask=None, *, target_sharding=None):
    """Return a globally reducible FP32 CE numerator and denominator."""

    logits_f32 = logits.astype(jnp.float32)
    log_probs = jax.nn.log_softmax(logits_f32, axis=-1)
    logits_sharding = getattr(logits, "sharding", None)
    logits_spec = getattr(logits_sharding, "spec", None)
    if (
        target_sharding is None
        and logits_spec is not None
        and logits_spec[-1] is not None
    ):
        target_sharding = NamedSharding(
            logits_sharding.mesh,
            P(*tuple(logits_spec[:-1])),
        )
    if target_sharding is not None:
        batch_index = jnp.arange(logits.shape[0])[:, None]
        token_index = jnp.arange(logits.shape[1])[None, :]
        target_log_probs = log_probs.at[
            batch_index,
            token_index,
            targets,
        ].get(out_sharding=target_sharding)
        nll = -target_log_probs
    else:
        nll = -jnp.take_along_axis(
            log_probs, targets[..., None], axis=-1
        ).squeeze(-1)
    if mask is not None:
        mask_f32 = mask.astype(jnp.float32)
        nll = nll * mask_f32
        return jnp.sum(nll), jnp.sum(mask_f32)
    return jnp.sum(nll), jnp.asarray(nll.size, dtype=jnp.float32)


def l2wrap_loss(logits, factor=L2WRAP_FACTOR):
    total, count = l2wrap_components(logits, factor=factor)
    return total / count


def l2wrap_components(logits, factor=L2WRAP_FACTOR):
    """Return the RWKV-LM L2Wrap numerator and position count in FP32."""

    max_logits = jnp.max(logits.astype(jnp.float32), axis=-1)
    return (
        0.5 * factor * jnp.sum(jnp.square(max_logits)),
        jnp.asarray(max_logits.size, dtype=jnp.float32),
    )
