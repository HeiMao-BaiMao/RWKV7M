"""NNX-native optimizer lifecycle and training step."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from ..model.nnx_model import (
    NNXScreenedRWKVModel,
    NNXShardingConfig,
    initialize_nnx_model,
)
from ..model.screened_rwkv import cross_entropy_components
from ..model.nnx_conversion import nnx_params_to_linen
from .train_state import create_optimizer
from .train_step import l2wrap_components


def _optimizer_config(config):
    return {
        "lr_init": getattr(config, "lr_init", 1e-3),
        "lr_final": getattr(config, "lr_final", 1e-5),
        "warmup_steps": getattr(config, "warmup_steps", 10),
        "lr_schedule": getattr(config, "lr_schedule", "optax_cosine"),
        "max_grad_norm": getattr(config, "max_grad_norm", 1.0),
        "weight_decay": getattr(config, "weight_decay", 0.001),
        "adam_beta1": getattr(config, "adam_beta1", 0.9),
        "adam_beta2": getattr(config, "adam_beta2", 0.999),
        "adam_eps": getattr(config, "adam_eps", 1e-8),
        "optimizer_state_dtype": getattr(
            config, "optimizer_state_dtype", "float32"
        ),
    }


@dataclass
class NNXTrainState:
    """Thin compatibility handle around an NNX model and NNX optimizer."""

    model: NNXScreenedRWKVModel
    optimizer: nnx.Optimizer

    @property
    def step(self):
        return self.optimizer.step[...]

    @property
    def params(self):
        # Preserve the established portable/checkpoint boundary while the
        # optimizer itself remains fully NNX-native.
        return nnx_params_to_linen(self.model)

    @property
    def nnx_params(self):
        return nnx.state(self.model, nnx.Param)

    @property
    def opt_state(self):
        return self.optimizer.opt_state

    @property
    def tx(self):
        return self.optimizer.tx

    def ready_state(self):
        """Return every updated array leaf for a measurement-window barrier."""

        return nnx.state((self.model, self.optimizer))


def create_nnx_train_state(
    model: NNXScreenedRWKVModel,
    config,
    *,
    total_steps: int = 10000,
) -> NNXTrainState:
    tx = create_optimizer(_optimizer_config(config), total_steps=total_steps)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return NNXTrainState(model=model, optimizer=optimizer)


def initialize_nnx_train_state(
    rng_key,
    config,
    *,
    total_steps: int = 10000,
    sharding: NNXShardingConfig | None = None,
) -> NNXTrainState:
    """Create model and Optax state without an unsharded materialization."""

    if sharding is None:
        model = initialize_nnx_model(rng_key, config)
        return create_nnx_train_state(model, config, total_steps=total_steps)
    tx = create_optimizer(_optimizer_config(config), total_steps=total_steps)

    @jax.jit
    def initialize(key):
        model = NNXScreenedRWKVModel(
            config,
            rngs=nnx.Rngs(params=key),
            sharding=sharding,
        )
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
        return model, optimizer

    with jax.set_mesh(sharding.mesh):
        model, optimizer = initialize(rng_key)
    return NNXTrainState(model=model, optimizer=optimizer)


def _dtype_from_name(name):
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    raise ValueError(f"unsupported training dtype: {name!r}")


def _slice_batch_tree(tree, start, stop):
    return jax.tree.map(lambda value: value[start:stop], tree)


def _concat_batch_trees(trees):
    if len(trees) == 1:
        return trees[0]
    return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *trees)


def _cast_gradient_tree(grads, dtype):
    return jax.tree.map(
        lambda value: value.astype(dtype)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact)
        else value,
        grads,
    )


def nnx_model_loss(
    active_model,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase,
    deterministic,
    include_l2wrap,
    ce_loss_scale=1.0,
    l2_loss_scale=1.0,
):
    """Exact top-level sequence chunking without truncated BPTT."""

    token_count = int(batch["input_ids"].shape[1])
    chunk_size = active_model.config.sequence_chunk_size or token_count
    chunk_size = min(int(chunk_size), token_count)
    ce_total = jnp.zeros((), dtype=jnp.float32)
    ce_count = jnp.zeros((), dtype=jnp.float32)
    l2_total = jnp.zeros((), dtype=jnp.float32)
    l2_count = jnp.zeros((), dtype=jnp.float32)
    stats_total = None
    current_rwkv = rwkv_state
    current_screen = screen_state
    target_sharding = None
    if active_model.config.vocab_parallel and active_model.sharding is not None:
        target_sharding = active_model.sharding.activation(
            2,
            model_sharded=False,
        )

    def apply_chunk(model, carry, chunk_batch):
        chunk_rwkv, chunk_screen = carry
        logits, new_rwkv, new_screen, stats = model(
            chunk_batch["input_ids"],
            chunk_rwkv,
            chunk_screen,
            phase=phase,
            deterministic=deterministic,
        )
        chunk_ce, chunk_ce_count = cross_entropy_components(
            logits,
            chunk_batch["target_ids"],
            chunk_batch.get("mask"),
            target_sharding=target_sharding,
        )
        chunk_l2, chunk_l2_count = l2wrap_components(logits)
        outputs = {
            "ce_total": chunk_ce,
            "ce_count": chunk_ce_count,
            "l2_total": chunk_l2,
            "l2_count": chunk_l2_count,
            "stats": stats,
        }
        return (new_rwkv, new_screen), outputs

    chunk_count, remainder = divmod(token_count, chunk_size)
    if chunk_count > 1 and remainder == 0:
        batch_size = int(batch["input_ids"].shape[0])
        chunked_batch = jax.tree.map(
            lambda value: jnp.swapaxes(
                value.reshape(
                    (batch_size, chunk_count, chunk_size, *value.shape[2:])
                ),
                0,
                1,
            ),
            batch,
        )

        @nnx.scan(
            in_axes=(nnx.Carry, 0, None),
            out_axes=(nnx.Carry, 0),
        )
        def scan_chunks(carry, chunk_batch, model):
            return apply_chunk(model, carry, chunk_batch)

        (current_rwkv, current_screen), chunk_outputs = scan_chunks(
            (current_rwkv, current_screen),
            chunked_batch,
            active_model,
        )
        ce_total = jnp.sum(chunk_outputs["ce_total"], axis=0)
        ce_count = jnp.sum(chunk_outputs["ce_count"], axis=0)
        l2_total = jnp.sum(chunk_outputs["l2_total"], axis=0)
        l2_count = jnp.sum(chunk_outputs["l2_count"], axis=0)
        stats_total = jax.tree.map(
            lambda value: jnp.sum(value, axis=0) * chunk_size,
            chunk_outputs["stats"],
        )
    else:
        # Keep exact behavior for a final short chunk. The tracked 7B setup is
        # divisible and takes the compact scan above; this fallback avoids
        # padding tokens mutating the returned recurrent state.
        for start in range(0, token_count, chunk_size):
            stop = min(start + chunk_size, token_count)
            chunk_batch = {
                key: value[:, start:stop] for key, value in batch.items()
            }
            (current_rwkv, current_screen), chunk_outputs = apply_chunk(
                active_model,
                (current_rwkv, current_screen),
                chunk_batch,
            )
            ce_total += chunk_outputs["ce_total"]
            ce_count += chunk_outputs["ce_count"]
            l2_total += chunk_outputs["l2_total"]
            l2_count += chunk_outputs["l2_count"]
            chunk_tokens = jnp.asarray(stop - start, dtype=jnp.float32)
            stats = chunk_outputs["stats"]
            if stats_total is None:
                stats_total = {
                    key: value * chunk_tokens for key, value in stats.items()
                }
            else:
                stats_total = {
                    key: stats_total[key] + value * chunk_tokens
                    for key, value in stats.items()
                }

    ce_loss = ce_total / jnp.maximum(ce_count, 1.0)
    l2_loss = l2_total / jnp.maximum(l2_count, 1.0)
    total_loss = ce_loss * ce_loss_scale
    if include_l2wrap:
        total_loss += l2_loss * l2_loss_scale
    stats_total = {} if stats_total is None else stats_total
    metrics = {
        "loss": ce_loss,
        **{
            key: value / jnp.asarray(token_count, dtype=jnp.float32)
            for key, value in stats_total.items()
        },
    }
    for key in (
        "rel_read_mean",
        "active_slots_mean",
        "u_norm_mean",
        "rel_write_mean",
        "rel_write_effective_mean",
        "slot_update_norm_mean",
        "slot_usage_ema_mean",
    ):
        metrics.setdefault(key, jnp.zeros(()))
    return total_loss, (metrics, current_rwkv, current_screen)


@nnx.jit(
    static_argnames=("phase", "gradient_accumulation_steps"),
    donate_argnums=(0, 1),
)
def _nnx_train_step(
    model: NNXScreenedRWKVModel,
    optimizer: nnx.Optimizer,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase: str,
    gradient_accumulation_steps: int,
):
    batch_size = int(batch["input_ids"].shape[0])
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(
            "batch size must be divisible by gradient_accumulation_steps"
        )
    microbatch_size = batch_size // gradient_accumulation_steps
    token_count = int(batch["input_ids"].shape[1])
    accum_dtype = _dtype_from_name(model.config.gradient_accum_dtype)
    update_dtype = _dtype_from_name(model.config.param_update_dtype)
    mask = batch.get("mask")
    total_ce_count = (
        jnp.asarray(batch_size * token_count, dtype=jnp.float32)
        if mask is None
        else jnp.sum(mask, dtype=jnp.float32)
    )
    total_ce_count = jnp.maximum(total_ce_count, 1.0)
    total_positions = jnp.asarray(batch_size * token_count, dtype=jnp.float32)
    accumulated_grads = None
    accumulated_metrics = None
    accumulated_loss = jnp.zeros((), dtype=jnp.float32)
    new_rwkv_states = []
    new_screen_states = []

    for microstep in range(gradient_accumulation_steps):
        start = microstep * microbatch_size
        stop = start + microbatch_size
        microbatch = _slice_batch_tree(batch, start, stop)
        micro_rwkv = _slice_batch_tree(rwkv_state, start, stop)
        micro_screen = _slice_batch_tree(screen_state, start, stop)
        micro_mask = microbatch.get("mask")
        micro_ce_count = (
            jnp.asarray(microbatch_size * token_count, dtype=jnp.float32)
            if micro_mask is None
            else jnp.sum(micro_mask, dtype=jnp.float32)
        )
        ce_weight = micro_ce_count / total_ce_count
        position_weight = (
            jnp.asarray(microbatch_size * token_count, dtype=jnp.float32)
            / total_positions
        )

        def loss_fn(active_model):
            return nnx_model_loss(
                active_model,
                microbatch,
                micro_rwkv,
                micro_screen,
                phase=phase,
                deterministic=False,
                include_l2wrap=True,
                ce_loss_scale=ce_weight,
                l2_loss_scale=position_weight,
            )

        (loss, (metrics, new_rwkv, new_screen)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        grads = _cast_gradient_tree(grads, accum_dtype)
        accumulated_grads = (
            grads
            if accumulated_grads is None
            else jax.tree.map(jnp.add, accumulated_grads, grads)
        )
        weighted_metrics = {
            key: value * (ce_weight if key == "loss" else position_weight)
            for key, value in metrics.items()
        }
        accumulated_metrics = (
            weighted_metrics
            if accumulated_metrics is None
            else jax.tree.map(jnp.add, accumulated_metrics, weighted_metrics)
        )
        accumulated_loss += loss
        new_rwkv_states.append(new_rwkv)
        new_screen_states.append(new_screen)

    accumulated_grads = _cast_gradient_tree(accumulated_grads, update_dtype)
    optimizer.update(model, accumulated_grads)
    metrics = accumulated_metrics
    metrics["total_loss"] = accumulated_loss
    return (
        _concat_batch_trees(new_rwkv_states),
        _concat_batch_trees(new_screen_states),
        metrics,
    )


def nnx_train_step(
    train_state: NNXTrainState,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase="read_screening_only",
    gradient_accumulation_steps=1,
):
    new_rwkv_state, new_screen_state, metrics = _nnx_train_step(
        train_state.model,
        train_state.optimizer,
        batch,
        rwkv_state,
        screen_state,
        phase=phase,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    return train_state, new_rwkv_state, new_screen_state, metrics


__all__ = [
    "NNXTrainState",
    "create_nnx_train_state",
    "initialize_nnx_train_state",
    "nnx_model_loss",
    "nnx_train_step",
]
