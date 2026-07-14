from dataclasses import replace

import jax
import jax.numpy as jnp
from flax import nnx

from ..model.screened_rwkv import cross_entropy_loss
from ..train.train_step import train_step
from ..train.nnx_train import NNXTrainState
from .metrics import aggregate_metrics
from .sharding import host_batch_to_global_arrays


def _state_inputs(dist, carry_state):
    if carry_state:
        return dist.rwkv_state, dist.screen_state
    return dist.initial_rwkv_state, dist.initial_screen_state


def _replace_training_state(dist, train_state, rwkv_state, screen_state, carry_state):
    if carry_state:
        return replace(
            dist,
            train_state=train_state,
            rwkv_state=rwkv_state,
            screen_state=screen_state,
        )
    return replace(dist, train_state=train_state)


def train_global_batch_data_parallel(
    dist,
    global_batch,
    *,
    phase="read_screening_only",
    carry_state=False,
):
    rwkv_state, screen_state = _state_inputs(dist, carry_state)
    train_state, rwkv_state, screen_state, metrics = train_step(
        dist.train_state,
        global_batch,
        rwkv_state,
        screen_state,
        phase=phase,
    )
    return (
        _replace_training_state(dist, train_state, rwkv_state, screen_state, carry_state),
        aggregate_metrics(metrics),
    )


def train_batch_data_parallel(
    dist,
    host_batch,
    layout,
    *,
    phase="read_screening_only",
    carry_state=False,
):
    global_batch = host_batch_to_global_arrays(host_batch, dist.batch_sharding, layout)
    return train_global_batch_data_parallel(
        dist,
        global_batch,
        phase=phase,
        carry_state=carry_state,
    )


@jax.jit(static_argnames=["phase"])
def _linen_eval_step_data_parallel(train_state, batch, rwkv_state, screen_state, phase="read_screening_only"):
    logits, new_rwkv_state, new_screen_state, stats = train_state.apply_fn(
        {"params": train_state.params},
        batch["input_ids"],
        rwkv_state,
        screen_state,
        phase=phase,
        deterministic=True,
    )
    loss = cross_entropy_loss(logits, batch["target_ids"], batch.get("mask"))
    return {
        "loss": loss,
        "rel_read_mean": stats.get("rel_read_mean", jnp.zeros(())),
        "active_slots_mean": stats.get("active_slots_mean", jnp.zeros(())),
        "u_norm_mean": stats.get("u_norm_mean", jnp.zeros(())),
        "rel_write_mean": stats.get("rel_write_mean", jnp.zeros(())),
        "rel_write_effective_mean": stats.get("rel_write_effective_mean", jnp.zeros(())),
    }, new_rwkv_state, new_screen_state


@nnx.jit(static_argnames=("phase",))
def _nnx_eval_step_data_parallel(
    model, batch, rwkv_state, screen_state, phase="read_screening_only"
):
    logits, new_rwkv_state, new_screen_state, stats = model(
        batch["input_ids"],
        rwkv_state,
        screen_state,
        phase=phase,
        deterministic=True,
    )
    loss = cross_entropy_loss(logits, batch["target_ids"], batch.get("mask"))
    return {
        "loss": loss,
        "rel_read_mean": stats.get("rel_read_mean", jnp.zeros(())),
        "active_slots_mean": stats.get("active_slots_mean", jnp.zeros(())),
        "u_norm_mean": stats.get("u_norm_mean", jnp.zeros(())),
        "rel_write_mean": stats.get("rel_write_mean", jnp.zeros(())),
        "rel_write_effective_mean": stats.get(
            "rel_write_effective_mean", jnp.zeros(())
        ),
    }, new_rwkv_state, new_screen_state


def eval_step_data_parallel(
    train_state, batch, rwkv_state, screen_state, phase="read_screening_only"
):
    if isinstance(train_state, NNXTrainState):
        return _nnx_eval_step_data_parallel(
            train_state.model,
            batch,
            rwkv_state,
            screen_state,
            phase=phase,
        )
    return _linen_eval_step_data_parallel(
        train_state,
        batch,
        rwkv_state,
        screen_state,
        phase=phase,
    )


def evaluate_global_batch_data_parallel(
    dist,
    global_batch,
    *,
    phase="read_screening_only",
    carry_state=False,
):
    rwkv_state, screen_state = _state_inputs(dist, carry_state)
    metrics, _, _ = eval_step_data_parallel(
        dist.train_state,
        global_batch,
        rwkv_state,
        screen_state,
        phase=phase,
    )
    return aggregate_metrics(metrics)


def evaluate_global_batch_data_parallel_with_state(
    dist,
    global_batch,
    rwkv_state,
    screen_state,
    *,
    phase="read_screening_only",
):
    metrics, rwkv_state, screen_state = eval_step_data_parallel(
        dist.train_state,
        global_batch,
        rwkv_state,
        screen_state,
        phase=phase,
    )
    return aggregate_metrics(metrics), rwkv_state, screen_state


def evaluate_batch_data_parallel(
    dist,
    host_batch,
    layout,
    *,
    phase="read_screening_only",
    carry_state=False,
):
    global_batch = host_batch_to_global_arrays(host_batch, dist.batch_sharding, layout)
    return evaluate_global_batch_data_parallel(
        dist,
        global_batch,
        phase=phase,
        carry_state=carry_state,
    )


def evaluate_batch_data_parallel_with_state(
    dist,
    host_batch,
    layout,
    rwkv_state,
    screen_state,
    *,
    phase="read_screening_only",
):
    global_batch = host_batch_to_global_arrays(host_batch, dist.batch_sharding, layout)
    return evaluate_global_batch_data_parallel_with_state(
        dist,
        global_batch,
        rwkv_state,
        screen_state,
        phase=phase,
    )
