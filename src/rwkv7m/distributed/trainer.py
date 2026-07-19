from dataclasses import replace

import jax
import jax.numpy as jnp
from flax import nnx

from ..model.screened_rwkv import cross_entropy_loss
from ..train.train_step import train_step
from ..train.nnx_train import NNXTrainState, nnx_model_loss
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
    gradient_accumulation_steps=1,
):
    rwkv_state, screen_state = _state_inputs(dist, carry_state)
    train_state, rwkv_state, screen_state, metrics = train_step(
        dist.train_state,
        global_batch,
        rwkv_state,
        screen_state,
        phase=phase,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
    gradient_accumulation_steps=1,
):
    global_batch = host_batch_to_global_arrays(host_batch, dist.batch_sharding, layout)
    return train_global_batch_data_parallel(
        dist,
        global_batch,
        phase=phase,
        carry_state=carry_state,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
    metrics = {
        "loss": loss,
        "rel_read_mean": stats.get("rel_read_mean", jnp.zeros(())),
        "active_slots_mean": stats.get("active_slots_mean", jnp.zeros(())),
        "u_norm_mean": stats.get("u_norm_mean", jnp.zeros(())),
        "rel_write_mean": stats.get("rel_write_mean", jnp.zeros(())),
        "rel_write_effective_mean": stats.get("rel_write_effective_mean", jnp.zeros(())),
    }
    for key in (
        "matched_route_mass",
        "novel_route_mass",
        "route_entropy",
        "route_top1_concentration",
        "admission_mean",
        "admission_low_rate",
        "admission_high_rate",
        "novel_token_rate",
        "rejected_write_rate",
        "short_bank_write_mass",
        "mid_bank_write_mass",
        "long_bank_write_mass",
        "eviction_age_mean",
        "eviction_usage_mean",
        "slot_utilization",
        "dead_slot_rate",
        "slot_cosine_redundancy",
        "matched_erase_mass",
        "matched_write_mass",
        "novel_erase_mass",
        "novel_write_mass",
        "accepted_novel_rate",
        "empty_allocation_rate",
        "occupied_eviction_rate",
        "read_energy_mean",
        "write_saturation_rate",
        "write_budget_rate",
        "write_budget_loss",
        "write_budget_target",
        "write_self_similarity",
        "read_self_similarity",
        "self_index_raw_loss",
        "self_index_loss",
        "self_index_weight",
        "read_soft_warmup_alpha",
        "screening_residual_rms",
        "base_residual_rms",
        "screening_base_rms_ratio",
        "screening_residual_scale",
    ):
        metrics[key] = stats.get(key, jnp.zeros(()))
    return metrics, new_rwkv_state, new_screen_state


@nnx.jit(static_argnames=("phase",))
def _nnx_eval_step_data_parallel(
    model,
    batch,
    rwkv_state,
    screen_state,
    training_step,
    phase="read_screening_only",
):
    _, (metrics, new_rwkv_state, new_screen_state) = nnx_model_loss(
        model,
        batch,
        rwkv_state,
        screen_state,
        phase=phase,
        deterministic=True,
        include_l2wrap=False,
        training_step=training_step,
    )
    return metrics, new_rwkv_state, new_screen_state


def eval_step_data_parallel(
    train_state, batch, rwkv_state, screen_state, phase="read_screening_only"
):
    if isinstance(train_state, NNXTrainState):
        return _nnx_eval_step_data_parallel(
            train_state.model,
            batch,
            rwkv_state,
            screen_state,
            train_state.step,
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
