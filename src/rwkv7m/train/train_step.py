import jax
import jax.numpy as jnp
from flax import struct

from ..model.losses import l2wrap_components, l2wrap_loss
from ..model.screened_rwkv import cross_entropy_loss
from ..model.state import init_screen_state, ModelScreenState


@struct.dataclass
class TrainMetrics:
    loss: jnp.ndarray
    total_loss: jnp.ndarray
    rel_read_mean: jnp.ndarray
    active_slots_mean: jnp.ndarray
    u_norm_mean: jnp.ndarray


@jax.jit(static_argnames=["phase"], donate_argnums=(0,))
def _linen_train_step(train_state, batch, rwkv_state, screen_state, phase="read_screening_only"):
    def loss_fn(params):
        logits, new_rwkv_state, new_screen_state, stats = train_state.apply_fn(
            {"params": params},
            batch["input_ids"],
            rwkv_state,
            screen_state,
            phase=phase,
            deterministic=False,
        )
        ce_loss = cross_entropy_loss(
            logits,
            batch["target_ids"],
            batch.get("mask"),
        )
        loss = ce_loss + l2wrap_loss(logits)
        metrics = {
            "loss": ce_loss,
            "rel_read_mean": stats.get("rel_read_mean", jnp.zeros(())),
            "active_slots_mean": stats.get("active_slots_mean", jnp.zeros(())),
            "u_norm_mean": stats.get("u_norm_mean", jnp.zeros(())),
            "rel_write_mean": stats.get("rel_write_mean", jnp.zeros(())),
            "rel_write_effective_mean": stats.get(
                "rel_write_effective_mean",
                jnp.zeros(()),
            ),
            "slot_update_norm_mean": stats.get(
                "slot_update_norm_mean",
                jnp.zeros(()),
            ),
            "slot_usage_ema_mean": stats.get(
                "slot_usage_ema_mean",
                jnp.zeros(()),
            ),
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
        return loss, (metrics, new_rwkv_state, new_screen_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_rwkv_state, new_screen_state)), grads = grad_fn(
        train_state.params
    )
    train_state = train_state.apply_gradients(grads=grads)

    metrics["total_loss"] = loss
    return train_state, new_rwkv_state, new_screen_state, metrics


def train_step(
    train_state,
    batch,
    rwkv_state,
    screen_state,
    phase="read_screening_only",
    gradient_accumulation_steps=1,
):
    """Dispatch to the NNX training path or frozen Linen reference path."""

    from .nnx_train import NNXTrainState, nnx_train_step

    if isinstance(train_state, NNXTrainState):
        return nnx_train_step(
            train_state,
            batch,
            rwkv_state,
            screen_state,
            phase=phase,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    if gradient_accumulation_steps != 1:
        raise ValueError("gradient accumulation is supported by the NNX path only")
    return _linen_train_step(
        train_state,
        batch,
        rwkv_state,
        screen_state,
        phase=phase,
    )


def compute_aux_losses(stats, phase="read_screening_only"):
    aux = {"dead_slot": jnp.zeros(()), "diversity": jnp.zeros(()), "total": jnp.zeros(())}
    return aux
