import jax
import jax.numpy as jnp
from flax import struct

from ..model.screened_rwkv import cross_entropy_loss
from ..model.state import init_screen_state, ModelScreenState


@struct.dataclass
class TrainMetrics:
    loss: jnp.ndarray
    total_loss: jnp.ndarray
    rel_read_mean: jnp.ndarray
    active_slots_mean: jnp.ndarray
    u_norm_mean: jnp.ndarray


_L2WRAP_FACTOR = 1e-4


def l2wrap_loss(logits, factor=_L2WRAP_FACTOR):
    """RWKV-LM L2Wrap: pull down the max logit per position.

    Equivalent to the upstream custom-gradient formulation, whose backward
    adds max_logit * factor / (B*T) at each argmax position.
    """
    max_logits = jnp.max(logits.astype(jnp.float32), axis=-1)
    return 0.5 * factor * jnp.mean(jnp.square(max_logits))


@jax.jit(static_argnames=["phase"], donate_argnums=(0,))
def train_step(train_state, batch, rwkv_state, screen_state, phase="read_screening_only"):
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
        return loss, (metrics, new_rwkv_state, new_screen_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_rwkv_state, new_screen_state)), grads = grad_fn(
        train_state.params
    )
    train_state = train_state.apply_gradients(grads=grads)

    metrics["total_loss"] = loss
    return train_state, new_rwkv_state, new_screen_state, metrics


def compute_aux_losses(stats, phase="read_screening_only"):
    aux = {"dead_slot": jnp.zeros(()), "diversity": jnp.zeros(()), "total": jnp.zeros(())}
    return aux
