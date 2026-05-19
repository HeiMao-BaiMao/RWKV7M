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


@jax.jit(static_argnames=["phase"])
def train_step(train_state, batch, rwkv_state, screen_state, phase="read_only"):
    def loss_fn(params):
        logits, new_rwkv_state, new_screen_state, stats = train_state.apply_fn(
            {"params": params},
            batch["input_ids"],
            rwkv_state,
            screen_state,
            phase=phase,
            deterministic=False,
        )
        loss = cross_entropy_loss(
            logits,
            batch["target_ids"],
            batch.get("mask"),
        )
        metrics = {
            "loss": loss,
            "rel_read_mean": stats.get("rel_read_mean", jnp.zeros(())),
            "active_slots_mean": stats.get("active_slots_mean", jnp.zeros(())),
            "u_norm_mean": stats.get("u_norm_mean", jnp.zeros(())),
            "rel_write_mean": stats.get("rel_write_mean", jnp.zeros(())),
        }
        return loss, (metrics, new_rwkv_state, new_screen_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, new_rwkv_state, new_screen_state)), grads = grad_fn(
        train_state.params
    )
    train_state = train_state.apply_gradients(grads=grads)

    metrics["total_loss"] = loss
    return train_state, new_rwkv_state, new_screen_state, metrics


def compute_aux_losses(stats, phase="read_only"):
    aux = {"dead_slot": jnp.zeros(()), "diversity": jnp.zeros(()), "total": jnp.zeros(())}
    return aux
