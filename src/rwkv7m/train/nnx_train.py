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
from ..model.screened_rwkv import cross_entropy_loss
from ..model.nnx_conversion import nnx_params_to_linen
from .train_state import create_optimizer
from .train_step import l2wrap_loss


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


@nnx.jit(static_argnames=("phase",), donate_argnums=(0, 1))
def _nnx_train_step(
    model: NNXScreenedRWKVModel,
    optimizer: nnx.Optimizer,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase: str,
):
    def loss_fn(active_model):
        logits, new_rwkv_state, new_screen_state, stats = active_model(
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
        total_loss = ce_loss + l2wrap_loss(logits)
        metrics = {
            "loss": ce_loss,
            "rel_read_mean": stats.get("rel_read_mean", jnp.zeros(())),
            "active_slots_mean": stats.get("active_slots_mean", jnp.zeros(())),
            "u_norm_mean": stats.get("u_norm_mean", jnp.zeros(())),
            "rel_write_mean": stats.get("rel_write_mean", jnp.zeros(())),
            "rel_write_effective_mean": stats.get(
                "rel_write_effective_mean", jnp.zeros(())
            ),
            "slot_update_norm_mean": stats.get(
                "slot_update_norm_mean", jnp.zeros(())
            ),
            "slot_usage_ema_mean": stats.get(
                "slot_usage_ema_mean", jnp.zeros(())
            ),
        }
        return total_loss, (metrics, new_rwkv_state, new_screen_state)

    (loss, (metrics, new_rwkv_state, new_screen_state)), grads = (
        nnx.value_and_grad(loss_fn, has_aux=True)(model)
    )
    optimizer.update(model, grads)
    metrics["total_loss"] = loss
    return new_rwkv_state, new_screen_state, metrics


def nnx_train_step(
    train_state: NNXTrainState,
    batch,
    rwkv_state,
    screen_state,
    *,
    phase="read_screening_only",
):
    new_rwkv_state, new_screen_state, metrics = _nnx_train_step(
        train_state.model,
        train_state.optimizer,
        batch,
        rwkv_state,
        screen_state,
        phase=phase,
    )
    return train_state, new_rwkv_state, new_screen_state, metrics


__all__ = [
    "NNXTrainState",
    "create_nnx_train_state",
    "initialize_nnx_train_state",
    "nnx_train_step",
]
