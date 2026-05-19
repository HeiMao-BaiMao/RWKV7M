import flax
import optax
import jax.numpy as jnp
from dataclasses import dataclass
from flax.training import train_state as flax_train_state


class TrainState(flax_train_state.TrainState):
    pass


def decay_mask_fn(params):
    flat = flax.traverse_util.flatten_dict(params)
    mask = {}
    for path, value in flat.items():
        name = "/".join(path)
        use_decay = (
            value.ndim >= 2
            and "slot_embed" not in name
            and "tau" not in name
            and "lambda" not in name
            and "norm" not in name.lower()
            and "bias" not in name.lower()
        )
        mask[path] = use_decay
    return flax.traverse_util.unflatten_dict(mask)


def create_optimizer(config, total_steps=10000):
    warmup_steps = config.get("warmup_steps", 100)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.get("lr_init", 6e-4),
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=config.get("lr_final", 1e-5),
    )

    tx = optax.chain(
        optax.clip_by_global_norm(config.get("max_grad_norm", 1.0)),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.get("weight_decay", 0.001),
            mask=decay_mask_fn,
        ),
    )
    return tx
