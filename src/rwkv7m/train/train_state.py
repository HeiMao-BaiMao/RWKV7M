import flax
import optax
import jax.numpy as jnp
from dataclasses import dataclass
from flax.training import train_state as flax_train_state


class TrainState(flax_train_state.TrainState):
    pass


def rwkv_warmup_cosine_schedule(lr_init, lr_final, warmup_steps, total_steps):
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)
    lr_init = float(lr_init)
    lr_final = float(lr_final)
    lr_final_factor = lr_final / lr_init if lr_init != 0.0 else 0.0

    def schedule(count):
        count = jnp.asarray(count, dtype=jnp.float32)
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = jnp.clip((count - warmup_steps) / float(decay_steps), 0.0, 1.0)
        lr_mult = (
            (0.5 + lr_final_factor / 2.0)
            + (0.5 - lr_final_factor / 2.0) * jnp.cos(jnp.pi * progress)
        )
        lr = lr_init * lr_mult
        if warmup_steps > 0:
            warmup_mult = 0.01 + 0.99 * count / float(warmup_steps)
            lr = jnp.where(count < warmup_steps, lr * warmup_mult, lr)
        return lr

    return schedule


def decay_mask_fn(params):
    # Upstream RWKV-LM decays only true matmul weights. Matching that here
    # keeps token-shift mix params, w0/a0/v0/k_k/k_a/r_k anchors, LoRA
    # matrices, norms, biases, tau/lambda scalars, and slot_embed decay-free.
    flat = flax.traverse_util.flatten_dict(params)
    mask = {}
    for path, value in flat.items():
        leaf = path[-1]
        mask[path] = value.ndim >= 2 and leaf in ("kernel", "embedding")
    return flax.traverse_util.unflatten_dict(mask)


def rwkv_w0_mask_fn(params):
    """Select the decay anchor that upstream RWKV trains at 2x base LR."""
    flat = flax.traverse_util.flatten_dict(params)
    mask = {path: path[-1] == "w0" for path in flat}
    return flax.traverse_util.unflatten_dict(mask)


def create_optimizer(config, total_steps=10000):
    total_steps = max(int(total_steps), 1)
    warmup_steps = min(config.get("warmup_steps", 100), max(total_steps - 1, 0))
    if config.get("lr_schedule", "optax_cosine") == "rwkv":
        lr_schedule = rwkv_warmup_cosine_schedule(
            config.get("lr_init", 6e-4),
            config.get("lr_final", 1e-5),
            warmup_steps,
            total_steps,
        )
    else:
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
            b1=config.get("adam_beta1", 0.9),
            b2=config.get("adam_beta2", 0.999),
            eps=config.get("adam_eps", 1e-8),
            mask=decay_mask_fn,
        ),
        optax.masked(optax.scale(2.0), rwkv_w0_mask_fn),
    )
    return tx
