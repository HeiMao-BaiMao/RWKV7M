import jax
import jax.numpy as jnp
from flax.training import train_state as flax_train_state
import optax

from ..model.screened_rwkv import (
    ModelConfig,
    ScreenedRWKVModel,
    init_rwkv_state,
    create_model_variables,
)
from ..model.state import init_screen_state
from .train_state import TrainState, create_optimizer
from .train_step import train_step


def build_train_state(key, model, variables, config, total_steps=10000):
    opt_config = {
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
    tx = create_optimizer(opt_config, total_steps=total_steps)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )
    return state


def generate_toy_batch(key, batch_size, seq_len, vocab_size):
    """Generate a random toy batch where target is input shifted by 1."""
    key1, key2 = jax.random.split(key)
    input_ids = jax.random.randint(key1, (batch_size, seq_len), 0, vocab_size)
    target_ids = jnp.roll(input_ids, shift=-1, axis=1)
    target_ids = target_ids.at[:, -1].set(0)
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "mask": mask,
    }


def run_toy_training(
    key,
    model_cfg: ModelConfig,
    batch_size: int = 4,
    seq_len: int = 8,
    num_steps: int = 100,
    print_every: int = 20,
):
    model = ScreenedRWKVModel(config=model_cfg)
    rwkv_state = init_rwkv_state(batch_size, model_cfg)
    screen_state = init_screen_state(batch_size, model_cfg.screening)

    key, subkey = jax.random.split(key)
    variables, _ = create_model_variables(subkey, model_cfg, batch_size)

    key, subkey = jax.random.split(key)
    train_state = build_train_state(
        subkey, model, variables, model_cfg, total_steps=num_steps
    )

    losses = []
    for step_i in range(num_steps):
        key, subkey = jax.random.split(key)
        batch = generate_toy_batch(subkey, batch_size, seq_len, model_cfg.vocab_size)

        rwkv_state = init_rwkv_state(batch_size, model_cfg)
        screen_state = init_screen_state(batch_size, model_cfg.screening)

        train_state, rwkv_state, screen_state, metrics = train_step(
            train_state, batch, rwkv_state, screen_state, phase="read_screening_only"
        )

        loss = float(metrics["loss"])
        losses.append(loss)

        if step_i % print_every == 0 or step_i == num_steps - 1:
            print(
                f"Step {step_i:4d} | loss={loss:.4f} | "
                f"rel_read={float(metrics.get('rel_read_mean', 0)):.4f} | "
                f"active_slots={float(metrics.get('active_slots_mean', 0)):.2f} | "
                f"u_norm={float(metrics.get('u_norm_mean', 0)):.4f}"
            )

    return losses, train_state
