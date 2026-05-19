from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .infer.generate import decode_one, generate, prefill
from .model.screened_rwkv import (
    ModelConfig,
    ScreenedRWKVModel,
    create_model_variables,
    init_rwkv_state,
)
from .model.state import init_screen_state
from .train.train_loop import build_train_state
from .train.train_step import train_step


@dataclass
class RWKV7MRuntime:
    model: ScreenedRWKVModel
    variables: dict
    rwkv_state: tuple
    screen_state: object


def create_runtime(
    rng_key,
    config: ModelConfig,
    *,
    batch_size: int = 1,
) -> RWKV7MRuntime:
    variables, model = create_model_variables(rng_key, config, batch_size)
    return RWKV7MRuntime(
        model=model,
        variables=variables,
        rwkv_state=init_rwkv_state(batch_size, config),
        screen_state=init_screen_state(batch_size, config.screening),
    )


def create_train_runtime(
    rng_key,
    config: ModelConfig,
    *,
    batch_size: int,
    total_steps: int = 10000,
):
    rng_key, init_key, train_key = jax.random.split(rng_key, 3)
    runtime = create_runtime(init_key, config, batch_size=batch_size)
    train_state = build_train_state(
        train_key,
        runtime.model,
        runtime.variables,
        config,
        total_steps=total_steps,
    )
    return runtime, train_state


def infer_prefill(runtime: RWKV7MRuntime, prompt_ids, *, phase="read_screening_only"):
    logits, rwkv_state, screen_state, stats = prefill(
        runtime.model,
        runtime.variables,
        prompt_ids,
        runtime.rwkv_state,
        runtime.screen_state,
        phase=phase,
    )
    runtime.rwkv_state = rwkv_state
    runtime.screen_state = screen_state
    return logits, stats


def infer_next(runtime: RWKV7MRuntime, token_ids, *, phase="read_screening_only"):
    logits, rwkv_state, screen_state, stats = decode_one(
        runtime.model,
        runtime.variables,
        token_ids,
        runtime.rwkv_state,
        runtime.screen_state,
        phase=phase,
    )
    runtime.rwkv_state = rwkv_state
    runtime.screen_state = screen_state
    return logits, stats


def generate_ids(
    runtime: RWKV7MRuntime,
    prompt_ids,
    *,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    rng_key=None,
    phase="read_screening_only",
):
    ids, rwkv_state, screen_state = generate(
        runtime.model,
        runtime.variables,
        prompt_ids,
        runtime.rwkv_state,
        runtime.screen_state,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        rng_key=rng_key,
        phase=phase,
    )
    runtime.rwkv_state = rwkv_state
    runtime.screen_state = screen_state
    return ids


def train_batch(
    train_state,
    batch,
    runtime: RWKV7MRuntime,
    *,
    phase="read_screening_only",
):
    train_state, rwkv_state, screen_state, metrics = train_step(
        train_state,
        batch,
        runtime.rwkv_state,
        runtime.screen_state,
        phase=phase,
    )
    runtime.rwkv_state = rwkv_state
    runtime.screen_state = screen_state
    runtime.variables = {"params": train_state.params}
    return train_state, metrics


def reset_runtime_state(runtime: RWKV7MRuntime, config: ModelConfig, *, batch_size: int):
    runtime.rwkv_state = init_rwkv_state(batch_size, config)
    runtime.screen_state = init_screen_state(batch_size, config.screening)
    return runtime


def tiny_config(
    *,
    vocab_size: int = 256,
    d_model: int = 64,
    n_layers: int = 3,
    n_heads: int = 4,
    head_size: int = 16,
    use_screening: bool = True,
) -> ModelConfig:
    from .model.screening import ScreeningConfig

    screening = ScreeningConfig(
        d_model=d_model,
        d_slot=max(16, d_model // 2),
        d_k=max(8, head_size),
        d_v=max(8, head_size),
        n_slots=4,
        screened_layers=(1,) if use_screening and n_layers > 1 else (),
        bank_ids=(0, 0, 1, 2),
    )
    return ModelConfig(
        d_model=d_model,
        d_ffn=d_model * 2,
        n_layers=n_layers,
        n_heads=n_heads,
        head_size=head_size,
        vocab_size=vocab_size,
        max_seq_len=128,
        dtype="float32",
        use_screening=use_screening,
        screening=screening,
    )


__all__ = [
    "RWKV7MRuntime",
    "create_runtime",
    "create_train_runtime",
    "infer_prefill",
    "infer_next",
    "generate_ids",
    "train_batch",
    "reset_runtime_state",
    "tiny_config",
]
