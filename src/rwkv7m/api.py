from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from .data import BinIdxBatchDataset, create_binidx_dataset
from .infer.generate import decode_one, generate, prefill
from .model.screened_rwkv import (
    ModelConfig,
    init_rwkv_state,
)
from .model.nnx_model import (
    NNXScreenedRWKVModel,
    NNXShardingConfig,
    initialize_nnx_model,
)
from .model.nnx_conversion import load_linen_params_into_nnx
from .model.state import init_screen_state
from .tokenizer import RWKVTokenizer
from .io.config import load_model_config
from .train.train_step import train_step
from .train.nnx_train import initialize_nnx_train_state


@dataclass
class RWKV7MRuntime:
    model: NNXScreenedRWKVModel
    variables: dict
    rwkv_state: tuple
    screen_state: object
    config: ModelConfig
    batch_size: int
    initial_rwkv_state: tuple
    initial_screen_state: object


def create_runtime(
    rng_key,
    config,
    *,
    batch_size: int = 1,
    sharding: NNXShardingConfig | None = None,
) -> RWKV7MRuntime:
    config = load_model_config(config)
    model = initialize_nnx_model(rng_key, config, sharding=sharding)
    variables = {"params": nnx.state(model, nnx.Param)}
    rwkv_state = init_rwkv_state(batch_size, config)
    screen_state = init_screen_state(batch_size, config.screening)
    return RWKV7MRuntime(
        model=model,
        variables=variables,
        rwkv_state=rwkv_state,
        screen_state=screen_state,
        config=config,
        batch_size=batch_size,
        initial_rwkv_state=rwkv_state,
        initial_screen_state=screen_state,
    )


def create_train_runtime(
    rng_key,
    config,
    *,
    batch_size: int,
    total_steps: int = 10000,
    sharding: NNXShardingConfig | None = None,
):
    config = load_model_config(config)
    _, init_key = jax.random.split(rng_key)
    train_state = initialize_nnx_train_state(
        init_key,
        config,
        total_steps=total_steps,
        sharding=sharding,
    )
    rwkv_state = init_rwkv_state(batch_size, config)
    screen_state = init_screen_state(batch_size, config.screening)
    runtime = RWKV7MRuntime(
        model=train_state.model,
        variables={"params": train_state.nnx_params},
        rwkv_state=rwkv_state,
        screen_state=screen_state,
        config=config,
        batch_size=batch_size,
        initial_rwkv_state=rwkv_state,
        initial_screen_state=screen_state,
    )
    return runtime, train_state


def load_runtime_params(runtime: RWKV7MRuntime, params):
    """Load a portable Linen-style parameter tree into an NNX runtime."""

    load_linen_params_into_nnx(runtime.model, params)
    runtime.variables = {"params": nnx.state(runtime.model, nnx.Param)}
    return runtime


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


def generate_text(
    runtime: RWKV7MRuntime,
    prompt: str,
    *,
    tokenizer: RWKVTokenizer | None = None,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    rng_key=None,
    phase="read_screening_only",
    include_prompt: bool = True,
):
    tokenizer = tokenizer or RWKVTokenizer()
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    prompt_array = jnp.asarray([prompt_ids], dtype=jnp.int32)
    new_ids = generate_ids(
        runtime,
        prompt_array,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        rng_key=rng_key,
        phase=phase,
    )
    generated_ids = [int(x) for x in list(new_ids[0])]
    output_ids = prompt_ids + generated_ids if include_prompt else generated_ids
    return tokenizer.decode(output_ids)


def train_batch(
    train_state,
    batch,
    runtime: RWKV7MRuntime,
    *,
    phase="read_screening_only",
    carry_state: bool = False,
    gradient_accumulation_steps: int = 1,
):
    if carry_state:
        rwkv_state = runtime.rwkv_state
        screen_state = runtime.screen_state
    else:
        batch_size = int(batch["input_ids"].shape[0])
        if batch_size == runtime.batch_size:
            rwkv_state = runtime.initial_rwkv_state
            screen_state = runtime.initial_screen_state
        else:
            rwkv_state = init_rwkv_state(batch_size, runtime.config)
            screen_state = init_screen_state(batch_size, runtime.config.screening)

    train_state, rwkv_state, screen_state, metrics = train_step(
        train_state,
        batch,
        rwkv_state,
        screen_state,
        phase=phase,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if carry_state:
        runtime.rwkv_state = rwkv_state
        runtime.screen_state = screen_state
    runtime.variables = {"params": train_state.nnx_params}
    return train_state, metrics


def train_binidx(
    rng_key,
    config: ModelConfig,
    data_file,
    *,
    ctx_len: int,
    batch_size: int,
    num_steps: int,
    magic_prime: int | None = None,
    phase="read_screening_only",
    carry_state: bool = False,
    sampling_mode: str = "magic",
    print_every: int | None = None,
    gradient_accumulation_steps: int = 1,
):
    if carry_state and sampling_mode != "sequential":
        raise ValueError("carry_state training requires sampling_mode='sequential'")
    dataset = create_binidx_dataset(
        data_file,
        ctx_len=ctx_len,
        batch_size=batch_size,
        magic_prime=magic_prime,
        epoch_steps=num_steps,
        sampling_mode=sampling_mode,
    )
    runtime, train_state = create_train_runtime(
        rng_key,
        config,
        batch_size=batch_size,
        total_steps=num_steps,
    )
    losses = []
    try:
        for step in range(num_steps):
            if carry_state and dataset.should_reset_state_before_step(step):
                runtime.rwkv_state = runtime.initial_rwkv_state
                runtime.screen_state = runtime.initial_screen_state
            batch = dataset.get_batch(step)
            train_state, metrics = train_batch(
                train_state,
                batch,
                runtime,
                phase=phase,
                carry_state=carry_state,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            loss = float(metrics["loss"])
            losses.append(loss)
            if print_every and (step % print_every == 0 or step == num_steps - 1):
                print(f"step={step} loss={loss:.6f}")
    finally:
        dataset.close()
    return losses, runtime, train_state


def reset_runtime_state(
    runtime: RWKV7MRuntime,
    config: ModelConfig | None = None,
    *,
    batch_size: int | None = None,
):
    config = config or runtime.config
    batch_size = batch_size or runtime.batch_size
    runtime.rwkv_state = init_rwkv_state(batch_size, config)
    runtime.screen_state = init_screen_state(batch_size, config.screening)
    runtime.config = config
    runtime.batch_size = batch_size
    runtime.initial_rwkv_state = runtime.rwkv_state
    runtime.initial_screen_state = runtime.screen_state
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
    "load_runtime_params",
    "infer_prefill",
    "infer_next",
    "generate_ids",
    "generate_text",
    "train_batch",
    "train_binidx",
    "reset_runtime_state",
    "tiny_config",
    "BinIdxBatchDataset",
    "create_binidx_dataset",
]
