# rwkv7m

JAX/Flax reference implementation of an RWKV-7 style recurrent language model with optional state-level screening memory.

This repository is research-oriented. The current implementation is optimized for correctness, tests, and API usability before custom kernels or checkpoint compatibility.

## Install

From this checkout:

```powershell
uv sync
uv run pytest -q
```

From Git:

```powershell
uv add git+https://github.com/HeiMao-BaiMao/RWKV7M.git
```

or:

```powershell
uv pip install git+https://github.com/HeiMao-BaiMao/RWKV7M.git
```

## Minimal Inference

```python
import jax
import jax.numpy as jnp
from rwkv7m import create_runtime, generate_ids, tiny_config

cfg = tiny_config(vocab_size=256, d_model=64, n_layers=3, n_heads=4, head_size=16)
runtime = create_runtime(jax.random.PRNGKey(0), cfg, batch_size=1)

prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
new_ids = generate_ids(runtime, prompt, max_new_tokens=8, temperature=0.0)
print(new_ids)
```

## Minimal Training

```python
import jax
from rwkv7m import create_train_runtime, tiny_config, train_batch
from rwkv7m.train import generate_toy_batch

cfg = tiny_config(vocab_size=256, d_model=64, n_layers=3, n_heads=4, head_size=16)
runtime, state = create_train_runtime(jax.random.PRNGKey(1), cfg, batch_size=2, total_steps=10)

batch = generate_toy_batch(jax.random.PRNGKey(2), batch_size=2, seq_len=8, vocab_size=cfg.vocab_size)
state, metrics = train_batch(state, batch, runtime)
print(float(metrics["loss"]))
```

## Public API

Common imports:

```python
from rwkv7m import (
    ModelConfig,
    ScreeningConfig,
    ScreenedRWKVModel,
    create_model_variables,
    create_runtime,
    create_train_runtime,
    generate_ids,
    train_batch,
    tiny_config,
)
```

Lower-level modules:

- `rwkv7m.model`: RWKV core, screening module, config, state helpers.
- `rwkv7m.infer`: `prefill`, `decode_one`, `generate`.
- `rwkv7m.train`: optimizer, train state, toy batch, train step.

## Current Scope

- Flax Linen implementation.
- Full-sequence training path with `jax.lax.scan` inside recurrent components.
- Chunked inference state carry for the reference RWKV state (`time_mix_x`, `channel_mix_x`, WKV matrix state).
- State-level screening with `read_screening_only` and `read_write` phases.
- Installable package layout for `from rwkv7m import ...`.

Not yet included:

- production fused RWKV kernels,
- pretrained RWKV checkpoint conversion,
- tokenizer integration,
- distributed training utilities,
- long-context evaluation harnesses.

## Tests

```powershell
uv run pytest -q
```

Current smoke coverage includes math helpers, shape checks, phase/config validation, scan consistency, public API inference, and public API training.
