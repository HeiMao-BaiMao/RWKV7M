# rwkv7m

[日本語 README](README.ja.md)

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

`train_batch` resets recurrent state by default, which is the correct mode for independently sampled training chunks. Use `carry_state=True` only for deliberate streaming/stateful training. The default path reuses immutable initial zero states in the runtime, so it does not rebuild zero states every step for the configured batch size.

## Training From RWKV-LM-V7 `.bin/.idx`

`rwkv7m` can read the same binidx dataset format used by RWKV-LM-V7.
Download the tokenized Minipile dataset into `data/`, then pass the prefix path without `.bin` / `.idx`.

```powershell
New-Item -ItemType Directory -Force data
wget -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

CLI smoke run:

```powershell
uv run rwkv7m-train-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 100 `
  --vocab-size 65536 `
  --d-model 128 `
  --n-layers 4 `
  --n-heads 4 `
  --head-size 32
```

`magic_prime` is computed automatically from dataset size and `ctx_len`; pass `--magic-prime` to force a specific value. The automatic value is constrained so every sampled `ctx_len + 1` span stays inside the `.bin` file.

For read/write screening from scratch, the write branch uses slot identity in its write key and a tiny `write_rel_floor` update floor. This avoids a dead write branch when slots are all zero at initialization.

Python API:

```python
import jax
from rwkv7m import tiny_config, train_binidx

cfg = tiny_config(vocab_size=65536, d_model=128, n_layers=4, n_heads=4, head_size=32)
losses, runtime, state = train_binidx(
    jax.random.PRNGKey(0),
    cfg,
    "data/minipile",
    ctx_len=512,
    batch_size=1,
    num_steps=100,
)
print(losses[-1])
```

Quick baseline/screening comparison on the same binidx data:

```powershell
uv run rwkv7m-bench-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 10 `
  --vocab-size 65536
```

To convert JSONL text with the repository copy of the RWKV tokenizer vocabulary:

```powershell
uv run rwkv7m-make-binidx data/my_corpus.jsonl --output-prefix data/my_corpus --ctx-len 512
```

Tokenizer API:

```python
from rwkv7m import RWKVTokenizer

tokenizer = RWKVTokenizer()
ids = tokenizer.encode("Hello RWKV", add_eos=True)
text = tokenizer.decode(ids[:-1])
```

## Public API

Common imports:

```python
from rwkv7m import (
    create_binidx_dataset,
    ModelConfig,
    ScreeningConfig,
    ScreenedRWKVModel,
    create_model_variables,
    create_runtime,
    create_train_runtime,
    generate_ids,
    train_batch,
    train_binidx,
    tiny_config,
)
```

Lower-level modules:

- `rwkv7m.model`: RWKV core, screening module, config, state helpers.
- `rwkv7m.data`: RWKV-LM-V7 compatible `.bin/.idx` reader and batch sampler.
- `rwkv7m.infer`: `prefill`, `decode_one`, `generate`.
- `rwkv7m.train`: optimizer, train state, toy batch, train step.

## Current Scope

- Flax Linen implementation.
- Full-sequence training path with `jax.lax.scan` inside recurrent components.
- RWKV-LM-V7 compatible `.bin/.idx` dataset reader and sampler.
- Chunked inference state carry for the reference RWKV state (`time_mix_x`, `channel_mix_x`, WKV matrix state).
- State-level screening with `read_screening_only` and `read_write` phases.
- RWKV tokenizer API and JSONL-to-binidx conversion.
- Installable package layout for `from rwkv7m import ...`.

Not yet included:

- production fused RWKV kernels,
- pretrained RWKV checkpoint conversion,
- high-level text generation API,
- distributed training utilities,
- long-context evaluation harnesses.

## Tests

```powershell
uv run pytest -q
```

Current smoke coverage includes math helpers, shape checks, phase/config validation, scan consistency, public API inference, public API training, and binidx data loading.
