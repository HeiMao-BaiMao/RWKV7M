# RWKV7M Implementation Specification

## 1. Purpose

`rwkv7m` is a JAX/Flax reference implementation for experimenting with RWKV-7 style recurrent language models plus optional state-level screening memory.

The implementation follows the research design in `RWKV7M.paper.md`, but this document is the concrete engineering contract for the current repository.

## 2. Current Status

Implemented:

1. Flax Linen model modules.
2. RWKV-7 inspired TimeMix / ChannelMix blocks.
3. Reference recurrent state carry for chunked inference:
   - previous TimeMix hidden input,
   - previous ChannelMix hidden input,
   - WKV matrix state.
4. State-level screening memory:
   - fixed slot bank,
   - unit-normalized query/key,
   - Trim-and-Square relevance,
   - non sum-to-one read aggregation,
   - TanhNorm,
   - slow slot update,
   - optional read/write separated phase.
5. Optax training step and toy training loop.
6. Inference helpers: `prefill`, `decode_one`, `generate`.
7. Installable library API: `from rwkv7m import ...`.

Important limitation:

The current code is a pure JAX/Flax reference path. It is full-sequence-training-first and suitable for correctness testing and small experiments. It does not yet provide production fused RWKV kernels, pretrained RWKV checkpoint conversion, tokenizer integration, or distributed training. The external recurrent state carry is implemented for this reference model, but it should not be treated as compatibility with upstream RWKV-7 production checkpoints.

## 3. Package Layout

```text
src/rwkv7m/
  __init__.py
  api.py
  model/
    rwkv_core.py
    screened_rwkv.py
    screening.py
    state.py
  infer/
    generate.py
  train/
    train_loop.py
    train_state.py
    train_step.py
tests/
```

Wheel packaging uses:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/rwkv7m"]
```

## 4. Public API

Top-level imports:

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

Runtime helper:

```python
runtime = create_runtime(rng_key, config, batch_size=1)
```

Training helper:

```python
runtime, train_state = create_train_runtime(
    rng_key,
    config,
    batch_size=2,
    total_steps=100,
)
train_state, metrics = train_batch(train_state, batch, runtime)
```

## 5. Model Config Validation

`ModelConfig` validates:

- `d_model == n_heads * head_size`
- if screening is enabled:
  - `screening.d_model == model.d_model`
  - `len(screening.bank_ids) == screening.n_slots`
  - `bank_ids` values are only `0`, `1`, `2`
  - `screened_layers` are within `[0, n_layers)`

`ScreeningConfig` validates:

- non-empty `bank_ids` must have length `n_slots`
- `bank_ids` values must be only `0`, `1`, `2`

## 6. Phase Semantics

Canonical phases:

- `read_screening_only`
- `read_write`

Compatibility alias:

- `read_only` maps to `read_screening_only`

Unknown phases raise `ValueError`.

Write branch activation is explicit:

```python
write_enabled = phase == "read_write" and cfg.use_write_screening
```

This prevents accidental write behavior from strings such as `"inference"`.

## 7. RWKV Reference State

Per layer:

```python
LayerRWKVState(
    time_mix_x,      # [B, d_model]
    channel_mix_x,   # [B, d_model]
    wkv,             # [B, n_heads, head_size, head_size]
)
```

`init_rwkv_state(batch_size, config)` returns a tuple of one `LayerRWKVState` per layer.

The current RWKV core uses this state to make chunked decode consistent with full prefill for the reference implementation.

## 8. Screening State

Per screened layer:

```python
LayerScreenState(
    slots,      # [B, M, d_slot]
    ages,       # [B, M]
    usage_ema,  # [B, M]
)
```

Model-level:

```python
ModelScreenState(layers=tuple(...))
```

Only screened layers have screening states.

## 9. Screening Math

Unit norm:

```python
unit_norm(x) = x / sqrt(sum(x * x) + eps * eps)
```

Bounded threshold:

```python
tau = 2 * sigmoid(theta) - 1
```

Trim-and-Square relevance:

```python
rel = relu((sim - tau) / (1 - tau + eps)) ** 2
```

Read aggregation is not normalized:

```python
z = einsum("bm,bmv->bv", rel_read, value)
u = tanh_norm(z)
```

This is intentionally not softmax attention. If all slots are irrelevant, read-out can remain near zero.

## 10. Slot Update

`read_screening_only`:

- read branch is active,
- write screening branch is inactive,
- slow slot updater runs,
- ages are preserved.

`read_write` with `use_write_screening=True`:

- write query/key/tau are used,
- write relevance modulates slot update,
- ages accumulate through the scan and reset only for active writes.

Initialization creates write branch parameters whenever `cfg.use_write_screening=True`, even if variables are initialized through `read_screening_only`. This allows later `read_write` apply calls without missing parameters.

## 11. Training

Training uses next-token cross entropy:

```python
loss = cross_entropy_loss(logits, target_ids, mask)
```

`train_step` is JIT-compiled with static `phase`.

Optimizer:

- global norm clipping,
- AdamW,
- warmup cosine decay,
- no weight decay on bias, norm, tau, lambda, and slot embeddings.

The optimizer clamps warmup steps for very short schedules so library smoke tests such as `total_steps=2` remain valid.

## 12. Inference

Prefill:

```python
logits, rwkv_state, screen_state, stats = prefill(...)
```

Decode one token:

```python
next_logits, rwkv_state, screen_state, stats = decode_one(...)
```

Generate:

```python
ids, rwkv_state, screen_state = generate(...)
```

The top-level `RWKV7MRuntime` stores variables and mutable recurrent states for simple library use.

## 13. Tests

Current tests cover:

- unit norm,
- tau conversion,
- Trim-and-Square,
- TanhNorm,
- forward shapes,
- state update shape,
- full prefill vs stepwise decode consistency,
- invalid phase rejection,
- write-age accumulation,
- read-screening-only initialization followed by read-write apply,
- config validation,
- public top-level runtime API,
- public top-level train API,
- toy training stability,
- forward/backward smoke benchmarks.

Run:

```powershell
uv run pytest -q
```

As of this document update, the full suite passes locally.

## 14. Design Rules

Do not:

1. apply softmax over slots in the screening module,
2. normalize relevance by slot sum,
3. hard-overwrite slots,
4. use in-place slot mutation,
5. silently accept unknown phase names,
6. initialize write parameters only in write phase.

Prefer:

1. `read_screening_only` as the default phase,
2. `read_write` only when write screening is explicitly enabled,
3. `float32` for norm/similarity/relevance,
4. small configs for correctness tests,
5. full `uv run pytest -q` before commits.

## 15. Roadmap

Next engineering steps:

1. tokenizer and checkpoint IO,
2. upstream RWKV-7 checkpoint mapping,
3. custom kernels or kernel-backed WKV recurrence,
4. long-context retrieval evaluation,
5. causal intervention hooks for slot ablation/patching,
6. richer train configs and checkpoint save/load,
7. distributed training support.
