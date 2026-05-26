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
7. RWKV-LM-V7 compatible `.bin/.idx` reader and sampler.
8. Sequential carry-state binidx training and validation with lane-wrap state reset.
9. RWKV tokenizer API and JSONL-to-binidx conversion using the repository vocabulary.
10. Safetensors export/import for Flax params with model config and tokenizer metadata.
11. PyTorch-readable safetensors loading helper for external runtime projects.
12. Single-process train checkpoints with optional runtime state for carry-state resume.
13. Local-testable distributed data-parallel training layer with mesh/sharding helpers.
14. Process-aware Flax and Orbax train-state checkpoints, metrics, summaries, validation hooks, checkpoint rotation, and run artifact audit tooling.
15. Installable library API: `from rwkv7m import ...`.

Important limitation:

The current code is still a JAX/Flax reference path. It is suitable for correctness testing, small experiments, portable artifact validation, and local distributed smoke tests. It does not yet provide production fused RWKV kernels, pretrained RWKV checkpoint conversion, tuned TPU sharding policy, real TPU pod checkpoint validation, or a PyTorch/non-JAX runtime. The external recurrent state carry is implemented for this reference model, but it should not be treated as compatibility with upstream RWKV-7 production checkpoints.

## 3. Package Layout

```text
src/rwkv7m/
  __init__.py
  api.py
  assets/
  backends/
    torch/
      checkpoint.py
  data/
    binidx.py
    dataset.py
  cli/
    audit_distributed_run.py
    bench_binidx.py
    eval_binidx.py
    generate.py
    make_binidx.py
    train_binidx.py
    train_binidx_distributed.py
  distributed/
    audit.py
    checkpoint.py
    input_pipeline.py
    mesh.py
    metrics.py
    partitioning.py
    sharding.py
    train_state.py
    trainer.py
  io/
    config.py
    flax_checkpoint.py
    safetensors.py
  model/
    rwkv_core.py
    screened_rwkv.py
    screening.py
    state.py
  infer/
    generate.py
  tokenizer/
    rwkv_tokenizer.py
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

`train_batch` resets recurrent state by default. This is intentional: normal RWKV-LM-V7 style binidx training samples shuffled independent chunks. Pass `carry_state=True` only for contiguous streaming/stateful training.

For binidx training and validation, carry-state mode must be paired with `sampling_mode="sequential"` / `--sampling-mode sequential`. The `magic` sampler is a cubic pseudo-shuffle, so carrying state through it would mix unrelated contexts. In sequential mode each batch row is a separate stream lane, and carried RWKV/screening state is reset automatically when that lane wraps from its tail back to its head.

For JAX efficiency, `RWKV7MRuntime` keeps reusable initial zero states for the configured batch size. The default stateless training path reuses those immutable JAX arrays instead of rebuilding zero states every step.

Binidx training helper:

```python
losses, runtime, train_state = train_binidx(
    rng_key,
    config,
    "data/minipile",
    ctx_len=512,
    batch_size=1,
    num_steps=100,
)
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

To keep from-scratch `read_write` training alive, write keys use `slots + slot_embed`, and update strength uses `max(rel_write, write_rel_floor)`. The default `write_rel_floor` is intentionally tiny and acts as a warm-up path when slots are initially zero.

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

## 11.1 RWKV-LM-V7 Binidx Training

The package includes a torch-free reader for RWKV-LM-V7 `.bin/.idx` files.

Reader:

```python
from rwkv7m.data import MMapIndexedDataset

data = MMapIndexedDataset("data/minipile")  # no extension
tokens = data.get(idx=0, offset=0, length=513)
```

Batch sampler:

```python
from rwkv7m import create_binidx_dataset

dataset = create_binidx_dataset(
    "data/minipile",
    ctx_len=512,
    batch_size=2,
)
batch = dataset.get_batch(0)
```

The batch has the same shape expected by `train_step`:

```text
input_ids:  int32[B, T]
target_ids: int32[B, T]
mask:       float32[B, T]
```

Sampling follows the RWKV-LM-V7 cubic shuffle:

```text
ii = 1 + epoch * samples_per_epoch + sample_index * world_size + rank
factor = int(magic_prime * ((sqrt(5) - 1) / 2))
offset = ((factor * ii^3) % magic_prime) * ctx_len
```

`magic_prime` is automatically computed as the largest `3n+2` prime not greater than `(data_size - 1) // ctx_len`, and can also be supplied explicitly. The extra `-1` ensures that every sampled `ctx_len + 1` span remains inside the `.bin` file.

For streaming/stateful training, use sequential sampling:

```powershell
uv run rwkv7m-train-binidx --data-file data/minipile --ctx-len 512 --batch-size 1 --steps 100 --sampling-mode sequential --carry-state --vocab-size 65536
```

Sequential sampling assigns a stream lane to each batch row. The sampler exposes `should_reset_state_before_step(step)` so training and validation reset carried state at lane wrap boundaries instead of carrying context from the end of a lane into its beginning.

CLI:

```powershell
uv run rwkv7m-train-binidx --data-file data/minipile --ctx-len 512 --batch-size 1 --steps 100 --vocab-size 65536
```

Validation:

```powershell
uv run rwkv7m-eval-binidx --data-file data/minipile --ctx-len 512 --batch-size 1 --steps 10 --vocab-size 65536
```

Stateful stream validation:

```powershell
uv run rwkv7m-eval-binidx --data-file data/minipile --ctx-len 512 --batch-size 1 --steps 10 --sampling-mode sequential --carry-state --vocab-size 65536
```

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
- tokenizer metadata and text round trip,
- binidx data loading and sequential carry-state reset behavior,
- safetensors export/import and train checkpoint round trip,
- stateful binidx validation CLI,
- local distributed mesh/sharding, train-state placement, trainer, checkpoint, run audit, and CLI boundaries,
- toy training stability,
- forward/backward smoke benchmarks.

Run:

```powershell
uv run pytest -q
```

As of this document update, the full suite passes locally: 88 tests.

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

1. Validate Orbax train-state checkpoint save/resume on real TPU pods, including sharded optimizer/parameter states.
2. Tune per-parameter TPU sharding rules beyond the current rule-based placement hooks.
3. Tune TPU pod throughput and document failure recovery drills.
4. Implement task-specific long-context evaluation harnesses beyond stateful binidx validation.
5. Add causal intervention hooks for slot ablation/patching, read shuffle, write suppression, and frozen-slot controls.
6. Implement upstream RWKV-7 checkpoint mapping only after conversion tests prove compatibility.
7. Add fused/custom kernels or kernel-backed WKV recurrence if JAX/XLA output is insufficient.
