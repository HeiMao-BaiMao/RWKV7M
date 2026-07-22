# RWKV7M Implementation Specification

## 1. Purpose

`rwkv7m` is a JAX/Flax NNX research implementation for experimenting with
RWKV-7 style recurrent language models plus optional state-level screening
memory. The former Linen implementation is retained as a numerical and
upstream-conversion reference rather than the training source of truth.

`RWKV7M.paper.md` is the broader research design, while this document is the
concrete engineering contract for the current repository. The implemented
Screening v2 revision is specified separately in
`docs/state_level_screening_v2_design.md`; it maps to the paper's
`screening-v4-legacy` and `screening-v4-competitive` predecessor semantics.
The paper's `screening-v5-core` and `screening-v5-retention` profiles are
design-only and are not current implementation behavior.

## 2. Current Status

Implemented:

1. Flax NNX model, runtime, optimizer, and distributed training modules, with
   Linen reference modules and full-model parity tests.
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
13. Local-testable distributed data/model-parallel training layer with explicit
    NNX sharding contracts and post-SPMD HLO collective auditing.
14. Process-aware local and Orbax train-state checkpoints, carry-state runtime
    checkpoints, metrics, summaries, validation hooks, checkpoint rotation, and
    run artifact audit tooling.
15. Functional Phase 3 validation on a four-device TPU v5e slice for a small
    complete-model forward/backward/optimizer step, plus a separate-process
    small NNX Orbax lifecycle probe.
16. Installable library API: `from rwkv7m import ...`.
17. Shared `0.185b`, `0.3b`, `1b`, `3b`, and `7b` `ModelConfig` presets used by
    public runtime/training APIs, evaluation and benchmark CLIs, distributed
    training, and the scale planner, with exact NNX-tree parameter-count tests.
18. BF16/FP32 dtype policies, vocabulary-parallel loss and L2Wrap, block
    rematerialization, exact sequence chunking, and microbatch accumulation in
    the NNX training path.
19. Persistent backend-specific Pallas WKV and projected-Screening forward and
    backward kernels, with portable reference fallbacks and explicit dispatch.
20. Real L40S and TPU v5e correctness/performance gates for the tracked
    projected Screening recurrence shape.
21. A synthetic delayed key-value retrieval data generator with document-aware
    sampling modes, staged Screening activation with a separate Screening
    learning-rate multiplier, resume-time execution overrides with preserved
    checkpoint metadata, non-finite gradient/parameter training diagnostics
    with fail-closed stopping, and a fail-closed AMD WKV dispatch
    (`pallas_gpu_triton_reference_vjp`: Triton Pallas forward with the
    portable reference VJP).

Important limitation:

The current code is a unified JAX/Flax NNX small-to-7B training path with
backend-specific Pallas kernels and real-TPU functional coverage for a small
model, not a TPU-validated production-scale 7B trainer. It is suitable for
correctness testing, small experiments, portable artifact validation, and
distributed smoke tests. It does not yet provide XProf-tuned pod-scale
sharding, full RWKV7M train/runtime-state checkpoint validation on TPU pods, 7B
or multi-host/multi-slice TPU evidence, Hopper/Blackwell Mosaic validation,
pretrained RWKV checkpoint conversion, or a PyTorch/non-JAX runtime. The
external recurrent state carry is implemented for this research model, but it
should not be treated as compatibility with upstream RWKV-7 production
checkpoints. Screening v2 is implemented as an opt-in configuration and has
portable-reference plus CPU Pallas interpret-mode parity coverage. Corrected
commit `f35f6fc` passed real TPU v5e-4 lowering, recurrence parity and timing,
checkpointed `T=512`, and four-way model-axis optimizer steps through the
tracked 512-token maximum on 2026-07-18. Direct single-kernel `T=2048` exceeds
v5e scoped VMEM; measured peak memory and steady-state complete-model
throughput gates have not run. On AMD MI300X, a 2026-07-22 fixed-batch
isolation showed that multiple Pallas WKV training pullbacks coexisting in
one full graph corrupted cotangents even though every captured call passed in
isolation; AMD automatic dispatch therefore fails closed to a
Pallas-forward/reference-VJP hybrid WKV backend, and its real-device
complete-step acceptance is still pending.

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
  kernels/
    wkv_backend.py
    wkv_pallas_gpu.py
    wkv_pallas_tpu.py
    screening_backend.py
    screening_pallas_gpu.py
    screening_pallas_tpu.py
    training_loss_backend.py
    optimizer_backend.py
  model/
    nnx_model.py
    presets.py
    rwkv_core.py
    screened_rwkv.py
    screening.py
    screening_recurrence.py
    state.py
  infer/
    generate.py
  tokenizer/
    rwkv_tokenizer.py
  train/
    nnx_train.py
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
- ChannelMix uses `d_ffn` as its hidden width; `d_ffn <= 0` falls back to `4 * d_model` for compatibility with the default config sentinel.
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

The current phase names predate the Screening v2 write-mode split. In the
current implementation, `read_screening_only` still runs the unconditional slow
slot updater; it does not freeze slots. The v2 migration maps it to
`legacy_unconditional`. `read_write` with `use_write_screening=True` maps to
`legacy_threshold`. Old configs and checkpoints must preserve those semantics.

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

The public runtime state stores slots, ages, and usage. Inside the projected
accelerator recurrence, initial slots and every candidate are also projected
into read keys, values, and write keys. The recurrent carry is therefore:

```text
slots, read_keys, values, write_keys, ages, usage_ema
```

All projected content states currently use the same scalar update strength.
That consistency is a required invariant across sequence chunks.

## 9. Screening Math

The equations in this section describe the implemented v4 predecessor. They
do not include the v5 paper's explicit occupancy, capacity-calibrated safe
threshold, ambiguity penalty, or post-aggregation vector-norm bound.

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

This prohibition applies to reads. Screening v2 may conditionally normalize
eligible *write* routes, but it must multiply the distribution by absolute
write confidence so weak eligibility is not promoted to unit write mass.

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

### 10.1 Implemented Opt-In Screening v2 Contract

In the paper's semantics vocabulary, the legacy modes below belong to
`screening-v4-legacy`, and `competitive_novel` belongs to
`screening-v4-competitive`. There is no implemented v5 semantics selector.

The implementation provides four explicit write modes:

```text
disabled
legacy_unconditional
legacy_threshold
competitive_novel
```

これらの明示modeは`read_write` phaseで使用する。歴史的な
`read_screening_only`は旧slow updaterを維持し、既存phaseの意味を変更しない。
`read_write`内では`disabled`だけがslot更新を完全に停止する。

`competitive_novel` first computes absolute Trim-and-Square eligibility. For a
matched write, eligible slots compete but the normalized route is multiplied by
`max(eligibility)`. Novelty and sigmoid admission both use hard-forward,
soft-backward gates; rejected tokens therefore apply exactly zero write mass,
while the selection boundaries retain surrogate gradients. A bank-aware top-1
victim is sparse in the forward pass and soft in the backward pass. Slot
updates, age reset, and write counts follow the applied route. Usage EMA instead
follows absolute read activity so frequently read slots are protected from
eviction.

The opt-in revision moves the gate from model space to value space, factorizes
the slot candidate through a configurable low-rank latent, adds interval
checkpointing for the six-value FP32 training tape, and finally adds multi-read
tiles with fixed total dimensions and `1 / sqrt(n_read_tiles)` residual
scaling. Group-wise slot updates are deferred because changing slot channels
without matching projected read-key/value/write-key updates would break chunk
invariance. Checkpointed inverse reconstruction is accepted only when the
configured maximum effective update strength is at most `0.95`.

The complete formulas, compatibility mapping, configuration surface, memory
equation, metrics, implementation status, and acceptance gates are in
[`docs/state_level_screening_v2_design.md`](docs/state_level_screening_v2_design.md).
The tracked small-model configuration is
[`configs/rwkv7m-0.185b-screening-v2.json.example`](configs/rwkv7m-0.185b-screening-v2.json.example).
CPU interpret mode verifies both GPU and TPU Pallas kernel equations, including
checkpointed backward gradients. TPU v5e-4 passed real-device v2 lowering,
all-output/all-input-gradient parity, the tracked recurrence benchmark, and a
four-device model-axis optimizer step before the routing correction. Corrected
commit `f35f6fc` was then revalidated on 2026-07-18: the default parity gate,
checkpointed `T=512`, and full 0.185B contexts 128 and 512 all passed. Direct
single-kernel `T=2048` exceeded v5e scoped VMEM. The existing L40S report
predates v2, and real GPU v2 remains unverified.

## 11. Training

Training uses next-token cross entropy plus the RWKV-LM L2Wrap max-logit
regularizer (equivalent auxiliary form, factor `1e-4`):

```python
loss = cross_entropy_loss(logits, target_ids, mask) + l2wrap_loss(logits)
```

Metrics report the pure cross entropy as `loss` and the regularized
objective as `total_loss`. Evaluation paths use `cross_entropy_loss` only.

`train_step` is JIT-compiled with static `phase`.

Optimizer:

- global norm clipping,
- AdamW,
- warmup cosine decay,
- weight decay only on dense kernels and the token embedding, matching
  upstream RWKV-LM: bias, norm, tau, lambda, slot embeddings, token-shift
  mix params, LoRA matrices, and w0/a0/v0/k_k/k_a/r_k anchors are excluded.

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
- local distributed mesh/sharding, train-state and runtime-state placement, trainer, checkpoint, run audit, and CLI boundaries,
- toy training stability,
- forward/backward smoke benchmarks.

Run:

```powershell
uv run pytest -q
```

As of 2026-07-22, the current worktree suite completed with 279 passing tests
and five accelerator/optional-runtime skips. This count includes GPU optimizer
and benchmark-harness coverage plus Screening v2 legacy migration, routing
invariants, hard-forward/soft-backward admission and novelty gradients,
checkpoint strength limits, fail-closed benchmark parity, sequence-chunk
parity, and CPU Pallas interpret-mode GPU/TPU checkpointed-gradient parity.
It also covers the synthetic retrieval generator and document sampling modes,
staged Screening activation, resume execution overrides with preserved
checkpoint metadata, non-finite training diagnostics, separated read/write
threshold warm-ups, tie-safe allocation statistics, the decay-underflow WKV
backward regression on both GPU and TPU Pallas backends, and the fail-closed
AMD `pallas_gpu_triton_reference_vjp` WKV dispatch. Real TPU/GPU evidence is
recorded separately and is not part of this local test count.

## 14. Design Rules

Do not:

1. apply softmax or slot-sum normalization to read relevance,
2. normalize write eligibility without restoring absolute write confidence,
3. hard-overwrite slots,
4. use in-place slot mutation,
5. silently accept unknown phase names,
6. initialize write parameters only in write phase,
7. use a hard-only admission decision without a soft backward surrogate,
8. introduce group-wise slot rates without a matching projected-state contract,
9. report CPU interpret-mode Screening v2 parity as real GPU/TPU validation or
   measured speedup.

Prefer:

1. explicit legacy write modes during migration,
2. `competitive_novel` only when explicitly configured,
3. `float32` for norm/similarity/relevance,
4. applied-route accounting for writes and absolute-read accounting for usage,
5. small configs for correctness tests,
6. full `uv run pytest -q` before commits.

## 15. Roadmap

Next engineering steps:

1. Validate Screening v2 on real NVIDIA GPUs. On TPU, measure checkpoint peak
   memory and steady-state 0.185B train-step throughput; corrected v5e-4 kernel
   parity, recurrence latency through `T=512`, and four-device model-axis
   execution through context 512 already pass.
2. Validate full RWKV7M Orbax train-state and runtime-state checkpoint
   save/resume on real TPU pods, including sharded optimizer, parameter,
   recurrent, and screening states. The independent small NNX lifecycle probe
   is already validated.
3. Use XProf to tune the explicit sharding contracts and the measured
   post-SPMD collectives, including the projected Screening boundary.
4. Validate the 7B configuration and multi-host execution, then tune TPU pod
   throughput and document failure recovery drills.
5. Validate Mosaic GPU kernels and the opt-in Pallas optimizer on actual
   Hopper/Blackwell hardware before changing defaults.
6. Implement task-specific long-context evaluation harnesses beyond stateful binidx validation.
7. Add causal intervention hooks for slot ablation/patching, read shuffle,
   admission/allocation suppression, memory reset, and frozen-slot controls.
8. Implement upstream RWKV-7 checkpoint mapping only after conversion tests prove compatibility.
