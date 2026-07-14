# TPU Research Cloud Training

This document describes the TPU path for `rwkv7m`. Flax NNX now owns the
runtime, optimizer, distributed train/eval, and checkpoint lifecycle. The Linen
implementation is frozen as the numerical and upstream-conversion reference.

Phase 3 now provides an executable Explicit model-parallel path and post-SPMD
HLO collective audit. Its small complete-model path has been functionally
validated on a four-device TPU v5e slice. Still pending: XProf-based collective
tuning, full RWKV7M train/runtime-state checkpoint validation on TPU pods, and
production-scale throughput tuning.

## Install

```powershell
uv sync --extra tpu
uv run pytest -q tests/test_distributed_skeleton.py tests/test_distributed_train_state.py tests/test_distributed_trainer.py tests/test_train_binidx_distributed_cli.py
```

Record the exact JAX, jaxlib, Flax, Optax, Orbax, Python, backend, and device
versions before a TPU run. Distributed `run_config.json` now includes this
environment manifest automatically.

## Model Presets

The common NNX configuration registry exposes `0.185b`, `0.3b`, `1b`, `3b`,
and `7b` through `model_preset()` and `--model-preset`. Their exact current
parameter counts are 184,985,222; 297,738,764; 985,479,192; 2,943,319,064; and
6,994,788,376 respectively. Suggested initial model-axis sizes are 1, 1, 2, 4,
and 8. Always run the estimator for the allocated topology because these axis
sizes do not account for activations or compiler temporaries.

```powershell
uv run rwkv7m-plan-scale `
  --model-preset 3b `
  --model-axis-size 4 `
  --dtype-profile config
```

The equivalent tracked JSON files live under `configs/`; the existing
`rwkv7m-7b-tpu.json.example` is the canonical 7B JSON preset.

## 7B Preflight

The tracked candidate is in `configs/rwkv7m-7b-tpu.json.example`. Count its
current abstract parameter tree and estimate parameter-related memory with:

```powershell
uv run rwkv7m-plan-scale `
  --model-config configs/rwkv7m-7b-tpu.json.example `
  --model-axis-size 8 `
  --dtype-profile config
```

The current implementation has 6,994,788,376 parameters for this config:

- RWKV core and common model parameters: 6,871,986,176.
- four screening layers: 122,802,200.

The count comes from `jax.eval_shape` over the current model initializer rather
than a separately maintained formula. The memory estimate includes parameter
storage, two Adam moments, FP32 accumulated gradients, and an update workspace.
It excludes activations, RWKV/screening state, compiler temporaries, and
collective buffers.

The same model artifact is accepted by the public Python runtime, the scale
planner, and both training CLIs. A complete distributed run example is tracked
in `configs/rwkv7m-7b-tpu-train.json.example`:

```powershell
uv run rwkv7m-train-binidx-dp `
  --config configs/rwkv7m-7b-tpu-train.json.example
```

The example uses an eight-way model axis, one sequence per microstep, eight
microsteps per optimizer update, vocabulary-parallel loss, block rematerialization,
and exact 256-token chunks. Replace its dataset paths and adapt its mesh to the
allocated TPU topology. The shared code path is tested on small shapes, but this
specific 7B launch remains an unexecuted preflight configuration.

## NNX Lifecycle Gate

Validate the small independent framework probe when changing JAX/Flax/Orbax:

```powershell
uv run rwkv7m-verify-nnx-lifecycle --checkpoint-dir out/nnx-probe --mode create
uv run rwkv7m-verify-nnx-lifecycle --checkpoint-dir out/nnx-probe --mode restore
```

The two commands run in different Python processes. Together they validate
topology-aware sharded initialization, forward/backward, one Adam update,
preservation of explicit output shardings, Orbax save, sharded restore, and a
second update. On a real TPU slice, inspect the reported `PartitionSpec` values
and device topology in addition to successful completion.

The repository also runs the same lifecycle against the complete NNX RWKV7M
model: Linen-to-NNX parameter conversion, forward/gradient parity, sharded
model and Adam initialization, update, Orbax save/restore, and a post-restore
update. Use `tests/test_nnx_model.py` as the Phase 2 acceptance test.

Run the Phase 3 full-model contract and HLO audit with at least two devices:

```powershell
$env:JAX_NUM_CPU_DEVICES="2"
uv run rwkv7m-audit-nnx-model-parallel `
  --model-axis-size 2 `
  --screening `
  --write-screening `
  --vocab-parallel `
  --remat-blocks `
  --sequence-chunk-size 2
```

This compiles the complete forward path, counts collectives in the compiled
post-SPMD HLO, executes RWKV and screening state updates, and completes one NNX
optimizer step. On TPU, omit the forced CPU environment variable and select a
model-axis size that divides `d_model`, `n_heads`, and `d_slot`.

## Real TPU Verification Record

The Phase 3 functional gate was run on 2026-07-14 on a single-host TPU v5e
`v5litepod-4` slice with four devices and a `data=1, model=4` mesh. The validated
environment was Python 3.13.14, JAX/jaxlib 0.10.0, Flax 0.12.7, Optax 0.2.8,
Orbax-checkpoint 0.11.39, and libtpu 0.0.40.

```bash
uv run --python 3.13 rwkv7m-audit-nnx-model-parallel \
  --model-axis-size 4 \
  --d-model 32 \
  --n-heads 4 \
  --head-size 8 \
  --screening \
  --write-screening
```

The small complete-model smoke configuration completed finite forward/backward
computation and optimizer step 1. The audit confirmed the row/column kernel
contracts, WKV state `P("data", "model", None, None)`, screening slots
`P("data", None, "model")`, and 16 post-SPMD collectives: 13 all-reduces,
3 all-to-alls, and no all-gathers.

The independent small NNX lifecycle probe was then run as separate `create` and
`restore` processes on the same slice. It restored the Orbax checkpoint and
completed optimizer step 2 with finite loss. This result validates the isolated
NNX/Optax/Orbax lifecycle only. It does not validate the full RWKV7M distributed
train-state checkpoint, carried recurrent/screening runtime-state checkpoint,
a 7B model, or multi-host TPU execution.

On TPU VMs, install from the checkout or package in the same way, then verify JAX sees TPU devices:

```powershell
uv run python -c "import jax; print(jax.devices()); print(jax.process_count(), jax.process_index())"
```

## Data

Use RWKV-LM-V7 compatible `.bin/.idx` files and pass the prefix path without suffix:

```powershell
uv run rwkv7m-make-binidx data/corpus.jsonl --output-prefix data/corpus --ctx-len 512
```

## Single-Host Smoke Run

```powershell
uv run rwkv7m-train-binidx-dp `
  --data-file data/corpus `
  --ctx-len 512 `
  --global-batch-size 8 `
  --steps 10 `
  --vocab-size 65536 `
  --output-dir out/tpu-smoke `
  --save-every 10 `
  --eval-every 10 `
  --eval-steps 2 `
  --eval-data-file data/corpus `
  --prefetch-size 2
```

## Multi-Host Environment

Set the JAX distributed environment before launching the same command on every host:

```powershell
$env:JAX_COORDINATOR_ADDRESS="host0:12345"
$env:JAX_NUM_PROCESSES="4"
$env:JAX_PROCESS_ID="0"
uv run rwkv7m-train-binidx-dp `
  --data-file data/corpus `
  --ctx-len 512 `
  --global-batch-size 128 `
  --steps 1000 `
  --output-dir out/tpu-run `
  --save-every 100 `
  --checkpoint-backend orbax `
  --keep-last-checkpoints 3 `
  --eval-every 100 `
  --eval-steps 10 `
  --eval-data-file data/validation `
  --summary-every 10 `
  --save-best-checkpoint `
  --prefetch-size 4
```

Use the same `JAX_COORDINATOR_ADDRESS` and `JAX_NUM_PROCESSES` on all hosts. Set `JAX_PROCESS_ID` to each host's zero-based rank.

`global_batch_size` must be divisible by `process_count`, and each process-local batch must be divisible by the local device count.

For future model-parallel experiments, the CLI can construct a multi-axis mesh:

```powershell
uv run rwkv7m-train-binidx-dp `
  --data-file data/corpus `
  --ctx-len 512 `
  --global-batch-size 128 `
  --mesh-axis-names data model `
  --mesh-axis-sizes 4 2 `
  --param-axis-name model
```

The default remains replicated parameters over a 1D `data` mesh.
`--param-axis-name model` constructs the NNX model and Adam state directly under
the target mesh using declared row/column logical axes; it does not first build
a full unsharded model. It also selects Explicit mesh axes, validates model-axis
divisibility, and carries RWKV heads and screening slots in model-sharded state.

## Current Implementation Boundary

Implemented:

- `jax.distributed.initialize()` environment-based setup.
- topology-aware 1D and multi-axis JAX mesh construction with `jax.make_mesh()`.
- complete NNX RWKV core, screening, model, optimizer, runtime, distributed
  train/eval, and checkpoint lifecycle.
- full-model Linen-to-NNX tensor-path conversion and forward/gradient parity.
- NNX sharded init/update/Orbax/restore lifecycle probe and full-model test.
- abstract parameter counting, dtype policy, and parameter-related HBM estimator.
- exact package/runtime manifest in distributed run configuration.
- declared row/column NNX parameter axes for embeddings, RWKV projections,
  screening projections, norms, states, and LM head.
- explicit output shardings for embedding gather, row/column linear operations,
  RWKV head reshapes, and screening delta projection.
- a shared `ModelConfig` artifact for public APIs, small runs, the 7B planner,
  and distributed training.
- BF16 parameter storage/compute with explicit FP32 update, Adam-state, and
  gradient-accumulation policies.
- vocabulary-parallel logits, distributed cross entropy and L2Wrap, exact
  state-carrying sequence chunks, block rematerialization, and microbatch
  gradient accumulation.
- activation placement `P("data", None, "model")`, RWKV state placement
  `P("data", "model", None, None)`, and screening slot placement
  `P("data", None, "model")` on the Phase 3 path.
- compiled post-SPMD HLO collective audit and a forced two-CPU full-model
  forward/backward/optimizer regression test.
- real four-device TPU v5e functional validation of the small complete-model
  Phase 3 forward/backward/optimizer path and its compiled collective audit.
- real TPU validation of the independent small NNX Orbax lifecycle probe across
  separate create/restore processes.
- process-aware host binidx sampling.
- data-parallel batch placement with `jax.make_array_from_process_local_data`.
- train/eval step boundaries for local distributed tests.
- recurrent/screening state reset by default for independent binidx chunks.
- sequential carry-state train/eval mode with automatic state reset at stream-lane wrap boundaries.
- validation with a separate carried eval state, so validation does not mutate the training stream state.
- process 0 checkpoint writing and checkpoint rotation.
- Orbax train-state checkpoint writing for TPU-scale runs.
- distributed carry-state runtime checkpoint/resume for local and single-process distributed runs.
- `run_config.json`, `run_summary.json`, `best_eval.json`, `metrics.jsonl`, and `metrics.csv` output.
- best validation checkpoint tracking with rotation protection.
- safetensors artifacts with model and tokenizer metadata for external runtimes.

Pending:

- XProf profiling and throughput tuning on multi-device TPU, including the
  three observed all-to-all collectives.
- real TPU pod validation of full RWKV7M sharded runtime-state checkpoint/resume
  for carried recurrent/screening state.
- real TPU pod validation of full RWKV7M Orbax checkpoint save/resume under
  sharded train states.
- 7B and multi-host TPU execution.
- TPU pod throughput tuning and failure recovery drills.
- task-specific long-context evaluation harnesses beyond stateful binidx validation.

## Resume

Two checkpoint backends are available. The local-compatible msgpack backend is
the CLI default; select Orbax explicitly for TPU-scale runs:

- `--checkpoint-backend flax`: process 0 writes the NNX model/optimizer pure
  state as `train_state.msgpack` plus `model.safetensors`; use it for small/local
  runs and portable artifact export. With `--carry-state`, it also writes
  `runtime_state.msgpack`.
- `--checkpoint-backend orbax`: all processes write an Orbax train-state checkpoint under `orbax_train_state`, while process 0 writes `checkpoint.json`; use this for TPU-scale runs where materializing the full train state on process 0 is not viable. With `--carry-state`, it also writes `orbax_runtime_state`.

Resume works for both backends:

```powershell
uv run rwkv7m-train-binidx-dp `
  --data-file data/corpus `
  --ctx-len 512 `
  --global-batch-size 128 `
  --steps 1000 `
  --resume out/tpu-run/ckpt-00001000 `
  --output-dir out/tpu-run `
  --save-every 100
```

The resume step is read from `checkpoint.json`; `--steps` means additional optimizer updates after the checkpoint.

## Logging

When `--output-dir` is set, the distributed CLI writes:

- `run_config.json`
- `run_summary.json`
- `best_eval.json` when validation has produced a best metric
- `metrics.jsonl`
- `metrics.csv`

`run_config.json` stores CLI args, model config, process count, device count, device names, and exact runtime/package versions. When `--param-axis-name` is set, it also records a parameter partition summary with shapes and `PartitionSpec` strings. `run_summary.json` is updated during training with status, current step, completed steps, token counts, latest checkpoint, last train/eval records, and best eval. Metric records include `split`, `step`, `loss`, screening metrics, tokens, and `tokens_per_sec` for train steps. The NNX distributed CLI synchronizes the complete updated model/optimizer state at the end of the timing window.

Use `--log-jsonl`, `--log-csv`, and `--summary-json` to override output paths. `--summary-every` controls periodic summary writes. `--best-metric` and `--best-mode` choose the validation metric to track, and `--save-best-checkpoint` saves a checkpoint when that metric improves. Best checkpoints are protected from checkpoint rotation.

Generated `.json` files are run artifacts and are ignored by git. Commit reusable config examples as `.json.example`.

## Local Run Audit

Before moving a run to TPU scale, audit the local artifact contract:

```powershell
uv run rwkv7m-audit-dp-run out/tpu-smoke `
  --require-complete `
  --require-best-checkpoint `
  --min-train-records 10
```

The audit checks `run_config.json`, `run_summary.json`, `best_eval.json`, `metrics.jsonl`, `ckpt-*` metadata, backend-specific checkpoint artifacts, summary step counts, token counts, monotonic train steps, latest checkpoint references, and best-eval checkpoint references. Use `--json` for machine-readable output.

## State Carry

The distributed trainer resets recurrent and screening state by default for each sampled binidx chunk, matching the reference `train_batch` behavior. Use `--carry-state` only for deliberate streaming/stateful training, and pair it with `--sampling-mode sequential`.

Sequential sampling assigns one stream lane per batch row. When a lane wraps from its tail back to its head, the trainer resets carried RWKV/screening state before the next step, avoiding context contamination between the end and beginning of the lane.

Periodic validation can also run in carry-state mode. It advances a separate eval state across validation batches and resets that eval state at the same lane-wrap boundaries, so validation does not mutate the training stream state.

Current checkpoint backends persist distributed carry-state runtime state and restore it before mesh placement on resume. The Flax backend writes `runtime_state.msgpack`; the Orbax backend writes `orbax_runtime_state`. This is covered by local distributed tests, but real TPU pod validation of the sharded runtime-state policy is still pending.

## Export Contract

Flax-backend checkpoint directories include `model.safetensors`. Orbax training
checkpoints contain sharded train state and metadata but do not implicitly
materialize portable weights. A final/export operation must write complete
tensors one at a time or in small groups without gathering the whole model at
once. `model.safetensors` remains the canonical external-runtime artifact:

- `rwkv7m_config_json` stores the full model config.
- top-level metadata stores architecture, dtype, dimensions, screening summary, and tokenizer vocabulary identity.
- external PyTorch projects can read tensors with `safetensors.torch`.

`.pth` export should be added only if an external runtime project needs that format.
