# TPU Research Cloud Training

This document describes the TPU path for `rwkv7m`. The current code provides a local-testable data-parallel training layer with process-aware checkpoints, checkpoint rotation, structured logs, run summaries, validation hooks, best-eval checkpoint tracking, device prefetching, and optional multi-axis mesh / rule-based parameter placement hooks.

Still pending: fully tuned per-parameter sharding rules, Orbax checkpoint policy validation on real TPU pods, and production-scale throughput tuning.

## Install

```powershell
uv sync --extra tpu
uv run pytest -q tests/test_distributed_skeleton.py tests/test_distributed_train_state.py tests/test_distributed_trainer.py tests/test_train_binidx_distributed_cli.py
```

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

The default remains replicated parameters over a 1D `data` mesh. `--param-axis-name` enables rule-based placement for model params and shape-based placement for optimizer leaves over that mesh axis; full tuned per-parameter sharding rules are still future work.

## Current Implementation Boundary

Implemented:

- `jax.distributed.initialize()` environment-based setup.
- 1D and multi-axis JAX mesh construction.
- rule-based parameter placement for embeddings, dense kernels, LM head weights, and fallback array leaves.
- process-aware host binidx sampling.
- data-parallel batch placement with `jax.make_array_from_process_local_data`.
- train/eval step boundaries for local distributed tests.
- recurrent/screening state reset by default for independent binidx chunks.
- process 0 checkpoint writing and checkpoint rotation.
- Orbax train-state checkpoint writing for TPU-scale runs.
- `run_config.json`, `run_summary.json`, `best_eval.json`, `metrics.jsonl`, and `metrics.csv` output.
- best validation checkpoint tracking with rotation protection.
- safetensors artifacts with model and tokenizer metadata for external runtimes.

Pending:

- tuned sharding specs per parameter group.
- real TPU pod validation of Orbax checkpoint save/resume under sharded train states.
- TPU pod throughput tuning and failure recovery drills.
- long-context evaluation harnesses.

## Resume

Two checkpoint backends are available:

- `--checkpoint-backend flax`: process 0 materializes and writes `train_state.msgpack` plus `model.safetensors`; this remains the default and is useful for small/local runs and portable artifact export.
- `--checkpoint-backend orbax`: all processes write an Orbax train-state checkpoint under `orbax_train_state`, while process 0 writes `checkpoint.json`; use this for TPU-scale runs where materializing the full train state on process 0 is not viable.

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

`run_config.json` stores CLI args, model config, process count, device count, and device names. When `--param-axis-name` is set, it also records a parameter partition summary with shapes and `PartitionSpec` strings. `run_summary.json` is updated during training with status, current step, completed steps, token counts, latest checkpoint, last train/eval records, and best eval. Metric records include `split`, `step`, `loss`, screening metrics, tokens, and `tokens_per_sec` for train steps.

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

The distributed trainer resets recurrent and screening state by default for each sampled binidx chunk, matching the reference `train_batch` behavior. Use `--carry-state` only for deliberate streaming/stateful training.

## Export Contract

Each checkpoint directory includes `model.safetensors`. This is the canonical artifact for external runtimes:

- `rwkv7m_config_json` stores the full model config.
- top-level metadata stores architecture, dtype, dimensions, screening summary, and tokenizer vocabulary identity.
- external PyTorch projects can read tensors with `safetensors.torch`.

`.pth` export should be added only if an external runtime project needs that format.
