# TPU Research Cloud Training

This document describes the TPU path for `rwkv7m`. Flax NNX now owns the
runtime, optimizer, distributed train/eval, and checkpoint lifecycle. The Linen
implementation is frozen as the numerical and upstream-conversion reference.

Phase 3 now provides an executable Explicit model-parallel path and post-SPMD
HLO collective audit. Its small complete-model path has been functionally
validated on both four-device and 16-device TPU v5e slices. The 16-device gate
crossed four JAX processes and exercised model axes spanning two and four
workers. The `0.185b` and `1b` presets have completed single-host v5e throughput
runs. Still pending: XProf-based collective tuning, a shared-storage Orbax
save/restore drill on a TPU pod, multi-slice DCN execution, 7B execution, and
production-scale throughput tuning.

The topology, input-replica, checkpoint-storage, and staged acceptance rules
for larger TRC jobs are specified in [TRC Scaling Design](trc_scaling_design.md).

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

After projected Pallas screening was integrated, the same small audit was
rerun on 2026-07-16. It completed finite loss `3.95950198` and optimizer step 1
with 18 post-SPMD collectives: 13 all-reduces, 3 all-to-alls, and 2 explicit
slot-feature all-gathers at the screening recurrence boundary.

The independent small NNX lifecycle probe was then run as separate `create` and
`restore` processes on the same slice. It restored the Orbax checkpoint and
completed optimizer step 2 with finite loss. This result validates the isolated
NNX/Optax/Orbax lifecycle only. It does not validate the full RWKV7M distributed
train-state checkpoint, carried recurrent/screening runtime-state checkpoint,
a 7B model, or multi-host TPU execution.

## Single-Host TPU v5e Performance Record

A performance pass was run later on 2026-07-14 at commit
`9faf9aa8e6c7db51a5e6fd4d39d586c835c71146`. It used a temporary single-host
`v5litepod-4` in `us-west4-a`, with four TPU v5e devices. The environment was
Python 3.13.14, JAX/jaxlib 0.10.0, Flax 0.12.7, Optax 0.2.8,
Orbax-checkpoint 0.11.39, and libtpu 0.0.40. Transparent huge pages were enabled
before measurement to reduce TPU runtime startup and shutdown overhead.

The input was a synthetic RWKV-compatible binidx file containing 4,194,304
tokens. It removes storage and network variability but does not provide a model
quality result. Preset runs kept the tracked dtype, rematerialization, sequence
chunking, vocabulary-parallel, and screening settings unchanged. The
no-screening comparison used the same L12-D768-FFN2688 core dimensions as the
`0.185b` preset.

The distributed CLI starts each step timer after yielding the prefetched global
batch and stops it only after blocking on the complete updated model and
optimizer state. The first step was excluded because it includes XLA
compilation. `Measured steps` below therefore counts steps after step 1. The
mean includes every post-compilation stall; the median shows the normal steady
step more clearly.

The main 30-step measurement used:

```bash
JAX_COMPILATION_CACHE_DIR=/home/rumia/jax_cache \
  uv run rwkv7m-train-binidx-dp \
  --data-file data/bench/synthetic \
  --model-preset 0.185b \
  --ctx-len 512 \
  --global-batch-size 4 \
  --steps 30 \
  --phase read_write \
  --mesh-axis-names data \
  --mesh-axis-sizes 4 \
  --prefetch-size 2
```

| Model and phase | Context | Global batch | Mesh | Measured steps | Median token/s | Mean token/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 0.185B core dimensions, no screening | 512 | 4 | `data=4` | 11 | 18,680 | 18,611 |
| `0.185b`, read-only screening | 512 | 4 | `data=4` | 11 | 13,021 | 12,315 |
| `0.185b`, read/write screening | 512 | 4 | `data=4` | 29 | 11,931 | 11,510 |
| `0.185b`, read/write screening | 512 | 4 | `data=1, model=4` | 11 | 8,812 | 8,811 |
| `0.185b`, read/write screening | 512 | 1 | `data=1, model=4` | 7 | 4,909 | 4,911 |
| `1b`, read/write screening | 256 | 1 | `data=1, model=4` | 5 | 879 | 786 |
| `1b`, read/write screening | 256 | 4 | `data=1, model=4` | 5 | 2,397 | 2,249 |

Main findings:

- The 30-step `0.185b` data-parallel run had 29 post-compilation observations.
  Twenty-seven formed a narrow 11,858-11,963 token/s band, while two isolated
  stalls reached 6,694 and 5,170 token/s. The mean of the non-stall observations
  was 11,923 token/s.
- At global batch 4, replicating the `0.185b` model over `data=4` was 1.35 times
  faster by median than four-way model sharding. The model fits on each device,
  so model-sharding communication was not throughput-justified in this test.
- Relative to the no-screening median, read-only screening reduced throughput
  by 30.3%. Enabling writes reduced the read-only result by a further 8.4%; the
  complete read/write path was 36.1% below the no-screening core.
- Increasing the model-sharded `1b` run from global batch 1 to 4 improved median
  throughput by 2.73 times. Both configurations fit in v5e HBM and completed
  finite forward, backward, and optimizer updates.
- A separate 2026-07-14 small-model audit with vocabulary parallelism,
  rematerialization, and sequence chunking enabled reported 15 forward
  collectives: 12 all-reduces, 3 all-to-alls, and no all-gathers. This is a
  historical, different compile configuration from both functional records
  above.
- No OOM, TPU runtime restart, or hardware health error was observed. Cloud
  Monitoring agent errors occurred near the test window, but no causal link to
  the isolated slow steps was established.

These results establish useful single-host v5e execution for the current
`0.185b` and `1b` paths. They do not establish multi-host scaling, 7B fit or
throughput, controlled run-to-run variance, power or cost efficiency, or
XProf-guided collective efficiency.

The temporary node `rwkv7m-bench-260714-9faf9aa` was deleted after the run. The
delete operation completed successfully, and the target name was absent from
all four zones in which allocation had been attempted.

## Multi-Host TPU v5e-16 Scaling Validation

A first larger-slice validation was attempted on 2026-07-16 from commit `7b67cb6`
using a temporary `v5litepod-16` in `us-central1-a`. The slice exposed 16 TPU
v5e chips through four worker VMs. The intended controlled run was the current
`0.185b` read/write-screening path at context 128, global batch 16, and a
`data=16` mesh, with one sample per device and six prefetched training steps:

```bash
JAX_COORDINATOR_ADDRESS=<worker-0-internal-ip>:12356 \
JAX_NUM_PROCESSES=4 \
JAX_PROCESS_ID=<0..3> \
JAX_COMPILATION_CACHE_DIR=/home/rumia/jax_cache \
  uv run rwkv7m-train-binidx-dp \
  --data-file data/bench/synthetic \
  --model-preset 0.185b \
  --ctx-len 128 \
  --global-batch-size 16 \
  --steps 6 \
  --phase read_write \
  --mesh-axis-names data \
  --mesh-axis-sizes 16 \
  --prefetch-size 2 \
  --disable-python-gc
```

No throughput value was produced. A preliminary worker-0-only launch was
invalid because a pod-slice runtime must be initialized by all workers and is
excluded from measurement. In the four-process launch, workers 1 and 2 failed
while evaluating `mesh.local_mesh` with:

```text
ValueError: devices connected to a single host must form a contiguous subcube
of the global device mesh
```

The other workers then exited through the expected JAX coordination-service
shutdown path. A provisional retry using
`create_device_mesh(contiguous_submeshes=True)` reproduced the same error.
Therefore this attempt establishes a multi-host mesh-ordering defect, not a
performance or scaling result. It must not be compared numerically with the
four-device records above.

That initial diagnosis was incomplete. A JAX process boundary inside one TPU
slice is not necessarily a DCN boundary. The actual failing call was the input
pipeline's use of `mesh.local_mesh`; process-local input assembly does not
require that convenience view. The corrected input path derives the logical
`data` coordinates owned by each process directly from `mesh.devices` and gives
replicated processes identical sampler streams.

The JAX 0.10.0 v5e devices in the corrected real-TPU run did not expose
`slice_index`, so the mesh used the compatibility
`create_hybrid_device_mesh(..., process_is_granule=True)` path. On this physical
`4x4` slice, each process was a contiguous `2x2` quadrant and the fallback
passed every gate below. When `slice_index` is available, one observed slice
uses topology-aware `jax.make_mesh()` and multiple slices use
`create_hybrid_device_mesh()` with the slice as the outer granule. If TPU
coordinates repeat without slice metadata, the builder now fails rather than
silently constructing a DCN-unaware process mesh. This follows the
[JAX multi-process guidance](https://docs.jax.dev/en/latest/multi_process.html)
and the [hybrid mesh API](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.mesh_utils.create_hybrid_device_mesh.html).

The v5e-16 slice was READY from approximately 13:39:38 JST until the delete
request at 14:07:46 JST. At the
[documented on-demand rate](https://cloud.google.com/tpu/pricing) of 1.20 USD
per v5e chip-hour, the estimated READY-state cost is about 9.00 USD; including
the delete-operation interval gives a conservative upper estimate of about
9.50 USD. This is an estimate rather than an exported billing record.

Fallback allocation requests did not start billable resources: `v5litepod-8`
was rejected by the serving quota, while `v6e-8` and `v5p-8` were rejected by
accelerator permissions. The v5e-16 delete operation completed successfully.
Final TPU VM listings for `us-central1-a`, `us-east5-b`, `us-east1-d`, and
`us-east5-a` were empty, and describing `rwkv7m-scale16-260716` returned
`NOT_FOUND`.

The corrected implementation was then validated later on 2026-07-16 on a new
temporary `v5litepod-16` in `us-central1-a`. It exposed a physical `4x4`
topology as four worker VMs with four v5e devices each. JAX 0.10.0 reported 16
global devices and four processes. `scripts/audit_multihost_mesh.py` executed
the following exact-value gates on every worker:

| Logical mesh | Process relationship on the data axis | Assembly | JIT elementwise | Global reduction |
| --- | --- | --- | --- | --- |
| `data=16` | four distinct data shards per process | pass | pass | pass |
| `data=4, model=4` | one distinct data shard per process | pass | pass | pass |
| `data=2, model=8` | each data shard replicated across two processes | pass | pass | pass |
| `data=1, model=16` | the same data shard replicated across all four processes | pass | pass | pass |

All four layouts completed with exact expected checksums. This validates
`jax.make_array_from_process_local_data(..., global_shape=...)` for both
distinct and cross-process replicated data shards. The replica rule is
essential: JAX can silently accept different process-local values for a
replicated global shard, as described in the
[distributed data loading guide](https://docs.jax.dev/en/latest/distributed_data_loading.html).

The complete NNX audit then used `data=1, model=16`, `d_model=128`, 16 heads,
read/write screening, vocabulary parallelism, recurrent chunk size 2, and head
chunk size 4. It completed a finite forward pass, backward pass, and optimizer
step 1 with loss `3.41725206`. The sharding contract contained both row- and
column-parallel kernels; WKV state remained
`P("data", "model", None, None)` and screening slots remained
`P("data", None, "model")`. The compiled forward executable contained 30
collectives: 18 all-gathers and 12 all-reduces. These counts are a correctness
record for this deliberately tiny, communication-heavy shape, not a tuned
production target.

Finally, the actual binidx distributed CLI completed one read/write-screening
training update on the same `data=1, model=16` mesh with finite loss
`4.276630`. The printed `0.36 token/s` included compilation and is deliberately
excluded from performance reporting. Its final checkpoint exposed two separate
TRC issues:

- the Flax backend cannot gather a non-addressable global train state on process
  0;
- Orbax cannot complete a multi-process checkpoint when each worker is given
  the same-looking path on its private local `/tmp` filesystem.

The CLI now rejects both configurations before training. Multi-process runs
must select Orbax and provide an explicit `--checkpoint-dir` on storage shared
by every worker. `gs://` paths are preserved through the Orbax boundary, and
the TPU extra installs `gcsfs` for the project-owned metadata files. Local run
artifacts remain under `--output-dir`. This storage correction was implemented
after the slice was deleted and therefore still requires a real GCS
save/restore and preemption drill.

The corrected slice was READY at 15:04:49 JST. Deletion was requested at
15:19:09 and completed at 15:21:58. At the
[documented on-demand rate](https://cloud.google.com/tpu/pricing) of 1.20 USD
per v5e chip-hour, READY-to-delete-request time is approximately 4.59 USD; a
conservative estimate through delete completion is approximately 5.49 USD.
These are rate-based estimates, not exported billing records. The delete
operation completed successfully, the target subsequently returned
`NOT_FOUND`, and a project-wide TPU API listing returned zero nodes.

This is a real multi-process validation within one TPU slice. It does not prove
multi-slice DCN placement, pod-scale collective efficiency, checkpoint I/O to
GCS, resume after preemption, 7B fit, or throughput scaling.

On TPU VMs, install from the checkout or package in the same way, then verify JAX sees TPU devices:

```powershell
uv run python -c "import jax; print(jax.devices()); print(jax.process_count(), jax.process_index())"
```

## Data

Use RWKV-LM-V7 compatible `.bin/.idx` files and pass the prefix path without suffix:

```powershell
uv run rwkv7m-make-binidx data/corpus.jsonl --output-prefix data/corpus --ctx-len 512
```

For accelerator-only throughput runs, create the deterministic synthetic input
used by the scaling procedure:

```powershell
uv run python scripts/prepare_synthetic_binidx.py `
  --output-prefix data/bench/synthetic `
  --tokens 4194304 `
  --vocab-size 65536
```

This input measures execution throughput only and must not be used for model
quality or convergence claims.

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
  --output-dir /tmp/rwkv7m-run `
  --checkpoint-dir gs://YOUR_BUCKET/rwkv7m/run-001 `
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

Each global microbatch must be divisible by the logical `data` mesh size. A
process loads the logical data shards addressed by its devices. When a model
replica spans processes, every process participating in that replica loads the
same deterministic data-shard stream.

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
- topology-aware `jax.make_mesh()` when one TPU slice is identified,
  slice-granule `create_hybrid_device_mesh()` across identified TPU slices, and
  a real-v5e-validated process-granule fallback when slice metadata is absent
  but physical coordinates are globally unique.
- logical data-shard ownership derived from the global mesh without requiring
  `mesh.local_mesh`, including deterministic input replication when a model
  axis spans processes.
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
- real single-host TPU v5e throughput measurements for the `0.185b` and `1b`
  presets, including data/model mesh and screening-phase comparisons.
- real four-process TPU v5e-16 exact-array validation for `data=16`,
  `data=4/model=4`, `data=2/model=8`, and `data=1/model=16` meshes.
- real v5e-16 `data=1/model=16` screening forward/backward/optimizer validation
  and an actual binidx CLI training update.
- real TPU validation of the independent small NNX Orbax lifecycle probe across
  separate create/restore processes.
- mesh-data-shard-aware host binidx sampling with cross-process replica safety.
- data-parallel batch placement with `jax.make_array_from_process_local_data`.
- train/eval step boundaries for local distributed tests.
- recurrent/screening state reset by default for independent binidx chunks.
- sequential carry-state train/eval mode with automatic state reset at stream-lane wrap boundaries.
- validation with a separate carried eval state, so validation does not mutate the training stream state.
- process 0 metadata writing and checkpoint rotation.
- Orbax train-state checkpoint paths that preserve GCS URIs, plus fail-fast
  rejection of private local checkpoint paths on multi-process runs.
- distributed carry-state runtime checkpoint/resume for local and single-process distributed runs.
- `run_config.json`, `run_summary.json`, `best_eval.json`, `metrics.jsonl`, and `metrics.csv` output.
- best validation checkpoint tracking with rotation protection.
- safetensors artifacts with model and tokenizer metadata for external runtimes.

Pending:

- XProf profiling and further throughput tuning on multi-device TPU, including the
  three observed all-to-all collectives.
- real TPU pod validation of full RWKV7M sharded runtime-state checkpoint/resume
  for carried recurrent/screening state.
- real TPU pod validation of full RWKV7M Orbax checkpoint save/resume under
  sharded train states.
- real shared-GCS Orbax save/resume and preemption recovery on a TPU pod.
- real multi-slice TPU/DCN execution and topology audit.
- 7B execution.
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

`--output-dir` is intentionally local and holds process-0 run logs. A
multi-process job that writes checkpoints must also provide an explicit
`--checkpoint-dir` visible at the same path from every worker. On TRC, use a
dedicated GCS prefix or a verified shared mount; identical strings pointing to
private worker-local disks are not shared storage. Orbax requires all hosts to
participate in a save, and its per-process OCDBT fragments are finalized into a
single global view. See the
[Orbax checkpoint format guide](https://orbax.readthedocs.io/en/latest/guides/checkpoint/checkpoint_format.html).

Resume works for both backends:

```powershell
uv run rwkv7m-train-binidx-dp `
  --data-file data/corpus `
  --ctx-len 512 `
  --global-batch-size 128 `
  --steps 1000 `
  --resume gs://YOUR_BUCKET/rwkv7m/run-001/ckpt-00001000 `
  --output-dir /tmp/rwkv7m-run `
  --checkpoint-dir gs://YOUR_BUCKET/rwkv7m/run-001 `
  --checkpoint-backend orbax `
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

`run_config.json` stores CLI args, model config, process count, device count,
exact runtime/package versions, logical mesh axes, and every device's logical
index, process index, physical coordinates, and slice index when available.
When `--param-axis-name` is set, it also records a parameter partition summary
with shapes and `PartitionSpec` strings. `run_summary.json` is updated during
training with status, current step, completed steps, token counts, latest
checkpoint, last train/eval records, and best eval. Metric records include
`split`, `step`, `loss`, screening metrics, tokens, and `tokens_per_sec` for
train steps. The NNX distributed CLI synchronizes the complete updated
model/optimizer state at the end of the timing window.

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

The audit reads `checkpoint_dir` from `run_config.json`; use
`--checkpoint-dir` to override it. With the TPU extra installed, this can be a
`gs://` prefix while `output_dir` remains local.

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
