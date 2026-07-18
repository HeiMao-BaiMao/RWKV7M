# NVIDIA L40S Pallas performance validation

> Results dated 2026-07-14 below are the historical two-L40S baseline. The
> 2026-07-16 gate used one L40S only and is the current result for projected
> Pallas screening, the tiled training head, and fixed-batch train compute.

The screening measurements here predate the corrected competitive Screening
v2 path. In the semantics vocabulary of
[`RWKV7M.paper.md`](../RWKV7M.paper.md) v5, they cover the
earlier projected legacy recurrence, not `screening-v4-competitive` and not the
design-only `screening-v5-core` or `screening-v5-retention` profiles.

## Summary

On 2026-07-16, the latest single-L40S path passed real Triton lowering and
gradient parity for WKV, projected state-level screening, and the tiled
training head. At the tracked `T=128, B=1` shapes, Pallas screening was 16.71x
faster in forward and 15.76x faster in forward plus backward than its projected
`lax.scan` reference. WKV was 8.04x and 14.92x faster, respectively.

The schema-v2, fixed-device-batch `0.185b` train-compute gate measured 11,561
token/s without screening, 9,708 token/s with read-only screening, and 10,365
token/s with read/write screening. These correspond to 19.1% and 11.5% higher
median complete-step latency than the no-screening core. The read/write result
being faster than read-only is an observed single-run compiler outcome, not
evidence that writes are intrinsically free or beneficial; a repeated profile
is required before attributing that difference.

This materially replaces the pre-projected result in which screening cost
roughly 2.4-2.5x throughput. The recurrence itself is no longer the dominant
GPU defect. The remaining full-step cost includes screening projections,
gradient work, and extra optimizer parameters, and has not yet been isolated
with a kernel timeline.

## Current single-L40S gate (2026-07-16)

| Item | Value |
| --- | --- |
| Host | disposable external server; no GCP GPU resources used |
| GPU | 1 x NVIDIA L40S, 46,068 MiB, 350 W limit |
| Driver | 580.159.03 |
| Stack | Python 3.13.14, JAX/jaxlib 0.10.0, CUDA 12 plugin |
| Source base | `33e5594`; the GPU screening lowering fix is committed with this report |
| Dataset | deterministic 4,194,304-token synthetic binidx |
| Fixed batch | batch 1 x 512 tokens, SHA-256 `e2ca3f794addd9202928689e675f6cd2f473e50107603369dc808bdd7f09fac9` |

The real accelerator suite ran with `interpret=False`:

```bash
uv run pytest -q \
  tests/test_wkv_pallas_accelerator.py \
  tests/test_screening_pallas_accelerator.py \
  tests/test_training_head_pallas_accelerator.py
```

All four tests passed. This includes WKV forward/state and gradient parity,
screening parity for all six outputs and all 15 differentiable inputs, the
tiled-head custom VJP, and NNX model integration.

The first screening compile exposed two Triton-only defects that interpret
mode did not reveal. Triton could not lower an eight-value `jnp.stack`, and
Pallas GPU stores require the value dtype to match the destination `Ref`
exactly. The GPU kernel now stores each statistic scalar directly and casts
FP32 internal results only at typed output boundaries. FP32 recurrence math is
unchanged.

### Projected screening recurrence

Inputs were fixed and device-resident, compilation was excluded, CPython
cyclic GC was disabled, and every one of 50 calls after 5 warmups was
synchronized.

| Shape | Pallas forward | Reference forward | Speedup | Pallas fwd+bwd | Reference fwd+bwd | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `T=128, B=1, slots=16, slot=128, key=value=64` | 0.247 ms | 4.127 ms | 16.71x | 0.620 ms | 9.773 ms | 15.76x |

Maximum output absolute error was `4.77e-7`. All gradients passed the
accelerator test tolerance; the largest absolute difference was `4.88e-4` for
the initial read keys, and the largest relative L2 difference was `0.00215`
for delta values.

### WKV recurrence rerun

The same method used 100 timed calls at `T=128, B=1, H=12, N=64`:

| Path | Forward | Forward + backward |
| --- | ---: | ---: |
| Triton Pallas | 0.202 ms | 0.518 ms |
| `lax.scan` reference | 1.623 ms | 7.728 ms |
| Speedup | 8.04x | 14.92x |

Activation and final-state maximum absolute errors were `1.19e-7` and
`4.66e-10`. All seven tested gradients passed tolerance.

### Tiled training-head gate

This gate used 128 positions, hidden size 768, vocabulary 65,536, BF16, 5
warmups, and 50 timed calls. Ratios are `full-XLA / tiled-Pallas`; values below
1.0 mean that tiling is slower. The untiled BF16 logits tensor is 16 MiB.

| Vocabulary tile | Largest tiled logits | Forward ratio | Forward + backward ratio |
| ---: | ---: | ---: | ---: |
| 4,096 | 1 MiB | 0.817x | 0.751x |
| 8,192 | 2 MiB | 0.888x | 0.847x |
| 16,384 | 4 MiB | 0.905x | 1.011x |
| 32,768 | 8 MiB | 0.588x | 0.771x |
| 65,536 | 16 MiB | 0.361x | 0.641x |

All component errors were zero; maximum gradient error was `1.91e-6`. Tile
16,384 is the only tested size that matched untiled forward-plus-backward
latency while reducing the largest logits tensor by 4x. Tiling remains an
opt-in memory path because forward alone was still 9.5% slower and the result
does not establish a speed advantage across larger head chunks.

### Fixed-batch complete train compute

These runs used the same fixed device-resident batch, `0.185b`, no
rematerialization, no recurrent chunking, a 512-token head chunk, vocabulary
tile 16,384, 5 warmups, 20 timed calls, disabled cyclic GC, and synchronization
after every call. Dataset sampling, host transfer, compilation, checkpointing,
logging, and host metrics are excluded.

| Variant | Parameters | Forward | Backward | Optimizer | Complete step | Median token/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No screening | 183,956,736 | 6.688 ms | 34.379 ms | 16.695 ms | 44.286 ms | 11,561 |
| Read-only screening | 184,878,725 | 7.278 ms | 38.184 ms | 17.005 ms | 52.741 ms | 9,708 |
| Read/write screening | 184,985,222 | 7.331 ms | 36.647 ms | 16.691 ms | 49.397 ms | 10,365 |

The separately timed phases are diagnostic windows and do not add to the
complete-step time because the complete step is compiled and measured without
intermediate barriers. One CUDA autotuning warning reported a delay-kernel
timeout during compilation for each screening variant; compilation was outside
the timing window, but an Nsight repeat is still needed before attributing the
read-only/read-write difference.

Only one GPU was used for this current gate. Two-GPU reruns were explicitly
out of scope for budget reasons; the older two-GPU section remains historical
evidence only.

## Historical 2026-07-14 validation

On 2026-07-14, the production Triton-Pallas WKV path was validated on a
disposable two-GPU L40S host. The real-GPU forward and gradient parity gate
passed, and the same-shape WKV microbenchmark showed a 15.09x median
forward-plus-backward speedup over the portable `lax.scan` reference. In a
memory-safe `0.185b` training configuration, replacing
only WKV increased median complete-step throughput from 764 to 3,181 token/s,
or 4.17x.

After removing recurrent and head chunking, disabling rematerialization, and
controlling CPython cyclic garbage collection, the screening-free local core
sustained 9,413 token/s. The directly executed upstream RWKV-LM-V7 fused CUDA
runner sustained 8,516 token/s in its corresponding run. These runs only show
that two different complete training stacks reached similar throughput on the
same L40S. Their 1.105 ratio is a reference comparison, not evidence that the
Pallas WKV kernel or the JAX stack is 1.105x faster.

At that revision, state-level screening dominated this small model's GPU cost.
Those values predate the projected Pallas screening recurrence and must not be
used as the current screening result.

## Historical environment and revisions

| Item | Value |
| --- | --- |
| Host | disposable external server; no GCP GPU resources used |
| GPUs | 2 x NVIDIA L40S, 46,068 MiB each, compute capability 8.9 |
| Interconnect | `SYS`; different NUMA nodes; no NVLink |
| Driver / system CUDA | 580.159.03 / 12.6 |
| Local stack | Python 3.13.14, JAX/jaxlib 0.10.0, CUDA 12 plugin, Flax 0.12.7 |
| Upstream stack | Python 3.12.3, PyTorch 2.13.0, DeepSpeed 0.19.2 |
| Upstream source | RWKV-LM-V7 `665472dab30952de9379a3a3a01eaa3453f1ad4e` plus the tracked Ada atomic patch |
| Measured local revision | `416f967` |
| Metadata-only follow-up | `52002de` records the actual upstream shape and fixes relative checkpoint resolution |
| Dataset | deterministic synthetic one-item RWKV binidx, 4,194,304 `uint16` tokens |

The data removes storage and network variance; it does not measure language
model quality. Step 1 was excluded from every train-step result because it
contains XLA or CUDA-extension compilation and initialization.

## Correctness gates

The real L40S test compiled with `interpret=False` and passed for BF16 vectors
and an FP32 state:

```bash
uv run pytest tests/test_wkv_pallas_accelerator.py -q
```

This checks activation and final-state parity plus gradients for all six WKV
vector inputs and the initial state. The benchmark's `T=128, B=1, H=12, N=64`
case measured maximum absolute differences of `9.5367e-7` for activation and
`2.3283e-10` for final state. All seven gradients passed the configured
tolerance; measured relative L2 errors ranged from approximately 0.00135 to
0.00924.

An explicit two-GPU `data=1, model=2` audit also completed finite forward,
loss `3.946409`, backward, and optimizer step 1 with screening and writes
enabled. Its compiled program contained three all-to-alls, two reduce-scatters,
and no all-gathers.

The first real-GPU backward compile exposed an integer-predicate remainder in
the Triton checkpoint-boundary lowering. Commit `fe22845` replaced that
expression with an equivalent division/multiply boundary test; the complete
accelerator parity test then passed. Interpret-mode parity alone would not have
found this lowering defect.

## WKV microbenchmark

The tracked harness is:

```bash
uv run python scripts/benchmark_wkv_accelerator.py \
  --time 128 --batch 1 --heads 12 --head-size 64 \
  --warmup 5 --iterations 100 --output out/wkv-l40s.json
```

Each path is independently JIT-compiled. Warmups are excluded, every one of
the 100 samples is synchronized, and the backward objective is the sum of
squared FP32-cast WKV activations. Inputs are BF16 and recurrent state and
accumulation are FP32.

| Shape `(T, B, H, N)` | Pallas forward | Reference forward | Speedup | Pallas fwd+bwd | Reference fwd+bwd | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128, 1, 12, 64 | 0.210 ms | 1.841 ms | 8.77x | 0.537 ms | 8.109 ms | 15.09x |
| 128, 4, 12, 64 | 0.200 ms | 1.788 ms | 8.95x | 0.550 ms | 9.325 ms | 16.97x |
| 512, 1, 12, 64 | 0.431 ms | 5.332 ms | 12.36x | 1.560 ms | 24.783 ms | 15.89x |
| 512, 4, 12, 64 | 0.433 ms | 6.689 ms | 15.46x | 1.623 ms | 33.215 ms | 20.47x |

These timings are one WKV recurrence, not model throughput. They establish
that the persistent Pallas structure removes the large GPU penalty of the
portable recurrence over all four tested shapes.

## Complete training steps

The distributed CLI starts timing after yielding the prefetched batch and
blocks on the complete updated model and optimizer state. `Time-weighted` below
means total measured tokens divided by total measured step time; unlike the
arithmetic mean of per-step token rates, it correctly charges slow steps.

| Implementation and configuration | Measured steps | Median token/s | Time-weighted token/s |
| --- | ---: | ---: | ---: |
| Local FFN-3072 no-screening core, Pallas, no chunk/remat, `--disable-python-gc` | 199 | 9,498 | 9,413 |
| Official fused RWKV-LM-V7 x070, actual FFN 3072 | 49 | 8,533 | 8,516 |
| Local canonical FFN-2688 core, no screening, Pallas, no chunk/remat, GC disabled | 99 | 9,523 | 9,378 |
| Local canonical `0.185b`, read-only screening, no chunk/remat, GC disabled | 99 | 3,960 | 3,875 |
| Local canonical `0.185b`, read/write screening, no chunk/remat, GC disabled | 199 | 3,763 | 3,747 |

The FFN-3072 local configuration contains 191,034,624 parameters. The official
runner contains 191,084,544. The upstream CLI accepts `--dim-ffn`, but the
validated x070 `ChannelMix` implementation actually constructs four-times-
width FFN matrices directly, so its actual FFN is 3072 rather than the
previously reported 2688. The comparison helper now records `actual_dim_ffn`
and `actual_parameter_count` instead of relying on the requested wrapper value.

The local-to-official sustained-throughput ratio is 1.105x. It is a reference
comparison only. The local window begins after a prefetched batch is available
and ends after synchronization of the complete updated state. The official
window also includes dataset sampling, CPU-to-GPU transfer, and the scalar
loss read; GC policy, measured step count, optimizer, and Python environment
also differ. The supported conclusion is therefore limited to: *different
complete learning stacks attained similar train throughput on the same L40S.*
It does not establish that Pallas WKV is faster than the official CUDA kernel,
or that JAX is faster than PyTorch.

## Strict compute-only comparison method

New measurements must use the tracked schema-v2 compute-only harnesses. First,
materialize one batch once and pass the identical NPZ to both full-step
runners:

```bash
uv run python scripts/prepare_compute_benchmark_batch.py \
  --data-file /data/train --ctx-len 512 --batch-size 1 \
  --magic-prime 999983 --output out/fixed-batch-b1-t512.npz

uv run python scripts/benchmark_local_train_compute.py \
  --fixed-batch out/fixed-batch-b1-t512.npz \
  --model-preset 0.185b --variant baseline \
  --ctx-len 512 --batch-size 1 --no-remat-blocks \
  --no-sequence-chunking --no-head-chunking \
  --benchmark-warmup 5 --benchmark-iterations 50 \
  --disable-python-gc --output out/local-train-compute.json

python scripts/benchmark_upstream_train_compute.py \
  --upstream-repo /opt/RWKV-LM/RWKV-v7/train_temp \
  --fixed-batch out/fixed-batch-b1-t512.npz \
  --n-layer 12 --n-embd 768 --dim-att 768 --dim-ffn 3072 \
  --vocab-size 65536 --benchmark-warmup 5 \
  --benchmark-iterations 50 --disable-python-gc \
  --output out/upstream-train-compute.json
```

Both outputs record the fixed-batch SHA-256, device/runtime identity, warmup,
iteration count, and GC policy. Sampling, host-to-device transfer, compilation,
checkpointing, logging, and host metrics are outside the timing windows.
Forward, backward, and optimizer are measured as independent synchronized
windows. A separate full-step window has no intermediate phase barriers and is
the throughput comparison value. The local backward window reuses one prepared
VJP residual; the official backward window prepares its autograd graph before
the timer. Optimizer windows likewise consume precomputed fixed gradients.
The records include optimizer identity and hyperparameters; Optax and FusedAdam
remain different implementations, so the phase split is required when
attributing a full-step difference.

Before quoting a ratio, run `scripts/compare_compute_benchmarks.py` on the two
JSON files. It refuses records with different schema, benchmark kind, device,
shape, input fingerprint, warmup, iteration count, or GC policy and maps the
framework-specific phase names explicitly.

WKV-only comparison uses the same PCG64-generated tensors, zero initial state,
seed, shape, warmup, iteration count, GC policy, and FP32-squared objective:

```bash
uv run python scripts/benchmark_wkv_accelerator.py \
  --time 128 --batch 1 --heads 12 --head-size 64 \
  --initial-state zero \
  --warmup 5 --iterations 100 --disable-python-gc \
  --output out/local-wkv-compute.json

python scripts/benchmark_upstream_wkv_compute.py \
  --upstream-repo /opt/RWKV-LM/RWKV-v7/train_temp \
  --time 128 --batch 1 --heads 12 --head-size 64 \
  --warmup 5 --iterations 100 --disable-python-gc \
  --output out/upstream-wkv-compute.json
```

The two WKV JSON records expose a pre-BF16 input fingerprint. A comparison is
invalid if that fingerprint, shape, method fields, or device identity differ.
Use local `pallas_training_forward`, rather than the tape-free
`pallas_forward`, against the autograd-enabled official forward. Backward and
combined windows expose gradients only for the six vector inputs on both
sides; the local zero initial state is fixed rather than a timed gradient
argument.
The 2026-07-16 local results above now satisfy the schema-v2 fixed-batch side of
this method. A matching upstream schema-v2 result has not been measured, so a
strict local-versus-official ratio is still unavailable.

The direct Pallas-versus-reference model comparison used identical local
configuration and data. With the tracked `0.185b` rematerialization and
recurrent-chunk defaults (`remat_blocks=True`, recurrent chunk 128) plus a
512-token head-chunk override, its 14 measured steps were:

| WKV backend | Median token/s | Time-weighted token/s | Range |
| --- | ---: | ---: | ---: |
| Triton Pallas | 3,181 | 3,131 | 2,594-3,198 |
| `lax.scan` reference | 764 | 746 | 570-780 |

The complete-step gain was 4.17x by median and 4.20x time-weighted. Increasing
the head chunk from the preset-inherited 128 tokens to 512 improved median
throughput from 2,962 to 3,181 token/s, or 7.4%. Removing both recurrent
chunking and rematerialization raised the short-run median to 3,686 token/s.

## CPython garbage collection

A 499-step screening-free run showed regular stalls approximately every 14 to
16 steps: median throughput remained 8,852 token/s, but 34 samples fell below
3,000 and the time-weighted result fell to 7,501 token/s. Disabling CPython's
cyclic collector eliminated those periodic stalls in a 199-step diagnostic.

The distributed training CLI therefore provides opt-in
`--disable-python-gc`. Reference counting remains active, and the prior cyclic-
GC state is restored in `finally` when the process exits. The 199-step tracked
run had no sample below 5,000 token/s and produced the 9,413 token/s result in
the table. The default remains unchanged because a 200-step test does not prove
that every long-running configuration is free of cyclic heap growth.

## Historical pre-projected screening result

The canonical FFN-2688 comparison isolates the state-level-screening feature:

| Configuration | Time-weighted token/s | Change from no screening |
| --- | ---: | ---: |
| No screening | 9,378 | baseline |
| Read-only screening | 3,875 | -58.7% |
| Read/write screening | 3,747 | -60.0% |

The write phase added 3.3% overhead relative to the read-only mechanism in
this old implementation. The current projected-Pallas gate supersedes these
throughput values: the corresponding complete-step latency overhead is now
19.1% for read-only and 11.5% for read/write in one schema-v2 run. The old
result remains useful only as the optimization baseline.

## Two-GPU behavior

The pair is connected through `SYS`, not NVLink. With the same memory-safe
screening configuration:

| Placement | Global / per-device batch | Median token/s | Scaling versus one GPU at the same per-device batch |
| --- | ---: | ---: | ---: |
| 1 L40S | 1 / 1 | 3,181 | baseline |
| 2 L40S, `data=2` | 2 / 1 | 2,960 | 0.93x total; 46.5% parallel efficiency |
| 1 L40S | 4 / 4 | 10,273 | baseline |
| 2 L40S, `data=2` | 8 / 4 | 10,283 | 1.00x total; 50.0% parallel efficiency |

The Pallas recurrence shortened local compute enough that replicated gradient
synchronization now dominates this small model on the slow cross-NUMA link.
Two-way model sharding is functionally valid and remains useful for capacity,
but data parallelism on this particular pair is not a throughput optimization
for `0.185b`.

## TPU comparison

The TPU v5e report used the same WKV shape, dtypes, objective, synchronized
per-call method, and 100 iterations. This is the cleanest available
cross-accelerator comparison:

| Accelerator and path | Forward | Forward + backward |
| --- | ---: | ---: |
| 1 L40S, Triton Pallas | 0.210 ms | 0.537 ms |
| 1 L40S, `lax.scan` reference | 1.841 ms | 8.109 ms |
| 1 TPU v5e device, TPU Pallas | 0.276 ms | 0.538 ms |
| 1 TPU v5e device, `lax.scan` reference | 0.307 ms | 0.766 ms |

The Pallas forward-plus-backward latencies are effectively equal for this one
small shape, while the generic recurrence is much less suitable for the GPU.
L40S Pallas forward latency is 24.0% lower. These figures do not compare device
price, full-model throughput, memory capacity, or pod scaling.

The older end-to-end TPU record used commit `9faf9aa`, four v5e devices, and the
pre-Pallas/pre-pipeline implementation. Its `0.185b` read/write data-parallel
batch-4 median was 11,931 token/s. A current single-L40S memory-safe batch-4 run
reached 10,273 token/s, but the code revisions and device counts differ. That
pair of numbers is historical context only and is not a controlled hardware
speedup result. A current-commit TPU train-step rerun is still required.

The temporary Google Cloud TPU VM used for the TPU report was deleted after
measurement, and the zone listing was verified empty. The L40S tests in this
report used only the user-provided external host.

## Profiling and limits

For the local matched-core run, active samples reported approximately 49%
median GPU utilization and 144 W median board power, with a 155 W maximum. The
official run reported approximately 39% median utilization, 140 W median power,
and a 146 W maximum under its sampling filter. Both are far below the 350 W
board limit, but board power alone cannot identify occupancy, dependency,
launch, or bandwidth limits.

Nsight Compute 2024.3 was present, but both normal and privileged invocations
returned `ERR_NVGPUCTRPERM`; the disposable environment did not expose the
required performance-counter capability. Profiler-instrumented elapsed times
were discarded. Register-spill, eligible-warp, and DRAM-throughput attribution
therefore remains unverified.

The evidence supports the following current decisions:

- keep Triton Pallas as the default Ada path; it passed real lowering and is
  compatible with a local complete stack that reached similar throughput to
  the directly executed fused upstream runner; strict compute-only ranking is
  still unmeasured;
- keep FFI as an explicit registration boundary, not an automatic path;
- profile and fuse the remaining screening projections/gradient work before
  spending effort on another WKV implementation for this shape;
- use one L40S for the `0.185b` throughput profile on this `SYS` host, and use
  two-way sharding only when capacity requires it;
- repeat complete train-step measurement on the current TPU commit before
  making full-stack TPU-versus-L40S claims.
