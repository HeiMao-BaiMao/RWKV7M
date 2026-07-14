# NVIDIA L40S validation (2026-07-13 to 2026-07-14)

This record covers the direct upstream training runner and the Ada compatibility
path. Generated datasets, checkpoints, extension binaries, and logs remained on
the disposable server and are not repository dependencies.

> **Historical baseline:** the main throughput section below predates the
> persistent Pallas WKV implementation. See the
> [post-Pallas L40S report](gpu_l40s_pallas_performance.md) for current results.

## Environment

- GPU: NVIDIA L40S, 46,068 MiB visible memory, compute capability 8.9.
- Driver: 580.159.03.
- System CUDA toolkit: 12.6.
- Official environment: Python 3.12 uv venv, PyTorch 2.13.0 and DeepSpeed 0.19.2.
- Local environment: Python 3.13 uv venv, JAX 0.10.0 CUDA 12.
- Official source: RWKV-LM-V7 commit
  `665472dab30952de9379a3a3a01eaa3453f1ad4e` plus the tracked Ada atomic patch.

The host required the OS `python3.12-dev` package in addition to the CUDA
compiler and normal build tools because PyTorch C++ extensions include
`Python.h`.

## Current NNX L40S x2 performance record (2026-07-14)

A follow-up pass at commit
`9faf9aa8e6c7db51a5e6fd4d39d586c835c71146` tested the current NNX model,
rematerialization, sequence chunking, screening, and explicit mesh paths. The
external disposable host had two L40S GPUs. `nvidia-smi topo -m` reported a
`SYS` connection between them rather than NVLink, with GPU 0 and GPU 1 attached
to different NUMA nodes. No GCP GPU resources were used.

The local environment was Python 3.13.14, JAX/jaxlib 0.10.0 with the CUDA 12
plugin, and Flax 0.12.7. The input was a synthetic RWKV-compatible binidx file
containing 4,194,304 tokens. It removes storage and network variability but is
not a model-quality measurement. The official comparison used RWKV-LM-V7
commit `665472dab30952de9379a3a3a01eaa3453f1ad4e` plus the tracked Ada atomic
patch and its BF16 fused CUDA kernels.

The distributed CLI starts each step timer after yielding the prefetched batch
and blocks on the complete updated model and optimizer state before stopping
it. Step 1 was excluded because it includes XLA compilation. The mean retains
all later stalls; the median better represents the normal step. The official
wrapper requested FFN 2688, but the validated upstream x070 `ChannelMix`
implementation actually constructs FFN 3072 matrices directly. It is not the
state-level-screening architecture and has a different model implementation
and parameterization. The post-Pallas comparison corrects this shape mismatch
and records the actual parameter count.

| Implementation and model | Context | Global batch | Mesh | Measured steps | Median token/s | Mean token/s | Post-step-1 range |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Current NNX core dimensions, no screening | 512 | 1 | 1 L40S | 14 | 1,058 | 1,088 | 1,038-1,159 |
| Current NNX `0.185b`, read/write screening | 512 | 1 | 1 L40S | 14 | 945 | 925 | 651-959 |
| Current NNX core dimensions, no screening | 512 | 4 | 1 L40S | 11 | 4,094 | 4,105 | 4,069-4,170 |
| Current NNX `0.185b`, read/write screening | 512 | 4 | 1 L40S | 11 | 3,334 | 3,226 | 2,331-3,365 |
| Current NNX `0.185b`, read/write screening | 512 | 2 | `data=2` | 14 | 1,332 | 1,307 | 979-1,362 |
| Current NNX core dimensions, no screening | 512 | 8 | `data=2` | 11 | 6,183 | 6,200 | 6,083-6,323 |
| Current NNX `0.185b`, read/write screening | 512 | 8 | `data=2` | 11 | 4,752 | 4,743 | 3,708-5,004 |
| Current NNX `1b`, read/write screening | 256 | 1 | `data=1, model=2` | 7 | 225 | 217 | 169-226 |
| Official fused RWKV-LM-V7 core | 512 | 1 | 1 L40S | 11 | 9,608 | 9,486 | 8,826-9,686 |

The same compiled official fused runner was extended to 200 steps for a longer
power sample. Excluding step 1, its median was 9,529 token/s and its mean was
9,691 token/s. This longer run is not substituted into the table because the
local rows are much shorter.

Main findings:

- At batch 1, the official fused core was 9.1 times faster by median than the
  current screening-free NNX core, and 10.2 times faster than the complete
  read/write path. Screening is therefore not the primary GPU bottleneck; the
  generic JAX recurrence path is already much slower before screening is added.
- Doubling from one GPU to two with the per-device batch held at one increased
  read/write throughput by 1.41 times, or 70.5% parallel efficiency. At a
  per-device batch of four, read/write scaling was 1.43 times (71.3% efficiency)
  and the no-screening core scaled by 1.51 times (75.5% efficiency). The lack of
  NVLink makes model parallelism particularly unattractive except where it is
  needed for capacity.
- The read/write mechanism reduced median throughput relative to the
  no-screening core by 10.6% at single-GPU batch 1, 18.6% at single-GPU batch 4,
  and 23.1% at two-GPU global batch 8. A validation or sample-efficiency gain is
  still required before this overhead can be judged worthwhile.
- The current `1b` preset completed finite forward, backward, and optimizer
  updates with two-way model sharding. Its first measured step, which included
  compilation, was 1.45 token/s, and the complete command took 358.5 seconds.
  Later steps normally reached approximately 225 token/s. The approximately
  35 GiB reported per device is primarily JAX/XLA's reserved memory pool and
  must not be interpreted as exact live tensor usage.
- During the `1b` intervals with at least 80% reported GPU utilization and the
  XLA pool resident, median utilization and power were 99% and 117.7 W on GPU 0,
  and 93% and 116.1 W on GPU 1. Maximum sampled power was 127.8 W, far below the
  350 W board limit. High `nvidia-smi` utilization alongside low power is
  consistent with latency, synchronization, or low-arithmetic-intensity work;
  it does not prove high tensor-core utilization.
- During the longer official fused run, active GPU-0 samples had median reported
  utilization of 44% and median power of 148.2 W, with a maximum of 159.6 W.
  Even the much faster fused reference did not approach the power limit in this
  small batch-1 profile. Nsight Systems/Compute or an equivalent profiler is
  required to separate kernel-launch, memory, recurrence-dependency, and
  collective limits.
- JAX emitted CUDA timer timeout warnings in some short runs, and isolated slow
  steps are visible in the ranges. The medians are suitable for engineering
  direction, but controlled repetitions and profiler traces are still needed
  for publication-grade claims.

At this historical revision, the architecture was sufficient for functional
research experiments and for fitting the tested 1B preset on two L40S GPUs,
but the generic GPU recurrence was not performance-sufficient. The subsequent
Pallas work resolved that primary bottleneck for the tested 0.185B shapes and
made screening the largest measured remaining cost. On this particular host,
two-way model parallelism still crosses a slow `SYS` link.

## Compatibility findings

The downloaded MiniPile index contains multiple items, while official
`MyDataset` always samples item zero with a global offset. The comparison path
now creates a 62-byte single-item index and a hard link to the original 2.8 GiB
`.bin`; token bytes are not copied or changed.

Three official CUDA helpers used vector `atomicAdd(float2*, float2)` without an
architecture guard. That operation does not compile for Ada `sm_89`. The tracked
patch retains the vector operation for `sm_90+` and performs two scalar float
atomics below `sm_90`. The actual upstream diff hash is recorded in run and
parity metadata.

## Direct upstream training

The Lightning/DeepSpeed-free runner completed all tested official BF16 fused
forward, backward, L2Wrap, FusedAdam, sampler, metric, and checkpoint paths.

| Profile | Global / micro batch | Steps | Loss sequence | Post-warmup throughput |
| --- | ---: | ---: | --- | ---: |
| smoke, L4-D256, ctx128 | 1 / 1 | 4 | 11.1746, 11.1508, 10.8458, 10.4493 | 6,801–7,596 token/s |
| smoke, L4-D256, ctx128 | 2 / 1 | 3 | 11.1796, 11.1912, 10.8199 | 7,241–7,840 token/s |
| small, L12-D768, ctx512 | 1 / 1 | 3 | 11.1904, 10.4899, 10.6619 | 8,319–9,348 token/s |

The global-batch-two case exercised two micro-batch backward passes per
optimizer step on the single GPU. These short runs establish execution and
artifact correctness, not converged model quality or paper-grade throughput.

## Core parity after the Ada patch

The fixed-weight official fused reference and local JAX verifier passed for seed
1234, batch 1, and 16 tokens:

- logit max absolute difference: 0.00613725;
- gradient cosine similarity: 0.999996;
- gradient relative L2: 0.00287618;
- optimizer-update cosine similarity: 0.997937;
- optimizer-update relative L2: 0.0642298;
- all 69 mapped tensors passed.

These values are consistent with the earlier Blackwell results and support the
claim that the scalar atomic fallback preserves the tested numerical behavior.

## Local RWKV7M CUDA smoke

The existing full CUDA smoke suite passed:

- L8-D512, batch 4 x 128 forward: 18.28 ms mean;
- L4-D256, batch 2 x 32 train step: 2,833.82 ms mean;
- L12-D768, batch 8 x 256 full screening forward: 107.14 ms mean and
  19,116 token/s;
- reported peak memory: 34,601 / 46,068 MiB.

JAX emitted CUDA timer timeout warnings, so the timing values are host-specific
smoke measurements. The capacity and successful execution results are valid;
controlled throughput claims still require repeated runs, variance reporting,
and an uncontended host.

## Full RWKV7M 0.19B training

The full local mechanism was subsequently tested at the same core dimensions as
the upstream `small` profile: L12-D768, FFN 2688, vocabulary 65,536, context
length 512, BF16, and global batch 1. State-level screening used 16 slots with
the read/write phase enabled on layer 6. The resulting model contained
184,985,222 trainable parameters (approximately 0.185B).
This architecture is now the canonical `model_preset("0.185b")` and
`configs/rwkv7m-0.185b.json.example` shape. The newer NNX remat/chunk execution
options still require a post-integration CUDA rerun before being described as
part of this historical validation record.

One optimizer step was run from initialization, then the Orbax train-state
checkpoint was restored and training continued through steps 2 and 3. The loss
sequence was 11.163989, 10.252852, and 10.743930. Both read and effective-write
relevance metrics were non-zero at every step, confirming that the full
read/write screening path participated in the compiled forward and backward
computation. The final step-3 checkpoint was 1.6 GiB, checkpoint rotation
removed the older steps, and `rwkv7m-audit-dp-run --require-complete` reported
zero errors and zero warnings across all three metric records.

Peak observed GPU allocation was 34,621 MiB. The second post-restore training
step reached 1,042.63 token/s, while the first step in each process included
JAX compilation and initialization and was approximately 9 token/s. These
short-run rates are execution diagnostics, not controlled throughput results.

This run exposed an Orbax boundary issue: recent Orbax/TensorStore versions
require absolute serialization paths, but the CLI allowed a relative
`--output-dir`. The wrapper now resolves only the internal paths passed to
Orbax while preserving the public checkpoint path representation. A regression
test covers save and restore with a relative output directory.
