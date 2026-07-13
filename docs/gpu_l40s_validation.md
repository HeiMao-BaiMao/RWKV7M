# NVIDIA L40S validation (2026-07-13)

This record covers the direct upstream training runner and the Ada compatibility
path. Generated datasets, checkpoints, extension binaries, and logs remained on
the disposable server and are not repository dependencies.

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
