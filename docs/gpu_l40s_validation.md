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
