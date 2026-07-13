# NVIDIA Blackwell validation (2026-07-13)

This record separates verified RWKV-7 core compatibility from training-stack and
throughput claims. Generated archives and logs remained on the disposable test
server and are not repository runtime dependencies.

## Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, compute capability 12.0.
- Driver: 580.159.03.
- Official CUDA build: CUDA 13.0 compiler and development libraries, PyTorch
  2.13.0+cu130.
- Official source: RWKV-LM-V7 commit
  `665472dab30952de9379a3a3a01eaa3453f1ad4e`.
- Official Python environment: isolated Python 3.12 `uv` venv.
- Local backend: JAX 0.10.0 CUDA 13, BF16 operations.

All eight official x070 CUDA extensions compiled for `sm_120`.

## Fixed-weight core results

The official fused CUDA model and the no-screening local JAX core used the same
converted BF16 weights, token IDs, zero recurrent state, cross-entropy plus RWKV
L2Wrap loss, and no optimizer state carried from an earlier step.

| Seed | Batch | Tokens | Logit max abs | Gradient cosine | Gradient relative L2 | Total-loss abs diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 1 | 16 | 0.00613737 | 0.99999595 | 0.00287596 | 0.00010109 |
| 2026 | 2 | 16 | 0.00669563 | 0.99999631 | 0.00272054 | 0.00013971 |
| 777 | 1 | 32 | 0.00581348 | 0.99999624 | 0.00275848 | 0.00011301 |

All 69 mapped tensors passed. The three layer-0 `v0`/`v1`/`v2` tensors allocated
but not read by official x070 were explicitly validated and ignored.

Direct official `deepspeed.ops.adam.FusedAdam` steps were also compared with the
local optimizer after gradient clipping, AdamW decay grouping, and the official
2x learning-rate rule for `att.w0`. Both updated parameter trees were cast to BF16
before comparing the pure update deltas.

| Seed | Batch | Update cosine | Update relative L2 | Update max abs |
|---:|---:|---:|---:|---:|
| 1234 | 1 | 0.99806010 | 0.06228937 | 0.00199890 |
| 2026 | 2 | 0.998252 | 0.0591314 | 0.00199890 |

All 69 tensors passed an absolute tolerance of 0.002. These are one-step
compatibility results, not evidence that long training trajectories are identical.

## Local CUDA load smoke

After warmup, the full screening model completed these existing CUDA smoke cases:

- L8-D512, batch 4 x 128 tokens: 17.36 ms mean forward time.
- L12-D768, batch 8 x 256 tokens: 66.43 ms mean forward time, 30,827 tokens/s.

JAX emitted repeated CUDA timer timeout warnings on the virtualized rental host.
The forward numbers are useful as a host-specific smoke result, but should not be
presented as a controlled hardware benchmark without rerunning on an uncontended,
non-virtualized machine with variance and power measurements.

## Training-stack limitation found

The official Lightning training CLI did not complete on this rental host:

- `deepspeed_stage_2` segfaulted while initializing DeepSpeed distributed state.
- the single-GPU Lightning `auto` strategy reached the official kernel but passed
  FP32 input to a BF16-only custom operator under Lightning 1.9.5 AMP.

Direct official BF16 fused forward, backward, L2Wrap, and FusedAdam execution all
succeeded. The observed CLI failures therefore do not contradict the core numeric
results, but upstream multi-step learning curves and official throughput remain
unverified on this machine. Those measurements require a compatible, preferably
non-virtualized NCCL/DeepSpeed environment.
