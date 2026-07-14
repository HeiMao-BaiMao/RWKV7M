# NVIDIA L40S Pallas performance validation

## Summary

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
runner sustained 8,516 token/s in its corresponding run. This 1.11x result is
useful evidence that Pallas is no longer the limiting issue for the tested
shape; it is not a claim that the two training stacks are instruction-for-
instruction identical.

The most important remaining result is negative: state-level screening now
dominates this small model's GPU cost. Against the canonical FFN-2688
screening-free core, read-only screening reduced time-weighted throughput by
58.7%, and read/write screening reduced it by 60.0%. The write phase accounted
for only another 3.3% relative to read-only screening.

## Environment and revisions

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

The local-to-official sustained-throughput ratio is 1.105x. This is a
controlled engineering comparison of the same context, batch, layer count,
model width, head shape, vocabulary, and actual FFN width. It still includes
different model wrappers, loss/optimizer implementations, parameterization,
and timing harnesses, so a small lead should not be interpreted as a universal
backend ranking.

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

## Screening result

The canonical FFN-2688 comparison isolates the state-level-screening feature:

| Configuration | Time-weighted token/s | Change from no screening |
| --- | ---: | ---: |
| No screening | 9,378 | baseline |
| Read-only screening | 3,875 | -58.7% |
| Read/write screening | 3,747 | -60.0% |

The write phase adds 3.3% overhead relative to the read-only mechanism. The
read/search path is therefore the primary remaining compute target. This
reverses the old pre-Pallas diagnosis: once WKV is persistent, screening is no
longer a secondary 10% effect. Its validation or sample-efficiency benefit must
justify a roughly 2.4-2.5x throughput cost in this batch-1 L40S profile, or its
implementation needs fusion and layout work.

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
  already competitive with the directly executed fused upstream runner;
- keep FFI as an explicit registration boundary, not an automatic path;
- tune or fuse state-level screening before spending effort on another WKV
  implementation for this shape;
- use one L40S for the `0.185b` throughput profile on this `SYS` host, and use
  two-way sharding only when capacity requires it;
- repeat complete train-step measurement on the current TPU commit before
  making full-stack TPU-versus-L40S claims.
