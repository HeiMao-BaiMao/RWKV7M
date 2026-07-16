# Single-GPU training optimization gate

This gate determines the production head, optimizer, and batch shape for one
NVIDIA GPU. It does not use or require a two-GPU host.

## Implemented comparison contract

`scripts/benchmark_gpu_train_matrix.py` runs the Cartesian product of:

- full-XLA and vocabulary-tiled Pallas training heads;
- Optax and the opt-in fused Pallas AdamW optimizer;
- requested per-device batch sizes.

Each cell runs in a fresh Python process. It receives a fixed device-resident
batch, excludes compilation and host transfer, disables rematerialization and
sequence chunking for the throughput gate, records forward, backward,
optimizer, and barrier-free complete-step latency, and requires exactly one
visible GPU. Capacity-probing failures, including OOM, are retained in the
summary instead of silently dropping the batch shape.

The fixed-batch contract now records two fingerprints. `sha256` identifies the
NPZ container bytes, while `content_sha256` hashes the ordered array names,
dtypes, shapes, and values. Cross-framework comparison uses the latter, so NPZ
serialization differences cannot make identical logical batches incomparable.
The comparator also requires matching layer count, model/FFN width, head
shape, and vocabulary size.

Example for the tracked 0.185B baseline:

```bash
uv run python scripts/benchmark_gpu_train_matrix.py \
  --data-file /data/synthetic \
  --model-preset 0.185b --variant baseline \
  --ctx-len 512 --batch-sizes 1 2 4 8 \
  --head-modes full_xla pallas_tiled \
  --optimizer-backends optax pallas_gpu_triton \
  --training-vocab-tile-size 16384 \
  --benchmark-warmup 5 --benchmark-iterations 20 \
  --disable-python-gc \
  --output-dir out/gpu-matrix \
  --summary out/gpu-matrix.json
```

For Blackwell/Hopper, replace `pallas_gpu_triton` with
`pallas_gpu_mosaic`. Do not select a backend only from its name: the complete
step and optimizer-only windows must both pass on the target architecture.

## Fused optimizer contract

The Pallas optimizer keeps the global gradient-norm reduction outside the
per-parameter kernel. For each local parameter shard, one Pallas call fuses:

```text
clip gradient
  -> update FP32 first and second moments
  -> bias correction
  -> Adam direction
  -> optional decoupled weight decay
  -> optional RWKV w0 2x multiplier
  -> emit FP32 update and moments
```

Its update equations, decay mask, `w0` multiplier, schedule count, and two-step
behavior pass the Optax parity tests in CPU interpret mode. The Pallas optimizer
is deliberately opt-in through `--optimizer-backend`; `optax` remains the
default until a real-GPU complete-step gate proves a net gain. TPU training
continues to use Optax.

## Nsight procedure

The benchmark compiles and warms up before opening a CUDA Profiler API range.
This keeps XLA/Pallas compilation out of Nsight Systems and lets the same
workload profile `forward`, `backward`, `optimizer`, or `full_step`.

First collect a Systems timeline:

```bash
uv run python scripts/profile_gpu_train_compute.py \
  --tool systems \
  --output-prefix out/nsight/full-step \
  --manifest out/nsight/full-step.json \
  -- uv run python scripts/benchmark_local_train_compute.py \
    --fixed-batch out/gpu-matrix/fixed-b1-t512.npz \
    --model-preset 0.185b --variant baseline \
    --ctx-len 512 --batch-size 1 \
    --no-remat-blocks --no-sequence-chunking --no-head-chunking \
    --no-training-vocab-tiling --optimizer-backend pallas_gpu_triton \
    --benchmark-warmup 5 --benchmark-iterations 20 \
    --profile-target full_step --profile-iterations 3 \
    --disable-python-gc --output out/nsight/full-step-benchmark.json
```

Use the generated CUDA-kernel summary to select one kernel, then run Compute:

```bash
uv run python scripts/profile_gpu_train_compute.py \
  --tool compute \
  --output-prefix out/nsight/optimizer-kernel \
  --manifest out/nsight/optimizer-kernel.json \
  --ncu-kernel-regex 'rwkv7_fused_adamw_gpu' \
  --ncu-set full --ncu-launch-count 1 \
  -- uv run python scripts/benchmark_local_train_compute.py \
    --fixed-batch out/gpu-matrix/fixed-b1-t512.npz \
    --model-preset 0.185b --variant baseline \
    --ctx-len 512 --batch-size 1 \
    --no-remat-blocks --no-sequence-chunking --no-head-chunking \
    --no-training-vocab-tiling --optimizer-backend pallas_gpu_triton \
    --benchmark-warmup 5 --benchmark-iterations 20 \
    --profile-target optimizer --profile-iterations 1 \
    --disable-python-gc --output out/nsight/optimizer-benchmark.json
```

Nsight Compute must report occupancy, eligible warps, register spills, and
DRAM throughput before a low-power result is assigned to a bottleneck. If the
host returns `ERR_NVGPUCTRPERM`, record the permission failure; do not replace
the missing counters with a power-only conclusion.

JAX documents that its own profiler must not coexist with Nsight. Use
`--profile-mode xprof --profile-output PATH` only in a separate run. The
programmatic JAX trace follows the official
[JAX profiling guidance](https://docs.jax.dev/en/latest/profiling.html), while
the external range follows NVIDIA's
[Nsight Systems CUDA Profiler API contract](https://docs.nvidia.com/nsight-systems/UserGuide/).

## Acceptance gate

For each production GPU architecture and batch shape:

1. Real-GPU forward/update parity must pass.
2. The largest selected batch must fit with production optimizer state.
3. Head and optimizer are selected by complete-step median, not their isolated
   microbenchmarks.
4. Pallas optimizer is adopted only when complete-step throughput is at least
   Optax throughput and optimizer latency does not regress.
5. Report absolute throughput, latency, memory-saving mode, and profiling
   permissions; do not extrapolate single-GPU batch scaling to multi-GPU.

The implementation and CPU parity portions of this gate are complete. Current
L40S/Mosaic real-hardware matrix and Nsight results remain pending until a
single-GPU host is available.
