# Accelerator kernel roadmap

This document records the implementation contract for moving the NNX training
path from the portable JAX reference toward tuned TPU and NVIDIA kernels. The
reference recurrence remains available for correctness and CPU tests; it is not
the expected production performance path.

## Dtype contract

The model must not depend on implicit type promotion at subsystem boundaries.

| Boundary | Input / storage | Internal computation | Output |
| --- | --- | --- | --- |
| WKV recurrence | BF16 vectors, FP32 state | FP32 state and accumulation | BF16 activation, FP32 next state |
| LayerNorm / GroupNorm | BF16 activation | FP32 statistics | BF16 activation |
| Linear | BF16 activation and weight | Backend high-precision accumulation | BF16 activation |
| LM head | BF16 hidden and weight | Backend high-precision accumulation | BF16 logits |
| Cross entropy | BF16 logits | FP32 log-sum-exp and reductions | FP32 loss |

The NNX reference path now enforces these boundaries explicitly. Screening
projections use the model compute dtype, while screening state, similarities,
normalization, and update statistics remain FP32.

## Backend ownership

The common layer owns the public WKV API, shape and dtype validation, state
checkpoint format, forward/backward equations, sharding contract, reference
implementation, and parity/performance tests. Kernel bodies are backend-owned:

```text
common WKV API + custom VJP + sharding contract
    TPU               -> TPU Pallas forward/backward
    NVIDIA Ada/Ampere -> Triton Pallas forward/backward
    NVIDIA Hopper+    -> Mosaic GPU forward/backward
    AMD (auto)        -> Triton Pallas forward + portable reference VJP
    NVIDIA opt-in     -> registered FFI forward/backward
    CPU/tests          -> lax.scan reference
```

The NVIDIA implementations now pass an explicit backend compiler object.
Triton uses `pltriton.CompilerParams`; Mosaic GPU uses
`plgpu.CompilerParams` with parallel grid semantics and 6 KiB of cross-warp
reduction scratch. Previously the Mosaic dispatch name did not pass Mosaic
compiler parameters, so it was not evidence of a distinct lowering. Real
Hopper/Blackwell validation is still required.

Forward and backward tuning are independent. A tuning record may select
different `rows_per_program`, warp count, pipeline depth, checkpoint interval,
and reduction strategy. Dispatch keys include GPU compute capability, batch
size, head count, head size, sequence length, compute dtype, and forward versus
backward execution.

`jax.custom_vjp` is the initial reverse-mode contract. Forward-mode AD is not a
current training requirement and must not be advertised until a JVP rule is
implemented.

The `wkv7()` contract accepts time-major BF16 or FP32 vectors shaped
`[time, batch, heads, head_size]` and an FP32 state shaped
`[batch, heads, head_size, head_size]`. It returns `(activation, final_state)`;
the activation follows the vector dtype and the state remains FP32. Accelerator
execution uses backend-specific persistent Pallas forward and backward kernels.
The forward rule saves FP32 `sa` values and interval state checkpoints; the
backward rule reconstructs states within each interval and implements the
reverse recurrence in Pallas. CPU execution retains the generic reference
pullback.

Automatic dispatch selects TPU Pallas on TPU, Mosaic GPU Pallas on recognized
Hopper/Blackwell devices, Triton Pallas on older NVIDIA devices including L40S,
the hybrid `pallas_gpu_triton_reference_vjp` on recognized AMD devices, and
the reference recurrence on CPU. The AMD hybrid keeps the Triton Pallas
forward but runs the portable reference VJP: a 2026-07-22 MI300X fixed-batch
isolation showed that multiple Pallas WKV pullbacks coexisting in one full
training graph corrupted cotangents even though all 20 captured pullback
calls passed in isolation, so AMD training pullbacks fail closed until a
real-device full-graph gate passes. `RWKV7M_WKV_BACKEND` can explicitly
select `reference`, `pallas_tpu`, `pallas_gpu_mosaic`, `pallas_gpu_triton`,
`pallas_gpu_triton_reference_vjp`, or `ffi`; explicit `pallas_gpu_triton`
remains available on AMD for benchmarking and investigation. FFI is never
selected automatically and fails early unless an application has registered
all three forward, forward-with-aux, and backward callables.

## Training pipeline boundary

The target training pipeline separates recurrent chunking from the LM head:

```text
sequence-chunked recurrent core
    -> retained or recomputable hidden sequence
    -> independently sized head chunks
    -> vocabulary-tiled streaming cross entropy
```

`compute_logits(hidden)` remains the evaluation/inference API. Training uses
`compute_training_loss(hidden, targets)` so logits do not escape the head/loss
boundary. Vocabulary-parallel loss must define the max and exponential-sum
collectives as part of its sharding contract.

The NNX training path now retains the recurrent hidden sequence and runs the LM
head with an independently selected `head_chunk_size`. When that value is
`None`, it inherits `sequence_chunk_size` to preserve the memory behavior of
existing presets.

An opt-in vocabulary-tiled training path keeps GEMMs in XLA and dispatches the
per-tile maximum, exponential sum, target logit, and maximum-count reduction to
separate TPU and GPU Pallas kernels. A common custom VJP reconstructs each tile
and computes hidden, head-weight, mask, cross-entropy, and L2Wrap gradients
without retaining a full FP32 vocabulary tensor.

Use `--training-vocab-tile-size N`; `--no-training-vocab-tiling` forces the
portable full-logits path. This is a memory option, not the speed default. On
TPU v5e, XLA's full-vocabulary head was faster for every measured 0.185B head
shape, so `training_vocab_tile_size` defaults to `None`. The tiled path also
stays out of explicitly sharded/vocabulary-parallel execution until that path
has a dedicated single-collective contract.

The GPU gate now measures the full-XLA and tiled paths at identical batch
sizes and optimizer backends. The matrix runner uses fresh processes and
complete steps so isolated head timings cannot select a slower production
stack.

## Optimizer boundary

The portable and TPU default remains Optax. NVIDIA has an opt-in Pallas AdamW
path that computes one global gradient norm, then fuses clipping, FP32 moment
updates, bias correction, decoupled decay, and the RWKV `w0` 2x multiplier in
one kernel per local parameter shard. This preserves the existing optimizer
equations and checkpoint-visible FP32 moment policy without claiming that a
per-leaf kernel is automatically faster than XLA fusion.

`--optimizer-backend` accepts `optax`, `pallas_gpu_triton`, or
`pallas_gpu_mosaic`. Pallas remains opt-in until real hardware passes update
parity, optimizer-only latency, and complete-step throughput gates.

## Screening recurrence boundary

State-level screening now separates sequence-wide dense projections from its
time-dependent recurrence. Layer normalization, read/write queries, gate,
slot-delta projection, projected slot targets, and the final output projection
run as batched XLA operations outside the scan. The recurrent API receives
time-major projected targets and maintains FP32 slot, read-key, value,
write-key, age, and usage state.

```text
XLA: norm + q/gate/delta projections
  -> XLA: project initial slots and all delta targets
  -> Pallas: normalize, score, aggregate, and update projected state over T
  -> XLA: one sequence-wide output projection and gated residual
```

This removes token-sized GEMMs and model-axis collectives from the persistent
loop. `RWKV7M_SCREENING_BACKEND` independently accepts `reference`,
`pallas_tpu`, `pallas_gpu_mosaic`, and `pallas_gpu_triton`; automatic selection
uses the same accelerator policy as WKV. GPU and TPU own separate kernel
bodies. CPU uses the projected `lax.scan` reference.

Accelerator training uses a Pallas forward that stores the FP32 per-token carry
and a dedicated reverse-time Pallas kernel. The reverse kernel performs the
sequential transpose in one persistent loop and applies the local transition
VJP at each step; it does not transpose the whole Pallas forward or recompute a
portable scan. CPU and explicit fallback execution retain the projected
reference pullback.

For model sharding, the slot feature shard is gathered once before the
recurrence. Each model device runs the persistent recurrence, replicated
outputs are averaged to encode one logical transpose contribution, and final
slots are sliced back to their owning shard. This makes the collective and VJP
contract explicit without inserting a collective inside the time loop. It
duplicates screening recurrence compute across the model axis, so a later
hardware profile may justify a more specialized distributed kernel. These new
forward/backward and sharded paths have passed forward/gradient parity,
four-device model-sharding, and recurrence-performance gates on TPU v5e. The
Triton path has also passed real L40S lowering, all-output/all-input-gradient
parity, and the tracked recurrence-performance gate. Mosaic GPU screening
validation remains pending on Hopper/Blackwell.

### Screening v2 kernel contract

The opt-in recurrence revision is specified in
[State-Level Screening v2 engineering contract](state_level_screening_v2_design.md).
It is implemented in the portable reference and in separate GPU/TPU Pallas
kernel bodies. CPU interpret-mode forward and all-input-gradient parity pass.
Corrected commit `f35f6fc` passed real TPU v5e-4 lowering, six-output/17-input
gradient parity, recurrence timing, checkpointed `T=512` parity, and full
0.185B model-axis steps at contexts 128 and 512 on 2026-07-18. Direct
single-kernel `T=2048` exceeds v5e scoped VMEM, while the production
128-token-chunk path reaches the configured 512-token maximum. Current L40S
records apply to the legacy projected recurrence. A 2026-07-19 AMD MI300X run
proved Triton v2 lowering and small-shape accelerator parity, but the
production 0.3B recurrence failed the fail-closed absolute-gradient/loss gate
and the learning runs exposed admission/read collapse and a v2 NaN. Real GPU
v2 production acceptance therefore remains pending.

The backend boundary remains:

```text
XLA outside time loop:
  read/write/admission/bank-route projections
  factorized candidate projections
  projected candidate read keys, values, and write keys

Pallas inside time loop:
  absolute read screening
  confidence-preserving matched write routing
  admission-controlled bank-aware sparse novel allocation
  projected-state, age, usage, and scalar-metric updates
```

GPU and TPU continue to own separate forward, reconstruction, and reverse
kernels. The write mode is a static kernel policy so legacy and v2 semantics do
not branch dynamically per step.

With checkpointing disabled, the accelerator forward stores slot, read-key,
value, write-key, age, and usage carry at every token. Its tape-only memory is:

```text
4 * T * B * screened_layers * M
  * (d_slot + 2*d_k + d_v + 2) bytes
```

This is about 10.06 MiB for the tracked 0.185B preset at batch 1 and 512 tokens,
but about 1.75 GiB for the 7B preset at batch 1 and 4,096 tokens before sharding
and other activations. The implemented interval path stores the four content
states at `C = ceil(T / I) + 1` boundaries plus per-token age, usage, and update
strength. Its tape-only memory is:

```text
4 * B * screened_layers * M
  * (C * (d_slot + 2*d_k + d_v) + 3*T) bytes
```

At interval 16 this is about 0.74 MiB and 115.44 MiB for the same tracked
0.185B and 7B cases. These are theoretical tape sizes; real-device peak memory
and recomputation cost remain to be measured.

Group-wise slot rates remain deferred. The projected slot, read-key, value, and
write-key states must receive one consistent update strength unless a new
projection-compatible grouped-state contract is proven chunk-invariant.

## Performance gate

Correctness is required before throughput comparison:

- forward numerical tolerance passes;
- gradient parity passes;
- the production memory limit passes;
- steady-state measurements exclude compilation and initialization.

Kernel microbenchmarks diagnose occupancy, register spilling, eligible warps,
and memory traffic. Final dispatch uses steady-state train-step throughput,
including forward, backward, checkpoint traffic, dtype conversions, collectives,
and dispatch overhead.

For every supported production GPU and shape, the performance target remains:

```text
Pallas train-step throughput >= CUDA FFI train-step throughput * 0.90
```

Missing this target triggers further Pallas tuning; it does not automatically
enable FFI. FFI is an explicit integration boundary only. Power draw alone is
not accepted as proof of the limiting resource.

## Implementation order and status

1. **Implemented:** explicit BF16/FP32 boundaries and dtype regression tests.
2. **Implemented:** opt-in CLI overrides for rematerialization and sequence
   chunking, without changing preset defaults.
3. **Implemented:** recurrent core, LM head, and training loss API separation,
   with independently sized recurrent and head chunks.
4. **Implemented:** common WKV API, Pallas-first dispatch, custom VJP boundary,
   reference implementation, dtype/shape validation, checkpointed dedicated
   backward, and forward/gradient parity tests.
5. **Implemented functional and performance gate on Ada:** persistent Triton
   Pallas GPU forward/backward pass interpret-mode parity, real L40S lowering,
   all-input gradient parity, a two-way model-sharding step, and controlled WKV
   and train-step benchmarks. Mosaic real-hardware validation, privileged GPU
   profiling, and independent autotuning remain pending.
6. **Implemented functional gate:** TPU-specific Pallas forward/backward pass
   interpret-mode parity, real TPU v5e lowering, full-model BF16 gradient, and
   four-way model-sharded train-step checks. Broader shape profiling and
   autotuning remain pending.
7. **Implemented, opt-in memory path:** token-axis head chunking and
   vocabulary-tiled online cross entropy/L2Wrap with a common analytic VJP and
   separate TPU/GPU Pallas reducers. The TPU speed gate failed, so full XLA
   logits remain the default. On L40S, tile 16,384 matched full-XLA
   forward-plus-backward latency at 128 positions while reducing the largest
   logits tensor by 4x; other tiles were slower. Vocabulary-parallel
   single-collective fusion and broader GPU shape gates remain pending.
8. **Implemented foundation:** optional FFI registration and explicit dispatch
   contracts exist; no native FFI implementation is bundled or selected by
   default.
9. **Partial:** an L40S/Ada measured dispatch record now covers four WKV shapes
   and tracked 0.185B train-step configurations. Hopper/Blackwell and broader
   production-shape records remain pending.
10. **Implemented; TPU and L40S gates passed:** screening dense projections are hoisted
    out of the time loop; the projected reference, independent backend
    dispatch, separate TPU/GPU persistent forward and reverse-time Pallas
    kernels, FP32 training carry tape, and explicit data/model sharding
    transpose are integrated. TPU v5e passed all-output and all-input-gradient
    parity, four-device model-sharding, a full optimizer step, and the tracked
    `0.185b` recurrence microbenchmark. L40S Triton lowering then passed all
    corresponding parity tests and achieved 16.71x forward and 15.76x
    forward-plus-backward speedup over the projected reference recurrence.
    Mosaic GPU validation remains pending.
11. **Implemented, hardware gate pending:** a fused GPU Pallas AdamW update,
    full-XLA/tiled-head x Optax/Pallas x batch-size matrix, logical fixed-batch
    fingerprint, strict model-shape comparator, and post-warmup Nsight capture
    range are integrated. CPU interpret parity passes; L40S and Mosaic
    complete-step and profiler records are still required before dispatch
    defaults change.
12. **Implemented; corrected TPU v5e-4 gate passed:** Screening v2 write-mode migration
    and metrics, value-space gate, factorized candidate, confidence-preserving
    matched routing, hard-forward/soft-backward novelty and admission with sparse bank-aware novel
    allocation, interval training-tape checkpointing, and fixed-total-dimension
    multi-read tiles are integrated. Portable/NNX tests and CPU Pallas
    interpret-mode GPU/TPU forward and gradient parity pass. Corrected commit
    `f35f6fc` passed real TPU v5e-4 lowering, six-output/17-input-gradient
    parity, tracked-shape and checkpointed-T=512 recurrence gates, and
    four-device 0.185B model-axis optimizer steps through context 512. Direct
    single-kernel T=2048 exceeds v5e scoped VMEM. MI300X Triton lowering and a
    small real-accelerator test pass, but production-shape parity and stable
    v2 learning do not. L40S/Mosaic v2, measured peak memory, stable
    complete-step throughput, and multi-host records remain required.

For a preset, omit execution flags to retain its tracked defaults. The following
flags make comparison runs explicit:

```text
--no-remat-blocks --no-sequence-chunking
--remat-blocks --no-sequence-chunking
--no-remat-blocks --sequence-chunk-size 128 --head-chunk-size 512
--remat-blocks --sequence-chunk-size 128 --head-chunk-size 512
```

Record the resolved `model_config` from `run_config.json` with every result.

## L40S Pallas validation record

The complete environment, correctness gates, WKV timings, full training-step
results, upstream RWKV comparison, TPU comparison, two-GPU scaling, and
profiling limitation are documented in
[NVIDIA L40S Pallas performance validation](gpu_l40s_pallas_performance.md).

For `T=128, B=1, H=12, N=64`, the real Triton-Pallas path was 8.77x faster in
forward and 15.09x faster in forward plus backward than the local `lax.scan`
reference. In an identical memory-safe 0.185B training configuration, Pallas
increased median complete-step throughput by 4.17x. A matched FFN-3072,
screening-free local run sustained 9,413 token/s versus 8,516 token/s for the
directly executed upstream fused RWKV-LM-V7 runner. This is a reference
comparison between different timing boundaries, not a Pallas-versus-CUDA or
JAX-versus-PyTorch ranking. Schema-v2 fixed-batch compute-only harnesses now
record WKV forward/backward and full-model forward/backward/optimizer/full-step
windows separately. The current single-L40S local side now measures 11,561
token/s without screening, 9,708 with read-only screening, and 10,365 with
read/write screening at batch 1 x 512. A matching schema-v2 upstream result is
still unmeasured, so these do not establish a strict JAX-versus-PyTorch ratio.

The projected recurrence changes the tuning priority for the tested small
shape. Its microbenchmark is no longer the bottleneck, but complete-step
latency remains 19.1% above baseline for read-only screening and 11.5% above
baseline for read/write in the single current run. Profiling and fusion of the
remaining projections, gradients, and optimizer work now precede further WKV
replacement work on Ada. The unexpected read/write advantage is not treated as
causal until repeated under a profiler.

## AMD MI300X learning-behavior record

The ROCm/Triton environment, production-shape parity diagnostic, compute-only
batch scaling, and short MiniPile training trajectories are documented in
[AMD MI300X training validation](gpu_mi300x_training_validation.md).

The MI300X path ran WKV, Screening, the full-XLA head, and Optax on one device.
At batch 32 it measured 35,312 token/s for the 0.185B v2 shape and 29,011
token/s for the 0.3B legacy shape under the fixed device-resident complete-step
boundary. These are capacity results, not evidence of Screening quality.
During real MiniPile training, 0.185B v2 rejected every write by step 26,
0.3B legacy stopped producing a measurable memory read by step 81, and the
longer 0.3B v2 attempt first became non-finite at step 7. The next GPU work is
therefore numerical and learning-dynamics diagnosis, not dispatch promotion.

A 2026-07-22 stability investigation on synthetic retrieval training then
isolated a recurring MI300X training NaN on a fixed batch. Along the way it
fixed an independent Pallas WKV backward defect (inverse state reconstruction
divided by a decay that FP32 rounds to zero; replaced with a per-token FP32
state tape on both GPU and TPU backends). The remaining failure was not a
single-kernel math bug: every captured WKV pullback passed in isolation on
MI300X for both Pallas and reference, yet any full graph keeping two or more
Pallas WKV pullbacks failed, while replacing all of them with the reference
VJP was finite. The first suspect is a ROCm/Triton lowering, buffer-aliasing,
or custom-call-scheduling interaction. AMD automatic dispatch therefore fails
closed to `pallas_gpu_triton_reference_vjp` (Pallas forward, reference
pullback); the real-device complete-step acceptance of this hybrid backend is
deferred to the next GPU session. Details are in the MI300X validation
report.

## TPU v5e Pallas validation record

The complete methodology, raw timings, correctness gates, limitations, and
resource cleanup record are documented in
[TPU Pallas WKV and Screening Performance Validation](tpu_pallas_performance.md).

The first real-hardware Pallas gate ran on 2026-07-14 using JAX/jaxlib 0.10.0
and libtpu 0.0.40 on a temporary single-host `v5litepod-4` in `us-west4-a`.
The following checks passed with the automatic `pallas_tpu` backend and
`interpret=False`:

- BF16 forward output and FP32 final-state parity against `lax.scan`;
- gradients for all six vector inputs and the initial FP32 state;
- a complete one-layer BF16 NNX model loss and parameter backward pass;
- the four-device `data=1, model=4` audit, including finite forward, loss
  3.9594927, and optimizer step 1;
- unchanged post-SPMD collective counts: 13 all-reduces, 3 all-to-alls, and no
  all-gathers.

The projected screening gate was then run on 2026-07-16 on the same TPU type.
It passed all six outputs, gradients for all 15 inputs, a four-device optimizer
step, and the tracked `0.185b` recurrence benchmark. Its explicit model-axis
screening boundary adds 2 all-gathers, bringing that audit to 18 collectives.

Pallas kernels require manual mesh axes in JAX 0.10. The NNX path therefore
wraps only WKV in `jax.shard_map`, with vectors using
`P(None, data, model, None)` and state using
`P(data, model, None, None)`. All kernel `Ref` accesses use slice-only
`pl.dslice` indexing so that the per-device references remain legal under that
manual boundary.

A WKV-only microbenchmark used random BF16 vectors with
`T=128, B=1, H=12, N=64`, FP32 state, 100 timed calls after compilation and
warmup, and host blocking on every result. Transparent hugepages were not
enabled, so these numbers are comparative rather than a production throughput
claim.

| Path | Forward | Forward + backward | Relative speed |
| --- | ---: | ---: | ---: |
| Pallas TPU | 0.276 ms | 0.538 ms | 1.11x / 1.42x |
| `lax.scan` reference | 0.307 ms | 0.766 ms | baseline |

This measures one WKV recurrence, not a complete model train step. It verifies
that the Pallas path is already faster for the tested shape but does not replace
the required end-to-end preset benchmarks or XProf tuning.

The temporary TPU VM `rwkv7m-pallas-260714` was deleted after validation. The
delete operation completed successfully, and the TPU VM list for `us-west4-a`
was empty afterward.

## TPU v5e training-head gate

The vocabulary-tiled path was tested on 2026-07-16 on a temporary
`v5litepod-4` named `rwkv7m-head-260716`. Inputs were fixed and resident on one
TPU device; JIT compilation was excluded; 10 calls were measured for the
initial 4,096 tile gate and 20 for each larger tile, with every call blocked on
all outputs. The tracked head shape used BF16, 128 positions, hidden size 768,
and vocabulary size 65,536. Ratios below are
`full-XLA median / tiled-Pallas median`, so values below 1.0 are regressions.

| Vocabulary tile | Largest BF16 logits | Forward ratio | Forward + backward ratio |
| ---: | ---: | ---: | ---: |
| 4,096 | 1 MiB | 0.40x | 0.38x |
| 8,192 | 2 MiB | 0.51x | 0.49x |
| 16,384 | 4 MiB | 0.59x | 0.59x |
| 32,768 | 8 MiB | 0.58x | 0.63x |
| 65,536 | 16 MiB | 0.79x | 0.83x |

The untiled BF16 logits tensor is 16 MiB for this head chunk. At tile 4,096,
the maximum component error was `3.66e-4` and the maximum gradient error was
`1.91e-6`, but median forward/backward time rose from 0.648 ms to 1.699 ms.
At 512 positions, tile 16,384 reduced the largest logits tensor from 64 MiB to
16 MiB but achieved only 0.54x of full-XLA forward/backward speed. These results
support the memory-saving implementation, but reject automatic speed dispatch
on TPU v5e.

The real-accelerator regression test passed both the standalone custom VJP and
the NNX model training-head integration with `pallas_tpu`. After the gate, the
temporary VM was deleted; the zone list was empty and describing the VM returned
`NOT_FOUND`.
