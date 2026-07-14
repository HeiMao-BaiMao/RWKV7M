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
    NVIDIA opt-in     -> registered FFI forward/backward
    CPU/tests          -> lax.scan reference
```

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
and the reference recurrence on CPU. `RWKV7M_WKV_BACKEND` can explicitly select
`reference`, `pallas_tpu`, `pallas_gpu_mosaic`, `pallas_gpu_triton`, or `ffi`.
FFI is never selected automatically and fails early unless an application has
registered all three forward, forward-with-aux, and backward callables.

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
existing presets. Each head chunk still creates its full BF16 vocabulary logits
and casts them to FP32 for cross entropy. Vocabulary tiling and online
log-sum-exp are therefore still pending.

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
forward/backward and sharded paths have not yet passed parity or hardware
performance gates.

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
7. **Partial:** token-axis head chunking is implemented; vocabulary-tiled
   streaming/fused cross entropy is pending.
8. **Implemented foundation:** optional FFI registration and explicit dispatch
   contracts exist; no native FFI implementation is bundled or selected by
   default.
9. **Partial:** an L40S/Ada measured dispatch record now covers four WKV shapes
   and tracked 0.185B train-step configurations. Hopper/Blackwell and broader
   production-shape records remain pending.
10. **Implemented, unvalidated:** screening dense projections are hoisted out
    of the time loop; the projected reference, independent backend dispatch,
    separate TPU/GPU persistent forward and reverse-time Pallas kernels, FP32
    training carry tape, and explicit data/model sharding transpose are
    integrated. Parity tests and real-hardware screening performance gates
    remain pending by request.

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
windows separately; no new strict result has been measured yet.

This changes the tuning priority for the tested small shape. The canonical
FFN-2688 read-only screening path reduced time-weighted throughput by 58.7%
relative to its screening-free core; read/write reduced it by 60.0%. Screening
fusion and layout work now precede further WKV replacement work on Ada.

## TPU v5e Pallas validation record

The complete methodology, raw timings, correctness gates, limitations, and
resource cleanup record are documented in
[TPU Pallas WKV Performance Validation](tpu_pallas_performance.md).

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
