# State-Level Screening v5 Phase 1 implementation contract

Status: portable research candidate. This is not a production or quality gate.

The normative design is [`RWKV7M.paper.md`](../RWKV7M.paper.md). This document
records what the repository implements now, what the tracked experiments
enable, and which claims remain unavailable.

## Implemented boundary

`screening-v5-core` is available in the NNX model and the portable JAX
recurrence. It requires `phase="read_write"` and carries an explicit FP32 hard
occupancy value for every slot. A v4 checkpoint without occupancy is rejected;
it is never interpreted as an all-empty v5 checkpoint.

The portable recurrence implements:

- occupied-only read and matched-write candidates;
- deterministic lowest-index empty allocation inside the selected bank;
- safe bounded, occupied-count/key-size/read-tile calibrated thresholds;
- load-dependent hard-read threshold warm-up;
- bounded, non-amplifying read aggregation;
- ambiguity-aware matched confidence;
- hard-forward/soft-backward novelty, admission, bank, and victim decisions;
- admission features from token context, matched confidence, total load, and
  per-bank load;
- projection-consistent slot/read-key/write-key/value updates;
- explicit age, usage, occupancy, erase/write, saturation, and residual metrics;
- versioned semantic golden data for the initial all-empty allocation case.

The tracked 0.185B and 0.3B configs deliberately use `edit_mode="tied"` and
`allocation_redundancy_weight=0`. This keeps the first experiment at the Phase
1 decision boundary. The portable recurrence contains experimental
`capacity_conserving`/`free_edit` and redundancy-aware victim branches, but
they are not headline-enabled until Phase 1 clears the paper's quality gate.

## Anti-starvation curriculum

An upper write budget cannot prevent the model from choosing never-write.
The NNX training loss therefore supports an opt-in temporary lower floor:

```text
target(step) =
    initial_target
    * max(1 - step / warmup_steps, 0)
    * remaining_empty_fraction

loss = weight * relu(target(step) - mean(novel_soft * admission_soft))^2
```

The loss is zero after warm-up and when the memory is full. It does not impose
a permanent write quota. The tracked examples use an initial target of 0.05,
weight 0.1, and 2,000 steps; these are experiment defaults, not validated
optima. `admission_floor_loss`, `admission_floor_target`, accepted novel rate,
read/write rates, residual RMS, and memory-on/off causal deltas must be reported
together. A nonzero admission rate alone is not evidence that memory helps.

During the same 2,000-step interval, the tracked configs keep the effective
screening residual scale at or above 0.01 (before the read-tile scaling). The
floor anneals to zero; inference and calls without a training step use the
learned scale only. This blocks the easiest early escape through
`lambda_screen -> 0` without imposing a permanent memory contribution.

## Deliberately gated paths

- `screening-v5-retention` raises `NotImplementedError`.
- v5 interval checkpointing is rejected because v4 reverse reconstruction is
  invalid for irreversible allocation and separated erase/write events.
- v5 does not dispatch to the existing GPU or TPU Pallas kernels. Those kernels
  implement v4 semantics, and the accelerator benchmark rejects a v5 label.
- the frozen Linen model rejects v5 rather than silently running v4 equations.
- large-mesh TPU/GPU lowering, real-device parity, throughput, and memory use
  have not been measured for v5.

## Tracked configurations

- [`rwkv7m-0.185b-screening-v5-core.json.example`](../configs/rwkv7m-0.185b-screening-v5-core.json.example)
- [`rwkv7m-0.3b-screening-v5-core.json.example`](../configs/rwkv7m-0.3b-screening-v5-core.json.example)

Missing `semantics_version` retains the v4 migration rule. Explicit
`screening-v5-core` is required to activate this path.

## Current validation

Local tests cover configuration round-trip and rejection, capacity threshold
behavior, bounded aggregation, ambiguity suppression, finite empty-memory
gradients, hard/soft admission gradients, empty allocation, rejected writes,
chunk invariance, a full tiny-model loss/gradient pass, the temporary admission
floor, and the first versioned golden vector.

The complete paper acceptance matrix is still open: all boundary golden cases,
five-seed synthetic memory tasks, three-seed matched language-model runs,
counterfactual memory ablation, parameter/compute-matched controls, and real
accelerator gates must pass before this path is called beneficial or
production-ready.
