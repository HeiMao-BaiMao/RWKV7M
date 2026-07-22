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
- load-dependent threshold warm-up;
- a training-only soft-to-hard read curriculum that restores query/key
  gradients below the hard threshold and expires to exact hard reads;
- bounded, non-amplifying read aggregation;
- ambiguity-aware matched confidence;
- hard-forward/soft-backward novelty, admission, bank, and victim decisions;
- admission features from token context, matched confidence, total load, and
  per-bank load;
- projection-consistent slot/read-key/write-key/value updates;
- explicit age, usage, occupancy, erase/write, saturation, and residual metrics;
- temporary candidate self-index supervision for both read and write keys;
- an upper write-budget penalty complementary to the empty-capacity floor;
- versioned semantic golden data for the initial all-empty allocation case.

The tracked 0.185B and 0.3B configs use `edit_mode="tied"` and enable a modest
redundancy-aware victim weight after the first MI300X runs reached slot cosine
redundancy 0.919 and 0.987. The portable recurrence contains experimental
`capacity_conserving`/`free_edit` and redundancy-aware victim branches, but
continuous edit modes are not headline-enabled until Phase 1 clears the
paper's quality gate.

The original analytic implementation used the loose bound
`sqrt(2 log(N/delta) / d)`. At the tracked full 4-tile/16-slot read shape with
tile dimension 16 this produced `tau=0.946`, a direct mechanism for the
observed read starvation. The implementation now follows the paper's null-CDF
contract more closely with a Gaussian family-wise quantile. The same example
starts at `tau=0.789`. Threshold lookup values for all occupancy counts are
constructed once outside the token scan. Learned key distributions still
require empirical false-read calibration; neither approximation is a
statistical guarantee after training.

The low-occupancy warm-up applies only to the read threshold. Write matching
and novelty classification always use their capacity-calibrated null
thresholds. Reusing the permissive read warm-up for writes makes the first
occupied slot match almost every later token and defeats empty-first
allocation; this failure mode was reproduced on MI300X before the contracts
were separated.

Long-context reverse scans use a stricter mixed-precision boundary than v4.
The surrounding RWKV block and stored model parameters remain BF16, while v5
Screening projection compute, recurrent vectors, slot state, and recurrent
cotangents are FP32. The Screening output is converted back at the residual
boundary. A real MI300X checkpoint exposed finite loss with 247/423 non-finite
gradient leaves when BF16 projection results were transported through the
long scan; the same checkpoint and batch passed 423/423 after this boundary
was enforced. Pure diagnostic norms are detached from the learning graph.

## Anti-starvation curriculum

An upper write budget cannot prevent the model from choosing never-write.
The NNX training loss therefore supports an opt-in temporary lower floor:

```text
target(layer, step) =
    initial_target
    * max(1 - step / warmup_steps, 0)
    * remaining_empty_fraction(layer)

realized_rate = soft_rate + stop_gradient(hard_rate - soft_rate)
loss = mean_layer(
    weight * relu(target(layer, step) - realized_rate(layer))^2
)
```

The loss is zero after warm-up and when the memory is full. It does not impose
a permanent write quota. The tracked examples use an initial target of 0.05,
weight 0.1, and 2,000 steps; these are experiment defaults, not validated
optima. `admission_floor_loss`, `admission_floor_target`, accepted novel rate,
read/write rates, residual RMS, and memory-on/off causal deltas must be reported
together. The forward constraint now observes actual hard allocations while
its gradient follows the continuous novelty/admission path. Applying the
hinge before the layer reduction prevents one active layer from satisfying the
floor for a collapsed layer. A nonzero admission rate alone is not evidence
that memory helps.

The upper write budget is also evaluated per layer with the same
hard-forward/soft-backward rate. A configurable occupied-slot threshold keeps
that upper penalty disabled during empty-memory bootstrap. The tracked
profiles use 50% utilization; this is an experiment setting, not a validated
optimum.

The tracked profiles now keep Screening recurrence, residual, auxiliary
losses, and optimizer updates disabled for the first 100 trunk steps. The
recurrence starts after that boundary with hard occupancy intact, while the
residual, auxiliary loss, and optimizer update scales ramp linearly over 100
steps. Screening optimizer updates additionally use a 0.1 multiplier. The
optimizer gate includes AdamW decay, so "disabled" does not silently modify
Screening kernels through weight decay. Curriculum clocks start at Screening
activation rather than at global step zero.

During the same 2,000-step interval, the tracked configs keep the effective
screening residual scale at or above 0.01 (before the read-tile scaling). The
floor anneals to zero; inference and calls without a training step use the
learned scale only. This blocks the easiest early escape through
`lambda_screen -> 0` without imposing a permanent memory contribution.

The same interval now performs smooth-forward read screening and anneals to
the exact hard Trim-and-Square operator. This intentionally changes training
semantics during the curriculum; deterministic evaluation and inference are
always hard. The smooth operator is a squared softplus approximation in the
same normalized relevance coordinate, so weak reads are not normalized to
unit mass.

Newly admitted candidates receive a temporary self-index objective. The
selected candidate read key must clear the current read threshold and its
write key must clear the novelty similarity threshold, each with a small
margin. Selection and admission are stopped hard decisions for this loss, so
the objective trains query/key geometry without rewarding admission collapse.
It anneals to zero after 2,000 steps.

An independent upper write budget penalizes the per-layer realized hard-write
surrogate above the configured ceiling. This is required by the 0.3B
all-novel/all-write observation; the lower admission floor alone cannot
distinguish healthy writes from saturation.

## Deliberately gated paths

- `screening-v5-retention` raises `NotImplementedError`.
- v5 interval checkpointing is rejected because v4 reverse reconstruction is
  invalid for irreversible allocation and separated erase/write events.
- v5 does not dispatch to the existing GPU or TPU Pallas kernels. Those kernels
  implement v4 semantics, and the accelerator benchmark rejects a v5 label.
- the frozen Linen model rejects v5 rather than silently running v4 equations.
- large-mesh TPU lowering, multi-device GPU lowering, v5 Pallas parity, and
  checkpoint peak memory have not been measured. Portable v5 has single-MI300X
  loss/gradient, curriculum-length training, counterfactual, and throughput
  measurements, but its memory-quality gate failed.

## Tracked configurations

- [`rwkv7m-0.185b-screening-v5-core.json.example`](../configs/rwkv7m-0.185b-screening-v5-core.json.example)
- [`rwkv7m-0.3b-screening-v5-core.json.example`](../configs/rwkv7m-0.3b-screening-v5-core.json.example)

Missing `semantics_version` retains the v4 migration rule. Explicit
`screening-v5-core` is required to activate this path.

## Current validation

Local tests cover configuration round-trip and rejection, capacity threshold
behavior, bounded aggregation, ambiguity suppression, finite empty-memory
and underflowed-route gradients, hard/soft admission gradients, empty
allocation, rejected writes, chunk invariance, the FP32-v5/BF16-parameter
boundary, a full tiny-model loss/gradient pass, the temporary admission floor,
the Gaussian-null threshold value, below-threshold smooth-read gradients,
self-index geometry gradients, the upper write budget, the memory-off residual
override, hard-forward layerwise budgets, staged recurrence/optimizer
activation, document-aligned state reset, answer-only loss masks, and the first
versioned golden vector.

On 2026-07-22, the tracked recovery profile was repeated on one MI300X. The
0.185B configuration completed all 2,000 curriculum steps (32.77M tokens), and
the 0.3B configuration completed 400 steps plus two same-seed 30-step runs.
All final production-shape diagnostics passed 423/423 and 480/480 finite
gradient leaves. The earlier 0.3B step-16 NaN did not recur in these runs,
although the two same-seed trajectories were not bitwise deterministic.

The memory-quality gate still failed. The 0.185B run converged to one occupied
slot out of 16 and a residual/base RMS ratio of 9.42e-6. The 0.3B aggregate
converged to 3.125% utilization across two screened layers, 50.0488% novel
tokens, 50% rejected writes, and a residual ratio of 5.37e-7. Exact aggregate
fractions are consistent with one layer rejecting nearly every token and only
one slot in the other layer being occupied, but layer-resolved metrics are
required to prove that interpretation. Memory-off loss deltas ranged from
-5e-6 to +1.6e-5 and did not show a consistent quality benefit. The recovery
profile therefore exchanged all-write/high-redundancy collapse for
under-allocation and likely layer-wise collapse. See
[`gpu_mi300x_training_validation.md`](gpu_mi300x_training_validation.md) for
the exact conditions and limitations.

The complete paper acceptance matrix is still open: all boundary golden cases,
five-seed synthetic memory tasks, three-seed matched language-model runs,
counterfactual memory ablation, parameter/compute-matched controls, and real
accelerator gates must pass before this path is called beneficial or
production-ready.

The next recovery iteration now exposes and constrains write/read behavior per
screened layer and bank and bootstraps empty capacity without enabling the
upper budget prematurely. GPU validation remains pending. Retention, v5
checkpoint reconstruction, and v5 Pallas optimization remain gated until
multi-slot use and a positive causal counterfactual are demonstrated.

## Retrieval evaluation vehicle

`rwkv7m-prepare-retrieval` creates document-aligned delayed key/value data with
distractors. Training uses one complete document in one optimizer step with
recurrent chunking and an answer-only mask. This is necessary because merely
carrying state between optimizer steps truncates the gradient from a later
answer back to an earlier write. Streaming evaluation separately uses
`document_sequential` sampling, row-selective state resets, and the same
answer-only mask. It reports retrieval accuracy and memory-off deltas in
addition to loss.
