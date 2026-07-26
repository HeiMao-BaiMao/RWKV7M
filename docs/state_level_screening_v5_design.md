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

The tracked 0.185B and 0.3B configs use `edit_mode="tied"`, a modest
redundancy-aware victim weight, a per-layer hard-write controller, and a
temporary CE-visible residual floor after the first MI300X runs reached slot
cosine redundancy 0.919 and 0.987 and later retrieval runs oscillated between
all-write and empty-memory states. The portable recurrence contains experimental
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

The first recovery profile tried to regulate writes through lower/upper
quadratic losses. MI300X runs found two failures in that contract:

```text
soft rate satisfied the loss while hard writes collapsed
hard utilization gating disabled the upper loss in a low-utilization all-write state
```

The current tracked profile therefore removes write-rate regulation from the
loss gradient. Every screened layer owns a bounded admission-bias controller.
After each train step it observes that layer's realized hard accepted-novel
rate and applies an incremental PI update:

```text
error_t = target_rate - hard_write_rate_t
delta_bias_t = clip(
    kp * (error_t - error_{t-1}) + ki * error_t,
    -max_step,
    +max_step,
)
bias_{t+1} = clip(
    bias_t + delta_bias_t,
    -bias_limit,
    +bias_limit,
)
```

The bias is stop-gradient inside the model loss. Consequently, the controller
determines how many tokens are admitted while CE gradients remain responsible
for ranking tokens and learning memory content. The tracked target is 0.05,
but it is an experiment control point rather than a claim that 5% is optimal.
The initial admission probability is 0.16: its logit is close to the lower 5%
tail of a unit-normal route score, which is only an initialization heuristic
and must be checked on the production shape.

The controller bias is an FP32 model parameter so increments smaller than one
BF16 ULP are not discarded and portable inference artifacts retain the learned
operating point. Its rate EMA and previous error are
non-parameter training state: full checkpoints restore them, while
safetensors intentionally omit them. The controller is inactive before
Screening activation and ignores non-finite observed rates.

The legacy admission-floor and write-budget losses remain available for
ablation, but config validation rejects enabling either alongside the
controller. The legacy utilization option now scales the upper penalty
continuously from zero to full strength; it no longer creates a hard off
region below the utilization threshold.

The tracked profiles now keep Screening recurrence, residual, auxiliary
losses, and optimizer updates disabled for the first 100 trunk steps. The
recurrence starts after that boundary with hard occupancy intact, while the
residual, auxiliary loss, and optimizer update scales ramp linearly over 100
steps. Screening optimizer updates additionally use a 0.1 multiplier. The
optimizer gate includes AdamW decay, so "disabled" does not silently modify
Screening kernels through weight decay. Curriculum clocks start at Screening
activation rather than at global step zero.

During the next 2,000 Screening steps, projections receive stop-gradient
copies of trunk inputs. The ordinary RWKV residual path still trains from CE,
but the memory branch cannot destabilize trunk activations through its input
features while routing and content are bootstrapped. The tracked configs keep
the pre-tile residual floor at 0.2, which is 0.1 after four-tile scaling at the
start of the curriculum. The floor anneals to zero; inference and calls
without a training step use the learned scale only. This creates a direct CE
signal for the memory output without imposing a permanent memory contribution.

The same interval now performs smooth-forward read screening and anneals to
the exact hard Trim-and-Square operator. This intentionally changes training
semantics during the curriculum; deterministic evaluation and inference are
always hard. The smooth operator is a squared softplus approximation in the
same normalized relevance coordinate, so weak reads are not normalized to
unit mass.

Temporary self-index supervision remains implemented as an ablation but is
disabled in the tracked configs. A self-index-only MI300X run produced raw
gradient L2 up to 5.56e11 and then collapsed toward all-reject. It is therefore
not used as the default content objective without a redesigned, bounded
gradient contract.

Gradient statistics use a scaled norm that does not overflow merely because
individual finite values are large. The tracked configs additionally set a
finite max-absolute-gradient guard of 1e6. Exceeding the guard skips the whole
optimizer update, including moments and optimizer step, and records
`gradient_spike_detected=1`; non-finite gradients remain fail-closed. This is
a corruption guard, not evidence that the underlying instability is solved.

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
versioned golden vector. Controller-specific tests cover update direction,
bounds, inactive steps, full train-step integration, config incompatibilities,
and full-checkpoint state round-trip. Gradient-guard tests cover finite values
whose naive FP32 squared norm would overflow and verify that a skipped update
does not alter parameters, optimizer moments, or optimizer step.

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
screened layer and bank, controls hard write rate outside the loss gradient,
and supplies CE signal through a temporary residual floor without propagating
the memory-input gradient into the trunk. GPU validation remains pending.
Retention, v5
checkpoint reconstruction, and v5 Pallas optimization remain gated until
multi-slot use and a positive causal counterfactual are demonstrated.

A 2026-07-22 MI300X stability investigation on the synthetic retrieval task
added two numeric hardenings to this path: victim-allocation age/usage
statistics use a tie-safe masked min--max (a span at or below `eps` yields
zero forward and gradient, removing artifact gradients up to about `1.17e6`
on tied statistics), and `norm_eps` is an independent setting with an
unchanged default. Neither was the cause of the recurring production training
NaN: that failure was isolated on a fixed batch to the coexistence of
multiple ROCm Pallas WKV pullbacks in one full graph, not to v5 recurrence
math, and AMD WKV dispatch now fails closed to a Pallas-forward/reference-VJP
hybrid backend (see
[`gpu_mi300x_training_validation.md`](gpu_mi300x_training_validation.md)).
The same investigation showed that while the memory residual is near zero the
auxiliary objectives are the effective primary training signal for routing:
with all auxiliaries off, routing saturated toward write-everything;
self-index alone produced raw-gradient spikes up to `5.56e11` followed by
collapse toward all-reject; no configuration has yet held a stable interior
write rate. On held-out streaming retrieval at an early checkpoint the memory
branch showed its first nonzero causal loss delta (`+1.04e-4`) with
chance-level retrieval accuracy, so the memory-quality gate remains open.

## Retrieval evaluation vehicle

`rwkv7m-prepare-retrieval` creates document-aligned delayed key/value data with
distractors. Training uses one complete document in one optimizer step with
recurrent chunking and an answer-only mask. This is necessary because merely
carrying state between optimizer steps truncates the gradient from a later
answer back to an earlier write. Streaming evaluation separately uses
`document_sequential` sampling, row-selective state resets, and the same
answer-only mask. It reports retrieval accuracy and memory-off deltas in
addition to loss.
