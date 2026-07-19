# State-Level Screening v2 engineering contract

Status: opt-in implementation complete. Corrected GPU/TPU Pallas equations
pass CPU interpret parity, and the corrected TPU path passes real v5e-4
lowering, all-output/all-input-gradient parity, recurrence timing, and a full
four-way model-axis step at the tracked maximum context. The Triton GPU path
also lowers and executes on AMD MI300X, but its production-shape fail-closed
parity and stable-learning gates remain open. Multi-host gates remain pending.

Here, "v2" names the second implemented Screening architecture. The
accompanying paper is now `design-locked-draft-v5`; its
`screening-v4-legacy` and `screening-v4-competitive` profiles describe the
predecessor implemented by this contract, while `screening-v5-core` and
`screening-v5-retention` are design-only profiles. The architecture revision,
paper draft, and semantics labels therefore track different artifacts.

This document is the implementation contract for State-Level Screening v2.
The research motivation and evaluation claims live in
[`RWKV7M.paper.md`](../RWKV7M.paper.md). The v5 equations in that paper must not
be read as current implementation behavior; this document remains the source
of truth for the implemented v4 predecessor semantics. Existing configurations
continue to select the legacy projected recurrence. V2 is available only
through explicit settings; the tracked example is
[`configs/rwkv7m-0.185b-screening-v2.json.example`](../configs/rwkv7m-0.185b-screening-v2.json.example).

## Scope and invariants

The revision keeps these properties:

- the RWKV recurrent core remains the primary path;
- memory capacity is fixed by the slot count, not context length;
- reads use absolute Trim-and-Square relevance and can reject every slot;
- read and write scores are separate;
- slots remain divided into short, mid, and long banks;
- dense projections stay in XLA while the time recurrence stays in Pallas;
- GPU and TPU use separate Pallas kernel bodies behind one recurrence API;
- state, normalization, routing, and recurrence statistics use FP32;
- activation inputs and outputs follow the model compute dtype.

The revision does not add a PyTorch runtime or make upstream checkpoint
compatibility claims.

## Current behavior that must remain reproducible

The current NNX path hoists dense projections out of the time loop and keeps
six recurrent values per screened layer:

```text
slots, read_keys, values, write_keys, ages, usage_ema
```

All four projected content states use the same scalar update strength. This is
why the projected keys and values remain consistent with the slot state across
sequence chunks.

The existing phase names do not mean "no write":

| Existing configuration | Current recurrence behavior | v2 compatibility mode |
| --- | --- | --- |
| `read_screening_only` | update every slot by bank `mu`; preserve age | `legacy_unconditional` |
| `read_write` and `use_write_screening=True` | use `max(write_relevance, write_rel_floor)` | `legacy_threshold` |
| `read_write` and `use_write_screening=False` | same slow update as read-only | `legacy_unconditional` |

Loading an old config or checkpoint must select the corresponding legacy mode.
It must not silently enable competitive routing or novel allocation.

## Configuration surface

These names are the implemented public `ScreeningConfig` contract.

```text
write_mode:
  disabled
  legacy_unconditional
  legacy_threshold
  competitive_novel

gate_space: model | value
gate_activation: sigmoid | tanh_silu
candidate_rank: None | 32 | 64 | 128
route_power: float
novelty_threshold: float
novelty_temperature: float
admission_init: float
allocation_temperature: float
bank_route_temperature: float
allocation_top_k: int
allocation_age_weight: float
allocation_usage_weight: float
admission_threshold: float for hard-forward/soft-backward admission
  (`None` is accepted only as a legacy-config fallback to 0.5)
checkpoint_interval: None | 8 | 16 | 32
n_read_tiles: int
```

Explicit write modes are consumed by `read_write`. The historical
`read_screening_only` phase always retains its slow legacy updater so old phase
semantics do not change. Within `read_write`, `disabled` means no slot update
and is distinct from `legacy_unconditional`.

## Read path

Read behavior remains absolute and non-sum-to-one. For each read tile:

```text
similarity_m = dot(unit(q), unit(k_m))
relevance_m = TrimSquare(similarity_m, tau_read)
read_latent = TanhNorm(sum_m relevance_m * unit(value_m))
```

No slot softmax or slot-sum normalization is allowed in this path. With no
eligible slots, the read latent can remain zero.

The gate moves before the output projection:

```text
gate_v = sigmoid(gate_proj(layer_norm(x)))       # d_model -> d_v
memory_out = out_proj(read_latent * gate_v)      # d_v -> d_model
hidden = hidden_base + lambda_screen * memory_out
```

The legacy model-space gate remains available for checkpoint compatibility and
ablation. `sigmoid` is the default activation; `tanh(silu(.))` is an independent
ablation rather than an implicit behavior change.

For multiple read tiles, total key and value dimensions remain fixed:

```text
d_k_tile = Dk_total / n_read_tiles
d_v_tile = Dv_total / n_read_tiles
effective_lambda = softplus(lambda_raw) / sqrt(n_read_tiles)
```

All tile projections must be produced by batched GEMMs and reshapes, not one
Dense call per tile. There is initially one shared write path. Screened-layer
count is not part of the default residual scaling.

## Candidate path

The factorized candidate combines the input, RWKV output, and slot identity in
a rank-`r` latent space:

```text
x_latent = x_proj(layer_norm(x))
h_latent = h_proj(hidden_base)
slot_latent = slot_proj(slot_embedding)

candidate = tanh(
    candidate_out(silu(x_latent + h_latent + slot_latent))
)
```

Ranks 32, 64, and 128 are evaluation candidates. `None` selects the legacy
candidate projection. A factorized configuration is accepted only when its
parameter count and principal projection FLOPs are lower than the legacy path
for that preset.

## Write routing

### Matched information

Absolute write eligibility is computed first:

```text
eligibility_m = TrimSquare(write_similarity_m, tau_write)
confidence = max_m eligibility_m
powered_m = eligibility_m ** route_power
denominator = where(sum_j powered_j > 0, sum_j powered_j, 1)
matched_distribution_m = powered_m / denominator
matched_route_m = confidence * matched_distribution_m
```

The conditional normalization decides *where* to write. Multiplying by
`confidence` retains the absolute decision of *how much* to write. A weak match
must not become a unit-mass write solely because it is the best available slot.

### Novel information

A token is novel when no existing slot reaches the configured eligibility:

```text
is_novel = confidence < novelty_threshold
novel_hard = float(is_novel)
novel_soft = sigmoid((novelty_threshold - confidence) / novelty_temperature)
novel_gate = novel_soft + stop_gradient(novel_hard - novel_soft)

admission_soft = sigmoid(admission_logit)
admission_hard = admission_soft >= admission_threshold
admission_gate = admission_soft + stop_gradient(
    admission_hard - admission_soft
)
```

Novelty is not sufficient for allocation. Both decisions are hard in the
forward pass and use continuous surrogate gradients in training. A rejected
token therefore has exactly zero applied write mass and cannot reset slot age,
while the admission projection and the confidence boundary remain trainable.
The novelty surrogate restores confidence gradients where Trim-and-Square has
nonzero derivative; the intentional hard reject region of Trim-and-Square
itself remains zero-gradient.

The first implementation uses a hierarchical bank-then-slot route. A small
three-way projection selects short, mid, or long from the current token. Within
each bank, victim scores use bank-normalized age and read usage:

```text
bank_soft = softmax(bank_logit / bank_temperature)
bank_hard = one_hot(argmax(bank_logit))
bank_route = bank_soft + stop_gradient(bank_hard - bank_soft)

slot_logit_m = (
    age_weight * normalized_age_within_bank_m
    - usage_weight * usage_ema_m
)
```

Slot selection is independently normalized within each bank and uses the same
straight-through pattern:

```text
slot_soft_b = masked_softmax(slot_logit / temperature, bank=b)
slot_hard_b = one_hot(argmax(slot_logit within bank b))
slot_route_b = slot_soft_b + stop_gradient(slot_hard_b - slot_soft_b)

victim_route_m = sum_b bank_route_b * slot_route_b,m
novel_route = novel_gate * admission_gate * victim_route
matched_applied = (1 - novel_gate) * matched_route
write_route = novel_route + matched_applied
```

Forward therefore chooses one bank and one victim slot; backward has soft paths
through novelty, admission, bank, and victim decisions. This avoids flattening
short/mid/long semantics into an unrestricted global slot softmax. Top-k and
bank quotas are ablations after top-1 correctness. A balance regularizer is
added only if bank-collapse metrics justify it.

### State update and accounting

```text
strength_m = mu_bank(m) * write_route_m
next_state_m = state_m + strength_m * (candidate_m - state_m)
```

The same `strength_m` updates slots, read keys, values, and write keys. Age
advances with time and resets only for an applied write. Write metrics are
derived from the applied route, not raw eligibility. Allocation usage is an EMA
of absolute read activity, because a frequently read slot should be protected
even if it has not been written recently; write eligibility alone must not mark
a slot as used. A rejected write therefore leaves slot content unchanged, does
not reset age, and does not increment write counts. It may still advance age or
record genuine read usage.

```text
next_age_m = where(write_route_m > write_epsilon, 0, age_m + 1)
read_activity_m = max_tile clip(read_relevance_tile,m, 0, 1)
next_usage_m = usage_decay * usage_m
               + (1 - usage_decay) * read_activity_m
```

Legacy modes retain their existing age and usage equations exactly; the revised
accounting applies to `competitive_novel`.

## Deferred group-wise update

Group-wise slot rates are not part of v2 implementation. Applying different
channel rates only to slots would break the invariant between slots and their
projected read-key, value, and write-key states. Reprojecting all states inside
the recurrent loop would also reintroduce token-sized dense work.

Group-wise updates may be reconsidered only with one of these proven contracts:

- a projection layout that commutes with the group update;
- an explicitly grouped projected state with matching rates; or
- an efficient re-projection kernel whose chunk invariance and accelerator cost
  have been measured.

## Training-tape checkpointing

The current training forward stores all six FP32 carry values at every token.
Its tape-only memory is:

```text
bytes = 4 * T * B * screened_layers * M
        * (d_slot + 2*d_k + d_v + 2)
```

For the tracked presets at batch 1 this is approximately 10.06 MiB for the
0.185B preset at 512 tokens and 1.75 GiB for the 7B preset at 4,096 tokens,
before sharding and excluding all other activations.

`checkpoint_interval` changes the accelerator backward contract. The
implemented path keeps the four projected content states only at interval
boundaries and keeps age, usage, and applied update strength as per-token
scalar tapes:

```text
forward:
  store interval-boundary slot/read-key/value/write-key states
  store per-token age, usage, and applied update strength

backward:
  reconstruct prior content from candidate and applied strength
  reset reconstruction drift at every saved boundary
  run the reverse recurrence for that interval
  continue with the previous boundary
```

Because reconstruction divides by `1 - strength`, public configuration
validation requires the maximum possible effective strength to be at most
`0.95` whenever checkpointing is enabled. The bound includes half-life-derived
update rates and, for `legacy_threshold`, `write_rel_floor`. Configurations over
the bound fail before model initialization; the reverse kernel does not clamp
the denominator and silently change gradients.

For interval `I`, let `C = ceil(T / I) + 1`. The checkpointed tape-only memory
is:

```text
bytes = 4 * B * screened_layers * M
        * (C * (d_slot + 2*d_k + d_v) + 3*T)
```

At interval 16 this is approximately 0.74 MiB for the tracked 0.185B preset at
batch 1 and 512 tokens, and 115.44 MiB for the 7B preset at batch 1 and 4,096
tokens with four screened layers. These are theoretical tape sizes, not
measured peak-device-memory reductions. GPU and TPU own separate checkpoint
forward and reverse kernels. Metrics are accumulated online; full per-token
route tensors remain debug-only outputs.

## Metrics contract

Normal training metrics add scalar aggregates only:

- matched and novel route mass;
- route entropy and top-1 concentration;
- admission mean and saturation fractions;
- novel-token and rejected-token rates;
- per-bank matched writes, allocations, and evictions;
- eviction age and usage;
- slot utilization and dead-slot rate;
- slot cosine redundancy.

Detailed per-token/per-slot traces are opt-in diagnostics and must not enlarge
the default training state or benchmark timing boundary.

## Implementation status and remaining gates

Implemented:

1. write-mode migration and scalar metrics without changing default legacy
   selection;
2. value-space gate with model-space compatibility path;
3. factorized candidate with validation that the selected rank reduces
   parameters;
4. confidence-preserving matched routing, hard-forward/soft-backward novelty
   and admission, and sparse bank-aware novel allocation in the portable
   reference;
5. separate GPU and TPU Pallas forward/backward kernel bodies;
6. interval Screening training-tape checkpointing;
7. fixed-total-dimension multi-read tiles and tile-count residual scaling.

Portable and NNX tests cover legacy mapping, rejected-write accounting,
confidence preservation, straight-through routing gradients, checkpoint
strength validation, and sequence chunk parity. CPU Pallas interpret mode
covers forward and all-input-gradient parity for the GPU and TPU checkpointed
multi-read paths. The benchmark is fail-closed by default: any configured
output, gradient, or loss threshold violation exits nonzero, with
`--no-require-parity` reserved for diagnostics.

On 2026-07-18, corrected commit `f35f6fc` passed real TPU v5e-4 lowering and
all-output/all-input-gradient parity. At the tracked recurrence shape, Pallas
median latency was 0.578 ms forward and 4.888 ms forward plus backward, versus
3.949 ms and 13.171 ms for the portable reference. The tracked 0.185B v2 model
also completed finite forward and optimizer step 1 with `data=1, model=4` at
contexts 128 and 512; context 512 exercised four configured 128-token chunks.
Direct single-kernel `T=2048` exceeds v5e scoped VMEM, so long contexts must
remain chunked on this device. On 2026-07-19, the real MI300X Triton path
passed the small accelerator test and ran complete 0.185B/0.3B steps. The
production 0.3B recurrence diagnostic did not pass the absolute-gradient/loss
parity gate, the 0.185B v2 training run collapsed to zero applied writes, and
the longer 0.3B v2 attempt became non-finite at step 7. These results establish
real ROCm lowering, not production v2 acceptance or a Screening quality gain.
Full TPU measurements and cleanup evidence are in
[the TPU report](tpu_pallas_performance.md); the GPU training behavior and
limitations are in
[the MI300X report](gpu_mi300x_training_validation.md).

Every stage must pass:

- legacy-unconditional output and gradient parity with the pre-v2 behavior;
- full-sequence versus multiple sequence-chunk-size parity;
- zero content update, age reset, and write count for rejected writes;
- usage driven by absolute read activity rather than rejected write eligibility;
- no amplification of weak eligibility into a strong matched write;
- reference versus accelerator forward and all-input gradient parity;
- checkpoint save/resume parity when the serialized config gains new fields;
- peak-memory and complete-step throughput records for production shapes.

Scientific adoption additionally requires matched baselines, multiple seeds,
long-context retrieval tasks, loss versus tokens and wall-clock, memory-off
counterfactuals, and slot/bank utilization evidence. A faster recurrence alone
does not establish that Screening improves model quality.
