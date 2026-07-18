# TPU Pallas WKV and Screening Performance Validation

## Summary

Naming note: "Screening v2" in this report is the implemented architecture
called `screening-v4-competitive` by
[`RWKV7M.paper.md`](../RWKV7M.paper.md) v5. None of the
measurements in this report validate the design-only `screening-v5-core` or
`screening-v5-retention` semantics.

On 2026-07-14, the new TPU Pallas WKV implementation was compared with the
portable `lax.scan` recurrence on a Google Cloud TPU v5e. For the tested
`T=128, B=1, H=12, N=64` BF16 WKV shape, Pallas reduced forward latency by
10.0% and forward-plus-backward latency by 29.8%.

| Operation | Pallas TPU | `lax.scan` reference | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Forward | 0.276 ms | 0.307 ms | 1.11x | 10.0% |
| Forward + backward | 0.538 ms | 0.766 ms | 1.42x | 29.8% |

These are WKV-only kernel measurements on one TPU device. They are not
complete-model tokens-per-second results and must not be combined with the
older end-to-end preset measurements as if they used the same workload.
The matching L40S shape and full GPU training results are recorded separately
in the [L40S Pallas performance report](gpu_l40s_pallas_performance.md).

On 2026-07-16, the projected state-level-screening recurrence was validated on
the same TPU type. For the `0.185b` preset recurrence shape
`T=128, B=1, M=16, d_slot=128, d_k=d_v=64`, the TPU Pallas path was 5.04x
faster in inference forward and 5.92x faster in synchronized
forward-plus-backward than the equivalent `lax.scan` reference.

| Screening operation | Pallas TPU | `lax.scan` reference | Speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Forward | 0.257 ms | 1.298 ms | 5.04x | 80.2% |
| Forward + backward | 0.380 ms | 2.251 ms | 5.92x | 83.1% |

These ratios apply only to the projected screening recurrence. A complete
`0.185b` train step includes the RWKV blocks, dense projections, vocabulary
head, loss, and optimizer, so its end-to-end ratio is necessarily different.

On 2026-07-18, the corrected hard-forward/soft-backward novelty and admission
equations were revalidated from public commit `f35f6fc` on a fresh
`v5litepod-4`. For the tracked Screening v2 recurrence shape, the corrected TPU
Pallas path passed the fail-closed output, gradient, and loss gate and retained
a 6.83x forward and 2.69x forward-plus-backward median speedup.

| Corrected Screening v2 operation | Pallas TPU median | reference median | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 0.578 ms | 3.949 ms | 6.83x |
| Forward + backward | 4.888 ms | 13.171 ms | 2.69x |

This latest table supersedes the 2026-07-16 table only for the corrected v2
routing equations. It is still a recurrence microbenchmark, not complete-model
tokens per second.

## Implementation change

The previous accelerator path expressed the time recurrence as `lax.scan` over
general JAX operations. The new path uses:

- a TPU-specific persistent Pallas forward kernel;
- a TPU-specific Pallas reverse recurrence connected through `custom_vjp`;
- BF16 vector storage with FP32 state and accumulation;
- FP32 `sa` values and interval state checkpoints for backward reconstruction;
- a local `jax.shard_map` boundary for sharded Pallas execution;
- `pl.dslice`-based, slice-only `Ref` access required by sharded JAX 0.10
  Pallas references.

Automatic dispatch now selects `pallas_tpu` on TPU. CPU continues to use the
reference recurrence. FFI is not selected automatically.

The screening kernel uses the same external BF16-vector/FP32-state contract,
but its TPU body keeps vector values in rank-two physical tiles such as
`[1, feature]` and `[slots, 1]`. The initial rank-one implementation triggered
a TPU compiler `VectorLayout::join` assertion before execution. Keeping the
rank-two layout inside Pallas, then squeezing only at the public boundary,
removed the compiler failure without changing the public shapes.

## Test environment

| Item | Value |
| --- | --- |
| Zone | `us-west4-a` |
| TPU | single-host `v5litepod-4` slice, four TPU v5e devices |
| Microbenchmark placement | one TPU device from the slice |
| Python | 3.13.14 |
| JAX / jaxlib | 0.10.0 / 0.10.0 |
| libtpu | 0.0.40 |
| Vector dtype | BF16 |
| State and accumulation dtype | FP32 |
| Transparent hugepages | disabled |

The WKV session used `rwkv7m-pallas-260714`. The screening session used
`rwkv7m-screening-260716`; both were `v5litepod-4` TPU VMs in `us-west4-a`.
The corrected Screening v2 session used `rwkv7m-screening-st-260718`, also a
single-host `v5litepod-4` in `us-west4-a`.

## WKV microbenchmark method

The benchmark used six random BF16 vector inputs and one random FP32 initial
state with the following WKV shape:

```text
time       T = 128
batch      B = 1
heads      H = 12
head size  N = 64
vectors        [T, B, H, N]
state          [B, H, N, N]
```

Pallas and reference functions were compiled independently with `jax.jit`.
Compilation and one warmup execution were excluded. Each reported latency is
the elapsed time for 100 sequential calls divided by 100, with
`jax.block_until_ready` applied to every result.

The backward measurement used `jax.value_and_grad` over all six vector inputs
and the initial state. Its scalar objective was the sum of squared FP32-cast WKV
activations. The final recurrent state did not contribute directly to this
microbenchmark loss.

Raw final measurements were:

```text
Pallas forward:              0.27634697 ms
reference forward:           0.30719018 ms
Pallas forward + backward:   0.53761736 ms
reference forward + backward: 0.76567165 ms
```

The Pallas and reference final FP32 states had a measured maximum absolute
difference of `0.0` in this run. Separate real-accelerator parity tests also
checked BF16 outputs and gradients for all seven differentiable inputs.

### 2026-07-16 WKV revalidation

The current schema-v2 benchmark harness was rerun on the screening-session VM
with the same WKV shape. It used five excluded warmups, 100 synchronized
iterations, a fixed PCG64-generated input, and disabled Python GC.

| Operation | Pallas TPU | `lax.scan` reference | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 0.276 ms | 0.309 ms | 1.12x |
| Forward + backward | 0.528 ms | 0.657 ms | 1.24x |

This revalidation is the current reproducible WKV result. The 2026-07-14 raw
numbers above are retained as the historical first validation rather than
silently replacing a prior measurement. The change in reference
forward-plus-backward latency reflects the revised measurement harness, so the
two dates should not be combined into a trend claim.

## Integration validation

The timing result was interpreted only after the following real-TPU checks
passed with `interpret=False`:

1. BF16 forward output and FP32 final-state parity against the reference.
2. Gradient parity for all six vector inputs and the initial FP32 state.
3. A complete one-layer BF16 NNX model loss and parameter backward pass.
4. A four-device `data=1, model=4` forward and optimizer step.

The four-device audit completed with finite outputs, loss `3.9594927`, and
optimizer step 1. Its post-SPMD executable retained the previous collective
counts: 13 all-reduces, 3 all-to-alls, and no all-gathers.

The real-accelerator parity test is invoked with:

```bash
uv run pytest tests/test_wkv_pallas_accelerator.py -q
```

The four-way TPU model-sharding audit used:

```bash
uv run rwkv7m-audit-nnx-model-parallel \
  --model-axis-size 4 \
  --d-model 32 \
  --n-heads 4 \
  --head-size 8 \
  --screening \
  --write-screening
```

## Screening correctness and performance

The screening accelerator test enabled writes, value normalization, leaky
warmup, and the age mask. It compared all six public outputs and gradients for
all 15 differentiable inputs with the reference on a real TPU. The test passed
with BF16 public output and FP32 state boundaries intact:

```bash
uv run pytest tests/test_screening_pallas_accelerator.py -q
```

The performance run used the `0.185b` screening recurrence shape, the fixed
random seed 31, three excluded warmups, 20 measured iterations, disabled
Python GC, and a synchronization after every call. Pallas and reference used
the same arrays and objective. Compilation, input construction, and host to
device transfer were outside the measurement windows.

```bash
uv run python scripts/benchmark_screening_accelerator.py \
  --warmup 3 \
  --iterations 20 \
  --disable-python-gc
```

The maximum absolute output error was `5.96e-8`; the maximum absolute gradient
error over all 15 inputs was also `5.96e-8`. The scalar training-loss
difference was `0.0`.

The four-device `data=1, model=4` audit also passed after the screening change.
It executed finite forward output and one complete optimizer step with loss
`3.95950198`. Screening slots had sharding
`P('data', None, 'model')`. The post-SPMD executable contained 2 all-gathers,
13 all-reduces, and 3 all-to-alls. The two all-gathers are the explicit slot
feature gathers at the screening recurrence boundary.

## Screening v2 real-hardware gate

On 2026-07-16, commit `6fc9a48a1d2dbdc82561c7aaad81d46ff5f0d4d4`
was validated on a temporary single-host `v5litepod-4` named
`rwkv7m-screening-v2-260716` in `us-west4-a`. The software versions matched the
earlier gate: Python 3.13.14, JAX/jaxlib 0.10.0, Flax 0.12.7, Optax 0.2.8, and
libtpu 0.0.40. Transparent hugepages remained disabled.

This is a historical record for that commit. The later
hard-forward/soft-backward novelty and admission correction changes both the
forward route and its VJP. CPU interpret-mode GPU/TPU parity passes for the
corrected equations, but this table must not be cited as real-v5e validation of
the corrected HEAD until the accelerator parity command is rerun.

The real-accelerator regression test used `competitive_novel`, two read tiles,
and interval-2 tape checkpointing. It passed all six public outputs and all 17
input gradients against the portable reference:

```bash
uv run pytest -q tests/test_screening_pallas_accelerator.py
```

The tracked recurrence measurement then used `T=128, B=1, M=16`, slot size
128, key/value size 64, route power 2, four fixed-total-dimension read tiles,
and interval-16 checkpointing. Inputs were device resident; two warmups and
five synchronized calls were excluded/included respectively; Python GC was
disabled.

| Screening v2 operation | Pallas TPU median | reference median | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 0.572 ms | 3.874 ms | 6.77x |
| Forward + backward | 4.856 ms | 12.380 ms | 2.55x |

The scalar loss difference was zero. The maximum absolute output error was
`3.81e-6`. The maximum gradient relative L2 error over all inputs was
`2.33e-5`; the largest absolute gradient difference was `0.003662` for the age
carry, whose relative L2 error was `1.32e-6`. These are recurrence-only
measurements, not a complete-model throughput result.

A separate four-device `data=1, model=4` small-model audit enabled value-space
gating, rank-4 factorized candidates, competitive novel routing, route power 2,
two read tiles, and interval-2 checkpointing. It completed finite forward,
backward, and optimizer step 1 with loss `4.34657097`. Screening slots retained
`P('data', None, 'model')`. The compiled forward contained 2 all-gathers,
14 all-reduces, and 6 all-to-alls.

The real TPU gate exposed two classes of issue hidden by CPU interpret mode:

- Mosaic TPU rejected float iota, feature-to-tile shape casts, short-vector
  relayouts/reductions, and `powf`; the TPU kernel now uses integer tie-break
  indices, rank-two static slices, scalar-unrolled slot/bank routing and
  statistics, and multiplication/log-exp route powers;
- the factorized slot projection initially treated its slot axis as a leading
  data axis under explicit sharding; its replicated output placement is now
  explicit and has a local regression test.

This establishes single-host v5e functional and recurrence-performance gates
for Screening v2 at commit `6fc9a48`. It does not establish the post-gate
novelty/admission correction, measured checkpoint peak-memory savings,
complete 0.185B train-step throughput, multi-host scaling, GPU v2 lowering, or
model-quality benefit.

The VM was READY at `2026-07-16T12:35:52Z`. The delete operation completed by
approximately `13:18:23Z`; the zone listing was empty and describing the VM
returned `NOT_FOUND`. The approximately 42.5-minute READY-to-delete interval is
about 3.4 USD at the previously documented 1.20 USD per v5e chip-hour rate;
this is an estimate, not an exported billing record.

## Corrected Screening v2 real-hardware revalidation

On 2026-07-18, public commit
`f35f6fca1461217d54fb3b494d68563eaf7e8a95` was cloned directly from
`https://github.com/HeiMao-BaiMao/RWKV7M.git` onto temporary TPU VM
`rwkv7m-screening-st-260718`. The VM exposed four TPU v5 lite devices in a
`2 x 2` topology. Python 3.13.14, JAX/jaxlib 0.10.0, libtpu 0.0.40, Flax
0.12.7, and Optax 0.2.8 were installed through the repository's TPU extra.

The real-accelerator regression test passed with native TPU lowering rather
than interpret mode:

```bash
uv run pytest -q tests/test_screening_pallas_accelerator.py
```

It completed as `1 passed` in 11.68 seconds and compared all six public outputs
and all 17 input gradients against the portable reference. The tracked
recurrence benchmark then used `T=128, B=1, M=16`, slot size 128, key/value
size 64, four read tiles, route power 2, competitive novel routing, and
interval-16 checkpointing. Three warmups and ten synchronized iterations were
used with Python GC disabled and device-resident inputs.

```bash
uv run python scripts/benchmark_screening_accelerator.py \
  --backend pallas_tpu \
  --write-mode competitive_novel \
  --time 128 --batch 1 --slots 16 \
  --slot-size 128 --key-size 64 --value-size 64 \
  --read-tiles 4 --route-power 2 \
  --checkpoint-interval 16 \
  --warmup 3 --iterations 10 --disable-python-gc
```

| Corrected v2 measurement | Pallas TPU median | reference median | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 0.5778045 ms | 3.9487745 ms | 6.834x |
| Forward + backward | 4.8878200 ms | 13.1706545 ms | 2.695x |

The fail-closed parity gate passed. The loss difference was `0.0`, maximum
output absolute error was `2.3842e-7`, maximum gradient absolute error was
`0.0078125`, and maximum gradient relative L2 error was `1.6613e-5`. These
figures are for the corrected novelty and admission VJP and therefore close
the real-v5e correctness gap left by the 2026-07-16 record.

### Checkpoint length and strength boundary

The same recurrence shape was also run at `T=512` with interval-8
checkpointing. It passed the default parity gate and produced these medians:

| `T=512` measurement | Pallas TPU median | reference median | Speedup |
| --- | ---: | ---: | ---: |
| Forward | 1.736230 ms | 15.425279 ms | 8.884x |
| Forward + backward | 18.854747 ms | 55.000542 ms | 2.917x |

The maximum output absolute error was `1.1921e-6`, maximum gradient absolute
error was `6.1035e-5`, maximum gradient relative L2 error was `9.1872e-6`, and
the loss difference was zero.

A direct single-kernel `T=2048` run did not compile on v5e. Both interval 16
and interval 32 failed with `CompileTimeScopedVmemOom`: the scoped allocation
was reported as `17.05M` against a `16.00M` limit. Repeating the failure with a
larger checkpoint interval shows that boundary checkpoint count is not the
dominant scoped allocation. `T=4096` was not attempted after the repeat
reproduced the same compiler limit. This is a direct long-kernel limit, not a
failure of the production chunked path: the tracked 0.185B config keeps
`sequence_chunk_size=128`.

Near-limit reconstruction was probed at `T=512`, interval 8, with every update
rate set to 0.90, 0.925, and 0.95. Loss difference remained zero and maximum
gradient relative L2 error remained at or below `1.43e-4`, but the default
independent absolute-error gate rejected every rate because the initial-age
gradient differed by about 0.03. At rate 0.95 the reference age gradient
reached 45,404.73, the absolute difference was 0.03223, relative L2 error was
`8.65e-7`, and a combined `atol=1e-2, rtol=5e-3` all-close check passed. This
does not demonstrate divergence at 0.95; it demonstrates that the deliberately
strict scale-independent absolute gate is conservative for large-magnitude
carry gradients. The public 0.95 configuration ceiling remains an algebraic
reconstruction bound, not a per-shape numerical-accuracy guarantee.

### Full tracked model and four-way mesh

The tracked `rwkv7m-0.185b-screening-v2` config was audited with
`data=1, model=4`, full vocabulary 65,536, write screening enabled,
rematerialization enabled, and its configured 128-token recurrent chunk. Both
context 128 and the configured maximum context 512 completed finite forward
and a complete backward/optimizer step:

```bash
uv run rwkv7m-audit-nnx-model-parallel \
  --model-axis-size 4 \
  --model-config configs/rwkv7m-0.185b-screening-v2.json.example \
  --ctx-len 512 \
  --vocab-size 65536 \
  --write-screening
```

| Context | Loss | Optimizer step | Forward collectives |
| ---: | ---: | ---: | --- |
| 128 | 20.728317 | 1 | 31 all-gathers, 54 all-reduces, 6 all-to-alls |
| 512 | 20.723475 | 1 | 40 all-gathers, 54 all-reduces, 6 all-to-alls |

Both compiled executables exercised row- and column-parallel kernels. BF16
logits had shapes `[1, 128, 65536]` and `[1, 512, 65536]`. FP32 screening slots
retained `P('data', None, 'model')`, WKV state retained
`P('data', 'model', None, None)`, and all arrays had four addressable shards.
The approximately 97- and 112-second process durations include compilation and
are intentionally not reported as training throughput.

This gate establishes corrected single-host v5e lowering, recurrence parity
and latency, checkpointed `T=512` parity, and full configured-context
four-device execution. It does not establish measured peak memory,
steady-state complete-model throughput, multi-host or multi-slice scaling,
model-quality benefit, or real-GPU v2 behavior.

## Complete-model compute-only check

A fixed device-resident `1 x 128` batch was used with the `0.185b` preset on
one TPU device. Both variants disabled block rematerialization, sequence
chunking, and head chunking so that their execution settings were identical.
Each result used two excluded warmups, ten synchronized iterations, and
disabled Python GC.

| Variant | Median full step | Median throughput |
| --- | ---: | ---: |
| Baseline | 20.394 ms | 6,276 token/s |
| Read-write screening | 21.002 ms | 6,095 token/s |

For this shape, read-write screening added about 3.0% median latency, or 2.9%
throughput cost, while adding 1,028,486 parameters. This is a complete-model
comparison between variants, not a Pallas-versus-reference comparison: both
variants use the production TPU Pallas WKV path, and the read-write variant
also uses Pallas screening.

The preset-default rematerialized read-write path was separately exercised
with five iterations and completed at a median 22.735 ms, or 5,630 token/s.
This shorter run is a compatibility check, not the primary comparison above.
The corrected compute-only harness now performs optimizer-only and full-step
windows through the functional NNX model/optimizer state, matching the actual
NNX optimizer node types instead of passing array gradients to an incompatible
raw Optax state tree.

## Interpretation and limitations

The measured result supports the narrower conclusion that the current Pallas
WKV recurrence is faster than the current `lax.scan` WKV reference for this one
v5e shape, especially in backward. It does not yet establish:

- complete-model or optimizer-step throughput improvement;
- performance for other batch, sequence, head, or checkpoint shapes;
- multi-host TPU scaling;
- XProf-confirmed bottleneck attribution;
- results with transparent hugepages enabled;
- production stability across future experimental Pallas API changes.

The new screening result likewise establishes a substantial recurrence-level
improvement and successful one-host SPMD integration for the tested shapes. It
does not establish multi-host scaling, other preset shapes, or that screening
improves model quality. The complete-model result shows low overhead for this
small batch; it is not evidence for screening's validation-loss benefit.

The next performance gate should compare complete steady-state training steps
for tracked presets, then use XProf to tune checkpoint intervals, VMEM usage,
and program layout. Kernel-level numbers should remain diagnostic evidence, not
the final dispatch criterion.

## Vocabulary-tiled training-head gate

On 2026-07-16, a separate `v5litepod-4` run evaluated the TPU-specific Pallas
tile reducer used by the opt-in streaming training head. With BF16, 128
positions, hidden size 768, and vocabulary 65,536, tile sizes from 4,096 through
65,536 all passed component and gradient parity. They did not pass the speed
gate: forward ratios ranged from 0.40x to 0.79x and forward/backward ratios from
0.38x to 0.83x relative to the full XLA head. Tile 4,096 reduced the largest
BF16 logits allocation from 16 MiB to 1 MiB, establishing its value as a memory
control rather than a default throughput optimization.

The tracked benchmark is `scripts/benchmark_training_head_accelerator.py`. It
uses fixed device-resident inputs, excludes compilation, records forward and
forward/backward independently, blocks on all leaves after each call, and
reports both component and gradient parity. The default model configuration
therefore leaves `training_vocab_tile_size=None`; users can opt in with
`--training-vocab-tile-size` when memory is the binding constraint.

## Resource cleanup

After all correctness, sharding, and timing checks completed, the temporary TPU
VM `rwkv7m-pallas-260714` was deleted. The Google Cloud delete operation
completed successfully, and a subsequent TPU VM listing for `us-west4-a`
returned no remaining TPU VMs.

The 2026-07-16 screening VM `rwkv7m-screening-260716` was likewise deleted
after its final four-device audit. A subsequent `tpu-vm list` for
`us-west4-a` returned no rows, and an explicit describe of that VM returned
`NOT_FOUND`.

The 2026-07-16 training-head VM `rwkv7m-head-260716` was deleted after its
standalone and NNX integration gates. The zone list returned no rows and an
explicit describe returned `NOT_FOUND`.

The 2026-07-18 corrected-v2 VM `rwkv7m-screening-st-260718` was deleted after
the context-512 model-axis audit. A subsequent TPU VM list for `us-west4-a`
returned `[]`, and an explicit describe returned `NOT_FOUND`.
