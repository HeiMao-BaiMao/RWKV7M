# RWKV7M.paper.md remaining implementation TODO (2026-05-20)

Current status: the core JAX/Flax implementation and state-level screening training path exist, but the full research proposal in `RWKV7M.paper.md` is not fully implemented or validated.

## Implemented baseline to preserve
- Installable `rwkv7m` package using `uv`.
- JAX/Flax + Optax training/inference runtime.
- `.bin/.idx` dataset training support modeled after RWKV-LM-V7 data flow.
- RWKV-style TimeMix/ChannelMix/recurrent state carry.
- State-level screening core: slots, read screening, normalized q/k, Trim-and-Square relevance, non-sum-to-one aggregation, TanhNorm, residual fusion.
- Screening phases: `read_screening_only` and `read_write`.
- Write screening with separate write q/k/tau parameters, bank-specific update rates, age updates, and config validation.
- Tests currently pass after the latest implementation round.

## High-priority missing work from RWKV7M.paper.md
1. Implement an evaluation harness for the paper's claims:
   - passkey retrieval
   - Needle-in-a-Haystack
   - RULER-style long-context tasks
   - MQAR / associative retrieval
   - long-form consistency tasks
2. Add comparison baselines:
   - vanilla RWKV-style model without screening
   - softmax-over-slots baseline
   - random / frozen / shuffled controls
   - optionally Mamba / RetNet / DeltaNet comparison hooks if practical
3. Add causal intervention and analysis hooks:
   - slot ablation
   - slot patching
   - read shuffle
   - write suppression
   - frozen-slot controls
   - logits delta / output sensitivity analysis
4. Improve memory hygiene mechanisms:
   - make `usage_ema` meaningful instead of only stored in state
   - implement/test dead-slot prevention
   - implement/test diversity or decorrelation auxiliary losses
   - decide and document age policy per bank
5. Strengthen observability:
   - richer screening metrics and histograms
   - per-bank usage/update statistics
   - read/write relevance diagnostics
   - optional debug dumps for screening behavior
6. Implement production training features:
   - checkpoint save/load
   - richer train config files
   - resume training
   - logging integration
   - tokenizer integration
7. Implement interoperability/performance work:
   - upstream RWKV checkpoint conversion or mapping, if required
   - fused/custom kernels where JAX/XLA output is insufficient
   - distributed training utilities
8. Update documentation after implementation:
   - keep `RWKV7M.md` as implementation spec
   - keep `RWKV7M.paper.md` aligned with the implemented optimized design
   - README should distinguish "implemented", "experimental", and "not yet implemented"

## Important interpretation
Do not claim that `RWKV7M.paper.md` has been fully realized until the evaluation plan, baselines, causal analyses, and acceptance criteria have been implemented and run. The current implementation demonstrates that state-level screening can be trained, but it does not yet prove the paper's long-context memory claims.