# RWKV7M Direction: TPU TRC, safetensors, and multi-backend

The project direction is to evolve from a JAX/Flax reference implementation into a research training stack that can scale on Google TPU Research Cloud while keeping model artifacts portable for non-JAX execution.

## Long-term target

- Large-scale training on Google TPU Research Cloud with a JAX-first distributed trainer.
- Portable artifacts using `safetensors` plus explicit config/tokenizer metadata.
- Backend separation so inference/runtime is not permanently tied to JAX/Flax.
- Reproducible experiments with resumable checkpoints, validation, and benchmark tooling.

## Current baseline

The current code remains a reference path: Flax Linen model, RWKV-7 style recurrent blocks, optional state-level screening memory, RWKV-LM-V7 compatible binidx reader/sampler, small train/benchmark CLIs, and repository-local RWKV tokenizer vocabulary at `data/rwkv_vocab_v20230424.txt`.

Do not use ignored `sample/` files as runtime dependencies. If useful material exists under `sample/`, copy required artifacts into tracked project-owned paths and depend on those tracked copies.

## Priority order

1. Promote tokenizer support into `src/rwkv7m/tokenizer` with stable `encode`, `decode`, and text generation helpers.
2. Add `safetensors` export/import for Flax params, with config and tokenizer metadata, plus small-model round-trip tests.
3. Add training checkpoint save/load for params, optimizer state, step, PRNG state, and dataset position, with CLI resume.
4. Add validation loss/perplexity and structured experiment reporting.
5. Add TPU Research Cloud training infrastructure: JAX distributed initialization, mesh/sharding helpers, sharded train state, host-aware binidx pipeline, TPU VM docs.
6. Add non-JAX execution after export is stable, starting with PyTorch inference from safetensors. Do not attempt full PyTorch training parity before checkpoint/export boundaries are solid.

## Suggested structure

Distributed TPU layer:

```text
src/rwkv7m/distributed/
  mesh.py
  sharding.py
  trainer.py
  checkpoint.py
  input_pipeline.py
  metrics.py
```

IO/export layer:

```text
src/rwkv7m/io/
  safetensors.py
  flax_checkpoint.py
  config.py
  conversion.py
```

## Engineering rules

- Keep `sample/` ignored and out of runtime imports, tests, and README command paths.
- Keep existing small reference APIs working while adding large-scale infrastructure.
- Add tests for tokenizer, binidx, safetensors, checkpoint restore, and backend conversion boundaries.
- Prefer explicit config files and metadata over implicit assumptions.
- Do not claim upstream RWKV-7 checkpoint compatibility until conversion tests prove it.
- Do not treat state-level screening as beneficial until benchmark and validation results support it.
