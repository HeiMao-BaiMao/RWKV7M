# AGENTS.md

## Project Direction

This repository is moving from a JAX/Flax reference implementation toward a research training stack that can scale on Google TPU Research Cloud while still allowing non-JAX execution paths through portable checkpoint/export formats.

The long-term target is:

- large-scale training on Google TPU Research Cloud with JAX-first distributed training,
- portable model artifacts using `safetensors` plus explicit config/tokenizer metadata,
- backend separation so training and inference are not permanently tied to JAX/Flax,
- reproducible experiments with resumable checkpoints, validation, and benchmark tooling.

## Current Baseline

The current implementation is still a reference path:

- Flax Linen model implementation.
- RWKV-7 style recurrent blocks.
- optional state-level screening memory.
- RWKV-LM-V7 compatible `.bin/.idx` dataset reader and sampler.
- small training and benchmark CLIs.
- repository-local RWKV tokenizer vocabulary at `data/rwkv_vocab_v20230424.txt`.

Do not use ignored `sample/` files as runtime dependencies. If useful material exists under `sample/`, copy the required artifact into tracked project-owned paths and make code depend on that tracked copy.

## Near-Term Priorities

1. Promote tokenizer support into a stable package module.
   - Move tokenizer logic out of CLI-only code.
   - Provide `encode`, `decode`, and text generation helpers.
   - Keep tokenizer metadata tied to checkpoints and exports.

2. Add portable checkpoint/export support.
   - Add `safetensors` dependency.
   - Implement Flax params to safetensors export.
   - Implement safetensors import back into Flax params.
   - Store model config and tokenizer metadata next to weights.
   - Add round-trip tests on small models.

3. Add training checkpointing.
   - Save and restore model params, optimizer state, step, PRNG state, and dataset position.
   - Support checkpoint rotation.
   - Support resume from CLI.

4. Add validation and experiment reporting.
   - Validation loss/perplexity CLI.
   - Baseline vs screening vs read-write comparison reports.
   - Throughput logging with tokens/sec.
   - Structured CSV/JSON logs.

5. Prepare TPU Research Cloud training.
   - Add JAX distributed initialization.
   - Add mesh and sharding helpers.
   - Add sharded train state handling.
   - Add host-aware binidx input pipeline.
   - Add TPU VM setup and run documentation.

6. Add non-JAX execution path after export is stable.
   - Start with PyTorch inference loading safetensors.
   - Keep JAX as the first large-scale training backend.
   - Avoid trying to maintain full JAX and PyTorch training parity before checkpoint/export boundaries are stable.

## TPU Training Design Notes

The TPU path should be implemented as a new training layer rather than stretching the existing smoke APIs.

Preferred structure:

```text
src/rwkv7m/distributed/
  mesh.py
  sharding.py
  trainer.py
  checkpoint.py
  input_pipeline.py
  metrics.py
```

The distributed trainer should own:

- `jax.distributed.initialize()` setup,
- `jax.sharding.Mesh` construction,
- global batch and per-host batch calculation,
- parameter and optimizer sharding,
- host-local input sampling,
- prefetch to devices,
- resumable checkpoint state,
- validation hooks,
- metric aggregation.

Keep the existing `train_binidx` API as a smoke/reference path.

## Export And Backend Design Notes

Add a dedicated IO layer:

```text
src/rwkv7m/io/
  safetensors.py
  flax_checkpoint.py
  config.py
  conversion.py
```

Minimum safetensors metadata should include:

- format name and version,
- architecture,
- dtype,
- model dimensions,
- screening configuration,
- tokenizer vocabulary identity,
- source training step if available.

The first non-JAX backend should be inference-only PyTorch support that reads the exported safetensors and config. Full PyTorch training should be treated as a later project.

## Engineering Rules

- Keep `sample/` ignored and out of runtime imports, tests, and README command paths.
- Keep small reference APIs working while adding large-scale infrastructure.
- Add tests for every format boundary: tokenizer, binidx, safetensors, checkpoint restore, and backend conversion.
- Prefer explicit config files and metadata over implicit assumptions.
- Do not claim upstream RWKV-7 checkpoint compatibility until conversion tests prove it.
- Do not treat state-level screening as proven beneficial until benchmark and validation results support it.
