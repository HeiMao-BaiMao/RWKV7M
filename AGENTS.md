# AGENTS.md

## Project Direction

This repository is moving from a JAX/Flax reference implementation toward a research training stack that can scale on Google TPU Research Cloud while still allowing external non-JAX runtimes to consume portable checkpoint/export formats.

The long-term target is:

- large-scale training on Google TPU Research Cloud with JAX-first distributed training,
- portable model artifacts using `safetensors` plus explicit config/tokenizer metadata, readable by external runtimes including PyTorch,
- artifact-level backend separation so training output is not permanently tied to JAX/Flax,
- reproducible experiments with resumable checkpoints, validation, and benchmark tooling.

PyTorch runtime implementation is out of scope for this repository. The priority is TPU-scale JAX training plus export artifacts that other runtime projects can load.

## Current Baseline

The current implementation is still a reference path:

- Flax Linen model implementation.
- RWKV-7 style recurrent blocks.
- optional state-level screening memory.
- RWKV-LM-V7 compatible `.bin/.idx` dataset reader and sampler.
- small training and benchmark CLIs.
- local-testable TPU/distributed data-parallel training CLI with process-aware Flax checkpointing, Orbax train-state checkpointing, checkpoint rotation with best-eval protection, validation hooks, run summaries, structured metrics, prefetching, and multi-axis mesh hooks.
- repository-local RWKV tokenizer vocabulary at `data/rwkv_vocab_v20230424.txt`.

Do not use ignored `sample/` files as runtime dependencies. If useful material exists under `sample/`, copy the required artifact into tracked project-owned paths and make code depend on that tracked copy.

## Near-Term Priorities

1. Prepare TPU Research Cloud training.
   - Replace the current basic rule-based placement with tuned per-parameter sharding rules validated on TPU.
   - Validate Orbax train-state checkpointing on real TPU pods and harden sharded checkpoint policy.
   - Validate and tune throughput on real TPU pods.
   - Add failure recovery drills and resume documentation from interrupted TPU runs.

2. Harden portable checkpoint/export support.
   - Keep `safetensors` as the canonical portable artifact format.
   - Implement Flax params to safetensors export.
   - Implement safetensors import back into Flax params.
   - Ensure exported safetensors can be read by external PyTorch runtimes.
   - Store model config and tokenizer metadata next to weights.
   - Add round-trip tests on small models.
   - Add `.pth` export only if an external runtime actually requires it.

3. Promote tokenizer support into a stable package module.
   - Move tokenizer logic out of CLI-only code.
   - Provide `encode`, `decode`, and text generation helpers.
   - Keep tokenizer metadata tied to checkpoints and exports.

4. Add training checkpointing.
   - Save and restore model params, optimizer state, step, PRNG state, and dataset position.
   - Support checkpoint rotation.
   - Support resume from CLI.

5. Add validation and experiment reporting.
   - Validation loss/perplexity CLI.
   - Baseline vs screening vs read-write comparison reports.
   - Throughput logging with tokens/sec.
   - Structured CSV/JSON logs.

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

Current distributed code already covers the local-testable data-parallel layer, structured metrics, run summaries, validation hook, best-eval checkpoint tracking, checkpoint rotation, Orbax train-state checkpointing, and mesh construction. The remaining TPU work is deeper sharding policy, checkpoint validation on real TPU pods, and throughput tuning.

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

The repository should not implement a PyTorch RWKV runtime. It may keep lightweight optional helpers that prove exported artifacts can be read through `safetensors.torch` or, if needed later, `torch.load` for `.pth` files. The runtime itself belongs in a separate project.

## Engineering Rules

- Keep `sample/` ignored and out of runtime imports, tests, and README command paths.
- Keep small reference APIs working while adding large-scale infrastructure.
- Add tests for every format boundary: tokenizer, binidx, safetensors, checkpoint restore, distributed checkpointing, and external-runtime artifact loading.
- Prefer explicit config files and metadata over implicit assumptions.
- Do not claim upstream RWKV-7 checkpoint compatibility until conversion tests prove it.
- Do not treat state-level screening as proven beneficial until benchmark and validation results support it.
