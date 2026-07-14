# rwkv7m

[日本語 README](README.ja.md)

JAX/Flax reference implementation of an RWKV-7 style recurrent language model with optional state-level screening memory.

This repository is research-oriented. The current implementation is optimized for correctness, tests, and API usability before custom kernels or checkpoint compatibility.

For a more detailed setup and training walkthrough, including config files, checkpoints, resume, distributed runs, and run artifact auditing, see the Japanese README.

## Install

From this checkout:

```powershell
uv sync
uv run pytest -q
```

From Git:

```powershell
uv add git+https://github.com/HeiMao-BaiMao/RWKV7M.git
```

or:

```powershell
uv pip install git+https://github.com/HeiMao-BaiMao/RWKV7M.git
```

## Minimal Inference

```python
import jax
import jax.numpy as jnp
from rwkv7m import create_runtime, generate_ids, tiny_config

cfg = tiny_config(vocab_size=256, d_model=64, n_layers=3, n_heads=4, head_size=16)
runtime = create_runtime(jax.random.PRNGKey(0), cfg, batch_size=1)

prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
new_ids = generate_ids(runtime, prompt, max_new_tokens=8, temperature=0.0)
print(new_ids)
```

## Minimal Training

```python
import jax
from rwkv7m import create_train_runtime, tiny_config, train_batch
from rwkv7m.train import generate_toy_batch

cfg = tiny_config(vocab_size=256, d_model=64, n_layers=3, n_heads=4, head_size=16)
runtime, state = create_train_runtime(jax.random.PRNGKey(1), cfg, batch_size=2, total_steps=10)

batch = generate_toy_batch(jax.random.PRNGKey(2), batch_size=2, seq_len=8, vocab_size=cfg.vocab_size)
state, metrics = train_batch(state, batch, runtime)
print(float(metrics["loss"]))
```

`train_batch` resets recurrent state by default, which is the correct mode for independently sampled training chunks. Use `carry_state=True` only for deliberate streaming/stateful training. For binidx training and evaluation, pair carry-state mode with `sampling_mode="sequential"` / `--sampling-mode sequential`; carrying state across the default shuffled `magic` sampler mixes unrelated chunks. Sequential sampling assigns one stream lane per batch row and automatically resets carried RWKV/screening state when a lane wraps back to its beginning. The default path reuses immutable initial zero states in the runtime, so it does not rebuild zero states every step for the configured batch size.

## Training From RWKV-LM-V7 `.bin/.idx`

`rwkv7m` can read the same binidx dataset format used by RWKV-LM-V7.
Download the tokenized Minipile dataset into `data/`, then pass the prefix path without `.bin` / `.idx`.

```powershell
New-Item -ItemType Directory -Force data
wget -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

CLI smoke run:

```powershell
uv run rwkv7m-train-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 100 `
  --vocab-size 65536 `
  --d-model 128 `
  --n-layers 4 `
  --n-heads 4 `
  --head-size 32
```

Longer run with reference checkpoints and periodic eval:

```powershell
uv run rwkv7m-train-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 1000 `
  --vocab-size 65536 `
  --output-dir out/minipile-smoke `
  --save-every 100 `
  --eval-every 100 `
  --eval-steps 10
```

The same CLI accepts JSON config files. Explicit CLI options override config values. Repository examples use `.json.example`; generated or local `.json` files are ignored.

```json
{
  "data_file": "data/minipile",
  "ctx_len": 512,
  "batch_size": 1,
  "steps": 1000,
  "vocab_size": 65536,
  "output_dir": "out/minipile-smoke",
  "save_every": 100,
  "eval_every": 100,
  "eval_steps": 10
}
```

```powershell
uv run rwkv7m-train-binidx --config configs/minipile-smoke.json.example --steps 2000
```

Resume runs continue for `--steps` additional optimizer updates:

```powershell
uv run rwkv7m-train-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 1000 `
  --resume out/minipile-smoke/ckpt-00001000 `
  --output-dir out/minipile-smoke
```

For long stream training, use sequential sampling and carry-state mode from the start:

```powershell
uv run rwkv7m-train-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 1000 `
  --sampling-mode sequential `
  --carry-state `
  --vocab-size 65536 `
  --output-dir out/minipile-stream `
  --save-every 100
```

`magic_prime` is computed automatically from dataset size and `ctx_len`; pass `--magic-prime` to force a specific value. The automatic value is constrained so every sampled `ctx_len + 1` span stays inside the `.bin` file.

For read/write screening from scratch, the write branch uses slot identity in its write key and a tiny `write_rel_floor` update floor. This avoids a dead write branch when slots are all zero at initialization.

For proof-oriented runs, `write_rel_floor=0` tests true write rejection. Memory-bank update rates can also be specified as interpretable token half-lives with `--short-half-life-tokens`, `--mid-half-life-tokens`, and `--long-half-life-tokens`. The legacy `mu_*_max` behavior remains available when half-lives are omitted.

## Comparison experiments

To download MiniPile, check out `RWKV-Vibe/RWKV-LM-V7`, and build upstream/local comparison checkpoints with a matched token budget:

```bash
bash scripts/compare_minipile.sh --profile smoke --no-run
bash scripts/compare_minipile.sh --profile small --seeds "42 43 44"
```

The local matrix contains `local_core_baseline`, `local_read_screening`, `local_mechanism`, and a widened no-screening `local_param_control`. The CUDA path can additionally run the upstream PyTorch/CUDA RWKV-LM-V7 reference. Each run writes checkpoints, JSONL/CSV metrics, parameter counts, the recorded protocol, and replay commands under `out/comparison/<run-id>/`.

The upstream target now defaults to `--upstream-launcher single_gpu`. This path does not invoke the Lightning trainer or a DeepSpeed distributed strategy. It still imports the pinned official checkout's x070 model, fused CUDA operators, fused L2Wrap cross entropy, `.bin/.idx` reader and cubic sampler, initialization, and `deepspeed.ops.adam.FusedAdam`; it also preserves the official `att.w0` 2x learning-rate group, AdamW decay grouping, gradient clipping, warmup, and cosine schedule. A larger global batch is reproduced on one GPU by gradient accumulation over the official global sample order. The runner writes `run_config.json`, `metrics.jsonl`, `metrics.csv`, `run_summary.json`, `rwkv-init.pth`, and `rwkv-final.pth` under `upstream_rwkv_lm_v7/`.

For the downloaded MiniPile, the wrapper automatically creates a zero-copy single-item binidx view required by the official sampler. It also applies the tracked Ada CUDA compatibility patch in `patches/upstream_rwkv7_ada_atomic.patch`; the upstream commit and actual diff hash are recorded with the run. Set `APPLY_UPSTREAM_ADA_PATCH=0` only when intentionally testing an unmodified upstream checkout on hardware where it compiles.

Use `--upstream-launcher deepspeed` only when the Lightning/DeepSpeed stack itself is part of the experiment or has been validated on the machine. That legacy mode remains available, but is no longer the paper-baseline default.

The `smoke` profile validates the pipeline; it is not research evidence. `small` is approximately the upstream 0.19B scale, but its default 2,000 steps should still be treated as a starting point rather than a sufficient paper training budget. Cached data and the upstream checkout live under `.comparison/`.

### NVIDIA CUDA example

The CUDA extras target Linux systems with an NVIDIA driver. Choose the extra matching the installed CUDA major version, then confirm that JAX sees the GPU:

```bash
uv sync --extra cuda12
uv run --extra cuda12 python -c "import jax; print(jax.devices())"
```

Run the local JAX variants with independent held-out evaluation alongside the same-token-budget upstream PyTorch/CUDA training reference:

```bash
LOCAL_PREFIX="uv run --extra cuda12" \
bash scripts/compare_minipile.sh \
  --profile small \
  --backend cuda \
  --run-targets both \
  --verify-core-parity \
  --seeds "42 43 44" \
  --steps 10000 \
  --eval-data-file /data/minipile-heldout \
  --eval-every 500 \
  --eval-steps 100
```

The command above uses the direct single-GPU upstream baseline by default. To make the choice explicit, add `--upstream-launcher single_gpu`; to replay the former launcher path, use `--upstream-launcher deepspeed`.

The local project requires Python 3.13 or newer, while the upstream repository pins older dependencies. The comparison script therefore creates `.comparison/venvs/rwkv-lm-v7` as a dedicated Python 3.12 environment with `uv venv` and installs the upstream requirements with `uv pip`. It also constrains `setuptools<81`, because the upstream Lightning 1.9.5 dependency still imports `pkg_resources`. The OS must provide Python 3.12 development headers (`python3.12-dev` on Ubuntu), a C++ build toolchain, and the CUDA toolkit. The environment is reused across seeds, and its dependencies are synchronized automatically when either the upstream requirements or this compatibility constraint changes. The following preparation invocation creates the venv without starting training:

```bash
bash scripts/compare_minipile.sh --profile smoke --no-run
```

Set `UPSTREAM_VENV=/path/to/venv` to change the cache location, `UPSTREAM_PYTHON_VERSION=3.12` to select the Python requested from `uv`, or `INSTALL_UPSTREAM_DEPS=1` to force dependency synchronization. The legacy `UPSTREAM_PYTHON` variable is accepted as the interpreter request used to create the venv, but upstream commands always execute with the venv Python. Use `cuda13` instead of `cuda12` in both `uv` commands and `LOCAL_PREFIX` when appropriate. The NVIDIA driver alone is not sufficient for the official fused extensions: the server must also provide a CUDA compiler and development headers compatible with the installed PyTorch CUDA build. For example, the Blackwell validation used PyTorch CUDA 13.0 with the CUDA 13.0 compiler and development libraries.

`--verify-core-parity` adds a separate implementation-equivalence check before interpreting the training curves. It creates a small deterministic official x070 model, runs the actual upstream fused BF16 CUDA forward and backward paths on a fixed 16-token sequence, and performs one official FusedAdam step. It exports every official tensor, logits, gradients, and updated weights; maps them into the no-screening Flax tree; and verifies the same loss, gradient, and optimizer rules from zero recurrent state. The machine-readable result is written to `upstream_core_parity/parity_report.json`; incomplete tensor coverage or tolerance, cosine-similarity, or relative-L2 failure makes the comparison command fail.

This check establishes fixed-weight, zero-initial-state sequence forward/backward parity and one-step BF16 optimizer compatibility for the tested upstream commit and recorded diff. It does not establish identical multi-step or multi-seed training dynamics. The capture uses the official JIT path, which binds the upstream fused operators unambiguously; on the validated upstream commit it produced exactly the same reference logits as the mechanically disambiguated non-JIT wrappers. The reference capture requires the upstream Python environment, PyTorch/CUDA, and a working compiler for the official extensions. The generated small model avoids copying a multi-billion-parameter training checkpoint solely for the parity check; `scripts/capture_upstream_rwkv7_reference.py --checkpoint ...` can be used separately when a particular official `.pth` must be checked. See [NVIDIA Blackwell validation](docs/gpu_blackwell_validation.md) and [NVIDIA L40S validation](docs/gpu_l40s_validation.md) for measured results and limitations.

The two parity stages can also be replayed directly:

```bash
.comparison/venvs/rwkv-lm-v7/bin/python scripts/capture_upstream_rwkv7_reference.py \
  --upstream-repo .comparison/RWKV-LM-V7 \
  --output out/upstream-core-parity/reference.npz \
  --optimizer-step

JAX_PLATFORMS=cuda uv run --extra cuda12 rwkv7m-verify-upstream-rwkv7 \
  out/upstream-core-parity/reference.npz \
  --json-out out/upstream-core-parity/report.json
```

### Google TPU example

Install the TPU extra on a TPU VM and confirm device discovery:

```bash
uv sync --extra tpu
uv run --extra tpu python -c "import jax; print(jax.devices()); print(jax.process_count(), jax.process_index())"
```

On a single TPU host, run the local comparison matrix as follows. Passing the detected local device count avoids the comparison script's conservative one-device default for TPU:

```bash
TPU_DEVICES=$(uv run --extra tpu python -c "import jax; print(jax.local_device_count())")

LOCAL_PREFIX="uv run --extra tpu" \
bash scripts/compare_minipile.sh \
  --profile small \
  --backend tpu \
  --run-targets local \
  --devices "$TPU_DEVICES" \
  --global-batch-size "$TPU_DEVICES" \
  --seeds "42 43 44" \
  --steps 10000 \
  --eval-data-file /data/minipile-heldout \
  --eval-every 500 \
  --eval-steps 100
```

`upstream_rwkv_lm_v7` is CUDA/PyTorch-only and cannot run on TPU through this script. A cross-backend study must run the local matrix on TPU and the upstream target on NVIDIA CUDA as separate runs; throughput from different accelerator types should be reported separately, not treated as a hardware-controlled comparison. For multi-host TPU launches, configure the JAX coordinator variables on every host and use the distributed CLI procedure in [TPU Research Cloud Training](docs/tpu_research_cloud.md); the wrapper above is the straightforward single-host comparison path.

### Paper-oriented run requirements and outputs

For research evidence, always provide a fixed, independent binidx prefix with `--eval-data-file`; omitting it evaluates on the training data and is only a plumbing check. Keep the dataset split, tokenizer, token budget, optimizer settings, dtype, context length, global batch size, sampler, evaluation cadence, and evaluation token count fixed across local variants.

After each seed, inspect:

- `parameter_counts.csv` for the baseline, mechanism, and parameter-control sizes.
- `local_metric_summary.csv` for final train/eval metrics.
- `learning_speed_summary.csv` for best/final eval loss, perplexity, eval-loss AUC over tokens, and throughput.
- `learning_target_hits.csv` for steps/tokens needed to reach common loss targets.
- `parameters.env`, `protocol.md`, and `commands.sh` for provenance and replay.
- `upstream_core_parity/parity_report.json` for official CUDA vs local JAX fixed-weight core parity.

`--seeds` creates one run directory per seed, but the current scripts do not aggregate across seeds. Before placing values in a paper, combine the per-seed files and report at least the number of seeds and a dispersion or uncertainty measure such as standard deviation or a confidence interval. The upstream run is not included in the local held-out CSV summaries, and the repository does not yet provide FLOPs-matched controls or task-specific long-context benchmarks; do not infer those claims from the generated tables.

Python API:

```python
import jax
from rwkv7m import tiny_config, train_binidx

cfg = tiny_config(vocab_size=65536, d_model=128, n_layers=4, n_heads=4, head_size=32)
losses, runtime, state = train_binidx(
    jax.random.PRNGKey(0),
    cfg,
    "data/minipile",
    ctx_len=512,
    batch_size=1,
    num_steps=100,
)
print(losses[-1])
```

Quick baseline/screening comparison on the same binidx data:

```powershell
uv run rwkv7m-bench-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 10 `
  --vocab-size 65536
```

Validation loss/perplexity on binidx data:

```powershell
uv run rwkv7m-eval-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 10 `
  --vocab-size 65536
```

Stateful stream validation uses the same sequential/carry-state contract as training:

```powershell
uv run rwkv7m-eval-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 10 `
  --sampling-mode sequential `
  --carry-state `
  --vocab-size 65536
```

To convert JSONL text with the repository copy of the RWKV tokenizer vocabulary:

```powershell
uv run rwkv7m-make-binidx data/my_corpus.jsonl --output-prefix data/my_corpus --ctx-len 512
```

Tokenizer API:

```python
from rwkv7m import RWKVTokenizer

tokenizer = RWKVTokenizer()
ids = tokenizer.encode("Hello RWKV", add_eos=True)
text = tokenizer.decode(ids[:-1])
```

Text generation from a safetensors checkpoint:

```powershell
uv run rwkv7m-generate `
  --checkpoint out/minipile-smoke/ckpt-00001000 `
  --prompt "Hello" `
  --max-new-tokens 64 `
  --temperature 0.8 `
  --top-p 0.9
```

Safetensors export/import:

```python
from rwkv7m import load_model_safetensors, save_model_safetensors

save_model_safetensors("out/model.safetensors", runtime.variables["params"], runtime.config)
params, config, metadata = load_model_safetensors("out/model.safetensors")
```

`safetensors` is the canonical exchange format for external runtimes. Add `.pth` export only if a downstream runtime project needs it.

External runtime artifact boundary:

```python
from rwkv7m.backends.torch import load_torch_safetensors

state_dict, config, metadata = load_torch_safetensors("out/model.safetensors")
```

This is an artifact compatibility helper only. A PyTorch/non-JAX RWKV7M runtime is out of scope for this repository and should live in a separate runtime project.

Reference training checkpoint:

```python
from rwkv7m import load_train_checkpoint, save_train_checkpoint

save_train_checkpoint("out/ckpt-000001", state, runtime.config)
state, config, metadata = load_train_checkpoint("out/ckpt-000001", state)
```

TPU/distributed skeleton:

```python
from rwkv7m.distributed import (
    compute_batch_layout,
    create_host_binidx_dataset,
    data_parallel_sharding,
    host_batch_to_global_arrays,
    make_1d_mesh,
)

mesh = make_1d_mesh("data")
layout = compute_batch_layout(128, process_count=1, local_device_count=mesh.devices.size)
dataset = create_host_binidx_dataset(
    "data/minipile",
    ctx_len=512,
    global_batch_size=128,
)
batch = host_batch_to_global_arrays(
    dataset.get_batch(0),
    data_parallel_sharding(mesh),
    dataset.layout,
)
```

Reference train objects can be placed on the mesh for the data-parallel skeleton:

```python
from rwkv7m.distributed import replicate_train_objects

dist = replicate_train_objects(runtime, train_state, mesh=mesh)
```

The local-testable data-parallel train step boundary is:

```python
from rwkv7m.distributed import train_batch_data_parallel

dist, metrics = train_batch_data_parallel(dist, dataset.get_batch(0), dataset.layout)
```

The same skeleton is available as a CLI:

```powershell
uv run rwkv7m-train-binidx-dp `
  --data-file data/minipile `
  --ctx-len 512 `
  --global-batch-size 128 `
  --steps 10 `
  --vocab-size 65536 `
  --output-dir out/minipile-dp `
  --save-every 10 `
  --checkpoint-backend orbax `
  --keep-last-checkpoints 3 `
  --eval-every 10 `
  --eval-steps 2 `
  --summary-every 1 `
  --save-best-checkpoint `
  --prefetch-size 2
```

This is the first local-testable layer for TPU Research Cloud work. It includes process-aware Flax checkpoints, Orbax train-state checkpoints for TPU-scale runs, carry-state runtime checkpoint/resume for distributed local runs, checkpoint rotation with best-eval protection, structured JSONL/CSV metrics, run summaries, periodic validation, device prefetching, and optional multi-axis mesh / rule-based parameter placement hooks. Full tuned sharded checkpoint policy validation on real TPU pods is still pending.

Audit distributed run artifacts locally:

```powershell
uv run rwkv7m-audit-dp-run out/minipile-dp --require-complete --min-train-records 10
```

TPU setup and run notes are in [docs/tpu_research_cloud.md](docs/tpu_research_cloud.md).

### 7B planning and the NNX scale path

The small/reference model API remains on Flax Linen while the TPU-scale path is
being migrated to Flax NNX. The migration is gated by numerical parity rather
than replacing the reference implementation in one step. The first NNX gate
already covers topology-aware sharded initialization, an Optax Adam update,
Orbax save, restore in a new Python process, a second update, and preservation
of parameter and optimizer-state sharding.

Estimate parameter-related memory for the tracked 7B candidate without
allocating its tensors:

```powershell
uv run rwkv7m-plan-scale `
  --model-config configs/rwkv7m-7b-tpu.json.example `
  --model-axis-size 8 `
  --dtype-profile memory
```

The estimate deliberately excludes activations, recurrent/screening runtime
state, compiler temporaries, and collective buffers. It is a launch gate, not
an HBM-fit guarantee.

Validate the NNX lifecycle in two separate processes:

```powershell
uv run rwkv7m-verify-nnx-lifecycle --checkpoint-dir out/nnx-probe --mode create
uv run rwkv7m-verify-nnx-lifecycle --checkpoint-dir out/nnx-probe --mode restore
```

The existing distributed Linen CLI now uses topology-aware `jax.make_mesh()`
and synchronizes the complete updated train state before reporting per-step
throughput. The future NNX scale trainer will use larger asynchronous timing
windows.

## Public API

Common imports:

```python
from rwkv7m import (
    create_binidx_dataset,
    ModelConfig,
    ScreeningConfig,
    ScreenedRWKVModel,
    create_model_variables,
    create_runtime,
    create_train_runtime,
    generate_ids,
    train_batch,
    train_binidx,
    tiny_config,
)
```

Lower-level modules:

- `rwkv7m.model`: RWKV core, screening module, config, state helpers.
- `rwkv7m.data`: RWKV-LM-V7 compatible `.bin/.idx` reader and batch sampler.
- `rwkv7m.infer`: `prefill`, `decode_one`, `generate`.
- `rwkv7m.train`: optimizer, train state, toy batch, train step.

## Current Scope

- Flax Linen implementation.
- Full-sequence training path with `jax.lax.scan` inside recurrent components.
- RWKV-LM-V7 compatible `.bin/.idx` dataset reader and sampler.
- Chunked inference state carry for the reference RWKV state (`time_mix_x`, `channel_mix_x`, WKV matrix state).
- State-level screening with `read_screening_only` and `read_write` phases.
- Sequential carry-state binidx training and validation with lane-wrap state reset.
- RWKV tokenizer API and JSONL-to-binidx conversion.
- Wheel-packaged RWKV tokenizer vocabulary fallback.
- Safetensors export/import for Flax params plus model config metadata.
- Safetensors artifact metadata for architecture, dtype, dimensions, screening summary, and tokenizer vocabulary identity.
- PyTorch-readable safetensors loading helper for external runtime projects.
- Single-process reference training checkpoint save/load.
- Binidx validation loss/perplexity CLI.
- Local-testable distributed mesh/sharding helpers for TPU work.
- Data-parallel distributed binidx training CLI with process-aware Flax checkpointing, Orbax train-state checkpointing, carry-state runtime checkpoint/resume, checkpoint rotation with best-eval protection, structured JSONL/CSV logs, run summaries, validation hooks, run artifact audit CLI, device prefetching, and optional multi-axis mesh / rule-based parameter placement hooks.
- Installable package layout for `from rwkv7m import ...`.

Not yet included:

- production fused RWKV kernels,
- pretrained RWKV checkpoint conversion,
- in-repository PyTorch/non-JAX runtime backend (intentionally out of scope),
- fully tuned per-parameter TPU sharding rules,
- real TPU pod validation of Orbax sharded optimizer/parameter/runtime-state checkpoint save/resume,
- production-scale distributed TPU trainer validation on real TPU pods,
- task-specific long-context evaluation harnesses beyond stateful binidx validation.

## Tests

```powershell
uv run pytest -q
```

Current smoke coverage includes math helpers, shape checks, phase/config validation, scan consistency, public API inference, public API training, binidx data loading, sequential carry-state reset/eval behavior, safetensors/checkpoint boundaries, local distributed training boundaries, screening algebra/gradient parity, 7B abstract memory planning, and the cross-process NNX lifecycle. The current full suite is 120 tests.
