# rwkv7m

[English README](README.md)

RWKV-7 風の recurrent language model に、任意で state-level screening memory を追加した JAX/Flax リファレンス実装です。

このリポジトリは研究用途を主目的にしています。現時点では custom kernel や既存 checkpoint 互換性よりも、正しさ、テスト、API の使いやすさを優先しています。

## インストール

この checkout から使う場合:

```powershell
uv sync
uv run pytest -q
```

Git から追加する場合:

```powershell
uv add git+https://github.com/HeiMao-BaiMao/RWKV7M.git
```

または:

```powershell
uv pip install git+https://github.com/HeiMao-BaiMao/RWKV7M.git
```

## 最小推論例

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

## 最小学習例

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

`train_batch` はデフォルトで recurrent state をリセットします。これは、独立にサンプリングされた training chunk で学習する通常の使い方に合わせた挙動です。連続した streaming/stateful training を意図する場合だけ `carry_state=True` を指定してください。

## RWKV-LM-V7 `.bin/.idx` からの学習

`rwkv7m` は RWKV-LM-V7 と同じ binidx dataset 形式を読み取れます。
tokenized Minipile を `data/` にダウンロードし、`.bin` / `.idx` を除いた prefix path を指定します。

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

reference checkpoint と periodic eval 付きで長めに回す例:

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

同じ CLI は JSON config file も受け取れます。明示した CLI option は config の値を上書きします:

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
uv run rwkv7m-train-binidx --config configs/minipile-smoke.json --steps 2000
```

resume 時の `--steps` は、checkpoint から追加で実行する optimizer update 数です:

```powershell
uv run rwkv7m-train-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 1000 `
  --resume out/minipile-smoke/ckpt-00001000 `
  --output-dir out/minipile-smoke
```

`magic_prime` は dataset size と `ctx_len` から自動計算されます。特定の値を強制したい場合は `--magic-prime` を指定してください。自動計算値は、すべての `ctx_len + 1` span が `.bin` ファイル内に収まるよう制約されます。

同じ binidx data 上で baseline / screening / read_write を素早く比較するには:

```powershell
uv run rwkv7m-bench-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 10 `
  --vocab-size 65536
```

binidx data 上で validation loss/perplexity を測るには:

```powershell
uv run rwkv7m-eval-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 10 `
  --vocab-size 65536
```

リポジトリ内の RWKV tokenizer vocabulary を使って JSONL text を binidx に変換するには:

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

safetensors checkpoint から text generation:

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

Optional PyTorch checkpoint loading boundary:

```python
from rwkv7m.backends.torch import load_torch_safetensors

state_dict, config, metadata = load_torch_safetensors("out/model.safetensors")
```

これは backend 境界だけです。完全な PyTorch RWKV7M runtime はまだ未実装です。

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

reference train object は data-parallel skeleton 用に mesh 上へ配置できます:

```python
from rwkv7m.distributed import replicate_train_objects

dist = replicate_train_objects(runtime, train_state, mesh=mesh)
```

これは TPU Research Cloud 対応に向けた、ローカルテスト可能な最初の層です。完全な sharded training はまだ未実装です。

## Python API

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

主な top-level import:

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

## モジュール構成

- `rwkv7m.model`: RWKV core、screening module、config、state helper。
- `rwkv7m.data`: RWKV-LM-V7 互換 `.bin/.idx` reader と batch sampler。
- `rwkv7m.infer`: `prefill`、`decode_one`、`generate`。
- `rwkv7m.train`: optimizer、train state、toy batch、train step。

## 現在の対応範囲

- Flax Linen 実装。
- recurrent component 内の `jax.lax.scan` による full-sequence training path。
- RWKV-LM-V7 互換 `.bin/.idx` dataset reader と sampler。
- reference RWKV state の chunked inference state carry。
- `read_screening_only` / `read_write` phase を持つ state-level screening。
- RWKV tokenizer API と JSONL-to-binidx 変換。
- Flax params と model config metadata の safetensors export/import。
- 単一プロセス用 reference training checkpoint save/load。
- binidx validation loss/perplexity CLI。
- TPU 作業向けのローカルテスト可能な distributed mesh/sharding helper。
- `from rwkv7m import ...` で使える installable package layout。

未対応:

- production fused RWKV kernels。
- pretrained RWKV checkpoint conversion。
- 完全な PyTorch/non-JAX runtime backend。
- sharded TPU checkpoint save/resume。
- 完全な distributed TPU trainer。
- long-context evaluation harnesses。

## テスト

```powershell
uv run pytest -q
```

現在の smoke coverage には、math helper、shape check、phase/config validation、scan consistency、public API inference、public API training、binidx data loading が含まれます。
