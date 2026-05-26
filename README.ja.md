# rwkv7m

[English README](README.md)

RWKV-7 風の recurrent language model に、任意で state-level screening memory を追加した JAX/Flax リファレンス実装です。

このリポジトリは研究用途を主目的にしています。現時点では custom kernel や既存 checkpoint 互換性よりも、正しさ、テスト、API の使いやすさを優先しています。

## この README の読み方

最初に試す場合は、次の順番で進めるのが安全です。

1. `uv sync` で環境を作り、`uv run pytest -q` でローカル環境の基本動作を確認する。
2. `.bin/.idx` dataset を `data/` に置くか、JSONL text から `rwkv7m-make-binidx` で作る。
3. まず `rwkv7m-train-binidx` で小さい smoke run を回す。
4. checkpoint、resume、validation が必要になったら `--output-dir`、`--save-every`、`--eval-every` を指定する。
5. TPU/分散学習に寄せる場合は `rwkv7m-train-binidx-dp` を使い、最後に `rwkv7m-audit-dp-run` で run artifact を監査する。

この repository の学習実装は JAX/Flax が主軸です。PyTorch runtime 本体はここでは実装しません。外部 runtime へ渡すための artifact 境界として、`safetensors` export と PyTorch-readable loading helper を提供します。

## 学習コマンドの選び方

| 目的 | 使うコマンド | 説明 |
| --- | --- | --- |
| まず学習が動くか確認する | `rwkv7m-train-binidx` | 単一プロセス向けの参照学習CLIです。小さいrun、debug、checkpointの基本確認に使います。 |
| TPU/分散学習に近い形で動かす | `rwkv7m-train-binidx-dp` | JAX distributed、mesh、device sharding、Orbax checkpoint、metrics、run summary を使うCLIです。実Pod投入前の主経路です。 |
| validation loss/perplexity を測る | `rwkv7m-eval-binidx` | `.bin/.idx` dataset と checkpoint から評価します。 |
| screening 設定を比較する | `rwkv7m-bench-binidx` | baseline / screening / read-write の簡易比較に使います。 |
| JSONL text を dataset 化する | `rwkv7m-make-binidx` | repository 同梱の RWKV tokenizer vocabulary で `.bin/.idx` を作ります。 |
| 分散runの成果物を検査する | `rwkv7m-audit-dp-run` | `run_summary.json`、metrics、checkpoint参照、best checkpoint などの整合性を確認します。 |

## ファイルとディレクトリの扱い

- `data/`: dataset を置く場所です。`.bin` / `.idx` は大きくなりやすいので git には含めません。
- `out/`: checkpoint、metrics、run summary などの学習成果物を置く場所です。git には含めません。
- `configs/*.json.example`: 共有したい設定例です。repository に含めるJSON例は `.json.example` にします。
- `*.json`: local config や学習成果物として扱い、git には含めません。必要なら `.json.example` を元に自分用の `.json` を作って使います。
- `sample/`: ignore されています。runtime dependency や README の前提にはしません。

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

TPU VM で使う場合は TPU extra を入れます。実Podなしのローカル検証では通常の `uv sync` で十分です。

```powershell
uv sync --extra tpu
uv run python -c "import jax; print(jax.devices()); print(jax.process_count(), jax.process_index())"
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

`train_batch` はデフォルトで recurrent state をリセットします。これは、独立にサンプリングされた training chunk で学習する通常の使い方に合わせた挙動です。連続した streaming/stateful training を意図する場合だけ `carry_state=True` を指定してください。binidx 学習で state を持ち越す場合は、疑似シャッフルされた `magic` sampler ではなく `sampling_mode="sequential"` / `--sampling-mode sequential` と組み合わせます。

## 設定項目の考え方

CLI の設定は、まず小さい値で smoke run を通し、checkpoint/resume/eval が安定してから大きくします。

| 項目 | 目安 |
| --- | --- |
| `--data-file` | `.bin` / `.idx` を除いた dataset prefix です。`data/minipile.bin` と `data/minipile.idx` なら `data/minipile` を指定します。 |
| `--ctx-len` | 1 sample の token 長です。メモリ使用量と計算量に強く効きます。まずは `128` や `512` で確認します。 |
| `--batch-size` | `rwkv7m-train-binidx` 用のbatch sizeです。単一プロセスの参照学習で使います。 |
| `--global-batch-size` | `rwkv7m-train-binidx-dp` 用の全体batch sizeです。`process_count` と local device 数で割り切れる値にします。 |
| `--steps` | 実行する optimizer update 数です。resume 時も「checkpointから追加で何step回すか」を意味します。 |
| `--sampling-mode` | `magic` は従来の疑似シャッフルsampling、`sequential` はstream laneごとに連続chunkを読むmodeです。 |
| `--carry-state` | chunk間で RWKV state / screening state を持ち越します。長期stream訓練用です。`--sampling-mode sequential` とセットで使います。 |
| `--vocab-size` | tokenizer/dataset の語彙サイズです。RWKV vocab の binidx では通常 `65536` です。 |
| `--d-model`, `--d-ffn`, `--n-layers` | model size を決める主要項目です。大きくするとメモリとcompile時間が増えます。 |
| `--n-heads`, `--head-size` | recurrent block のhead構成です。`d_model` と整合する小さい構成から始めます。 |
| `--dtype` | `float32` または `bfloat16` です。TPUを意識するrunでは `bfloat16` を検討します。 |
| `--phase` | `read_screening_only` または `read_write` です。まずは安定確認に `read_screening_only` を使い、screening memory の書き込みを試す時に `read_write` を使います。 |
| `--no-screening` | screening memory を使わないbaselineを回す時に指定します。 |

config file では CLI option の destination 名をJSON keyにします。たとえば `--ctx-len` は `ctx_len`、`--save-every` は `save_every` です。明示したCLI optionはconfigの値を上書きします。

```powershell
uv run rwkv7m-train-binidx --config configs/minipile-smoke.json.example --steps 2000
```

`carry_state` は、長文や章単位のstreamを学習させるための入口です。ただし、疑似シャッフルされたchunkへstateを持ち越すと別文脈が混ざるため、CLIでは `--carry-state` と `--sampling-mode sequential` を同時に指定する必要があります。checkpointには `runtime_state.msgpack` も保存され、単一プロセスCLIではresume時に carried state を復元します。

## RWKV-LM-V7 `.bin/.idx` からの学習

`rwkv7m` は RWKV-LM-V7 と同じ binidx dataset 形式を読み取れます。
tokenized Minipile を `data/` にダウンロードし、`.bin` / `.idx` を除いた prefix path を指定します。

```powershell
New-Item -ItemType Directory -Force data
wget -O data/minipile.idx https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx
wget -O data/minipile.bin https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin
```

### 1. 小さい smoke run

まず checkpoint なしで短く回し、dataset path、model shape、JAX環境に問題がないことを確認します。

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

### 2. checkpoint と validation 付きのrun

次に `--output-dir` を指定して、checkpoint と validation を有効にします。`--save-every 100` は100stepごとに checkpoint を保存し、`--eval-every 100` は100stepごとに validation を走らせます。

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

checkpoint directory は `out/minipile-smoke/ckpt-00000100` のような名前になります。各 checkpoint には `checkpoint.json`、`train_state.msgpack`、外部runtime向けの `model.safetensors` が入ります。

同じ CLI は JSON config file も受け取れます。明示した CLI option は config の値を上書きします。repository に含める例は `.json.example` にし、生成物やlocal用の `.json` は ignore します:

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

### 3. checkpoint から再開する

resume 時の `--steps` は、checkpoint から追加で実行する optimizer update 数です。下の例は `ckpt-00001000` からさらに1000step進めます。

```powershell
uv run rwkv7m-train-binidx `
  --data-file data/minipile `
  --ctx-len 512 `
  --batch-size 1 `
  --steps 1000 `
  --resume out/minipile-smoke/ckpt-00001000 `
  --output-dir out/minipile-smoke
```

長期stream訓練を試す場合は、最初から次のように `sequential` sampling と `carry-state` を使います。`batch-size=1` ならstep間で隣接chunkをそのまま読みます。`batch-size>1` ではbatch rowごとに別stream laneを割り当てます。

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

外部 runtime 向けの標準交換形式は `safetensors` とします。`.pth` export は downstream runtime project で必要になった場合だけ追加します。

外部 runtime artifact 境界:

```python
from rwkv7m.backends.torch import load_torch_safetensors

state_dict, config, metadata = load_torch_safetensors("out/model.safetensors")
```

これは artifact 互換性確認用 helper です。この repository では PyTorch/non-JAX RWKV7M runtime は実装対象外で、別 runtime project 側で扱います。

Reference training checkpoint:

```python
from rwkv7m import load_train_checkpoint, save_train_checkpoint

save_train_checkpoint("out/ckpt-000001", state, runtime.config)
state, config, metadata = load_train_checkpoint("out/ckpt-000001", state)
```

## TPU / 分散学習に寄せる場合

実TPU Pod がない環境でも、`rwkv7m-train-binidx-dp` で分散学習用の境界をローカル検証できます。この経路は、通常の参照CLIよりも TPU Research Cloud を意識した構成です。

このCLIが扱う主なもの:

- `jax.distributed.initialize()` による process 初期化。
- JAX mesh と sharding。
- host-aware な `.bin/.idx` sampling。
- device への batch 配置と prefetch。
- Orbax train-state checkpoint。
- JSONL/CSV metrics、`run_config.json`、`run_summary.json`、`best_eval.json`。
- best validation checkpoint の保存と rotation 保護。

低レベルAPIでbatch配置を確認する場合:

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

ローカルテスト可能な data-parallel train step 境界は次です:

```python
from rwkv7m.distributed import train_batch_data_parallel

dist, metrics = train_batch_data_parallel(dist, dataset.get_batch(0), dataset.layout)
```

### 分散/TPU向けCLIの例

ローカルでは `global_batch_size` を小さくして動作確認できます。TPU Pod に載せる時は、`global_batch_size` が process 数と各hostのdevice数で割り切れるようにします。

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

よく使う分散CLI option:

| option | 説明 |
| --- | --- |
| `--global-batch-size` | 全process合計のbatch sizeです。分散runでは `--batch-size` ではなくこちらを使います。 |
| `--sampling-mode` | `magic` または `sequential` です。`--carry-state` を使う場合は `sequential` が必要です。 |
| `--carry-state` | 分散train step間でstateを持ち越します。ローカル検証用の入口で、実TPU podでのsharded runtime-state checkpoint/resume検証はまだ未完了です。 |
| `--checkpoint-backend flax` | process 0 が `train_state.msgpack` と `model.safetensors` を書きます。小さいローカルrunやartifact export確認向けです。 |
| `--checkpoint-backend orbax` | 全processで Orbax train-state checkpoint を書きます。TPU-scale run の主経路です。 |
| `--keep-last-checkpoints` | 古いcheckpointを残す数です。best checkpoint はrotationから保護されます。 |
| `--eval-every`, `--eval-steps` | periodic validation の頻度と評価step数です。 |
| `--save-best-checkpoint` | validation metric が改善した時にcheckpointを保存します。 |
| `--best-metric`, `--best-mode` | best判定に使うmetricと方向です。通常は `loss` / `min` です。 |
| `--summary-every` | `run_summary.json` を更新する間隔です。 |
| `--prefetch-size` | deviceへ先読みするbatch数です。 |
| `--mesh-axis-names`, `--mesh-axis-sizes` | multi-axis mesh 実験用です。 |
| `--param-axis-name` | rule-based parameter placement を有効にするmesh axis名です。 |

出力先には次の成果物ができます:

- `run_config.json`: 実行時のCLI args、model config、process/device情報、parameter partition summary。
- `run_summary.json`: 現在step、完了step、tokens、最新checkpoint、最後のtrain/eval metric、best eval。
- `best_eval.json`: best validation metric と対応checkpoint。
- `metrics.jsonl`: train/eval metric のJSONL log。
- `metrics.csv`: 表計算や簡易確認用のCSV log。
- `ckpt-*`: checkpoint directory。

これは TPU Research Cloud 対応に向けた、ローカルテスト可能な最初の層です。実 TPU pod 上での完全な sharded checkpoint policy 検証はまだ未実施です。

distributed run artifact はローカルで監査できます:

```powershell
uv run rwkv7m-audit-dp-run out/minipile-dp --require-complete --min-train-records 10
```

TPU setup と実行メモは [docs/tpu_research_cloud.md](docs/tpu_research_cloud.md) にあります。

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
- wheel に同梱される RWKV tokenizer vocabulary fallback。
- Flax params と model config metadata の safetensors export/import。
- architecture、dtype、model dimensions、screening summary、tokenizer vocabulary identity を含む safetensors artifact metadata。
- 外部 runtime project 向けの PyTorch-readable safetensors loading helper。
- 単一プロセス用 reference training checkpoint save/load。
- binidx validation loss/perplexity CLI。
- TPU 作業向けのローカルテスト可能な distributed mesh/sharding helper。
- process-aware Flax checkpoint、Orbax train-state checkpoint、best-eval protection 付き checkpoint rotation、structured JSONL/CSV logs、run summary、validation hook、run artifact audit CLI、device prefetching、optional multi-axis mesh / rule-based parameter placement hook を持つ data-parallel distributed binidx training CLI。
- `from rwkv7m import ...` で使える installable package layout。

未対応:

- production fused RWKV kernels。
- pretrained RWKV checkpoint conversion。
- repository 内 PyTorch/non-JAX runtime backend（意図的に対象外）。
- 完全に調整された per-parameter TPU sharding rules。
- 実 TPU pod 上での Orbax sharded optimizer/parameter checkpoint save/resume 検証。
- 実 TPU pod 上で検証済みの production-scale distributed TPU trainer。
- long-context evaluation harnesses。

## テスト

```powershell
uv run pytest -q
```

現在の smoke coverage には、math helper、shape check、phase/config validation、scan consistency、public API inference、public API training、binidx data loading、safetensors/checkpoint boundary、local distributed training boundary が含まれます。
