# rwkv7m

[English README](README.md)

RWKV-7風のrecurrent language modelに、任意でstate-level screening memoryを
追加したJAX/Flax NNX研究実装です。Linenは数値比較とupstream変換のreference
として維持しています。

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

## RWKV7M と RWKV-7 のアーキテクチャ差分

この repository の RWKV7M は、RWKV-7 風の recurrent core を置き換えるものではなく、その上に任意の state-level screening memory を追加して検証する研究用アーキテクチャです。`--no-screening` を指定した場合は、この実装内で最も RWKV-7 baseline に近い構成になります。ただし現時点の実装は JAX/Flax NNX研究経路であり、upstream RWKV-LM-V7 の fused CUDA kernel や既存 `.pth` checkpoint との互換性を主張するものではありません。

| 観点 | RWKV-7 | この repository の RWKV7M | 評価上の注意 |
| --- | --- | --- | --- |
| 基本構造 | token embedding、RWKV-7 block stack、final layer norm、LM head で構成される recurrent LM。 | 同じ形の RWKV-7 風 core を `ScreenedRWKVLayer` で包み、指定 layer だけに screening module を挿入できる。 | `--no-screening` を必ず baseline として走らせる。 |
| Layer 内の処理 | TimeMix/WKV recurrent update と ChannelMix FFN が主な処理。 | 各 layer はまず RWKV block を実行し、その出力に対して必要な layer だけ screening read/write を加える。 | core の差ではなく screening 追加分の寄与を分けて見る。 |
| Recurrent state | layer ごとに前回の TimeMix input、ChannelMix input、WKV matrix state を持つ。 | RWKV state に加えて、screened layer ごとに slot bank、age、usage EMA を持つ。 | screening state は token列のKV cacheではなく、固定サイズの圧縮memory。 |
| Memory の表現 | WKV state に履歴情報を圧縮する。明示的な候補slot集合は持たない。 | `n_slots` 個の固定slotに情報を蓄え、query/key similarity と閾値で読むslotを選ぶ。 | slot数やslot次元を増やすとparameter数とstate量も増える。 |
| Read mechanism | WKV recurrence の結果が hidden update に入る。 | layer input から read query、slotから read key/value を作り、unit-normalized similarity を閾値 `tau` でscreeningする。 | softmax attention ではない。全slotが不要なら総和1に正規化して無理に読む挙動を避ける設計。 |
| Relevance | 標準attentionのような候補集合softmaxではなく、RWKV recurrent update が系列情報を処理する。 | `Trim-and-Square` relevance を使い、slotごとに独立した採否強度を作る。 | 「相対的に一番ましなslot」ではなく「読む価値があるslot」を選ぶ仮説。 |
| Read 出力の合成 | RWKV block の出力がそのまま次段へ渡る。 | slot value の重み付き和を TanhNorm で bounded にし、gate と `lambda_screen` を通して RWKV block 出力へ残差的に足す。 | screeningが不安定な初期段階でもcore経路は残る。 |
| Write / slot update | WKV state はTimeMix内の recurrence で更新される。 | `read_screening_only` ではslow updaterでslotをゆっくり更新する。`read_write` では別の write query/key による write relevance も使える。 | write branch は `read_write` phase かつ `use_write_screening=True` の時だけ有効。 |
| Phase | architecture上のread/write phaseはない。 | `read_screening_only` と `read_write` を明示的に分ける。互換aliasとして `read_only` は `read_screening_only` に対応する。 | まず `read_screening_only` で安定性を見てから `read_write` を比較する。 |
| Multi-timescale bank | 標準のRWKV state更新に依存する。 | slotごとに short / mid / long bank id を持ち、bank別の最大更新率でslotの寿命を変えられる。 | 長期memory仮説の検証点。まだ有効性はbenchmarkで確認する必要がある。 |
| Parameter count | `n_layers`、`d_model`、`d_ffn`、vocab size が主なparameter規模を決める。 | screening layer ごとに read/write projection、gate、delta projection、slot embedding などが増える。 | 機構の有効性を主張するには、単純なparameter増加と分離する必要がある。 |
| Training state carry | 独立chunk学習ではstateを持ち越さない運用が一般的。 | デフォルトではstateを毎batchリセットし、streaming用途では `--sampling-mode sequential --carry-state` で明示的に持ち越す。 | `magic` sampler でcarryすると無関係chunkが混ざるため禁止している。 |
| Backend / artifacts | upstream RWKV-LM-V7 は PyTorch + CUDA kernel を主経路にする。 | この repository は JAX/Flax + TPU Research Cloud を主経路にし、外部runtime向けには `safetensors` artifact境界を用意する。 | PyTorch runtime 本体や upstream checkpoint変換はこのrepositoryの対象外。 |

### 1. 共通している RWKV-7 風 core

RWKV7M の core は、RWKV-7 風の `TimeMix` / `ChannelMix` block を積む構成です。TimeMix は layerごとの recurrent state と WKV matrix state を使い、ChannelMix は FFN 経路を担当します。token embedding、block stack、final layer norm、LM head という大枠は RWKV-7 baseline と同じです。

このため、screening の寄与を見たい場合は、同じ `d_model`、`d_ffn`、`n_layers`、`n_heads`、`head_size`、`vocab_size` で `--no-screening` を指定した run を基準にします。これは「この実装内の RWKV-7 風 baseline」です。

### 2. RWKV7M 固有の state-level screening memory

RWKV7M が追加する主な機構は、selected layer にだけ配置される固定slot型の memory です。screened layer では `LayerScreenState` が `slots`、`ages`、`usage_ema` を持ちます。slot は過去tokenをそのまま保存する KV cache ではなく、固定個数・固定次元の圧縮状態です。

Read時は layer input から query を作り、slot から key/value を作ります。query/key は unit norm 化され、similarity が閾値を超えた分だけ `Trim-and-Square` relevance として採用されます。この設計は softmax attention と違い、全候補slotが不要な場合に総和1の重みを強制しません。

### 3. Read 出力の入り方

screening module は RWKV block の代わりではありません。まず RWKV block が通常通り hidden を更新し、その後で slot read の結果を residual として加えます。slot value の合成結果は TanhNorm でboundedにされ、gate と `lambda_screen` を通ってから hidden に足されます。

この構造により、screening が学習初期に有用なslotを作れていない場合でも、core RWKV経路は残ります。一方で、screening 経路が有効なら、core stateだけでは拾いにくい固定slot memoryから追加情報を戻せる、という仮説を検証できます。

### 4. Write と phase の分離

RWKV7M には `read_screening_only` と `read_write` の2つの主要phaseがあります。

`read_screening_only` では、read relevance は使いますが、write relevance branch は使いません。slot は bank別の小さい更新率で全slotがゆっくり更新されます。厳密な read-only ではなく「screened read + uniform slow write」です。

`read_write` では、`use_write_screening=True` の場合に write専用の query/key branch が有効になります。write branch は、どのslotをどれだけ更新するかを別途screeningします。`write_rel_floor` は初期slotがゼロに近い場合の互換用設定で、学術比較では `0` を指定して完全なwrite棄却を評価できます。bank更新率は従来の `mu_*_max` に加え、`--short-half-life-tokens` / `--mid-half-life-tokens` / `--long-half-life-tokens` でtoken半減期として明示できます。

### 5. Parameter 増加との分離

screening module は追加parameterを持つため、単に「RWKV7M が RWKV-7 baseline より loss が低い」だけでは、機構の有効性を示したことになりません。parameter数が増えただけで改善した可能性が残るためです。

そのため、学術用の比較では最低限次の3本を同じdata、token budget、optimizer、dtype、global batch、samplerで比較します。

| 比較run | 目的 |
| --- | --- |
| `local_core_baseline` | screeningなしのRWKV-7風baseline。 |
| `local_mechanism` | 同じcore幅にscreeningを追加した本命run。 |
| `local_param_control` | screeningなしでFFN幅を増やし、`local_mechanism` 以上のparameter数にしたcontrol。 |

`local_mechanism` が `local_core_baseline` だけでなく `local_param_control` にも held-out validation loss / perplexity で勝つ場合、機構そのものの寄与を示す材料になります。さらに、同じlossへ到達するstep/token数や eval loss curve のAUCが小さければ、state-level screening memory が学習を速くしている材料になります。`comparison.sh` はこの比較行列、parameter count、run protocol、metrics summary、learning speed summary を出すための入口です。

指定の MiniPile と `RWKV-Vibe/RWKV-LM-V7` を自動取得して比較モデルを作る推奨入口は次です。

```bash
bash scripts/compare_minipile.sh --profile smoke --no-run
bash scripts/compare_minipile.sh --profile small --seeds "42 43 44"
```

local比較行列には `local_core_baseline`、`local_read_screening`、`local_mechanism`、screeningなしでFFNを広げた `local_param_control` が含まれます。CUDA経路では、これにupstreamのPyTorch/CUDA RWKV-LM-V7を加えて実行できます。runごとのcheckpoint、JSONL/CSV metric、parameter count、protocol、再実行commandは `out/comparison/<run-id>/` に保存されます。

upstream target の既定は `--upstream-launcher single_gpu` です。この経路は Lightning trainer と DeepSpeed distributed strategy を起動しません。一方で、固定した本家checkoutから x070 model、fused CUDA operator、fused L2Wrap cross entropy、`.bin/.idx` reader と cubic sampler、初期化、`deepspeed.ops.adam.FusedAdam` をそのままimportします。`att.w0` の2倍LR group、AdamW decay group、gradient clipping、warmup、cosine scheduleも本家規則を維持します。global batchが1 GPUのmicro batchより大きい場合は、本家のglobal sample順序を保ったgradient accumulationで再現します。成果物は `upstream_rwkv_lm_v7/` 配下の `run_config.json`、`metrics.jsonl`、`metrics.csv`、`run_summary.json`、`rwkv-init.pth`、`rwkv-final.pth` です。

downloadしたMiniPileについては、本家samplerが必要とするsingle-item binidx viewをtoken byteのcopyなしで自動生成します。また、追跡済みのAda CUDA互換patch `patches/upstream_rwkv7_ada_atomic.patch` を適用し、本家commitと実際のdiff hashをrunへ記録します。未変更の本家sourceを、それがcompileできるhardwareで意図的に試す場合だけ `APPLY_UPSTREAM_ADA_PATCH=0` を指定します。

Lightning/DeepSpeed stack自体を評価する場合、または対象machineでその構成を検証済みの場合に限り `--upstream-launcher deepspeed` を使います。従来経路は残していますが、論文baselineの既定からは外しました。

`smoke` は配線確認用であり、研究結果には使いません。`small` は約0.19B規模ですが、既定の2,000 stepsだけで論文に十分な学習量だとは限らないため、予備実験を基にtoken budgetを決めます。datasetとupstream checkoutは `.comparison/` にcacheされます。

#### NVIDIA CUDA環境での例

CUDA extraはNVIDIA driverが利用できるLinux環境を想定しています。導入済みCUDAのmajor versionに合うextraを選び、最初にJAXからGPUが見えることを確認します。

```bash
uv sync --extra cuda12
uv run --extra cuda12 python -c "import jax; print(jax.devices())"
```

local JAXの4変種には独立held-out評価を指定し、それと並行して同一token budgetのupstream PyTorch/CUDA学習参照を実行する例です。

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

上のcommandでは本家baselineにdirect single-GPU経路を既定で使います。明示する場合は `--upstream-launcher single_gpu`、従来のlauncherを再実行する場合は `--upstream-launcher deepspeed` を追加します。

local projectはPython 3.13以上を必要としますが、upstream repositoryは古い依存関係を固定しています。そのためcomparison scriptは、`uv venv`で専用Python 3.12環境 `.comparison/venvs/rwkv-lm-v7` を作り、`uv pip`でupstream依存関係を導入します。upstreamのLightning 1.9.5が現在も `pkg_resources` をimportするため、`setuptools<81` も互換制約として適用します。OS側にはPython 3.12 development header（Ubuntuでは`python3.12-dev`）、C++ build toolchain、CUDA toolkitも必要です。venvはseed間で再利用され、upstreamのrequirementsまたはこの互換制約が変わった場合は自動的に再同期されます。次の準備commandはvenvを作成しますが、学習は開始しません。

```bash
bash scripts/compare_minipile.sh --profile smoke --no-run
```

保存先を変える場合は `UPSTREAM_VENV=/path/to/venv`、`uv`に要求するPythonを指定する場合は `UPSTREAM_PYTHON_VERSION=3.12`、依存関係を強制的に再同期する場合は `INSTALL_UPSTREAM_DEPS=1` を使います。従来の `UPSTREAM_PYTHON` もvenv作成時のinterpreter指定として受理しますが、本家commandは常にvenv内のPythonで実行されます。CUDA 13環境では、両方の `uv` commandと `LOCAL_PREFIX` の `cuda12` を `cuda13` に置き換えます。本家fused extensionのbuildにはNVIDIA driverだけでなく、PyTorchのCUDA buildと互換性のあるCUDA compilerおよびdevelopment headerも必要です。Blackwellでの実機検証では、PyTorch CUDA 13.0にCUDA 13.0 compiler/development librariesを組み合わせました。

`--verify-core-parity` は、学習曲線を解釈する前に実装同等性を別途検証します。小型で決定的な公式x070 modelを生成し、固定16-token系列を本家の実fused BF16 CUDA forward/backward経路へ通して、公式FusedAdamを1 step実行します。全公式tensor、logits、gradient、更新後weightを出力し、screeningなしのFlax treeへ変換して、ゼロrecurrent stateから同じloss・gradient・optimizer規則を検証します。機械可読な結果は `upstream_core_parity/parity_report.json` に保存され、tensor coverage、許容誤差、cosine similarity、relative L2のいずれかが基準を外れると比較command全体が失敗します。

この検証が確認するのは、記録されたupstream commitとdiffに対する「固定weight・ゼロ初期state・系列forward/backward」とBF16 optimizer 1 stepの互換性です。複数step・複数seedのtraining dynamicsまで同一だと証明するものではありません。captureには、本家fused operatorを曖昧なくbindingする公式JIT経路を使います。実機検証したupstream commitでは、名前衝突を機械的に解消したnon-JIT wrapperとreference logitsが完全一致しました。reference captureにはupstream用Python環境、PyTorch/CUDA、本家extensionをbuildできるcompilerが必要です。既定ではparity確認だけのために巨大checkpointを複製しないよう小型modelを使います。特定の公式 `.pth` を検査する場合は、`scripts/capture_upstream_rwkv7_reference.py --checkpoint ...` を別途使用できます。実測値と制約は [NVIDIA Blackwell実機検証](docs/gpu_blackwell_validation.md) と [NVIDIA L40S実機検証](docs/gpu_l40s_validation.md) に記録しています。

parityの2段階だけを直接再実行する例です。

```bash
.comparison/venvs/rwkv-lm-v7/bin/python scripts/capture_upstream_rwkv7_reference.py \
  --upstream-repo .comparison/RWKV-LM-V7 \
  --output out/upstream-core-parity/reference.npz \
  --optimizer-step

JAX_PLATFORMS=cuda uv run --extra cuda12 rwkv7m-verify-upstream-rwkv7 \
  out/upstream-core-parity/reference.npz \
  --json-out out/upstream-core-parity/report.json
```

#### Google TPU環境での例

TPU VMではTPU extraを導入し、deviceとprocessの認識を確認します。

```bash
uv sync --extra tpu
uv run --extra tpu python -c "import jax; print(jax.devices()); print(jax.process_count(), jax.process_index())"
```

単一TPU hostでlocal比較行列を動かす例です。comparison scriptのTPU向けdevice数の既定値は安全側の1なので、検出したlocal device数を明示します。

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

`upstream_rwkv_lm_v7` はCUDA/PyTorch専用であり、このscriptからTPU上では実行できません。backendをまたぐ研究では、TPUでlocal比較行列を、NVIDIA CUDAでupstream targetを別runとして実行します。異なるaccelerator間のthroughputはhardware条件を揃えた比較ではないため、性能値とは分けて報告します。multi-host TPUでは全hostにJAX coordinator環境変数を設定し、[TPU Research Cloud Training](docs/tpu_research_cloud.md) のdistributed CLI手順を使います。上のwrapper例は単一host向けです。

#### 論文向けrunの条件と成果物

研究結果を取る場合は、固定した独立binidxを必ず `--eval-data-file` で指定します。省略時はtrain datasetをevalにも使うため、配線確認にしか使えません。local変種間ではdataset split、tokenizer、token budget、optimizer、dtype、context length、global batch、sampler、評価間隔、評価token数を固定します。

各seedの実行後は次を確認します。

- `parameter_counts.csv`: baseline、mechanism、parameter controlのparameter数。
- `local_metric_summary.csv`: 最終train/eval metric。
- `learning_speed_summary.csv`: best/final eval loss、perplexity、tokenに対するeval-loss AUC、throughput。
- `learning_target_hits.csv`: 共通loss目標への到達step/token数。
- `parameters.env`、`protocol.md`、`commands.sh`: 実行条件、source commit、再現用command。
- `upstream_core_parity/parity_report.json`: 本家CUDAとlocal JAXの固定weight core parity。

`--seeds` はseedごとのrun directoryを作りますが、現行scriptはseed横断集計を行いません。論文へ載せる前に各runのCSVを結合し、seed数と、標準偏差や信頼区間などのばらつき・不確実性を報告します。またupstream runはlocal held-out CSV summaryには含まれず、FLOPs-matched controlとtask-specific long-context benchmarkも未実装です。生成された表からそれらの比較結果まで主張しないようにします。

### 6. 互換性と主張範囲

この repository は RWKV-LM-V7 compatible な `.bin/.idx` dataset reader と RWKV tokenizer vocabulary を持ちますが、upstream RWKV-7 checkpoint互換や PyTorch runtime互換をまだ主張しません。学習主経路は JAX/Flax、TPU Research Cloud を意識した distributed training、外部runtimeへ渡すartifact境界は `safetensors` です。

したがって、現時点で主張できるのは「RWKV-7 風 recurrent core に state-level screening memory を追加する研究実装があり、その有効性をparameter control付きで検証できる」という範囲です。性能改善や長期記憶の有効性は、validation / benchmark / ablation の結果で示す必要があります。

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

`train_batch` はデフォルトで recurrent state をリセットします。これは、独立にサンプリングされた training chunk で学習する通常の使い方に合わせた挙動です。連続した streaming/stateful training を意図する場合だけ `carry_state=True` を指定してください。binidx 学習と評価で state を持ち越す場合は、疑似シャッフルされた `magic` sampler ではなく `sampling_mode="sequential"` / `--sampling-mode sequential` と組み合わせます。`sequential` では batch row ごとに stream lane を割り当て、lane が末尾から先頭へ wrap する境界で carried state を自動リセットします。

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
| `--carry-state` | chunk間で RWKV state / screening state を持ち越します。長期stream訓練/評価用です。`--sampling-mode sequential` とセットで使います。 |
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

`carry_state` は、長文や章単位のstreamを学習・評価させるための入口です。ただし、疑似シャッフルされたchunkへstateを持ち越すと別文脈が混ざるため、CLIでは `--carry-state` と `--sampling-mode sequential` を同時に指定する必要があります。sequential stream lane が wrap する境界では state を自動リセットします。checkpointには `runtime_state.msgpack` も保存され、単一プロセスCLIではresume時に carried state を復元します。

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

長期stream訓練を試す場合は、最初から次のように `sequential` sampling と `carry-state` を使います。`batch-size=1` ならstep間で隣接chunkをそのまま読みます。`batch-size>1` ではbatch rowごとに別stream laneを割り当てます。各 lane が末尾から先頭へ戻る時は state がリセットされるため、stream 終端と先頭の文脈は混ざりません。

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

stateful stream として validation する場合は、training と同じく `sequential` sampling と `carry-state` を使います。

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
| `--carry-state` | 分散train/eval step間でstateを持ち越します。validation は評価用 state を別に進め、lane wrap 境界では state をリセットします。分散ローカルrunでは runtime state も checkpoint/resume されます。実TPU podでのsharded runtime-state checkpoint/resume検証はまだ未完了です。 |
| `--checkpoint-backend flax` | process 0 が `train_state.msgpack` と `model.safetensors` を書きます。小さいローカルrunやartifact export確認向けです。 |
| `--checkpoint-backend orbax` | 全processで Orbax train-state checkpoint を書きます。TPU-scale run の主経路です。 |
| `--keep-last-checkpoints` | 古いcheckpointを残す数です。best checkpoint はrotationから保護されます。 |
| `--eval-every`, `--eval-steps` | periodic validation の頻度と評価step数です。 |
| `--save-best-checkpoint` | validation metric が改善した時にcheckpointを保存します。 |
| `--best-metric`, `--best-mode` | best判定に使うmetricと方向です。通常は `loss` / `min` です。 |
| `--summary-every` | `run_summary.json` を更新する間隔です。 |
| `--prefetch-size` | deviceへ先読みするbatch数です。 |
| `--mesh-axis-names`, `--mesh-axis-sizes` | multi-axis mesh 実験用です。 |
| `--param-axis-name` | NNX Explicit model parallelを有効にするmesh axis名です。 |

出力先には次の成果物ができます:

- `run_config.json`: 実行時のCLI args、model config、process/device情報、parameter partition summary。
- `run_summary.json`: 現在step、完了step、tokens、最新checkpoint、最後のtrain/eval metric、best eval。
- `best_eval.json`: best validation metric と対応checkpoint。
- `metrics.jsonl`: train/eval metric のJSONL log。
- `metrics.csv`: 表計算や簡易確認用のCSV log。
- `ckpt-*`: checkpoint directory。

これは TPU Research Cloud 対応に向けた、ローカルテスト可能な最初の層です。分散ローカルrunでは carry-state runtime state も checkpoint/resume 対象です。実 TPU pod 上での完全な sharded checkpoint policy 検証はまだ未実施です。

distributed run artifact はローカルで監査できます:

```powershell
uv run rwkv7m-audit-dp-run out/minipile-dp --require-complete --min-train-records 10
```

TPU setup と実行メモは [docs/tpu_research_cloud.md](docs/tpu_research_cloud.md) にあります。

### 7B見積りとNNX scale経路

runtime、inference、training、distributed train/eval、optimizer、checkpoint
lifecycleはFlax NNXへ移行しました。旧Linen modelは数値検証とupstream互換性確認の
referenceとしてのみ残します。全parameter pathを厳密に照合するconverterを設け、
小型full modelでread-only/read-write両方のforward、RWKV/screening state、統計、
gradient parityを検証します。NNX lifecycleではtopology-aware init、Optax update、
Orbax save/restore、restore後step、logical sharding metadata維持までを対象にします。

tracked 7B候補のtensorを実体化せず、parameter関連memoryを見積もるには:

```powershell
uv run rwkv7m-plan-scale `
  --model-config configs/rwkv7m-7b-tpu.json.example `
  --model-axis-size 8 `
  --dtype-profile memory
```

この値にはactivation、RWKV/screening runtime state、compiler一時領域、collective
bufferを含みません。実行前のgateには使えますが、HBMへ収まる保証値ではありません。

NNX lifecycleを別processで検証するには:

```powershell
uv run rwkv7m-verify-nnx-lifecycle --checkpoint-dir out/nnx-probe --mode create
uv run rwkv7m-verify-nnx-lifecycle --checkpoint-dir out/nnx-probe --mode restore
```

distributed CLIもNNX-nativeとなり、TPU-scale checkpointにはOrbaxを使います。
topology-awareな`jax.make_mesh()`を使い、timing window終了時に更新済みNNX
model/optimizer state全体の完了を待ちます。独立した`--param-axis-name`を指定した
場合、Phase 3経路はExplicit mesh、row/column `dot_general`の明示的出力配置、
hidden activationとRWKV head、screening slotのmodel shardingを使用します。
post-SPMD executable HLOのcollective監査とoptimizer 1 stepは次で実行できます。

```powershell
$env:JAX_NUM_CPU_DEVICES="2"
uv run rwkv7m-audit-nnx-model-parallel `
  --model-axis-size 2 `
  --screening `
  --write-screening
```

強制2 CPU実行はlocalで再現可能な契約試験です。2026-07-14には実TPU v5e
（`v5litepod-4`、4 devices）でも、`data=1, model=4` meshとread/write
screeningを有効にした小型の完全RWKV7M構成を検証しました。finiteな
forward/backwardとoptimizer 1 stepが完了し、WKV stateは
`P("data", "model", None, None)`、screening slotは
`P("data", None, "model")`を維持しました。compiled collectiveは16個で、
all-reduce 13、all-to-all 3、all-gather 0でした。

同じTPU sliceでは、独立した小型NNX lifecycle probeについても、別processでの
Orbax create/restoreと、restore後のoptimizer step 1から2への進行を確認しました。
これはframework lifecycle probeの検証であり、RWKV7M本体のdistributed
train-stateやrecurrent/screening runtime-state checkpoint経路の検証では
ありません。7B実行、multi-host、3個のall-to-allに対するXProf tuning、
production throughput、failure recoveryは未検証です。

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
    NNXScreenedRWKVModel,
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

- Flax NNX runtime/training実装。Linenは数値比較とupstream変換のreferenceとして維持。
- recurrent component 内の `jax.lax.scan` による full-sequence training path。
- RWKV-LM-V7 互換 `.bin/.idx` dataset reader と sampler。
- reference RWKV state の chunked inference state carry。
- `read_screening_only` / `read_write` phase を持つ state-level screening。
- lane wrap reset 付きの sequential carry-state binidx training / validation。
- RWKV tokenizer API と JSONL-to-binidx 変換。
- wheel に同梱される RWKV tokenizer vocabulary fallback。
- Flax params と model config metadata の safetensors export/import。
- architecture、dtype、model dimensions、screening summary、tokenizer vocabulary identity を含む safetensors artifact metadata。
- 外部 runtime project 向けの PyTorch-readable safetensors loading helper。
- 単一プロセス用 reference training checkpoint save/load。
- binidx validation loss/perplexity CLI。
- TPU 作業向けのローカルテスト可能な distributed mesh/sharding helper。
- process-aware Flax checkpoint、Orbax train-state checkpoint、carry-state runtime checkpoint/resume、best-eval protection 付き checkpoint rotation、structured JSONL/CSV logs、run summary、validation hook、run artifact audit CLI、device prefetching、optional NNX Explicit data/model mesh経路を持つ data-parallel distributed binidx training CLI。
- `from rwkv7m import ...` で使える installable package layout。

未対応:

- production fused RWKV kernels。
- pretrained RWKV checkpoint conversion。
- repository 内 PyTorch/non-JAX runtime backend（意図的に対象外）。
- 完全に調整された per-parameter TPU sharding rules。
- 実TPU pod上でのRWKV7M本体train-stateおよびrecurrent/screening runtime-stateの
  Orbax checkpoint save/resume検証。
- 実 TPU pod 上で検証済みの production-scale distributed TPU trainer。
- 7B・multi-host TPU実行、XProfによるcollective tuning、failure recovery drill。
- stateful binidx validation を超える task-specific long-context evaluation harnesses。

## テスト

```powershell
uv run pytest -q
```

現在の smoke coverage には、math helper、shape check、phase/config validation、scan consistency、NNX public inference/training、binidx data loading、sequential carry-state reset/eval behavior、safetensors/checkpoint boundary、local distributed training boundary、full-model Linen/NNX forward・gradient parity、screening algebra/gradient parity、7B abstract memory planning、NNX Orbax lifecycleが含まれます。現時点の full suite は126 testsです。
