#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./comparison.sh [options]

Fetch RWKV-Vibe/RWKV-LM-V7 from GitHub and run a comparison matrix designed to
separate architectural/mechanism gains from trainable-parameter-count gains.

Default matrix:
  1. upstream_rwkv_lm_v7       CUDA PyTorch RWKV-LM-V7 baseline
  2. local_core_baseline       this repo, RWKV core only, no screening mechanism
  3. local_read_screening      this repo, screened reads with uniform slow writes
  4. local_mechanism           this repo, same core params, selected screening mechanism phase
  5. local_param_control       this repo, no mechanism, FFN widened until params >= mechanism

Default data:
  If --data-file is omitted, the script downloads BlinkDL/minipile-tokenized
  rwkv_vocab_v20230424/minipile.bin and minipile.idx from Hugging Face into
  .comparison/data/rwkv_vocab_v20230424/.

Default model:
  A roughly 2B-parameter RWKV-7-sized setting: L24-D2560, head_size=64,
  dim_ffn=8960, vocab_size=65536.

Options:
  --backend cuda|tpu          Local JAX backend assumption. Default: cuda.
  --run-targets MODE          both, local, upstream, or commands.
                              Default: both for cuda, local for tpu.
  --data-file PREFIX          Train dataset prefix without .bin/.idx.
  --eval-data-file PREFIX     Held-out eval dataset prefix. If omitted, train data is used.
  --eval-every N              Periodic eval interval for local variants.
                              Default: max(1, steps / 10).
  --eval-steps N              Eval steps for local variants. Default: 10.
  --loss-targets CSV          Optional explicit eval-loss targets for time-to-loss metrics.
  --ctx-len N                 Context length. Default: 512.
  --steps N                   Local train steps and upstream token budget. Default: 20.
  --seed N                    Random seed passed to both implementations. Default: 42.
  --per-device-batch-size N   Per accelerator batch size. Default: 1.
  --global-batch-size N       Override global batch size.
  --devices N                 Devices per node/host. Default: nvidia-smi count for cuda, else 1.
  --num-nodes N               Node count. Default: 1.
  --n-layer N                 Layer count. Default: 24.
  --d-model N                 Embedding width / n_embd. Default: 2560.
  --head-size N               RWKV head size. Default: 64.
  --d-ffn N                   Base FFN width. Default: upstream RWKV-LM-V7 formula.
  --vocab-size N              Vocabulary size. Default: 65536.
  --screened-layers "IDS"     Space-separated screened layer ids. Default: middle layer.
  --mechanism-phase PHASE     read_screening_only or read_write. Default: read_write.
  --no-read-screening-run     Skip the separate read_screening_only screening run.
  --rwkv-lm-v7-ref REF        Git ref for RWKV-LM-V7. Default: main.
  --rwkv-lm-v7-repo PATH      Use an existing checkout instead of cloning/fetching.
  --output-root DIR           Output root. Default: out/comparison.
  --download-data             Download default Minipile binidx if missing. Default.
  --no-download-data          Do not download default Minipile binidx.
  --no-param-control          Skip the widened no-mechanism control.
  --verify-core-parity       Run official CUDA vs local JAX fixed-weight core parity.
  --no-run                    Only fetch, validate, and write commands.
  --help                      Show this help.

Common environment overrides:
  RWKV_LM_V7_URL, RWKV_LM_V7_REF, COMPARE_ROOT, OUTPUT_ROOT, RUN_ID
  PRECISION=bf16|fp32
  LR_INIT, LR_FINAL, WARMUP_STEPS, ADAM_BETA1, ADAM_BETA2, ADAM_EPS
  WEIGHT_DECAY, GRAD_CLIP, VOCAB_SIZE, MAGIC_PRIME
  LOSS_TARGETS="3.2,3.0"
  MINIPILE_BIN_URL, MINIPILE_IDX_URL
  PYTHON_BIN=python3
  D_SLOT, D_K, D_V, N_SLOTS, WRITE_REL_FLOOR
  SHORT_HALF_LIFE_TOKENS, MID_HALF_LIFE_TOKENS, LONG_HALF_LIFE_TOKENS
  LOCAL_PREFIX="uv run"
  UPSTREAM_VENV=.comparison/venvs/rwkv-lm-v7
  UPSTREAM_PYTHON_VERSION=3.12
  INSTALL_UPSTREAM_DEPS=1  # force dependency resync; initial install is automatic
  CORE_PARITY_ATOL=0.08 CORE_PARITY_RTOL=0.08 CORE_PARITY_TOKENS=16
USAGE
}

die() {
  echo "comparison.sh: $*" >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

RWKV_LM_V7_URL="${RWKV_LM_V7_URL:-https://github.com/RWKV-Vibe/RWKV-LM-V7.git}"
RWKV_LM_V7_REF="${RWKV_LM_V7_REF:-main}"
COMPARE_ROOT="${COMPARE_ROOT:-$script_dir/.comparison}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$COMPARE_ROOT/RWKV-LM-V7}"
UPSTREAM_VENV="${UPSTREAM_VENV:-$COMPARE_ROOT/venvs/rwkv-lm-v7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-out/comparison}"
MINIPILE_DATA_DIR="${MINIPILE_DATA_DIR:-$COMPARE_ROOT/data/rwkv_vocab_v20230424}"
MINIPILE_PREFIX="${MINIPILE_PREFIX:-$MINIPILE_DATA_DIR/minipile}"
MINIPILE_IDX_URL="${MINIPILE_IDX_URL:-https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.idx}"
MINIPILE_BIN_URL="${MINIPILE_BIN_URL:-https://huggingface.co/datasets/BlinkDL/minipile-tokenized/resolve/main/rwkv_vocab_v20230424/minipile.bin}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BACKEND="${BACKEND:-cuda}"
RUN_TARGETS="${RUN_TARGETS:-}"
FETCH_UPSTREAM="${FETCH_UPSTREAM:-1}"
RUN_PARAM_CONTROL="${RUN_PARAM_CONTROL:-1}"
RUN_READ_SCREENING="${RUN_READ_SCREENING:-1}"
RUN_CORE_PARITY="${RUN_CORE_PARITY:-0}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"

DATA_FILE="${DATA_FILE:-$MINIPILE_PREFIX}"
EVAL_DATA_FILE="${EVAL_DATA_FILE:-}"
EVAL_EVERY="${EVAL_EVERY:-}"
EVAL_STEPS="${EVAL_STEPS:-10}"
LOSS_TARGETS="${LOSS_TARGETS:-}"
CTX_LEN="${CTX_LEN:-512}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
NUM_NODES="${NUM_NODES:-1}"
DEVICES="${DEVICES:-}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-}"
VOCAB_SIZE="${VOCAB_SIZE:-65536}"
N_LAYER="${N_LAYER:-24}"
D_MODEL="${D_MODEL:-2560}"
HEAD_SIZE="${HEAD_SIZE:-64}"
D_FFN="${D_FFN:-}"
SCREENED_LAYERS="${SCREENED_LAYERS:-}"
MECHANISM_PHASE="${MECHANISM_PHASE:-read_write}"
D_SLOT="${D_SLOT:-256}"
D_K="${D_K:-64}"
D_V="${D_V:-128}"
N_SLOTS="${N_SLOTS:-16}"
WRITE_REL_FLOOR="${WRITE_REL_FLOOR:-0}"
SHORT_HALF_LIFE_TOKENS="${SHORT_HALF_LIFE_TOKENS:-64}"
MID_HALF_LIFE_TOKENS="${MID_HALF_LIFE_TOKENS:-512}"
LONG_HALF_LIFE_TOKENS="${LONG_HALF_LIFE_TOKENS:-4096}"
PRECISION="${PRECISION:-bf16}"
LR_INIT="${LR_INIT:-1e-3}"
LR_FINAL="${LR_FINAL:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
ADAM_EPS="${ADAM_EPS:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAGIC_PRIME="${MAGIC_PRIME:-}"

LOCAL_PREFIX="${LOCAL_PREFIX:-uv run}"
LOCAL_LR_SCHEDULE="${LOCAL_LR_SCHEDULE:-rwkv}"
LOCAL_CHECKPOINT_BACKEND="${LOCAL_CHECKPOINT_BACKEND:-orbax}"
LOCAL_PREFETCH_SIZE="${LOCAL_PREFETCH_SIZE:-2}"
LOCAL_PARAM_AXIS_NAME="${LOCAL_PARAM_AXIS_NAME:-}"
CORE_PARITY_ATOL="${CORE_PARITY_ATOL:-0.08}"
CORE_PARITY_RTOL="${CORE_PARITY_RTOL:-0.08}"
CORE_PARITY_TOKENS="${CORE_PARITY_TOKENS:-16}"
# UPSTREAM_PYTHON previously selected the runtime directly. Keep accepting it as
# the uv interpreter request, but always execute upstream inside UPSTREAM_VENV.
UPSTREAM_PYTHON_REQUEST="${UPSTREAM_PYTHON_VERSION:-${UPSTREAM_PYTHON:-3.12}}"
UPSTREAM_PYTHON=""
UPSTREAM_VENV_BIN=""
PYTHON_BIN="${PYTHON_BIN:-}"
UPSTREAM_STRATEGY="${UPSTREAM_STRATEGY:-deepspeed_stage_2}"
UPSTREAM_GRAD_CP="${UPSTREAM_GRAD_CP:-0}"
UPSTREAM_HEAD_CHUNK="${UPSTREAM_HEAD_CHUNK:-0}"
UPSTREAM_KERNEL="${UPSTREAM_KERNEL:-@rwkv3}"
UPSTREAM_MODEL_TYPE="${UPSTREAM_MODEL_TYPE:-x070}"
UPSTREAM_TRAIN_STAGE="${UPSTREAM_TRAIN_STAGE:-3}"
UPSTREAM_DS_BUCKET_MB="${UPSTREAM_DS_BUCKET_MB:-200}"
INSTALL_UPSTREAM_DEPS="${INSTALL_UPSTREAM_DEPS:-0}"
ENABLE_PROGRESS_BAR="${ENABLE_PROGRESS_BAR:-True}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --run-targets) RUN_TARGETS="$2"; shift 2 ;;
    --data-file) DATA_FILE="$2"; shift 2 ;;
    --eval-data-file) EVAL_DATA_FILE="$2"; shift 2 ;;
    --eval-every) EVAL_EVERY="$2"; shift 2 ;;
    --eval-steps) EVAL_STEPS="$2"; shift 2 ;;
    --loss-targets) LOSS_TARGETS="$2"; shift 2 ;;
    --ctx-len) CTX_LEN="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --per-device-batch-size) PER_DEVICE_BATCH_SIZE="$2"; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --num-nodes) NUM_NODES="$2"; shift 2 ;;
    --n-layer) N_LAYER="$2"; shift 2 ;;
    --d-model) D_MODEL="$2"; shift 2 ;;
    --head-size) HEAD_SIZE="$2"; shift 2 ;;
    --d-ffn) D_FFN="$2"; shift 2 ;;
    --vocab-size) VOCAB_SIZE="$2"; shift 2 ;;
    --screened-layers) SCREENED_LAYERS="$2"; shift 2 ;;
    --mechanism-phase) MECHANISM_PHASE="$2"; shift 2 ;;
    --rwkv-lm-v7-ref) RWKV_LM_V7_REF="$2"; shift 2 ;;
    --rwkv-lm-v7-repo) UPSTREAM_DIR="$2"; FETCH_UPSTREAM=0; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --download-data) DOWNLOAD_DATA=1; shift ;;
    --no-download-data) DOWNLOAD_DATA=0; shift ;;
    --no-param-control) RUN_PARAM_CONTROL=0; shift ;;
    --no-read-screening-run) RUN_READ_SCREENING=0; shift ;;
    --verify-core-parity) RUN_CORE_PARITY=1; shift ;;
    --no-run) RUN_TARGETS="commands"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$BACKEND" in
  cuda|tpu) ;;
  *) die "--backend must be cuda or tpu" ;;
esac

if [[ -z "$RUN_TARGETS" ]]; then
  if [[ "$BACKEND" == "cuda" ]]; then
    RUN_TARGETS="both"
  else
    RUN_TARGETS="local"
  fi
fi

case "$RUN_TARGETS" in
  both|local|upstream|commands) ;;
  *) die "--run-targets must be both, local, upstream, or commands" ;;
esac

case "$MECHANISM_PHASE" in
  read_screening_only|read_write) ;;
  *) die "--mechanism-phase must be read_screening_only or read_write" ;;
esac

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_BIN="uv run python"
  else
    die "python3, python, or uv is required for dataset metadata and parameter-count calculation"
  fi
fi
# shellcheck disable=SC2206
PYTHON_BIN_ARRAY=($PYTHON_BIN)

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$script_dir/$1" ;;
  esac
}

if [[ -z "$DEVICES" ]]; then
  if [[ "$BACKEND" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    DEVICES="$(nvidia-smi -L | wc -l | tr -d ' ')"
    [[ -n "$DEVICES" && "$DEVICES" != "0" ]] || DEVICES=1
  else
    DEVICES=1
  fi
fi

if [[ -z "$GLOBAL_BATCH_SIZE" ]]; then
  GLOBAL_BATCH_SIZE=$((NUM_NODES * DEVICES * PER_DEVICE_BATCH_SIZE))
fi

MICRO_BSZ="${MICRO_BSZ:-$PER_DEVICE_BATCH_SIZE}"
if (( GLOBAL_BATCH_SIZE != NUM_NODES * DEVICES * MICRO_BSZ )); then
  die "GLOBAL_BATCH_SIZE must equal NUM_NODES * DEVICES * MICRO_BSZ for upstream parity"
fi
if (( 40320 % GLOBAL_BATCH_SIZE != 0 )); then
  die "GLOBAL_BATCH_SIZE must divide 40320 because RWKV-LM-V7 hard-codes epoch_steps=40320/real_bsz"
fi
if [[ -z "$EVAL_EVERY" ]]; then
  EVAL_EVERY=$((STEPS / 10))
  if (( EVAL_EVERY < 1 )); then
    EVAL_EVERY=1
  fi
fi
if (( EVAL_EVERY <= 0 )); then
  die "EVAL_EVERY must be positive"
fi
if (( D_MODEL % HEAD_SIZE != 0 )); then
  die "D_MODEL must be divisible by HEAD_SIZE"
fi
N_HEADS=$((D_MODEL / HEAD_SIZE))
if [[ -z "$D_FFN" ]]; then
  D_FFN=$(( (D_MODEL * 35 / 10) / 32 * 32 ))
fi
if (( D_MODEL % 32 != 0 || D_FFN % 32 != 0 )); then
  die "D_MODEL and D_FFN must be multiples of 32 for RWKV-LM-V7"
fi

if [[ -z "$SCREENED_LAYERS" ]]; then
  if (( N_LAYER > 1 )); then
    SCREENED_LAYERS="$((N_LAYER / 2))"
  else
    SCREENED_LAYERS="0"
  fi
fi
read -r -a SCREENED_LAYERS_ARRAY <<< "$SCREENED_LAYERS"

case "$PRECISION" in
  bf16) LOCAL_DTYPE="bfloat16"; UPSTREAM_PRECISION="bf16" ;;
  fp32) LOCAL_DTYPE="float32"; UPSTREAM_PRECISION="fp32" ;;
  *) die "PRECISION must be bf16 or fp32" ;;
esac

if [[ "$BACKEND" == "tpu" && ( "$RUN_TARGETS" == "both" || "$RUN_TARGETS" == "upstream" ) ]]; then
  die "RWKV-LM-V7 upstream training is CUDA/PyTorch-only; use --run-targets local on TPU and run the generated upstream command on CUDA"
fi
if [[ "$RUN_CORE_PARITY" == "1" && ( "$BACKEND" != "cuda" || ( "$RUN_TARGETS" != "both" && "$RUN_TARGETS" != "upstream" && "$RUN_TARGETS" != "commands" ) ) ]]; then
  die "--verify-core-parity requires CUDA and an upstream/both target (or --no-run to emit commands)"
fi
if (( CORE_PARITY_TOKENS <= 0 || CORE_PARITY_TOKENS % 16 != 0 )); then
  die "CORE_PARITY_TOKENS must be positive and divisible by the official x070 chunk length 16"
fi
if [[ "$RUN_TARGETS" != "commands" && "$RUN_TARGETS" != "local" && "$STEPS" -le "$WARMUP_STEPS" ]]; then
  die "upstream my_exit_tokens exits before training when STEPS <= WARMUP_STEPS; increase --steps or lower WARMUP_STEPS"
fi

download_file() {
  local url="$1"
  local target="$2"
  mkdir -p "$(dirname "$target")"
  if [[ -s "$target" ]]; then
    echo "Using existing $target"
    return
  fi
  local partial="$target.part"
  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - --output "$partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$partial" "$url"
  else
    "${PYTHON_BIN_ARRAY[@]}" - "$url" "$partial" <<'PY'
import shutil
import sys
import urllib.request

url, target = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as response, open(target, "wb") as out:
    shutil.copyfileobj(response, out)
PY
  fi
  mv "$partial" "$target"
}

OUTPUT_ROOT="$(abs_path "${OUTPUT_ROOT%/}")"
DATA_FILE="$(abs_path "${DATA_FILE%/}")"
if [[ "$DATA_FILE" == "$MINIPILE_PREFIX" && "$DOWNLOAD_DATA" == "1" ]]; then
  download_file "$MINIPILE_IDX_URL" "$MINIPILE_PREFIX.idx"
  download_file "$MINIPILE_BIN_URL" "$MINIPILE_PREFIX.bin"
fi
[[ -f "$DATA_FILE.bin" ]] || die "missing dataset file: $DATA_FILE.bin"
[[ -f "$DATA_FILE.idx" ]] || die "missing dataset file: $DATA_FILE.idx"

if [[ -z "$EVAL_DATA_FILE" ]]; then
  EVAL_DATA_FILE="$DATA_FILE"
  EVAL_HELDOUT=0
else
  EVAL_DATA_FILE="$(abs_path "${EVAL_DATA_FILE%/}")"
  EVAL_HELDOUT=1
  [[ -f "$EVAL_DATA_FILE.bin" ]] || die "missing eval dataset file: $EVAL_DATA_FILE.bin"
  [[ -f "$EVAL_DATA_FILE.idx" ]] || die "missing eval dataset file: $EVAL_DATA_FILE.idx"
fi

prepare_upstream() {
  mkdir -p "$COMPARE_ROOT"
  if [[ "$FETCH_UPSTREAM" == "1" ]]; then
    if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
      git clone "$RWKV_LM_V7_URL" "$UPSTREAM_DIR"
    fi
    git -C "$UPSTREAM_DIR" fetch --tags origin
    git -C "$UPSTREAM_DIR" checkout --detach "$RWKV_LM_V7_REF" \
      || git -C "$UPSTREAM_DIR" checkout --detach "origin/$RWKV_LM_V7_REF"
  elif [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
    die "--rwkv-lm-v7-repo must point to a git checkout"
  fi
}

prepare_upstream
UPSTREAM_COMMIT="$(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
LOCAL_COMMIT="$(git rev-parse HEAD 2>/dev/null || printf unknown)"

upstream_venv_python() {
  local candidate
  for candidate in "$UPSTREAM_VENV/bin/python" "$UPSTREAM_VENV/Scripts/python.exe"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  die "uv created an upstream venv without a usable Python executable: $UPSTREAM_VENV"
}

prepare_upstream_venv() {
  command -v uv >/dev/null 2>&1 || die "uv is required to create the upstream RWKV-LM-V7 venv"
  local created=0
  if [[ ! -f "$UPSTREAM_VENV/pyvenv.cfg" ]]; then
    mkdir -p "$(dirname "$UPSTREAM_VENV")"
    echo "Creating upstream RWKV-LM-V7 venv with uv: $UPSTREAM_VENV"
    uv venv --no-project --python "$UPSTREAM_PYTHON_REQUEST" "$UPSTREAM_VENV"
    created=1
  fi

  UPSTREAM_PYTHON="$(upstream_venv_python)"
  local python_version
  python_version="$("$UPSTREAM_PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$python_version" == "3.12" ]] || die "upstream venv must use Python 3.12, found $python_version at $UPSTREAM_PYTHON"

  local requirements_file="$UPSTREAM_DIR/requirements.txt"
  [[ -f "$requirements_file" ]] || die "missing upstream requirements file: $requirements_file"
  local requirements_hash
  requirements_hash="$(git -C "$UPSTREAM_DIR" hash-object requirements.txt)"
  local dependency_fingerprint="${requirements_hash}:setuptools<81"
  local requirements_marker="$UPSTREAM_VENV/.rwkv7m-requirements.hash"
  local installed_hash=""
  if [[ -f "$requirements_marker" ]]; then
    installed_hash="$(<"$requirements_marker")"
  fi
  if [[ "$created" == "1" || "$INSTALL_UPSTREAM_DEPS" == "1" || "$installed_hash" != "$dependency_fingerprint" ]]; then
    echo "Syncing upstream RWKV-LM-V7 dependencies into $UPSTREAM_VENV"
    # Lightning 1.9.5 imports pkg_resources, which setuptools 81+ removed.
    uv pip install --python "$UPSTREAM_PYTHON" -r "$requirements_file" "setuptools<81"
    printf '%s\n' "$dependency_fingerprint" > "$requirements_marker"
  else
    echo "Using cached upstream RWKV-LM-V7 venv: $UPSTREAM_VENV"
  fi
}

if [[ "$RUN_TARGETS" == "both" || "$RUN_TARGETS" == "upstream" || "$RUN_TARGETS" == "commands" || "$RUN_CORE_PARITY" == "1" ]]; then
  prepare_upstream_venv
  UPSTREAM_VENV_BIN="$(dirname "$UPSTREAM_PYTHON")"
  # torch.utils.cpp_extension invokes ninja by name even when Python is absolute.
  export PATH="$UPSTREAM_VENV_BIN:$PATH"
fi

dataset_metadata() {
  "${PYTHON_BIN_ARRAY[@]}" - "$1" "$CTX_LEN" "$MAGIC_PRIME" <<'PY'
import os
import struct
import sys

prefix = sys.argv[1]
ctx_len = int(sys.argv[2])
forced = sys.argv[3]
dtype_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 4, 7: 8, 8: 2}

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def find_magic_prime(data_size, ctx_len):
    dataset_slot = (int(data_size) - 1) // int(ctx_len)
    for candidate in range(dataset_slot, 1, -1):
        if candidate % 3 == 2 and is_prime(candidate):
            return candidate
    raise SystemExit("could not find a 3n+2 prime for this data_size and ctx_len")

with open(prefix + ".idx", "rb") as f:
    if f.read(9) != b"MMIDIDX\x00\x00":
        raise SystemExit("idx file does not match MMIDIDX format")
    version = struct.unpack("<Q", f.read(8))[0]
    if version != 1:
        raise SystemExit(f"unsupported idx version: {version}")
    dtype_code = struct.unpack("<B", f.read(1))[0]
    dtype_size = dtype_sizes.get(dtype_code)
    if dtype_size is None:
        raise SystemExit(f"unsupported dtype code: {dtype_code}")
    item_count = struct.unpack("<Q", f.read(8))[0]
    _doc_count = struct.unpack("<Q", f.read(8))[0]
    first_size = struct.unpack("<i", f.read(4))[0] if item_count else 0

data_size = os.path.getsize(prefix + ".bin") // dtype_size
if item_count != 1 or first_size != data_size:
    raise SystemExit(
        "RWKV-LM-V7 samples with data.get(idx=0, offset=...), so comparison "
        "requires a single binidx item covering the whole .bin file"
    )

magic_prime = int(forced) if forced else find_magic_prime(data_size, ctx_len)
dataset_slot = data_size // ctx_len
if not is_prime(magic_prime) or magic_prime % 3 != 2:
    raise SystemExit("MAGIC_PRIME must be prime and satisfy prime % 3 == 2")
if not (0.9 < magic_prime / max(dataset_slot, 1) <= 1.0):
    raise SystemExit("MAGIC_PRIME must be close to data_size // ctx_len for RWKV-LM-V7")
if magic_prime * ctx_len + 1 > data_size:
    raise SystemExit("MAGIC_PRIME can sample beyond the end of the dataset")

print(f"DATA_TOKENS={data_size}")
print(f"DATASET_SLOT={dataset_slot}")
print(f"MAGIC_PRIME={magic_prime}")
PY
}

metadata="$(dataset_metadata "$DATA_FILE")"
metadata="${metadata//$'\r'/}"
eval "$metadata"

param_plan="$(
  "${PYTHON_BIN_ARRAY[@]}" - "$D_MODEL" "$D_FFN" "$N_LAYER" "$HEAD_SIZE" "$VOCAB_SIZE" "$D_SLOT" "$D_K" "$D_V" "$N_SLOTS" "$MECHANISM_PHASE" "$SCREENED_LAYERS" <<'PY'
import math
import sys

C = int(sys.argv[1])
d_ffn = int(sys.argv[2])
n_layer = int(sys.argv[3])
head_size = int(sys.argv[4])
vocab = int(sys.argv[5])
d_slot = int(sys.argv[6])
d_k = int(sys.argv[7])
d_v = int(sys.argv[8])
n_slots = int(sys.argv[9])
phase = sys.argv[10]
screened_layers = tuple(int(x) for x in sys.argv[11].split())
H = C // head_size
N = head_size

def round_mult32(value):
    return int(round(value / 32.0) * 32)

def helper_dim(scale):
    return max(32, round_mult32(scale * math.sqrt(C)))

def rwkv_core_params(ffn):
    total = vocab * C  # token embedding
    total += vocab * C  # lm head
    total += 2 * C      # final layer norm
    d_decay = helper_dim(2.5)
    d_aaa = helper_dim(2.5)
    d_gate = helper_dim(5.0)
    d_mv = helper_dim(1.7)
    for layer in range(n_layer):
        total += 4 * C  # ln1 + ln2
        if layer == 0:
            total += 2 * C  # ln0
        total += 6 * C          # x_r/x_w/x_k/x_v/x_a/x_g
        total += C * C          # receptance
        total += 2 * C * d_decay + C  # w1/w2/w0
        total += 2 * C * C      # key/value
        if layer > 0:
            total += 2 * C * d_mv + C  # v1/v2/v0
        total += 2 * C * d_aaa + C     # a1/a2/a0
        total += 2 * C * d_gate        # g1/g2
        total += 2 * C                 # k_k/k_a
        total += 2 * C                 # group norm ln_x
        total += H * N                 # r_k
        total += C * C                 # output
        total += C                     # channel mix x_k
        total += 2 * C * ffn           # channel mix key/value
    return int(total)

def screening_params(phase):
    per_layer = 0
    per_layer += 2 * C                    # screen_ln
    per_layer += C * d_k                  # q_proj_r
    per_layer += d_slot * d_k             # k_proj_r
    per_layer += d_slot * d_v             # v_proj
    per_layer += d_v * C                  # out_proj
    per_layer += C * C + C                # gate_proj
    per_layer += (2 * C + d_slot) * d_slot + d_slot  # delta_proj
    per_layer += 1 + 1                    # tau_r_raw + lambda_raw
    per_layer += n_slots * d_slot         # slot_embed
    per_layer += 3                        # mu_by_bank_raw
    if phase == "read_write":
        per_layer += (2 * C) * d_k        # q_proj_w
        per_layer += d_slot * d_k         # k_proj_w
        per_layer += 1                    # tau_w_raw
    return per_layer * len(screened_layers)

baseline = rwkv_core_params(d_ffn)
read_screening = baseline + screening_params("read_screening_only")
mechanism = baseline + screening_params(phase)
param_target = max(read_screening, mechanism)
control_d_ffn = d_ffn
control = baseline
while control < param_target:
    control_d_ffn += 32
    control = rwkv_core_params(control_d_ffn)

print(f"LOCAL_BASELINE_PARAMS={baseline}")
print(f"LOCAL_READ_SCREENING_PARAMS={read_screening}")
print(f"LOCAL_MECHANISM_PARAMS={mechanism}")
print(f"LOCAL_SCREENING_EXTRA_PARAMS={mechanism - baseline}")
print(f"LOCAL_PARAM_TARGET_PARAMS={param_target}")
print(f"LOCAL_CONTROL_D_FFN={control_d_ffn}")
print(f"LOCAL_CONTROL_PARAMS={control}")
print(f"LOCAL_CONTROL_EXTRA_PARAMS={control - baseline}")
PY
)"
param_plan="${param_plan//$'\r'/}"
eval "$param_plan"

MY_EXIT_TOKENS=$((STEPS * GLOBAL_BATCH_SIZE * CTX_LEN))
run_root="$OUTPUT_ROOT/$RUN_ID"
local_baseline_out="$run_root/local_core_baseline"
local_read_screening_out="$run_root/local_read_screening"
local_mechanism_out="$run_root/local_mechanism"
local_control_out="$run_root/local_param_control"
upstream_out="$run_root/upstream_rwkv_lm_v7"
core_parity_out="$run_root/upstream_core_parity"
mkdir -p "$local_baseline_out" "$local_mechanism_out" "$upstream_out"
if [[ "$RUN_CORE_PARITY" == "1" ]]; then
  mkdir -p "$core_parity_out"
fi
if [[ "$RUN_READ_SCREENING" == "1" ]]; then
  mkdir -p "$local_read_screening_out"
fi
if [[ "$RUN_PARAM_CONTROL" == "1" ]]; then
  mkdir -p "$local_control_out"
fi

# shellcheck disable=SC2206
LOCAL_PREFIX_ARRAY=($LOCAL_PREFIX)

make_local_cmd() {
  local -n cmd_ref=$1
  local out_dir="$2"
  local phase="$3"
  local ffn="$4"
  local use_screening="$5"

  cmd_ref=(
    "${LOCAL_PREFIX_ARRAY[@]}" rwkv7m-train-binidx-dp
    --data-file "$DATA_FILE"
    --ctx-len "$CTX_LEN"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --steps "$STEPS"
    --magic-prime "$MAGIC_PRIME"
    --sampling-mode magic
    --seed "$SEED"
    --phase "$phase"
    --dtype "$LOCAL_DTYPE"
    --lr-init "$LR_INIT"
    --lr-final "$LR_FINAL"
    --warmup-steps "$WARMUP_STEPS"
    --lr-schedule "$LOCAL_LR_SCHEDULE"
    --max-grad-norm "$GRAD_CLIP"
    --weight-decay "$WEIGHT_DECAY"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
    --adam-eps "$ADAM_EPS"
    --vocab-size "$VOCAB_SIZE"
    --d-model "$D_MODEL"
    --d-ffn "$ffn"
    --n-layers "$N_LAYER"
    --n-heads "$N_HEADS"
    --head-size "$HEAD_SIZE"
    --print-every 1
    --output-dir "$out_dir"
    --save-every "$STEPS"
    --keep-last-checkpoints 1
    --checkpoint-backend "$LOCAL_CHECKPOINT_BACKEND"
    --prefetch-size "$LOCAL_PREFETCH_SIZE"
    --eval-every "$EVAL_EVERY"
    --eval-steps "$EVAL_STEPS"
    --eval-data-file "$EVAL_DATA_FILE"
    --log-jsonl "$out_dir/metrics.jsonl"
    --log-csv "$out_dir/metrics.csv"
    --summary-json "$out_dir/run_summary.json"
    --summary-every 1
  )
  if [[ "$use_screening" == "1" ]]; then
    cmd_ref+=(
      --d-slot "$D_SLOT"
      --d-k "$D_K"
      --d-v "$D_V"
      --n-slots "$N_SLOTS"
      --screened-layers "${SCREENED_LAYERS_ARRAY[@]}"
      --write-rel-floor "$WRITE_REL_FLOOR"
      --short-half-life-tokens "$SHORT_HALF_LIFE_TOKENS"
      --mid-half-life-tokens "$MID_HALF_LIFE_TOKENS"
      --long-half-life-tokens "$LONG_HALF_LIFE_TOKENS"
    )
  else
    cmd_ref+=(--no-screening)
  fi
  if [[ -n "$LOCAL_PARAM_AXIS_NAME" ]]; then
    cmd_ref+=(--param-axis-name "$LOCAL_PARAM_AXIS_NAME")
  fi
}

local_baseline_cmd=()
local_read_screening_cmd=()
local_mechanism_cmd=()
local_control_cmd=()
make_local_cmd local_baseline_cmd "$local_baseline_out" read_screening_only "$D_FFN" 0
if [[ "$RUN_READ_SCREENING" == "1" ]]; then
  make_local_cmd local_read_screening_cmd "$local_read_screening_out" read_screening_only "$D_FFN" 1
fi
make_local_cmd local_mechanism_cmd "$local_mechanism_out" "$MECHANISM_PHASE" "$D_FFN" 1
if [[ "$RUN_PARAM_CONTROL" == "1" ]]; then
  make_local_cmd local_control_cmd "$local_control_out" read_screening_only "$LOCAL_CONTROL_D_FFN" 0
fi

upstream_common=(
  --wandb ""
  --proj_dir "$upstream_out"
  --my_testing "$UPSTREAM_MODEL_TYPE"
  --ctx_len "$CTX_LEN"
  --epoch_count 999999
  --epoch_begin 0
  --data_file "$DATA_FILE"
  --data_type binidx
  --my_exit_tokens "$MY_EXIT_TOKENS"
  --magic_prime "$MAGIC_PRIME"
  --num_nodes "$NUM_NODES"
  --micro_bsz "$MICRO_BSZ"
  --n_layer "$N_LAYER"
  --n_embd "$D_MODEL"
  --dim_att "$D_MODEL"
  --dim_ffn "$D_FFN"
  --kernel "$UPSTREAM_KERNEL"
  --lr_init "$LR_INIT"
  --lr_final "$LR_FINAL"
  --warmup_steps "$WARMUP_STEPS"
  --beta1 "$ADAM_BETA1"
  --beta2 "$ADAM_BETA2"
  --adam_eps "$ADAM_EPS"
  --vocab_size "$VOCAB_SIZE"
  --weight_decay "$WEIGHT_DECAY"
  --grad_clip "$GRAD_CLIP"
  --epoch_save 999999
  --head_size "$HEAD_SIZE"
  --head_chunk "$UPSTREAM_HEAD_CHUNK"
  --accelerator gpu
  --devices "$DEVICES"
  --precision "$UPSTREAM_PRECISION"
  --strategy "$UPSTREAM_STRATEGY"
  --grad_cp "$UPSTREAM_GRAD_CP"
  --ds_bucket_mb "$UPSTREAM_DS_BUCKET_MB"
  --enable_progress_bar "$ENABLE_PROGRESS_BAR"
  --random_seed "$SEED"
)
upstream_prepare_cmd=("$UPSTREAM_PYTHON" train.py "${upstream_common[@]}" --train_stage 1)
upstream_train_cmd=("$UPSTREAM_PYTHON" train.py --load_model 0 "${upstream_common[@]}" --train_stage "$UPSTREAM_TRAIN_STAGE")
upstream_parity_cmd=(
  env TORCH_EXTENSIONS_DIR="$COMPARE_ROOT/torch_extensions"
  "$UPSTREAM_PYTHON" "$script_dir/scripts/capture_upstream_rwkv7_reference.py"
  --upstream-repo "$UPSTREAM_DIR"
  --output "$core_parity_out/official_reference.npz"
  --tokens "$CORE_PARITY_TOKENS"
  --seed "$SEED"
  --kernel "$UPSTREAM_KERNEL"
)
local_parity_cmd=(
  "${LOCAL_PREFIX_ARRAY[@]}" rwkv7m-verify-upstream-rwkv7
  "$core_parity_out/official_reference.npz"
  --atol "$CORE_PARITY_ATOL"
  --rtol "$CORE_PARITY_RTOL"
  --json-out "$core_parity_out/parity_report.json"
)

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

{
  echo "# Generated by comparison.sh"
  echo "RUN_ID=$RUN_ID"
  echo "LOCAL_COMMIT=$LOCAL_COMMIT"
  echo "UPSTREAM_URL=$RWKV_LM_V7_URL"
  echo "UPSTREAM_REF=$RWKV_LM_V7_REF"
  echo "UPSTREAM_COMMIT=$UPSTREAM_COMMIT"
  echo "UPSTREAM_VENV=$UPSTREAM_VENV"
  echo "UPSTREAM_PYTHON=$UPSTREAM_PYTHON"
  echo "MINIPILE_IDX_URL=$MINIPILE_IDX_URL"
  echo "MINIPILE_BIN_URL=$MINIPILE_BIN_URL"
  echo "BACKEND=$BACKEND"
  echo "RUN_TARGETS=$RUN_TARGETS"
  echo "DATA_FILE=$DATA_FILE"
  echo "EVAL_DATA_FILE=$EVAL_DATA_FILE"
  echo "EVAL_HELDOUT=$EVAL_HELDOUT"
  echo "EVAL_EVERY=$EVAL_EVERY"
  echo "EVAL_STEPS=$EVAL_STEPS"
  echo "LOSS_TARGETS=$LOSS_TARGETS"
  echo "RUN_CORE_PARITY=$RUN_CORE_PARITY"
  echo "CORE_PARITY_ATOL=$CORE_PARITY_ATOL"
  echo "CORE_PARITY_RTOL=$CORE_PARITY_RTOL"
  echo "CORE_PARITY_TOKENS=$CORE_PARITY_TOKENS"
  echo "DATA_TOKENS=$DATA_TOKENS"
  echo "CTX_LEN=$CTX_LEN"
  echo "STEPS=$STEPS"
  echo "SEED=$SEED"
  echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE"
  echo "PER_DEVICE_BATCH_SIZE=$PER_DEVICE_BATCH_SIZE"
  echo "NUM_NODES=$NUM_NODES"
  echo "DEVICES=$DEVICES"
  echo "MICRO_BSZ=$MICRO_BSZ"
  echo "VOCAB_SIZE=$VOCAB_SIZE"
  echo "N_LAYER=$N_LAYER"
  echo "D_MODEL=$D_MODEL"
  echo "D_FFN=$D_FFN"
  echo "LOCAL_CONTROL_D_FFN=$LOCAL_CONTROL_D_FFN"
  echo "HEAD_SIZE=$HEAD_SIZE"
  echo "N_HEADS=$N_HEADS"
  echo "SCREENED_LAYERS=\"$SCREENED_LAYERS\""
  echo "MECHANISM_PHASE=$MECHANISM_PHASE"
  echo "D_SLOT=$D_SLOT"
  echo "D_K=$D_K"
  echo "D_V=$D_V"
  echo "N_SLOTS=$N_SLOTS"
  echo "WRITE_REL_FLOOR=$WRITE_REL_FLOOR"
  echo "SHORT_HALF_LIFE_TOKENS=$SHORT_HALF_LIFE_TOKENS"
  echo "MID_HALF_LIFE_TOKENS=$MID_HALF_LIFE_TOKENS"
  echo "LONG_HALF_LIFE_TOKENS=$LONG_HALF_LIFE_TOKENS"
  echo "PRECISION=$PRECISION"
  echo "LR_INIT=$LR_INIT"
  echo "LR_FINAL=$LR_FINAL"
  echo "WARMUP_STEPS=$WARMUP_STEPS"
  echo "LOCAL_LR_SCHEDULE=$LOCAL_LR_SCHEDULE"
  echo "ADAM_BETA1=$ADAM_BETA1"
  echo "ADAM_BETA2=$ADAM_BETA2"
  echo "ADAM_EPS=$ADAM_EPS"
  echo "WEIGHT_DECAY=$WEIGHT_DECAY"
  echo "GRAD_CLIP=$GRAD_CLIP"
  echo "MAGIC_PRIME=$MAGIC_PRIME"
  echo "MY_EXIT_TOKENS=$MY_EXIT_TOKENS"
  echo "LOCAL_BASELINE_PARAMS=$LOCAL_BASELINE_PARAMS"
  echo "LOCAL_READ_SCREENING_PARAMS=$LOCAL_READ_SCREENING_PARAMS"
  echo "LOCAL_MECHANISM_PARAMS=$LOCAL_MECHANISM_PARAMS"
  echo "LOCAL_SCREENING_EXTRA_PARAMS=$LOCAL_SCREENING_EXTRA_PARAMS"
  echo "LOCAL_PARAM_TARGET_PARAMS=$LOCAL_PARAM_TARGET_PARAMS"
  echo "LOCAL_CONTROL_PARAMS=$LOCAL_CONTROL_PARAMS"
  echo "LOCAL_CONTROL_EXTRA_PARAMS=$LOCAL_CONTROL_EXTRA_PARAMS"
  echo "LOCAL_BASELINE_OUT=$local_baseline_out"
  echo "LOCAL_READ_SCREENING_OUT=$local_read_screening_out"
  echo "LOCAL_MECHANISM_OUT=$local_mechanism_out"
  echo "LOCAL_CONTROL_OUT=$local_control_out"
  echo "UPSTREAM_OUT=$upstream_out"
  echo "CORE_PARITY_OUT=$core_parity_out"
} > "$run_root/parameters.env"

{
  echo "variant,trainable_params,d_ffn,use_screening,phase,note"
  echo "upstream_rwkv_lm_v7_estimate,$LOCAL_BASELINE_PARAMS,$D_FFN,0,read_screening_only,core architecture estimate"
  echo "local_core_baseline,$LOCAL_BASELINE_PARAMS,$D_FFN,0,read_screening_only,no mechanism"
  if [[ "$RUN_READ_SCREENING" == "1" ]]; then
    echo "local_read_screening,$LOCAL_READ_SCREENING_PARAMS,$D_FFN,1,read_screening_only,screened reads with uniform slow writes"
  fi
  echo "local_mechanism,$LOCAL_MECHANISM_PARAMS,$D_FFN,1,$MECHANISM_PHASE,same core width plus screening mechanism"
  if [[ "$RUN_PARAM_CONTROL" == "1" ]]; then
    echo "local_param_control,$LOCAL_CONTROL_PARAMS,$LOCAL_CONTROL_D_FFN,0,read_screening_only,no mechanism widened to >= largest screening params"
  fi
} > "$run_root/parameter_counts.csv"

{
  echo "# Comparison Protocol"
  echo
  echo "Goal: test whether state-level screening memory, adapted from Multi Screening to RWKV-style recurrent blocks, improves learning speed and final loss/perplexity beyond gains explained by extra trainable parameters."
  echo
  echo "Primary evidence should compare held-out eval curves for:"
  echo
  echo "1. local_core_baseline: no screening mechanism."
  echo "2. local_read_screening: same RWKV core width, screened reads, and uniform slow writes."
  echo "3. local_mechanism: same RWKV core width plus the selected screening phase, read_write by default."
  echo "4. local_param_control: no screening mechanism, widened FFN, trainable params >= the largest screening variant."
  echo
  echo "Interpretation rule:"
  echo
  echo "- Learning-speed evidence is strongest when a screening variant reaches the same held-out eval loss in fewer optimizer steps/tokens and has lower eval-loss AUC over tokens."
  echo "- Mechanism evidence is strongest when a screening variant beats both local_core_baseline and local_param_control on held-out eval at matched tokens, optimizer settings, dtype, context length, global batch, and data sampler."
  echo "- If --eval-data-file is omitted, eval uses the train dataset; those results are useful for debugging but not proof-grade."
  echo "- upstream_rwkv_lm_v7 is a CUDA RWKV-LM-V7 reference run for external baseline/throughput context. It does not isolate this repository's screening mechanism by itself."
  echo
  echo "Artifacts:"
  echo
  echo "- parameters.env records all run parameters and source commits."
  echo "- parameter_counts.csv records computed trainable-parameter counts for the local ablation matrix."
  echo "- local_metric_summary.csv is written after local runs and contains final train/eval metrics."
  echo "- learning_speed_summary.csv records best/final eval, eval-loss AUC over tokens, improvement per token, and throughput."
  echo "- learning_target_hits.csv records steps/tokens/estimated seconds needed to reach baseline/control/explicit loss targets."
  echo "- commands.sh records the exact commands for replay."
  echo "- upstream_core_parity/parity_report.json records fixed-weight official CUDA vs local JAX logits parity when requested."
} > "$run_root/protocol.md"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd $(printf '%q' "$script_dir")"
  if [[ -n "$UPSTREAM_VENV_BIN" ]]; then
    printf 'export PATH=%q:$PATH\n' "$UPSTREAM_VENV_BIN"
  fi
  echo
  echo "# Local core baseline"
  quote_cmd "${local_baseline_cmd[@]}"
  if [[ "$RUN_READ_SCREENING" == "1" ]]; then
    echo
    echo "# Local screened-read / uniform-slow-write control"
    quote_cmd "${local_read_screening_cmd[@]}"
  fi
  echo
  echo "# Local mechanism"
  quote_cmd "${local_mechanism_cmd[@]}"
  if [[ "$RUN_PARAM_CONTROL" == "1" ]]; then
    echo
    echo "# Local parameter-count control"
    quote_cmd "${local_control_cmd[@]}"
  fi
  echo
  echo "# Upstream RWKV-LM-V7 prepare command"
  echo "cd $(printf '%q' "$UPSTREAM_DIR")"
  quote_cmd "${upstream_prepare_cmd[@]}"
  echo
  echo "# Upstream RWKV-LM-V7 train command"
  quote_cmd "${upstream_train_cmd[@]}"
  if [[ "$RUN_CORE_PARITY" == "1" ]]; then
    echo
    echo "# Official fused CUDA core reference capture"
    echo "cd $(printf '%q' "$script_dir")"
    quote_cmd "${upstream_parity_cmd[@]}"
    echo
    echo "# Local JAX core parity verification"
    printf 'JAX_PLATFORMS=cuda '
    quote_cmd "${local_parity_cmd[@]}"
  fi
} > "$run_root/commands.sh"
chmod +x "$run_root/commands.sh"

run_logged() {
  local log_file="$1"
  shift
  set +e
  "$@" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

summarize_local_metrics() {
  "${PYTHON_BIN_ARRAY[@]}" - "$run_root" "$LOSS_TARGETS" "$GLOBAL_BATCH_SIZE" "$CTX_LEN" <<'PY'
import csv
import json
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
explicit_targets = []
for raw in sys.argv[2].split(","):
    raw = raw.strip()
    if not raw:
        continue
    explicit_targets.append(("explicit_" + raw.replace(".", "p"), float(raw)))
tokens_per_step = int(sys.argv[3]) * int(sys.argv[4])

def load_records(variant):
    path = root / variant / "metrics.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def cumulative_seconds(train_records):
    seconds_by_step = {}
    total = 0.0
    usable = True
    for record in sorted(train_records, key=lambda item: int(item.get("step", 0))):
        tps = record.get("tokens_per_sec")
        tokens = record.get("tokens") or tokens_per_step
        step = int(record.get("step", 0))
        if tps is None or float(tps) <= 0:
            usable = False
            seconds_by_step[step] = None
            continue
        if usable:
            total += float(tokens) / float(tps)
            seconds_by_step[step] = total
        else:
            seconds_by_step[step] = None
    return seconds_by_step, total if usable else None

def auc_over_tokens(eval_records):
    points = [
        (int(record["step"]) * tokens_per_step, float(record["loss"]))
        for record in eval_records
        if record.get("loss") is not None
    ]
    points.sort()
    if len(points) < 2:
        return None, None
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    span = points[-1][0] - points[0][0]
    return auc, auc / span if span > 0 else None

def first_hit(eval_records, target_loss, seconds_by_step):
    for record in sorted(eval_records, key=lambda item: int(item.get("step", 0))):
        loss = record.get("loss")
        if loss is None:
            continue
        if float(loss) <= float(target_loss):
            step = int(record["step"])
            return {
                "hit_step": step,
                "hit_tokens": step * tokens_per_step,
                "hit_estimated_train_seconds": seconds_by_step.get(step),
                "hit_eval_loss": float(loss),
            }
    return {
        "hit_step": None,
        "hit_tokens": None,
        "hit_estimated_train_seconds": None,
        "hit_eval_loss": None,
    }

rows = []
speed_rows = []
target_rows = []
variant_order = [
    "local_core_baseline",
    "local_read_screening",
    "local_mechanism",
    "local_param_control",
]
variant_payloads = {}

for variant in variant_order:
    records = load_records(variant)
    if not records:
        continue
    train = [r for r in records if r.get("split") == "train"]
    evals = [r for r in records if r.get("split") == "eval"]
    variant_payloads[variant] = {"train": train, "evals": evals}
    last_train = train[-1] if train else {}
    last_eval = evals[-1] if evals else {}
    rows.append({
        "variant": variant,
        "train_step": last_train.get("step"),
        "train_loss": last_train.get("loss"),
        "train_tokens_per_sec": last_train.get("tokens_per_sec"),
        "eval_step": last_eval.get("step"),
        "eval_loss": last_eval.get("loss"),
        "eval_perplexity": last_eval.get("perplexity"),
    })

    eval_losses = [float(r["loss"]) for r in evals if r.get("loss") is not None]
    train_losses = [float(r["loss"]) for r in train if r.get("loss") is not None]
    train_tps = [
        float(r["tokens_per_sec"])
        for r in train
        if r.get("tokens_per_sec") is not None and float(r["tokens_per_sec"]) > 0
    ]
    seconds_by_step, total_seconds = cumulative_seconds(train)
    auc, mean_token_loss = auc_over_tokens(evals)
    best_eval = None
    if evals:
        best_eval = min(
            (r for r in evals if r.get("loss") is not None),
            key=lambda item: float(item["loss"]),
            default=None,
        )
    improvement = None
    if len(eval_losses) >= 2:
        improvement = eval_losses[0] - eval_losses[-1]
    speed_rows.append({
        "variant": variant,
        "eval_points": len(evals),
        "first_eval_step": evals[0].get("step") if evals else None,
        "first_eval_loss": eval_losses[0] if eval_losses else None,
        "final_eval_step": last_eval.get("step"),
        "final_eval_loss": last_eval.get("loss"),
        "final_eval_perplexity": last_eval.get("perplexity"),
        "best_eval_step": best_eval.get("step") if best_eval else None,
        "best_eval_tokens": int(best_eval["step"]) * tokens_per_step if best_eval else None,
        "best_eval_loss": float(best_eval["loss"]) if best_eval else None,
        "eval_loss_auc_tokens": auc,
        "eval_loss_mean_over_tokens": mean_token_loss,
        "eval_loss_improvement": improvement,
        "eval_loss_improvement_per_billion_tokens": (
            improvement / max((int(last_eval["step"]) - int(evals[0]["step"])) * tokens_per_step, 1) * 1_000_000_000
            if improvement is not None and last_eval.get("step") is not None and evals
            else None
        ),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "mean_train_tokens_per_sec": statistics.fmean(train_tps) if train_tps else None,
        "median_train_tokens_per_sec": statistics.median(train_tps) if train_tps else None,
        "estimated_train_seconds": total_seconds,
    })

baseline_final = None
if variant_payloads.get("local_core_baseline", {}).get("evals"):
    baseline_evals = variant_payloads["local_core_baseline"]["evals"]
    if baseline_evals and baseline_evals[-1].get("loss") is not None:
        baseline_final = float(baseline_evals[-1]["loss"])
control_final = None
if variant_payloads.get("local_param_control", {}).get("evals"):
    control_evals = variant_payloads["local_param_control"]["evals"]
    if control_evals and control_evals[-1].get("loss") is not None:
        control_final = float(control_evals[-1]["loss"])

targets = []
if baseline_final is not None:
    targets.append(("baseline_final_eval_loss", baseline_final))
if control_final is not None:
    targets.append(("param_control_final_eval_loss", control_final))
targets.extend(explicit_targets)

for variant, payload in variant_payloads.items():
    seconds_by_step, _ = cumulative_seconds(payload["train"])
    for target_name, target_loss in targets:
        hit = first_hit(payload["evals"], target_loss, seconds_by_step)
        target_rows.append({
            "target_name": target_name,
            "target_loss": target_loss,
            "variant": variant,
            **hit,
        })

if rows:
    out = root / "local_metric_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"local metric summary: {out}")
if speed_rows:
    out = root / "learning_speed_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(speed_rows[0]))
        writer.writeheader()
        writer.writerows(speed_rows)
    print(f"learning speed summary: {out}")
if target_rows:
    out = root / "learning_target_hits.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    print(f"learning target hits: {out}")
PY
}

echo "Comparison run: $RUN_ID"
echo "Parameters: $run_root/parameters.env"
echo "Parameter counts: $run_root/parameter_counts.csv"
echo "Protocol: $run_root/protocol.md"
echo "Commands: $run_root/commands.sh"
echo "RWKV-LM-V7 commit: $UPSTREAM_COMMIT"
echo "magic_prime=$MAGIC_PRIME data_tokens=$DATA_TOKENS global_batch=$GLOBAL_BATCH_SIZE"
if [[ "$EVAL_HELDOUT" == "0" ]]; then
  echo "WARNING: --eval-data-file was not provided; local eval uses the train dataset and is not proof-grade."
fi

if [[ "$RUN_TARGETS" == "commands" ]]; then
  echo "No commands executed (--no-run/commands mode)."
  exit 0
fi

if [[ "$RUN_TARGETS" == "local" || "$RUN_TARGETS" == "both" ]]; then
  local_env=()
  if [[ "$BACKEND" == "cuda" ]]; then
    local_env+=(JAX_PLATFORMS=cuda)
  else
    local_env+=(JAX_PLATFORMS=tpu)
  fi
  echo "Running local core baseline..."
  run_logged "$local_baseline_out/train.log" env "${local_env[@]}" "${local_baseline_cmd[@]}"
  if [[ "$RUN_READ_SCREENING" == "1" ]]; then
    echo "Running local screened-read / uniform-slow-write control..."
    run_logged "$local_read_screening_out/train.log" env "${local_env[@]}" "${local_read_screening_cmd[@]}"
  fi
  echo "Running local mechanism variant..."
  run_logged "$local_mechanism_out/train.log" env "${local_env[@]}" "${local_mechanism_cmd[@]}"
  if [[ "$RUN_PARAM_CONTROL" == "1" ]]; then
    echo "Running local parameter-count control..."
    run_logged "$local_control_out/train.log" env "${local_env[@]}" "${local_control_cmd[@]}"
  fi
  summarize_local_metrics
fi

if [[ "$RUN_TARGETS" == "upstream" || "$RUN_TARGETS" == "both" ]]; then
  echo "Running upstream RWKV-LM-V7 prepare target..."
  (
    cd "$UPSTREAM_DIR"
    run_logged "$upstream_out/prepare.log" env TORCH_EXTENSIONS_DIR="$COMPARE_ROOT/torch_extensions" "${upstream_prepare_cmd[@]}"
  )
  echo "Running upstream RWKV-LM-V7 train target..."
  (
    cd "$UPSTREAM_DIR"
    run_logged "$upstream_out/train.log" env TORCH_EXTENSIONS_DIR="$COMPARE_ROOT/torch_extensions" "${upstream_train_cmd[@]}"
  )
  if [[ "$RUN_CORE_PARITY" == "1" ]]; then
    echo "Capturing official fused CUDA RWKV-7 core reference..."
    run_logged "$core_parity_out/capture.log" "${upstream_parity_cmd[@]}"
    echo "Verifying local JAX RWKV-7 core parity..."
    run_logged "$core_parity_out/verify.log" env JAX_PLATFORMS=cuda "${local_parity_cmd[@]}"
  fi
fi

echo "Comparison artifacts written under: $run_root"
