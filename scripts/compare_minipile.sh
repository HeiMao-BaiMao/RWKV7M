#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/compare_minipile.sh [--profile smoke|small|medium|large]
      [--seeds "42 43 44"] [comparison.sh options]

Downloads BlinkDL MiniPile, checks out RWKV-Vibe/RWKV-LM-V7, and runs the
same-token-budget upstream/local comparison matrix. The downloaded dataset and
upstream checkout are cached under .comparison/.

Profiles:
  smoke   L4-D256, ctx128, 50 steps. Pipeline and kernel sanity check only.
  small   L12-D768, ctx512, 2000 steps. About the upstream 0.19B scale.
  medium  L24-D2048, ctx512, 2000 steps. About the upstream 1.5B scale.
  large   L24-D2560, ctx512, 2000 steps. Expensive multi-GPU research run.

Examples:
  bash scripts/compare_minipile.sh --profile smoke --no-run
  bash scripts/compare_minipile.sh --profile small --seeds "42 43 44"
  bash scripts/compare_minipile.sh --profile small --steps 10000 \
      --eval-data-file /data/heldout

The upstream PyTorch runtime is created automatically with uv under
.comparison/venvs/rwkv-lm-v7 using Python 3.12. Set INSTALL_UPSTREAM_DEPS=1 to
force a dependency resync, or UPSTREAM_VENV to change the venv location.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
profile="small"
seeds="42"
forward_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) profile="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) forward_args+=("$1"); shift ;;
  esac
done

case "$profile" in
  smoke)
    n_layer=4; d_model=256; d_ffn=896; ctx_len=128; steps=50
    d_slot=64; d_k=32; d_v=32; n_slots=8
    ;;
  small)
    n_layer=12; d_model=768; d_ffn=2688; ctx_len=512; steps=2000
    d_slot=128; d_k=64; d_v=64; n_slots=16
    ;;
  medium)
    n_layer=24; d_model=2048; d_ffn=7168; ctx_len=512; steps=2000
    d_slot=256; d_k=64; d_v=128; n_slots=16
    ;;
  large)
    n_layer=24; d_model=2560; d_ffn=8960; ctx_len=512; steps=2000
    d_slot=256; d_k=64; d_v=128; n_slots=16
    ;;
  *)
    echo "unknown profile: $profile" >&2
    usage >&2
    exit 2
    ;;
esac

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
for seed in $seeds; do
  run_id="${timestamp}-${profile}-seed${seed}"
  echo "Starting profile=$profile seed=$seed run_id=$run_id"
  env \
    RUN_ID="$run_id" \
    D_SLOT="${D_SLOT:-$d_slot}" \
    D_K="${D_K:-$d_k}" \
    D_V="${D_V:-$d_v}" \
    N_SLOTS="${N_SLOTS:-$n_slots}" \
    bash "$repo_root/comparison.sh" \
      --seed "$seed" \
      --ctx-len "$ctx_len" \
      --steps "$steps" \
      --n-layer "$n_layer" \
      --d-model "$d_model" \
      --d-ffn "$d_ffn" \
      "${forward_args[@]}"
done
