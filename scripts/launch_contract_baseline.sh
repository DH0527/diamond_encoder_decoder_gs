#!/usr/bin/env bash
# Contract baseline: same 16k layout as S16k (cen 4 | occ 1 | shape 3 | app 8),
# from-scratch, with the A/B/C audit contracts on. Does not change unpack or
# use_fixed_anchor_center. Do not confuse with E1 (schedule-only).
#
#   bash scripts/launch_contract_baseline.sh --detached
#   tail -F runs/C0_*.log
#
# This script is the launch path. It is not started by the implementation
# session that added the contracts.
#
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/S16k_20260910_012456/args.json}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-16000}
TAG=${TAG:-C0_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29731}

export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 SRC="$SRC" GPUS="$GPUS" NPROC="$NPROC" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
    MASTER_PORT="$MASTER_PORT" TORCHRUN="$TORCHRUN" PYTHON="$PYTHON" \
    nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"
  echo "Log: $LOG"
  echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT"
if [[ ! -f "$SRC" ]]; then
  echo "missing args.json: $SRC" >&2
  exit 1
fi

mapfile -d '' ARGV < <(
  "$PYTHON" scripts/argv_from_argsjson.py --src "$SRC" \
    --set "out_dir=$OUT" \
    --set "resume=" \
    --set "max_steps=$MAX_STEPS" \
    --set "normalize_pooler_xyz=1" \
    --set "count_aware_template=1" \
    --set "attr_slot_mask=1" \
    --set "shared_cell_owner=1" \
    --set "holdout_own_photo=1" \
    --set "keep_extra_fullres=1" \
    --set "eval_pred_mask=1" \
    --set "reuse_prev_vis=0" \
    --set "use_fixed_anchor_center=0"
)

echo "launch contract baseline | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  src $SRC  (from-scratch, no resume)"
echo "  layout unchanged: 16x32x32, group 256, shape 3, appearance 8"
echo "  A+B+C contracts on; unpack / fixed-anchor-center unchanged"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}" \
  $EXTRA_ARGS
