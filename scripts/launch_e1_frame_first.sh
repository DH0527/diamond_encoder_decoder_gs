#!/usr/bin/env bash
# E1: same 16k layout as S16k/T16k (shape 3 / appearance 8, 1024 x 256),
# from-scratch, but the folding residual and decoder refine stay closed until
# step 8000 so the 6-DoF frame head can own the envelope.
#
#   bash scripts/launch_e1_frame_first.sh --detached
#   tail -F runs/E1_*.log
#
# Why this run exists:
#   T16k kept appearance 8 and opened folding_res_start=0 / refine_start=400.
#   The residual then outruns the frame (compressor.py comments, measured on an
#   earlier attempt). E1 changes only that schedule. Layout, appearance, and
#   point count stay put. Gate at eval 8000: thin structure / shape-channel std,
#   not pooled PSNR. VGG/detail never turns on in this 16k-step confirmation
#   (detail_start=20000).
#
# vs S16k_20260910_012456:
#   resume                     (none)     from-scratch
#   folding_res_start          0       -> 8000
#   decoder_refine_start       400     -> 8000
#   detail_start               40000   -> 20000  (past max_steps)
#   score_metric               psnr    -> psnr_gap_worst
#   max_steps                  64000   -> 16000  (gate + short residual look)
#
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/S16k_20260910_012456/args.json}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-16000}
TAG=${TAG:-E1_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29721}

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
    --set "folding_res_start=8000" \
    --set "folding_res_ramp_steps=400" \
    --set "decoder_refine_start=8000" \
    --set "decoder_refine_ramp_steps=800" \
    --set "detail_start=20000" \
    --set "score_metric=psnr_gap_worst" \
    --set "eval_milestones=500,1000,2000,4000,6000,8000,10000,12000,14000,16000" \
    --set "normalize_pooler_xyz=0" \
    --set "count_aware_template=0" \
    --set "attr_slot_mask=0" \
    --set "shared_cell_owner=0" \
    --set "holdout_own_photo=0" \
    --set "keep_extra_fullres=0" \
    --set "eval_pred_mask=0" \
    --set "reuse_prev_vis=0"
)

echo "launch E1 frame-first | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  src $SRC  (from-scratch, no resume)"
echo "  layout unchanged: 16x32x32, group 256, shape 3, appearance 8"
echo "  folding_res_start=8000 refine_start=8000 detail_start=20000"
echo "  score=psnr_gap_worst max_steps=$MAX_STEPS"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}" \
  $EXTRA_ARGS
