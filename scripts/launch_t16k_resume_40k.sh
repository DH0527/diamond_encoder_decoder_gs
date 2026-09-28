#!/usr/bin/env bash
# Resume S16k from step 40000 with the three unused terms that actually
# target floaters / mushy railings. Same 16k layout (shape 3 / appearance 8),
# so the checkpoint loads. Do NOT change budget_shape here -- that needs a
# from-scratch run.
#
#   bash scripts/launch_t16k_resume_40k.sh --detached
#   tail -F runs/T16k_*.log
#
# Why 40000, not 62000:
#   detail_start=40000 is when VGG / edge_gain / full-res render turned on and
#   PSNR rose while floaters grew. Resume from the last clean envelope, turn
#   splat+spacing on immediately, anneal Sinkhorn from this step (not from 0,
#   or eps would jump to 0.005 on the first iteration), and delay the detail
#   phase 4k steps so large splats start shrinking first.
#
# What changes vs S16k_20260910_012456:
#   w_splat_area              0 -> 1.0
#   w_intra_spacing           0 -> 2.0
#   sinkhorn_epsilon_final    0 -> 0.005   (0 means "no anneal")
#   sinkhorn_anneal_start     * -> 40000
#   sinkhorn_anneal_steps     * -> 12000
#   detail_start          40000 -> 44000
#   score_metric           psnr -> psnr_gap_worst
#   max_steps             64000 -> 72000   (32k more steps from the resume)
#
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/S16k_20260910_012456/args.json}
RESUME=${RESUME:-runs/S16k_20260910_012456/ckpt_step00040000.pt}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-72000}
TAG=${TAG:-T16k_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29711}

export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 SRC="$SRC" RESUME="$RESUME" GPUS="$GPUS" NPROC="$NPROC" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
    MASTER_PORT="$MASTER_PORT" TORCHRUN="$TORCHRUN" PYTHON="$PYTHON" \
    nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"
  echo "Log: $LOG"
  echo "Out: $OUT"
  echo "Resume: $RESUME"
  exit 0
fi

mkdir -p "$OUT"
if [[ ! -f "$RESUME" ]]; then
  echo "missing checkpoint: $RESUME" >&2
  exit 1
fi

mapfile -d '' ARGV < <(
  "$PYTHON" scripts/argv_from_argsjson.py --src "$SRC" \
    --set "out_dir=$OUT" \
    --set "resume=$RESUME" \
    --set "max_steps=$MAX_STEPS" \
    --set "w_splat_area=1.0" \
    --set "w_intra_spacing=2.0" \
    --set "sinkhorn_epsilon_final=0.005" \
    --set "sinkhorn_anneal_start=40000" \
    --set "sinkhorn_anneal_steps=12000" \
    --set "detail_start=44000" \
    --set "score_metric=psnr_gap_worst"
)

echo "launch T16k | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  resume $RESUME"
echo "  splat=1.0 spacing=2.0 sinkhorn 0.08->0.005 from 40000 over 12k"
echo "  detail_start=44000 score=psnr_gap_worst max_steps=$MAX_STEPS"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}" \
  $EXTRA_ARGS
