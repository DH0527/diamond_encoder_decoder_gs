#!/usr/bin/env bash
# Resume B1 from step 30000, before the grad-norm blow-up at ~34100.
# Same vanilla / bgcap config. Two guards are on by default and set here
# explicitly: skip a step whose pre-clip grad norm is >8x its running median,
# and detach render gradients of points closer than 0.1x the view median depth.
#
#   bash scripts/launch_b1g_resume_30k.sh --detached
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/B1_bgcap_20260922_024641/args.json}
RESUME=${RESUME:-runs/B1_bgcap_20260922_024641/ckpt_step00030000.pt}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-80000}
TAG=${TAG:-B1g_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
MASTER_PORT=${MASTER_PORT:-29521}

export PYTHONUNBUFFERED=1
export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 SRC="$SRC" RESUME="$RESUME" GPUS="$GPUS" NPROC="$NPROC" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" \
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
    --set "grad_spike_mult=3" \
    --set "grad_spike_abs=6000" \
    --set "near_detach_frac=0.1" \
    --set "w_splat_area=1.0"
)

echo "launch B1g | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  resume $RESUME"
echo "  w_splat_area=1 grad_spike_mult=3 grad_spike_abs=6000 near_detach_frac=0.1 max_steps=$MAX_STEPS"

CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}"
