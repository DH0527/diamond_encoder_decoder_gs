#!/usr/bin/env bash
# L1 step 56000 재개: 이웃 attr_set + w_sobel=1, max_steps=80000.
#
#   bash scripts/launch_c1_l1_80k.sh --detached
#
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/C1_l1_20260918_150236/args.json}
RESUME=${RESUME:-runs/C1_l1_20260918_150236/ckpt_step00056000.pt}
GPUS=${GPUS:-1,2,3}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-80000}
W_SOBEL=${W_SOBEL:-1.0}
W_LOCAL=${W_LOCAL:-3.0}
K_LOCAL=${K_LOCAL:-16}
TAG=${TAG:-C1_l1_80k_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
MEAS=${MEAS:-mdmd/c1_ablation_20260917}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29722}

export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 SRC="$SRC" RESUME="$RESUME" GPUS="$GPUS" NPROC="$NPROC" \
    MAX_STEPS="$MAX_STEPS" W_SOBEL="$W_SOBEL" W_LOCAL="$W_LOCAL" K_LOCAL="$K_LOCAL" \
    TAG="$TAG" OUT="$OUT" LOG="$LOG" MEAS="$MEAS" EXTRA_ARGS="$EXTRA_ARGS" \
    MASTER_PORT="$MASTER_PORT" TORCHRUN="$TORCHRUN" PYTHON="$PYTHON" \
    nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"
  echo "Log: $LOG"
  echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT" "$MEAS"
if [[ ! -f "$RESUME" ]]; then
  echo "missing checkpoint: $RESUME" >&2
  exit 1
fi

mapfile -d '' ARGV < <(
  "$PYTHON" scripts/argv_from_argsjson.py --src "$SRC" \
    --set "out_dir=$OUT" \
    --set "resume=$RESUME" \
    --set "max_steps=$MAX_STEPS" \
    --set "stop_at=0" \
    --set "w_sobel=$W_SOBEL" \
    --set "w_attr_local_set=$W_LOCAL" \
    --set "attr_local_set_k=$K_LOCAL" \
    --set "save_every=1000" \
    --set "eval_every=1000" \
    --set "log_every=50"
)

echo "launch C1 L1 56000->80000 | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  resume $RESUME"
echo "  w_attr_local_set=$W_LOCAL k=$K_LOCAL w_sobel=$W_SOBEL"
echo "  max_steps=$MAX_STEPS stop_at=0"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}" \
  $EXTRA_ARGS
train_rc=$?

if [[ -f "$OUT/ckpt_step00080000.pt" ]]; then
  echo "measuring step 80000 -> $MEAS/c1_l1_80000"
  REPO="$PWD" CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" \
    mdmd/c1_ablation_20260917/measure_c1_gate.py \
    --ckpt "$OUT/ckpt_step00080000.pt" \
    --out "$MEAS/c1_l1_80000" || true
fi
exit $train_rc
