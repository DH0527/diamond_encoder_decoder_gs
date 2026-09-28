#!/usr/bin/env bash
# C1 step 40000 재개 + w_sobel=1, 2000스텝만. LR 코사인은 max_steps=64000 그대로.
#
#   bash scripts/launch_c1_sobel_42k.sh --detached
#
# 왜 새 out_dir 인가: C1 원본 train.log / eval 을 덮지 않는다.
# 왜 stop_at=42000 인가: max_steps 를 줄이면 코사인이 점프한다.
# 왜 3-rank 인가: C1 이 GPUs=0,1,2 nproc=3 으로 학습됐다.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/C1_detail_20260916_021829/args.json}
RESUME=${RESUME:-runs/C1_detail_20260916_021829/ckpt_step00040000.pt}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-64000}
STOP_AT=${STOP_AT:-42000}
W_SOBEL=${W_SOBEL:-1.0}
TAG=${TAG:-C1_sobel_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
MEAS=${MEAS:-mdmd/c1_ablation_20260917}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29717}

export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 SRC="$SRC" RESUME="$RESUME" GPUS="$GPUS" NPROC="$NPROC" \
    MAX_STEPS="$MAX_STEPS" STOP_AT="$STOP_AT" W_SOBEL="$W_SOBEL" \
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
    --set "stop_at=$STOP_AT" \
    --set "w_sobel=$W_SOBEL" \
    --set "save_every=1000" \
    --set "eval_every=1000" \
    --set "log_every=50"
)

echo "launch C1+sobel | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  resume $RESUME"
echo "  w_sobel=$W_SOBEL stop_at=$STOP_AT max_steps=$MAX_STEPS (LR unchanged)"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}" \
  $EXTRA_ARGS
train_rc=$?

if [[ -f "$OUT/ckpt_step00042000.pt" ]]; then
  echo "measuring step 42000 -> $MEAS/c1_sobel_42000"
  REPO="$PWD" CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" \
    mdmd/c1_ablation_20260917/measure_c1_gate.py \
    --ckpt "$OUT/ckpt_step00042000.pt" \
    --out "$MEAS/c1_sobel_42000" || true
fi
exit $train_rc
