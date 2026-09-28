#!/usr/bin/env bash
# L1 recipe from scratch on vanilla 3DGS npz (train+truck), not Speedy-Splat.
# Speedy checkpoints are not resumed: density / stats / anchors differ.
#
#   bash scripts/launch_c1_l1_vanilla_80k.sh --detached
#
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/C1_l1_20260918_150236/args.json}
GPUS=${GPUS:-1,2,3}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-80000}
W_SOBEL=${W_SOBEL:-1.0}
W_LOCAL=${W_LOCAL:-3.0}
K_LOCAL=${K_LOCAL:-16}
TAG=${TAG:-C1_l1_vanilla_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
MEAS=${MEAS:-mdmd/c1_ablation_20260917}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29723}

ROOT_VANILLA="/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/train/replay,/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/truck/replay"

export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 SRC="$SRC" GPUS="$GPUS" NPROC="$NPROC" \
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
for p in \
  assets/split_vanilla_both_blockB.json \
  assets/stats_vanilla_train.json \
  assets/stats_vanilla_truck.json \
  assets/anchors_vanilla_train_n1024.npy \
  assets/anchors_vanilla_truck_n1024.npy \
  assets/npz_to_image_vanilla_both.json \
  assets/view_pool_vanilla_both.json
do
  if [[ ! -f "$p" ]]; then
    echo "missing vanilla asset: $p" >&2
    echo "run: bash scripts/build_vanilla_c1_assets.sh" >&2
    exit 1
  fi
done

mapfile -d '' ARGV < <(
  "$PYTHON" scripts/argv_from_argsjson.py --src "$SRC" \
    --set "out_dir=$OUT" \
    --set "resume=" \
    --set "init_from=" \
    --set "root=$ROOT_VANILLA" \
    --set "split_path=assets/split_vanilla_both_blockB.json" \
    --set "stats_path=assets/stats_vanilla_train.json,assets/stats_vanilla_truck.json" \
    --set "scene_anchors=assets/anchors_vanilla_train_n1024.npy,assets/anchors_vanilla_truck_n1024.npy" \
    --set "photo_map=assets/npz_to_image_vanilla_both.json" \
    --set "view_pool=assets/view_pool_vanilla_both.json" \
    --set "max_steps=$MAX_STEPS" \
    --set "stop_at=0" \
    --set "w_sobel=$W_SOBEL" \
    --set "w_attr_local_set=$W_LOCAL" \
    --set "attr_local_set_k=$K_LOCAL" \
    --set "save_every=1000" \
    --set "eval_every=1000" \
    --set "log_every=50" \
    --set "eval_milestones=500,1000,2000,4000,8000,12000,16000,20000,24000,28000,32000,36000,38000,40000,41000,42000,43000,44000,48000,52000,56000,60000,64000,68000,72000,76000,80000"
)

echo "launch C1 L1 vanilla-3DGS from scratch | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  root $ROOT_VANILLA"
echo "  w_attr_local_set=$W_LOCAL k=$K_LOCAL w_sobel=$W_SOBEL"
echo "  max_steps=$MAX_STEPS (no Speedy resume)"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}" \
  $EXTRA_ARGS
train_rc=$?

if [[ -f "$OUT/ckpt_step00080000.pt" ]]; then
  echo "measuring step 80000 -> $MEAS/c1_l1_vanilla_80000"
  REPO="$PWD" CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" \
    mdmd/c1_ablation_20260917/measure_c1_gate.py \
    --ckpt "$OUT/ckpt_step00080000.pt" \
    --out "$MEAS/c1_l1_vanilla_80000" || true
fi
exit $train_rc
