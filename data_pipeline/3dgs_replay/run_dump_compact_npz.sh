#!/usr/bin/env bash
# Vanilla 3DGS 를 학습하면서 10 iter 마다 compact float16 zip-npz 를 떨어뜨린다.
# 씬당 30,000 iter -> step_000010.npz ... step_030000.npz (3,000 개).
# SH-DC 와 reward 는 넣고, SH-rest 와 action delta 는 뺀다 (디스크).
#
# 서버에서 실제로 쓴 원본: /data/daeho/aaaa_proj/seondo/vanilla-3dgs/run_dump_compact_npz.sh
# 여기서는 경로만 환경변수로 뺐다. 기본값은 원래 서버 경로.
#
#   GS_ROOT=/path/to/gaussian-splatting PY=/path/to/envs/3dgs/bin/python \
#   OUT_ROOT=/path/to/vanilla-3dgs/output \
#   TRAIN_SRC=/path/to/train_colmap TRUCK_SRC=/path/to/truck_colmap \
#   bash run_dump_compact_npz.sh
set -euo pipefail

GS_ROOT="${GS_ROOT:-/data/daeho/gaussian-splatting}"
PY="${PY:-/home/super/anaconda3/envs/3dgs/bin/python}"
OUT_ROOT="${OUT_ROOT:-/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output}"
TRAIN_SRC="${TRAIN_SRC:-/data/daeho/train_colmap}"
TRUCK_SRC="${TRUCK_SRC:-/data/daeho/truck_colmap}"
mkdir -p "${OUT_ROOT}"

run_one() {
  local gpu="$1" scene="$2" source="$3"
  local model_path="${OUT_ROOT}/${scene}"
  local log_path="${OUT_ROOT}/${scene}.log"
  mkdir -p "${model_path}"
  echo "GPU ${gpu}  scene=${scene}  source=${source}  out=${model_path}"
  (
    cd "${GS_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    exec "${PY}" -u train.py \
      -s "${source}" \
      -m "${model_path}" \
      --eval \
      --iterations 30000 \
      --log_every 10 \
      --compact_npz \
      --disable_viewer \
      --test_iterations 7000 30000 \
      --save_iterations 7000 30000
  ) > "${log_path}" 2>&1 &
  echo $! > "${model_path}/train.pid"
  echo "started pid=$(cat "${model_path}/train.pid") log=${log_path}"
}

run_one 0 train "${TRAIN_SRC}"
run_one 1 truck "${TRUCK_SRC}"
echo "launched train@GPU0 and truck@GPU1  -> ${OUT_ROOT}/{train,truck}/replay/step_*.npz"
