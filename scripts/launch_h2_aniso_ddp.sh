#!/usr/bin/env bash
# H2: T16k 70000 재개 + 이방성 항 + Hungarian scale, 쓸 만한 LR. GPUs 0,1,2 nproc=3.
#
#   bash scripts/launch_h2_aniso_ddp.sh --detached
#   tail -F runs/H2_*.log
#
# 왜 3-rank 인가: T16k 자체가 GPUs=0,1,2 nproc=3 으로 학습됐다. 단일 GPU 로 재개하면
# 유효 배치가 1/3 이 되어 학습이 바뀐다. 같은 world size 여야 재개가 재개다.
#
# 무엇을 바꾸는가 (T16k_20260913_093332 대비):
#   w_aniso                 없음 -> 5.0    새 항. 크기와 방향을 나눠낸 순수 모양.
#   w_attr_hung_scale       없음 -> 5.0    sinkhorn 평균화가 이방성을 지우는 것을 1:1 로 막는다.
#   w_attr_hung_opacity     없음 -> 5.0
#   w_attr_hung_rot         없음 -> 0.0    의도적. 아래 3번 참고.
#   max_steps             72000 -> 140000  LR 코사인만 늘린다. 손실 램프는 명시적 스텝이라 무관.
#
# 근거 (모두 T16k step 70000 에서 측정. mdmd/detail_bottleneck_20260917/ 의 스크립트들):
#   1. 위치는 이미 난간을 덮고 있다. 예측 위치에 GT 속성을 최근접으로 옮기면 핸드레일이
#      선으로 나온다 (swap.py). 실패는 속성이다.
#   2. 회전은 무감독이다. w_rot=0 이고, 유일한 경로인 covariance3d_loss 는 회전에 0.0% 만
#      반응한다 -- GT 회전을 줘도 손실 불변, GT 스케일을 주면 99.1% 감소 (covcheck.py).
#      정규화 오차가 목표 노름의 7.7배라 크기 불일치가 방향을 덮는다.
#   3. 손실 형태를 바꿔도 안 된다. trace 정규화해도 회전 민감도 1.6%. 이유는 예측이 구에
#      가깝다는 것이다: 이방성 p50 3.73 vs GT 15.51, 4배 미만이 52.4% vs GT 13.2%.
#      R S S^T R^T 는 S 가 구면이면 R 을 잃는다. 구를 돌려도 구다. 그래서 순서가 강제된다.
#      이방성을 먼저 세우고, 그 다음에 회전을 켠다.
#   4. 왜 여태 안 보였나: A[rot] 이 1-|dot| 이라 20.5도 오차를 0.016 으로 보고했고,
#      집합 매칭이 8배 봐줬다 (정직한 값 0.1288). 결과적으로 이방성 8배 초과 구간(전체
#      70%)에서 방향 오차 p50 41.5도, 그룹 평균 하나 쓰는 것(35.0도)보다 나쁘다 (rotcheck.py).
#   5. H1 이 아무것도 못 움직인 이유: LR 이 max_steps 기준 코사인이라 70000/72000 에서
#      이미 lr_min(1.04e-5) 이었다. max_steps=140000 이면 1.06e-4, 10.2배다.
#
# 게이트는 PSNR 이 아니다. 난간을 제대로 그리는 오라클 렌더가 base 보다 1.5dB 낮게 나온다.
#   1단계  covcheck.py : 예측 이방성 p50 3.73 -> GT 15.51 쪽, 4배 미만 52.4% -> 13.2% 쪽
#   2단계  rotcheck.py : 그 다음에 방향 오차 p50 41.5도 -> 35.0도 아래. 그때 hung_rot 을 켠다.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
PYTHON=${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}
SRC=${SRC:-runs/T16k_20260913_093332/args.json}
RESUME=${RESUME:-runs/T16k_20260913_093332/ckpt_step00070000.pt}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
MAX_STEPS=${MAX_STEPS:-140000}
W_ANISO=${W_ANISO:-5.0}
W_HUNG_SCALE=${W_HUNG_SCALE:-5.0}
W_HUNG_OPACITY=${W_HUNG_OPACITY:-5.0}
W_HUNG_ROT=${W_HUNG_ROT:-0.0}
TAG=${TAG:-H2_aniso_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29715}

# Hungarian 짝짓기 루프의 스레드 수. scipy 의 linear_sum_assignment 는 C++ 이라 GIL 을
# 놓으므로 그룹 1024개를 스레드로 나눌 수 있고, 배정 결과와 기울기는 비트 단위로 같다
# (verify_hung.py: 기울기 최대 절대차 0.000e+00). 8 이 최적이고 32 이상은 경합으로
# 오히려 느려진다 (176ms -> 248ms). rank 3개 x 8 = 24 스레드, 코어는 128개다.
export CAN3TOK_HUNG_WORKERS=${CAN3TOK_HUNG_WORKERS:-8}
export CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
    CAN3TOK_DETACHED=1 SRC="$SRC" RESUME="$RESUME" GPUS="$GPUS" NPROC="$NPROC" \
    CAN3TOK_HUNG_WORKERS="$CAN3TOK_HUNG_WORKERS" \
    MAX_STEPS="$MAX_STEPS" W_ANISO="$W_ANISO" W_HUNG_SCALE="$W_HUNG_SCALE" \
    W_HUNG_OPACITY="$W_HUNG_OPACITY" W_HUNG_ROT="$W_HUNG_ROT" \
    TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
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
if [[ ! -f "$RESUME" ]]; then
  echo "missing checkpoint: $RESUME" >&2
  exit 1
fi

# 아래 --set 중 compat 계약 부분은 R1/H1 이 쓴 것과 같다. T16k args.json 에 없는 키들이라
# 그냥 두면 argparse 기본값이 끼어들어 체크포인트가 학습된 조건과 달라진다.
mapfile -d '' ARGV < <(
  "$PYTHON" scripts/argv_from_argsjson.py --src "$SRC" \
    --set "out_dir=$OUT" \
    --set "resume=$RESUME" \
    --set "max_steps=$MAX_STEPS" \
    --set "save_every=500" \
    --set "eval_every=10000" \
    --set "log_every=50" \
    --set "shared_cell_owner=0" \
    --set "normalize_pooler_xyz=0" \
    --set "count_aware_template=0" \
    --set "attr_slot_mask=0" \
    --set "holdout_own_photo=0" \
    --set "keep_extra_fullres=0" \
    --set "reuse_prev_vis=0" \
    --set "eval_pred_mask=0" \
    --set "w_attr_spread=0.0" \
    --set "w_attr_slope=0.0" \
    --set "w_aniso=$W_ANISO" \
    --set "w_attr_hung_scale=$W_HUNG_SCALE" \
    --set "w_attr_hung_opacity=$W_HUNG_OPACITY" \
    --set "w_attr_hung_rot=$W_HUNG_ROT"
)

echo "launch H2 | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  resume $RESUME"
echo "  w_aniso=$W_ANISO hung[scale=$W_HUNG_SCALE opacity=$W_HUNG_OPACITY rot=$W_HUNG_ROT]"
echo "  max_steps=$MAX_STEPS -> step 70000 의 lr 은 1.06e-4 (H1 은 1.04e-5)"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  "${ARGV[@]}" \
  $EXTRA_ARGS
