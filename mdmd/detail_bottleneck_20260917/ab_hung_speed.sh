#!/usr/bin/env bash
# Hungarian 이 스텝 시간에 실제로 얼마를 더하는가. GPU 3, 단일 rank, 40 스텝씩.
#
# 유휴 GPU 에서 함수만 재면 706ms 인데 학습에서는 T16k 4.68s -> H2 9.4s 로 4.7초가
# 늘었다. 차이가 7배라 함수 비용만으로는 설명되지 않으므로, 같은 코드에서 가중치만
# 껐다 켜서 in-situ 증분을 직접 잰다. 세 조건:
#   off     : w_attr_hung_* = 0            기준선
#   w1      : 켜고 스레드 1개              고친 전송 + 예전 병렬도
#   w8      : 켜고 스레드 8개              지금 기본값
set -u
cd "$(dirname "$0")/../.."
OUTDIR=mdmd/detail_bottleneck_20260917
PY=/home/super/anaconda3/envs/can3tok/bin/python

run() {
  local tag=$1 hung=$2 workers=$3
  local out=runs/ABspd_${tag}
  rm -rf "$out"
  GPUS=3 NPROC=1 MAX_STEPS=140000 TAG="ABspd_${tag}" OUT="$out" LOG="$OUTDIR/abspd_${tag}.log" \
    W_HUNG_SCALE="$hung" W_HUNG_OPACITY="$hung" MASTER_PORT=29731 \
    CAN3TOK_HUNG_WORKERS="$workers" \
    EXTRA_ARGS="--patch_chunk 64 --log_every 10 --save_every 100000 --eval_every 100000" \
    timeout 420 bash scripts/launch_h2_aniso_ddp.sh > "$OUTDIR/abspd_${tag}.log" 2>&1
  local sit
  sit=$(rg -o "[0-9.]+s/it" "$OUTDIR/abspd_${tag}.log" 2>/dev/null | tail -3 | tr -d 's/it' | \
        awk '{s+=$1; n++} END {if (n) printf "%.2f", s/n; else print "n/a"}')
  echo "  ${tag}: hung=${hung} workers=${workers}  ->  ${sit} s/it"
}

echo "[A/B] GPU 3, 단일 rank, 마지막 3개 로그의 평균 s/it"
# w1(스레드 1개) 은 함수 단위로 이미 재어 두었다: 706ms -> 179ms, 값과 기울기 동일.
# 여기서 궁금한 것은 in-situ 증분이므로 두 조건이면 된다.
run off 0.0 8
run w8  5.0 8
