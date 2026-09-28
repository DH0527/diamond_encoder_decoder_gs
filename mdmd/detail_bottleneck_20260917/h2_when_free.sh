#!/usr/bin/env bash
# GPU 3 이 비면 H2 를 올린다. 학습 설정은 하나도 바꾸지 않는다.
#
# 지금 막힌 이유는 성능이 아니라 자리다. GPU 3 은 32760MiB 인데 다른 사용자의 평가
# 루프가 6.1GB 를 잡고 있어 25.4GB 만 남고, 이 학습은 25.8GB 에서 정점을 찍는다.
# expandable_segments 와 patch_chunk 192->64 로 400~500MB 까지 좁혔지만 넘지 못했다.
# 남의 작업은 죽이지 않고, 시점 수처럼 학습 신호를 바꾸는 knob 도 손대지 않는다.
#
# 멈추려면:  pkill -f h2_when_free.sh
set -u
REPO=/data/daeho/aacd_proj/can3tok_encoder_decoder_new_fix_7
LOG=$REPO/mdmd/detail_bottleneck_20260917/h2_watch.log
NEED=27000          # MiB. 25.8GB 정점에 여유 1GB.
GIVE_UP=$((12*60))  # 분. 이만큼 기다려도 안 비면 포기하고 기록만 남긴다.
cd "$REPO" || exit 1

echo "[watch] $(date +%F_%T) 시작. GPU3 에 ${NEED}MiB 이상 비기를 기다린다" >> "$LOG"
for ((i=0; i<GIVE_UP*30; i++)); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 3)
  free=$((total - used))
  if (( free >= NEED )); then
    echo "[watch] $(date +%F_%T) ${free}MiB 확보. H2 실행" >> "$LOG"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=3 \
      nohup /home/super/anaconda3/envs/can3tok/bin/python \
      mdmd/detail_bottleneck_20260917/launch_h2_aniso.py \
      >> mdmd/detail_bottleneck_20260917/h2_launch.log 2>&1 &
    echo "[watch] pid $! . 진행은 mdmd/detail_bottleneck_20260917/h2_launch.log" >> "$LOG"
    exit 0
  fi
  (( i % 30 == 0 )) && echo "[watch] $(date +%F_%T) free ${free}MiB, 대기" >> "$LOG"
  sleep 2
done
echo "[watch] $(date +%F_%T) ${GIVE_UP}분 동안 자리가 나지 않았다. 포기" >> "$LOG"
