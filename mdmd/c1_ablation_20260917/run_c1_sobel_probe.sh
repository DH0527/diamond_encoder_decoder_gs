#!/usr/bin/env bash
# C1 40000 에서 T16k 와 같은 Sobel 프로브를 다시 돌린다.
# 이전 c1hf 는 holdout_own_photo=1 때문에 MAIN 시점이 런마다 달라 비교가 무효였다.
# ctl_s 는 데이터 로딩 중 끊겼다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/mdmd/c1_ablation_20260917"
PY="${PYTHON:-/home/super/anaconda3/envs/can3tok/bin/python}"
CKPT="${CKPT:-runs/C1_detail_20260916_021829/ckpt_step00040000.pt}"
PROBE="$ROOT/mdmd/detail_bottleneck_20260917/viewgen_mv.py"
mkdir -p "$OUT"

wave="${1:-hf}"   # hf | ctl
if [[ "$wave" == "hf" ]]; then
  # T16k 와 같은 손실: 평범한 L1 + Sobel. w_sobel = 0, 1, 5
  edges=(0 1 5); prefix=c1_hf; real=0
else
  # C1 학습 손실 위: edge_gain=4, DSSIM 0.2, 그 위에 w_sobel = 0, 1, 2
  edges=(0 1 2); prefix=c1_ctl; real=1
fi

echo "wave=$wave real_loss=$real out=$OUT"
for i in 0 1 2; do
  e="${edges[$i]}"
  tag="${prefix}_e${e}"
  echo "GPU $i -> $tag w_sobel=$e"
  REPO="$ROOT" CUDA_VISIBLE_DEVICES="$i" "$PY" "$PROBE" \
    --ckpt "$CKPT" --scope attrd --views 20 --fit 14 --per_step 3 --iters 600 \
    --real_loss "$real" --edge_gain 4.0 --lam_dssim 0.2 --edge "$e" \
    --view_seed 0 --out "$OUT/$tag" \
    >"$OUT/${tag}.log" 2>&1 &
  echo $! >"$OUT/${tag}.pid"
done
wait
echo "wave $wave done"
for e in "${edges[@]}"; do
  echo "===== ${prefix}_e${e} ====="
  rg -o "^\[base.*|^\[attrd\] *600/600.*|^base .*|^attrd .*" "$OUT/${prefix}_e${e}.log" || true
done
