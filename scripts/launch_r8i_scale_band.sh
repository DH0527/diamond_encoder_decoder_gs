#!/usr/bin/env bash
# R8I: stop R8H's giant-splat compensation.
#
# codec_own on R8H is fat and blurry because appearance covers geometry holes
# with huge opaque Gaussians. Three stacked causes, all measured:
#   1. attr_scale_cap_down/up were 0, so the band was a GLOBAL ±3 (~20x) around
#      bias -7.58. The group-relative asymmetric band was already implemented
#      but never turned on.
#   2. w_splat_area=2 logged splat=0.00000 every step: the hinge compared against
#      GT p99, which is the legitimate giant tail, so hole-covering blobs never
#      paid.
#   3. attr_param_floor=0.15 let scale/opacity L1 decay away once render was live.
#
# Warm-start R8H ckpt_best (step 9500, 16.57 dB photo). --init_from, not --resume:
# the scale-head bias would keep its absolute-log meaning under the new formula.
# Step resets to 0, so attr/render gates must be opened (HISTORY trap #9).
# Do not raise w_render_attr (R9's 40 was the wrong lesson). Same latent budget.
#
#   bash scripts/launch_r8i_scale_band.sh --detached
set -euo pipefail
cd "$(dirname "$0")/.."

# fix_4 starts from a checkpoint produced in fix_3, and the caller always passes
# its own --init_from in EXTRA_ARGS, which overrides the one built here. The
# existence check therefore has to accept an override rather than insisting on a
# file this tree never had. R8H_CKPT=skip disables it.
R8H_CKPT="${R8H_CKPT:-runs/R8H_20260817_154941/ckpt_best.pt}"
if [ "$R8H_CKPT" = "skip" ]; then R8H_CKPT=""; fi
if [ -n "$R8H_CKPT" ] && [ ! -f "$R8H_CKPT" ]; then
  echo "missing R8H checkpoint: $R8H_CKPT" >&2
  exit 1
fi

export TAG="${TAG:-R8I_$(date +%Y%m%d_%H%M%S)}"
export MAX_STEPS="${MAX_STEPS:-6000}"
export MASTER_PORT="${MASTER_PORT:-29522}"
export GPUS="${GPUS:-0,1,2}"
export NPROC="${NPROC:-3}"
USER_EXTRA="${EXTRA_ARGS:-}"

R8I_ARGS="\
 ${R8H_CKPT:+--init_from $R8H_CKPT} \
 --init_skip attr_decoder.head_scale \
 --attr_scale_cap_down 2.5 --attr_scale_cap_up 1.5 \
 --w_splat_area 4.0 --splat_area_quantile 0.80 \
 --attr_param_floor 0.50 --attr_param_decay_steps 1000 \
 --attr_start 0 --attr_force_steps 0 --attr_anneal_steps 1 \
 --render_start 0 --render_ramp_steps 400 \
 --render_downscale_start 4 --render_downscale 2 --render_downscale_steps 1500 \
 --geo_end 0 \
 --attr_detach_geometry 1 --attr_detach_release -1 \
 --w_render_attr 20.0 \
 --polish_start 1 \
 --polish_encoder_scale 0.02 --polish_attr_encoder_scale 0.50 \
 --polish_compressor_scale 0.05 --polish_decompressor_scale 0.05 \
 --polish_decoder_scale 0.10 --polish_attr_scale 1.0 \
 --lr 5e-5 --lr_min 5e-6 --lr_warmup_steps 200 \
 --eval_milestones 100,250,500,1000,1500,2000,3000,4000,6000"

export EXTRA_ARGS="$R8I_ARGS $USER_EXTRA"
exec bash scripts/launch_r8h_ddp.sh "${1:---detached}"
