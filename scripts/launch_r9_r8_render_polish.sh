#!/usr/bin/env bash
# R9: continue R8 with render/attribute-focused multi-view fine-tuning.
set -euo pipefail
cd "$(dirname "$0")/.."

R8_CKPT="${R8_CKPT:-runs/R8_anchor_ot_render_20260816_165127/ckpt_latest.pt}"
if [[ ! -f "$R8_CKPT" ]]; then
  echo "missing R8 checkpoint: $R8_CKPT" >&2
  exit 1
fi

OUT="${CAN3TOK_OUT:-runs/R9_r8_render_polish_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-16000}"
USER_EXTRA="${EXTRA_ARGS:-}"

# Resume at global step 10000.  Re-enter the render curriculum at coarse 8x:
# attributes get a clean 1000-step window, then photo gradients may make small
# geometry corrections.  Point-space losses stay as guards, not the objective.
R9_ARGS="\
 --resume $R8_CKPT \
 --polish_start 10000 --gen_end 10000 \
 --polish_encoder_scale 0.02 --polish_attr_encoder_scale 0.20 \
 --polish_compressor_scale 0.05 --polish_decompressor_scale 0.05 \
 --polish_decoder_scale 0.20 --polish_attr_scale 1.0 \
 --w_chamfer 1.0 --w_coverage 0.5 --w_intra_chamfer 2.0 \
 --w_intra_sinkhorn 2.0 --w_group_centroid 1.0 --w_shape_sinkhorn 2.0 \
 --w_latent_std 0.0 --w_latent_decorr 0.0 --w_equiv 0.0 \
 --w_render_attr 40.0 --render_start 10000 --render_ramp_steps 1000 \
 --render_downscale_start 8 --render_downscale 2 --render_downscale_steps 2000 \
 --attr_detach_geometry 1 --attr_detach_release 11000 \
 --attr_param_decay_steps 1 --attr_param_floor 0.05 \
 --attr_anchor_covered 0.10 \
 --lr 3e-5 --lr_min 3e-6 --lr_warmup_steps 1"

RUN_LABEL="R9 R8 render polish" \
  CURRICULUM_LABEL="curriculum: resume@10000 render-ramp@10000 geometry-release@11000 end@${MAX_STEPS}" \
  CAN3TOK_OUT="$OUT" \
  MAX_STEPS="$MAX_STEPS" \
  EXTRA_ARGS="$R9_ARGS $USER_EXTRA" \
  exec bash scripts/launch_r8_anchor_ot_render.sh "${1:---foreground}"
