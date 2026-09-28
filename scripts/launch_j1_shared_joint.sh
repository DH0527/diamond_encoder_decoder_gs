#!/usr/bin/env bash
# J1: R8-preserving shared-27 joint Gaussian refiner.
#
# This is intentionally launched with --init_from (not --resume): the R8 network
# is loaded as a function-preserving base and the zero-output joint refiner is the
# only newly initialised module.  xyz and attributes use one balanced OT plan;
# the old many-to-one scale-growing responsibility target is disabled.
set -euo pipefail
cd "$(dirname "$0")/.."

R8_CKPT="${R8_CKPT:-/data/daeho/aacd_proj/can3tok_encoder_decoder_new_fix/runs/R8_anchor_ot_render_20260816_165127/ckpt_latest.pt}"
if [[ ! -f "$R8_CKPT" ]]; then
  echo "missing R8 checkpoint: $R8_CKPT" >&2
  exit 1
fi

OUT="${CAN3TOK_OUT:-runs/J1_shared27_ot_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-6000}"
USER_EXTRA="${EXTRA_ARGS:-}"

J1_ARGS="\
 --init_from $R8_CKPT \
 --joint_shared_decoder 1 --joint_decoder_dim 192 --joint_decoder_layers 2 \
 --joint_decoder_chunk 256 --joint_nbr_window 1 \
 --joint_xyz_cap 0.10 --joint_scale_delta_cap 1.0 \
 --joint_opacity_delta_cap 2.0 --joint_color_delta_cap 0.5 \
 --joint_rot_delta_cap 0.25 --joint_detach_render_xyz 1 \
 --attr_read_shape 0 --attr_match_mode sinkhorn --attr_responsibility 0 \
 --attr_start 0 --attr_force_steps 0 --attr_anneal_steps 1 \
 --render_start 0 --render_ramp_steps 500 --render_downscale_start 0 \
 --decoder_refine_start 0 --decoder_refine_ramp_steps 1 \
 --folding_res_start 0 --folding_res_ramp_steps 1 \
 --geo_end 1 --latent_end 0 --late_decode_steps 0 \
 --w_chamfer 2.0 --w_coverage 1.0 --w_intra_chamfer 4.0 \
 --w_intra_sinkhorn 4.0 --w_group_centroid 2.0 --w_shape_sinkhorn 4.0 \
 --w_scale 1.0 --w_rot 0.5 --w_opacity 1.0 --w_color 1.0 \
 --attr_param_decay_steps 1 --attr_param_floor 0.50 --attr_anchor_covered 1.0 \
 --w_render_attr 20.0 --w_splat_area 2.0 \
 --attr_detach_geometry 1 --attr_detach_release -1 \
 --w_latent_std 0.5 --latent_std_floor 0.35 --w_latent_decorr 0.05 \
 --w_equiv 0.0 \
 --polish_start 1 --polish_encoder_scale 0.10 --polish_attr_encoder_scale 0.20 \
 --polish_compressor_scale 0.20 --polish_decompressor_scale 0.20 \
 --polish_decoder_scale 0.50 --polish_attr_scale 1.0 \
 --lr 5e-5 --lr_min 5e-6 --lr_warmup_steps 200 \
 --max_steps $MAX_STEPS --eval_every 500 \
 --eval_milestones 0,100,250,500,1000,1500,2000,3000,4000,5000,6000"

RUN_LABEL="${RUN_LABEL:-J1 shared-27 joint Gaussian decoder}" \
  CURRICULUM_LABEL="${CURRICULUM_LABEL:-R8-preserving init | inference-conditioned attrs | balanced OT | 5-view render}" \
  CAN3TOK_OUT="$OUT" MAX_STEPS="$MAX_STEPS" EXTRA_ARGS="$J1_ARGS $USER_EXTRA" \
  exec bash scripts/launch_r8_anchor_ot_render.sh "${1:---foreground}"
