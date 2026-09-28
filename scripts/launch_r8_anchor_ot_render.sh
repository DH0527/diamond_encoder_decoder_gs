#!/usr/bin/env bash
# R8: repair R7's fixed-anchor geometry path.
# Reuses the fully specified R7 launcher and overrides only the diagnosed faults;
# argparse resolves repeated options by their final value.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${CAN3TOK_OUT:-runs/R8_anchor_ot_render_$(date +%Y%m%d_%H%M%S)}"
USER_EXTRA="${EXTRA_ARGS:-}"

# - Open the folding residual immediately, while keeping decoder refinement late
#   enough that it cannot hide an unused compact shape code.
# - Use balanced local transport on both the direct shape path and final output.
# - Release the render gradient to geometry only after coarse geometry converges.
R8_ARGS="\
 --folding_res_start 0 --folding_res_ramp_steps 400 \
 --decoder_refine_start 400 --decoder_refine_ramp_steps 800 \
 --geo_end 1200 --geo_weak_scale 0.35 --detail_ramp_steps 100 \
 --w_chamfer 5.0 --w_coverage 2.0 --w_intra_chamfer 6.0 \
 --w_intra_sinkhorn 8.0 --w_group_centroid 4.0 \
 --w_shape_sinkhorn 12.0 --sinkhorn_epsilon 0.08 \
 --sinkhorn_iterations 6 --sinkhorn_chunk 256 \
 --attr_detach_geometry 1 --attr_detach_release 4000 \
 --w_latent_std 1.0 --latent_std_floor 0.35 --w_latent_decorr 0.05 \
 --w_equiv 0.5 --equiv_start 1200 --equiv_ramp_steps 800"

RUN_LABEL="${RUN_LABEL:-R8 anchor OT/render}" CURRICULUM_LABEL="${CURRICULUM_LABEL:-}" \
  CAN3TOK_OUT="$OUT" EXTRA_ARGS="$R8_ARGS $USER_EXTRA" \
  exec bash scripts/launch_r7_geom_render_fix.sh "${1:---foreground}"
