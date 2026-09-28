#!/usr/bin/env bash
# F3D: preserve four local codes through the compact bottleneck itself.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${CAN3TOK_OUT:-runs/F3D_structured_local_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-10000}"
USER_EXTRA="${EXTRA_ARGS:-}"

F3_ARGS="\
 --min_snapshot_step 12000 \
 --anchor_assignment spill --anchor_spill 8 \
 --use_fixed_anchor_center 1 \
 --pool_queries 16 --attr_pack_hidden 128 \
 --compress_merge_stages 1 --compress_head_stages 256 128 \
 --shared_free_head 1 \
 --structured_local_code 1 \
 --compact_global_dim 3 --compact_local_tokens 4 --compact_local_dim 6 \
 --joint_shared_decoder 0 --joint_direct_decoder 1 \
 --joint_local_memory 1 --joint_memory_layers 2 \
 --joint_decoder_dim 256 --joint_decoder_layers 3 \
 --joint_decoder_chunk 128 --joint_nbr_window 1 \
 --joint_direct_xyz_cap 1.5 --joint_translation_cap 0.25 \
 --joint_detach_render_xyz 1 \
 --attr_decoder_layers 0 --attr_read_shape 0 \
 --attr_match_mode sinkhorn --attr_responsibility 0 \
 --attr_start 1500 --attr_loss_ramp_steps 1000 \
 --attr_force_steps 0 --attr_anneal_steps 1 \
 --attr_param_decay_steps 1500 --attr_param_floor 0.0 \
 --attr_anchor_covered 0.0 \
 --w_scale 1.0 --w_rot 0.5 --w_opacity 1.0 --w_color 1.0 \
 --w_chamfer 4.0 --w_coverage 2.0 --w_intra_chamfer 6.0 \
 --w_intra_sinkhorn 8.0 --w_group_centroid 20.0 \
 --geometry_target_scale 1 --geometry_center_local 1 \
 --geometry_centroid_absolute 1 \
 --w_shape_sinkhorn 0.0 --w_presence 0.5 \
 --sinkhorn_epsilon 0.08 --sinkhorn_iterations 6 --sinkhorn_chunk 128 \
 --geo_end 1000 --geo_weak_scale 0.35 --detail_ramp_steps 300 \
 --render_start 2500 --render_ramp_steps 1000 \
 --render_downscale_start 8 --render_downscale_steps 3000 \
 --render_downscale 2 --render_views 1 --extra_real_views 1 \
 --w_render 0 --w_render_attr 20.0 --w_splat_area 2.0 \
 --render_presence 1 \
 --w_latent_std 0.5 --latent_std_floor 0.35 --latent_std_ceil 2.0 \
 --w_latent_decorr 0.0 --w_equiv 0.5 \
 --equiv_start 1200 --equiv_ramp_steps 800 \
 --folding_aniso_log_cap 1.5 \
 --polish_start 0 \
 --lr 2e-4 --lr_min 1e-5 --lr_warmup_steps 500 \
 --max_steps $MAX_STEPS --eval_every 500 --save_every 500 \
 --eval_milestones 100,250,500,1000,1500,2000,3000,4000,6000,8000,10000 \
 --eval_val_indices 0,4,7,31,63,95,127,159"

RUN_LABEL="${RUN_LABEL:-F3D structured compact-local shared-27 codec}" \
CURRICULUM_LABEL="${CURRICULUM_LABEL:-mature-only | pruning-free | fixed anchor + translation | encoder 8 tokens -> compact global[3]+local[4x6] -> 64 query cross-attention | centered shape OT}" \
CAN3TOK_OUT="$OUT" MAX_STEPS="$MAX_STEPS" EXTRA_ARGS="$F3_ARGS $USER_EXTRA" \
  exec bash scripts/launch_r7_geom_render_fix.sh "${1:---foreground}"
