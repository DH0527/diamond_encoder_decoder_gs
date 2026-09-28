#!/usr/bin/env bash
# WM-ready residual autoencoder on GPU 0,1,2.
#
# Fixes the six invertibility failures of F3D / R7:
#   1. decode(z_compact) is CodecDecoder identity-unpack. folding_decode is OFF:
#      the shared Fibonacci ball was still being stamped into every cell
#      (WM_residual @500: tmpl 5.6/97.6, nn_unique 0.36, photo 8.32 dB flat).
#   2. that same path is what training and a later world model both call
#   3. encoder and decoder see the same 262144 Gaussians (max_input_points=0)
#   4. no F3D structured-local mid-concat (structured_local_code off)
#   5. w_z_raw / w_z_residual / w_xyz_residual teach invertibility; Chamfer/OT are aux
#   6. fixed anchors + cache_slots; empty/overflow stay (capacity unchanged)
#
# Latent 32x64x64 and K=262144 are unchanged.
#
#   bash scripts/launch_wm_residual_ddp.sh --detached
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
ROOT=${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}
GPUS=${GPUS:-0,1,2}
NPROC=${NPROC:-3}
WORKERS=${WORKERS:-6}
MAX_STEPS=${MAX_STEPS:-10000}
TAG=${TAG:-WM_unpack_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}
MASTER_PORT=${MASTER_PORT:-29517}

PATCH_CHUNK=${PATCH_CHUNK:-512}
INTRA_CHUNK=${INTRA_CHUNK:-1024}
CHAMFER_SCALES=${CHAMFER_SCALES:-4096,16384,65536}
EVAL_CHAMFER_SAMPLES=${EVAL_CHAMFER_SAMPLES:-65536}
CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-4096}
export CHAMFER_PAIR_CHUNK
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 ROOT="$ROOT" GPUS="$GPUS" NPROC="$NPROC" WORKERS="$WORKERS" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
    PATCH_CHUNK="$PATCH_CHUNK" INTRA_CHUNK="$INTRA_CHUNK" \
    CHAMFER_SCALES="$CHAMFER_SCALES" EVAL_CHAMFER_SAMPLES="$EVAL_CHAMFER_SAMPLES" \
    CHAMFER_PAIR_CHUNK="$CHAMFER_PAIR_CHUNK" MASTER_PORT="$MASTER_PORT" \
    TORCHRUN="$TORCHRUN" nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"
  echo "Log: $LOG"
  echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT"
echo "launch WM unpack codec | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  decode=unpack(z_raw_hat)  no shape_xyz generator  refine=off  folding=off"

if [[ -n "${CAN3TOK_DETACHED:-}" ]]; then SINK=(cat); else SINK=(tee -a "$LOG"); fi

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" train.py \
  --root "$ROOT" \
  --out_dir "$OUT" \
  --split_path assets/replay_seg_2700_300_seed42.json \
  --stats_path assets/stats_replay_seg.json \
  --num_val_files 300 \
  --min_snapshot_step 12000 \
  --max_points 262144 \
  --max_input_points 0 \
  --drop_outside \
  --sample_mode stratified \
  --crop_prob 0.0 \
  --density_aware_sample \
  --no_slot_redistribute \
  --partition_mode morton \
  --partition_block 32 \
  --slot_sort morton \
  --cache_slots \
  --scene_anchors assets/scene_anchors.npy \
  --anchor_assignment spill --anchor_spill 8 \
  --use_fixed_anchor_center 1 \
  --workers "$WORKERS" \
  --batch_size 1 \
  --chunk_size 64 \
  --group_size 64 \
  --local_tokens_per_group 8 \
  --latent_channels 32 \
  --latent_hw 256 128 \
  --compact_latent_channels 32 \
  --compact_latent_hw 64 64 \
  --budget_centroid 4 \
  --budget_occupancy 1 \
  --budget_shape 19 \
  --budget_appearance 8 \
  --attr_pack_dim 11 \
  --attr_pack_hidden 128 \
  --pool_dim 128 \
  --pool_queries 16 \
  --compress_merge_stages 1 \
  --compress_head_stages 256 128 \
  --shared_free_head 0 \
  --structured_local_code 0 \
  --joint_shared_decoder 0 --joint_direct_decoder 0 \
  --attr_decoder_layers 4 \
  --attr_decoder_dim 256 \
  --attr_cond xattn \
  --attr_local_pe 1 \
  --attr_scale_log_cap 3.0 \
  --attr_nudge_cap 0.05 \
  --model_dim 448 \
  --heads 8 \
  --compress_intra_layers 3 \
  --compress_window_layers 2 \
  --compress_window 8 \
  --compress_mid_channels 256 \
  --decompress_intra_layers 3 \
  --decompress_window_layers 2 \
  --decoder_layers 10 \
  --residual_scale 0.6 \
  --patch_chunk "$PATCH_CHUNK" \
  --no_gen_branch \
  --stage geometry \
  --latent_end 0 \
  --late_decode_steps 0 \
  --geo_end 1500 \
  --geo_weak_scale 0.35 \
  --gen_start 999999999 \
  --gen_end 999999999 \
  --late_codec_start 999999999 \
  --encoder_residual \
  --encoder_residual_start 0 \
  --decoder_refine_start 999999999 \
  --decoder_refine_ramp_steps 1 \
  --shortcut_alpha_init 0.0 \
  --shortcut_alpha_final 0.0 \
  --shortcut_alpha_eval 0.0 \
  --shortcut_alpha_start 999999999 \
  --w_xyz 0 \
  --w_xyz_mse 0 \
  --w_xyz_hard 0 \
  --w_xyz_residual 12 \
  --w_xyz_residual_hard 6 \
  --w_chamfer 3.0 \
  --w_coverage 1.0 \
  --w_intra_chamfer 2.0 \
  --w_intra_sinkhorn 2.0 \
  --w_group_centroid 8.0 \
  --w_shape_sinkhorn 0.0 \
  --geometry_target_scale 1 --geometry_center_local 1 \
  --geometry_centroid_absolute 1 \
  --w_presence 0.5 \
  --chamfer_scales "$CHAMFER_SCALES" \
  --chamfer_scale_weights 0.2,0.3,0.5 \
  --balanced_chamfer \
  --intra_chamfer_chunk "$INTRA_CHUNK" \
  --detail_ramp_steps 300 \
  --w_scale 1.0 \
  --w_rot 0.5 \
  --w_opacity 1.0 \
  --w_color 1.0 \
  --w_sh 0.0 \
  --attr_start 1500 \
  --attr_loss_ramp_steps 1000 \
  --attr_force_steps 500 \
  --attr_anneal_steps 1500 \
  --attr_param_decay_steps 2000 \
  --attr_param_floor 0.15 \
  --attr_responsibility 1.0 \
  --attr_anchor_covered 0.25 \
  --attr_detach_geometry 1 \
  --attr_detach_release -1 \
  --attr_match_mode sinkhorn \
  --photo_map assets/npz_to_image.json \
  --view_pool assets/view_pool.json \
  --extra_real_views 1 \
  --render_views 1 \
  --view_downscale 2 \
  --w_render 0 \
  --w_render_attr 20.0 \
  --w_splat_area 2.0 \
  --render_presence 1 \
  --render_start 2500 \
  --render_ramp_steps 1000 \
  --render_downscale 2 \
  --render_downscale_start 8 \
  --render_downscale_steps 3000 \
  --render_lam_dssim 0.2 \
  --render_min_coverage 0.25 \
  --w_z_raw 8 \
  --w_z_raw_mse 1.0 \
  --w_z_hard 2.0 \
  --w_z_token_var 0.5 \
  --w_z_std_ratio 0.5 \
  --w_z_residual 20 \
  --w_z_residual_hard 8 \
  --w_z_intra_chamfer 0 \
  --w_learned_residual 0 \
  --w_shape_direct 0 \
  --pretrain_w_z_raw 8 \
  --pretrain_w_z_raw_mse 1.0 \
  --pretrain_w_z_hard 2.0 \
  --pretrain_w_z_token_var 0.5 \
  --pretrain_w_z_std_ratio 0.5 \
  --pretrain_w_z_residual 20 \
  --pretrain_w_z_residual_hard 8 \
  --pretrain_w_learned_residual 0 \
  --pretrain_w_shape_direct 0 \
  --w_plane_chamfer 0 \
  --w_proj_hist 0 \
  --w_dispersion 0 \
  --w_voxel_occ 0 \
  --w_teacher_cycle 0 \
  --w_res_ratio 0 \
  --w_kl 0 \
  --w_latent_std 1.0 \
  --latent_std_floor 0.35 \
  --latent_std_ceil 2.0 \
  --w_latent_decorr 0.05 \
  --w_equiv 0.5 \
  --equiv_start 1200 \
  --equiv_ramp_steps 800 \
  --equiv_every 4 \
  --equiv_rot_deg 10.0 \
  --equiv_shape_weight 0.0 \
  --lr 2e-4 \
  --lr_min 1e-5 \
  --lr_warmup_steps 500 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --amp bf16 \
  --seed 42 \
  --encoder_warmup_steps 0 \
  --score_metric psnr \
  --eval_view_count 8 \
  --eval_view_seed 1234 \
  --max_steps "$MAX_STEPS" \
  --log_every 50 \
  --eval_every 500 \
  --eval_milestones 100,250,500,1000,1500,2000,3000,4000,6000,8000,10000 \
  --eval_val_indices 0,4,7,31,63,95,127,159 \
  --save_every 500 \
  $EXTRA_ARGS \
  2>&1 | "${SINK[@]}"
