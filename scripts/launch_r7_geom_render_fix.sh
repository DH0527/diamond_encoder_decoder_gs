#!/usr/bin/env bash
# R7: fixed-anchor geometry first, then appearance/rendering.
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/launch_r7_geom_render_fix.sh --detached
#   tail -F runs/R7_geom_render_fix_*.log
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CHAMFER_PAIR_CHUNK="${CHAMFER_PAIR_CHUNK:-4096}"
export DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

ROOT="${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"
OUT="${CAN3TOK_OUT:-runs/R7_geom_render_fix_$(date +%Y%m%d_%H%M%S)}"
LOG="${OUT}.log"
MAX_STEPS="${MAX_STEPS:-10000}"
WORKERS="${WORKERS:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${1:-}" == "--detached" ]]; then
  mkdir -p runs
  RUN_LABEL="${RUN_LABEL:-}" CURRICULUM_LABEL="${CURRICULUM_LABEL:-}" \
    CAN3TOK_OUT="$OUT" CUDA_VISIBLE_DEVICES="$GPU" MAX_STEPS="$MAX_STEPS" \
    WORKERS="$WORKERS" EXTRA_ARGS="$EXTRA_ARGS" \
    nohup setsid bash "$0" --foreground >"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  echo "PID=$(cat "${OUT}.pid")"
  echo "OUT=$OUT"
  echo "LOG=$LOG"
  exit 0
fi

mkdir -p "$OUT"
echo "${RUN_LABEL:-R7 fixed-anchor geometry/render} | GPU=$GPU | out=$OUT"
echo "budget: centroid=4 occupancy=1 geometry=19 appearance=8"
echo "${CURRICULUM_LABEL:-curriculum: geometry[0,1500) attrs@1500 render@2500 equiv@1000}"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPU" \
  /home/super/anaconda3/envs/can3tok/bin/python -u -m can3tok.train \
  --root "$ROOT" \
  --out_dir "$OUT" \
  --split_path assets/replay_seg_2700_300_seed42.json \
  --stats_path assets/stats_replay_seg.json \
  --num_val_files 300 \
  --max_points 262144 \
  --max_input_points 589824 \
  --drop_outside \
  --sample_mode stratified \
  --crop_prob 0.0 \
  --density_aware_sample \
  --no_slot_redistribute \
  --partition_mode morton \
  --partition_block 32 \
  --slot_sort template \
  --scene_anchors assets/scene_anchors.npy \
  --anchor_spill 8 \
  --augment \
  --aug_rot_deg 8.0 \
  --aug_scale_jitter 0.03 \
  --aug_shift 0.01 \
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
  --attr_pack_hidden 64 \
  --pool_dim 128 \
  --attr_decoder_layers 4 \
  --attr_decoder_dim 256 \
  --attr_cond xattn \
  --attr_local_pe 1 \
  --attr_scale_log_cap 3.0 \
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
  --patch_chunk 512 \
  --folding_decode \
  --folding_res_cap 1.0 \
  --folding_frame_channels 6 \
  --folding_res_start 500 \
  --folding_res_ramp_steps 750 \
  --no_gen_branch \
  --stage geometry \
  --latent_end 0 \
  --late_decode_steps 0 \
  --geo_end 1500 \
  --geo_weak_scale 0.25 \
  --gen_start 999999999 \
  --gen_end 999999999 \
  --late_codec_start 999999999 \
  --encoder_residual \
  --encoder_residual_start 0 \
  --decoder_refine_start 500 \
  --decoder_refine_ramp_steps 1000 \
  --shortcut_alpha_start 999999999 \
  --pack_trainable \
  --w_xyz 0 \
  --w_xyz_mse 0 \
  --w_xyz_hard 0 \
  --w_chamfer 8.0 \
  --w_coverage 4.0 \
  --w_intra_chamfer 14.0 \
  --w_presence 0.5 \
  --chamfer_scales 4096,16384,65536 \
  --chamfer_scale_weights 0.2,0.3,0.5 \
  --balanced_chamfer \
  --intra_chamfer_chunk 1024 \
  --detail_ramp_steps 300 \
  --w_scale 1.0 \
  --w_rot 0.5 \
  --w_opacity 1.0 \
  --w_color 1.0 \
  --w_sh 0.0 \
  --attr_start 1500 \
  --attr_force_steps 500 \
  --attr_anneal_steps 1500 \
  --attr_param_decay_steps 2000 \
  --attr_param_floor 0.15 \
  --attr_responsibility 1.0 \
  --attr_anchor_covered 0.25 \
  --attr_detach_geometry 1 \
  --attr_detach_release -1 \
  --attr_nudge_cap 0.05 \
  --photo_map assets/npz_to_image.json \
  --view_pool assets/view_pool.json \
  --extra_real_views 4 \
  --render_views 5 \
  --view_downscale 2 \
  --w_render 0 \
  --w_render_attr 20.0 \
  --render_start 2500 \
  --render_ramp_steps 1500 \
  --render_downscale 2 \
  --render_downscale_start 8 \
  --render_downscale_steps 3000 \
  --render_lam_dssim 0.2 \
  --render_min_coverage 0.25 \
  --w_z_raw 0 \
  --w_z_raw_mse 0 \
  --w_z_hard 0 \
  --w_z_token_var 0 \
  --w_z_std_ratio 0 \
  --w_z_residual 0 \
  --w_z_residual_hard 0 \
  --w_z_intra_chamfer 0 \
  --w_learned_residual 0 \
  --w_shape_direct 0 \
  --pretrain_w_z_raw 0 \
  --pretrain_w_z_raw_mse 0 \
  --pretrain_w_z_hard 0 \
  --pretrain_w_z_token_var 0 \
  --pretrain_w_z_std_ratio 0 \
  --pretrain_w_z_residual 0 \
  --pretrain_w_z_residual_hard 0 \
  --pretrain_w_learned_residual 0 \
  --pretrain_w_shape_direct 0 \
  --w_xyz_residual 0 \
  --w_xyz_residual_hard 0 \
  --w_plane_chamfer 0 \
  --w_proj_hist 0 \
  --w_dispersion 0 \
  --w_voxel_occ 0 \
  --w_teacher_cycle 0 \
  --w_res_ratio 0 \
  --w_kl 0 \
  --w_latent_std 2.0 \
  --latent_std_floor 0.5 \
  --latent_std_ceil 3.0 \
  --w_latent_decorr 0.1 \
  --w_equiv 1.0 \
  --equiv_start 1000 \
  --equiv_ramp_steps 1000 \
  --equiv_every 4 \
  --equiv_rot_deg 10.0 \
  --equiv_shape_weight 0.0 \
  --lr 2e-4 \
  --lr_min 1e-5 \
  --lr_warmup_steps 300 \
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
  --eval_milestones 100,250,500,1000,1500,2000,2500,3000,4000,6000,8000,10000 \
  --eval_val_indices 0,4,7,63,74,138,183,288 \
  --save_every 1000 \
  $EXTRA_ARGS
