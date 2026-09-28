#!/usr/bin/env bash
# Measure s/it in the steady state (decode + gen both running), so the knobs can
# be picked on data instead of intuition. Forces decode on from step 0 and gen at
# full weight from step 0; everything else matches the real launch.
set -u
cd "$(dirname "$0")/.."
TORCHRUN=${TORCHRUN:-/home/super/anaconda3/bin/torchrun}
ROOT=/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg
STEPS=${STEPS:-30}

bench () {
  local tag=$1 pair=$2 pchunk=$3 gchunk=$4 ichunk=$5; shift 5
  echo "--- $tag  (CHAMFER_PAIR_CHUNK=$pair patch_chunk=$pchunk gen_region_chunk=$gchunk $*)"
  rm -rf "/tmp/bench_$tag"
  CUDA_VISIBLE_DEVICES=0,1,2,3 CHAMFER_PAIR_CHUNK=$pair \
  "$TORCHRUN" --standalone --nproc_per_node=4 train.py \
    --root "$ROOT" --out_dir "/tmp/bench_$tag" \
    --split_path assets/replay_seg_2700_300_seed42.json \
    --stats_path assets/stats_replay_seg.json \
    --drop_outside --sample_mode stratified --crop_prob 0.35 \
    --augment --aug_rot_deg 8 --aug_scale_jitter 0.03 --aug_shift 0.01 \
    --max_points 262144 --chunk_size 64 \
    --density_aware_sample --density_importance_weight 0.35 \
    --group_size 64 --local_tokens_per_group 8 --latent_channels 32 \
    --compact_latent_channels 32 --compact_latent_hw 64 64 \
    --budget_centroid 4 --budget_occupancy 0 --budget_shape 28 \
    --no_slot_redistribute --partition_mode morton --partition_block 32 --slot_sort morton \
    --model_dim 448 --heads 8 --map_blocks 0 \
    --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
    --compress_mid_channels 32 --decompress_intra_layers 3 --decompress_window_layers 2 \
    --decoder_layers 10 --residual_scale 0.6 --patch_chunk "$pchunk" \
    --gen_window_layers 4 --gen_group_layers 1 --gen_cross_layers 4 \
    --gen_region_chunk "$gchunk" --gen_offset_scale 0.35 \
    --encoder_residual \
    --shortcut_alpha_init 0 --shortcut_alpha_final 0 --shortcut_alpha_eval 0 \
    --stage xyz --batch_size 1 --workers 6 --amp bf16 --seed 42 \
    --max_steps "$STEPS" \
    `# force the steady state: decode from step 0, gen at full weight from step 0` \
    --latent_end 2 --late_decode_steps 2 --decoder_refine_start 0 --decoder_refine_ramp_steps 1 \
    --geo_end 3 --geo_weak_scale 1.0 --gen_start 0 --gen_ramp_steps 1 --gen_end 4 \
    --latent_decay_steps 1 --diversity_decay_steps 1 --encoder_residual_start 1000000 \
    --lr 2e-4 --lr_min 1e-5 --lr_warmup_steps 500 --weight_decay 1e-4 --grad_clip 1.0 \
    --w_xyz 24 --w_xyz_mse 1 --w_xyz_hard 4 --xyz_beta 0.0005 \
    --w_chamfer 20 --w_coverage 12 --w_voxel_occ 2 --w_plane_chamfer 15 --w_proj_hist 1.0 \
    --chamfer_scales 4096,16384,65536 --chamfer_scale_weights 0.2,0.3,0.5 --balanced_chamfer \
    --coverage_samples 32768 --voxel_occ_bins 24 --plane_chamfer_samples 65536 \
    --proj_hist_samples 32768 --proj_hist_bins 64 --proj_hist_sigma 0.025 --proj_hist_scale 5 \
    --spatial_bins 32 --spatial_weight_power 0 --spatial_weight_max 8 \
    --w_intra_chamfer 20 --w_group_cov 12 --intra_chamfer_chunk "$ichunk" \
    --w_dispersion 0 --dispersion_group_size 64 \
    --w_z_raw 10 --w_z_raw_mse 1.5 --w_z_hard 3 --w_z_token_var 1.5 --w_z_std_ratio 1.5 \
    --w_kl 0 --w_latent_std 8 --latent_std_floor 0.25 --latent_std_ceil 1.5 \
    --w_z_residual 40 --w_z_residual_hard 20 --w_shape_direct 15 \
    --w_res_ratio 0.5 --res_ratio_floor 0.12 --residual_hard_frac 0.2 \
    --w_direct_ratio 4 --direct_ratio_floor 0.10 \
    --w_xyz_residual 30 --w_xyz_residual_hard 24 \
    --w_equiv 1.0 --equiv_every 4 --equiv_rot_deg 10 --equiv_shape_weight 0 \
    --w_gen_xyz 28 --w_gen_xyz_mse 3 --w_gen_xyz_hard 10 \
    --w_gen_chamfer 22 --w_gen_coverage 14 --w_gen_voxel_occ 2 \
    --w_gen_plane_chamfer 18 --w_gen_proj_hist 1.0 --w_gen_dispersion 0 --w_gen_presence 0.2 \
    --w_gen_intra_chamfer 16 --w_gen_group_cov 9 \
    --w_gen_xyz_residual 24 --w_gen_xyz_residual_hard 12 --w_distill 3 \
    --log_every 10 --eval_every 1000000 --eval_milestones "" --save_every 1000000 \
    "$@" > "/tmp/bench_$tag.log" 2>&1
  local last
  last=$(grep -oE "[0-9.]+s/it [0-9.]+GB" "/tmp/bench_$tag.log" | tail -1)
  if [[ -z "$last" ]]; then
    echo "    FAILED: $(grep -oE 'CUDA out of memory|Error[^\"]{0,60}|Traceback' /tmp/bench_$tag.log | head -1)"
  else
    echo "    $last"
  fi
}

bench A_current   4096  512  256  1024
bench B_bigchunk 16384 1024  512  2048
bench C_nockpt   16384 1024  512  2048 --no_checkpoint_decode
bench D_nockpt2  16384 1024  512  2048 --no_checkpoint_decode --no_gen_branch
