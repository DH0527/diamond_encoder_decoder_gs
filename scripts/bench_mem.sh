#!/usr/bin/env bash
# Which component eats the memory once decode + gen + refine are all live?
# One GPU: DDP gives every rank its own full copy at batch 1, so per-process
# memory is the same as in the 4-GPU run.
set -u
cd "$(dirname "$0")/.."
TORCHRUN=${TORCHRUN:-/home/super/anaconda3/bin/torchrun}
PY=${PY:-/home/super/anaconda3/bin/python}
ROOT=/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg
STEPS=${STEPS:-6}

probe () {
  local tag=$1; shift
  rm -rf "/tmp/mem_$tag"
  CUDA_VISIBLE_DEVICES=0 CHAMFER_PAIR_CHUNK=${PAIR:-4096} \
  "$PY" -u train.py \
    --root "$ROOT" --out_dir "/tmp/mem_$tag" \
    --split_path assets/replay_seg_2700_300_seed42.json \
    --stats_path assets/stats_replay_seg.json \
    --drop_outside --sample_mode stratified --crop_prob 0.35 \
    --max_points 262144 --chunk_size 64 --density_aware_sample \
    --group_size ${GS:-64} --local_tokens_per_group ${TPG:-8} --latent_channels 32 \
    --compact_latent_channels 32 --compact_latent_hw 64 64 \
    --budget_centroid 4 --budget_occupancy 0 --budget_shape ${SHP:-28} \
    --no_slot_redistribute --partition_mode morton --partition_block 32 --slot_sort morton \
    --model_dim 448 --heads 8 --decoder_layers 10 --residual_scale 0.6 \
    --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
    --compress_mid_channels 32 --decompress_intra_layers 3 --decompress_window_layers 2 \
    --patch_chunk ${PC:-512} --gen_window_layers 4 --gen_group_layers 1 --gen_cross_layers 4 \
    --gen_region_chunk ${GRC:-256} --gen_offset_scale 0.35 --encoder_residual --checkpoint_gen \
    --shortcut_alpha_init 0 --shortcut_alpha_final 0 --shortcut_alpha_eval 0 \
    --stage xyz --batch_size 1 --workers 3 --amp bf16 --seed 42 --max_steps "$STEPS" \
    --latent_end 2 --late_decode_steps 2 --decoder_refine_start 0 --decoder_refine_ramp_steps 1 \
    --geo_end 3 --geo_weak_scale 1.0 --gen_start 0 --gen_ramp_steps 1 --gen_end 4 \
    --latent_decay_steps 1 --diversity_decay_steps 1 --encoder_residual_start 1000000 \
    --lr 2e-4 --grad_clip 1.0 \
    --w_xyz 24 --w_xyz_mse 1 --w_xyz_hard 4 --xyz_beta 0.0005 \
    --w_chamfer 20 --w_coverage 12 --w_voxel_occ 2 --w_plane_chamfer 15 --w_proj_hist 1.0 \
    --chamfer_scales 4096,16384,65536 --chamfer_scale_weights 0.2,0.3,0.5 --balanced_chamfer \
    --coverage_samples 32768 --voxel_occ_bins 24 --plane_chamfer_samples 65536 \
    --proj_hist_samples 32768 --proj_hist_scale 5 --spatial_weight_power 0 \
    --w_intra_chamfer 20 --w_group_cov 12 --intra_chamfer_chunk 1024 --w_dispersion 0 \
    --dispersion_group_size 64 \
    --w_z_raw 10 --w_z_residual 40 --w_z_residual_hard 20 --w_shape_direct 15 \
    --w_latent_std 8 --latent_std_floor 0.25 --latent_std_ceil 1.5 \
    --w_res_ratio 0.5 --res_ratio_floor 0.12 --w_direct_ratio 4 --direct_ratio_floor 0.10 \
    --w_xyz_residual 30 --w_xyz_residual_hard 24 --w_equiv 0 \
    --w_gen_xyz 28 --w_gen_xyz_mse 3 --w_gen_xyz_hard 10 \
    --w_gen_chamfer 22 --w_gen_coverage 14 --w_gen_voxel_occ 2 \
    --w_gen_plane_chamfer 18 --w_gen_proj_hist 1.0 --w_gen_dispersion 0 --w_gen_presence 0.2 \
    --w_gen_intra_chamfer 16 --w_gen_group_cov 9 \
    --w_gen_xyz_residual 24 --w_gen_xyz_residual_hard 12 --w_distill 3 \
    --log_every 2 --eval_every 1000000 --eval_milestones "" --save_every 1000000 \
    "$@" > "/tmp/mem_$tag.log" 2>&1
  local r
  r=$(grep -oE "[0-9.]+s/it [0-9.]+GB" "/tmp/mem_$tag.log" | tail -1)
  printf "%-26s %s\n" "$tag" "${r:-OOM/FAIL}"
}

echo "== which piece needs the memory (G=64, decode+gen+refine all live) =="
probe all
probe no_gen                 --no_gen_branch
probe no_codec_chamfer       --w_chamfer 0 --w_plane_chamfer 0 --w_coverage 0 --w_voxel_occ 0 --w_proj_hist 0
probe no_gen_chamfer         --w_gen_chamfer 0 --w_gen_plane_chamfer 0 --w_gen_coverage 0 --w_gen_voxel_occ 0 --w_gen_proj_hist 0
probe no_my_new_terms        --w_intra_chamfer 0 --w_group_cov 0 --w_gen_intra_chamfer 0 --w_gen_group_cov 0
echo
echo "== is it G=64? same probe at the old G=32 layout =="
GS=32 TPG=4 SHP=12 PC=1024 probe g32_all
