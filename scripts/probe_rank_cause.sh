#!/usr/bin/env bash
# Why does the shape block sit at effective rank ~5 of 28?
#
# Two candidate causes:
#   (a) reconstruction never trains the encoder. Measured on the compressor's 104
#       tensors: w_latent_std 73.1%, w_latent_decorr 25.5%, geometry 1.4%. The
#       geometry terms average over ~1e6 elements and the regularisers over 32, so
#       their gradients differ by ~1e4 and no weighting balances them.
#   (b) the architecture cannot carry more than ~5 dims through the compress path.
#
# ARM A  regularisers OFF, pure reconstruction. If lrank climbs, the cause is (a).
# ARM B  same, plus a real staged mid (448 -> 256 -> 256 -> budget instead of
#        448 -> 32 -> 32). Separates (b) once reconstruction is actually driving.
# ARM C  the current setting, as the reference.
#
# lrank is logged every step, so 1500 steps is enough to see the trend.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/home/super/anaconda3/bin/python}
ROOT=/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg
STEPS=${STEPS:-1500}

arm () {
  local tag=$1; shift
  local out=/tmp/rank_$tag
  rm -rf "$out"
  echo "--- $tag : $* ---"
  CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" -u train.py \
    --root "$ROOT" --out_dir "$out" \
    --split_path assets/replay_seg_2700_300_seed42.json \
    --stats_path assets/stats_replay_seg.json \
    --drop_outside --sample_mode stratified --crop_prob 0.0 \
    --max_points 262144 --chunk_size 64 --density_aware_sample \
    --group_size 64 --local_tokens_per_group 8 --latent_channels 32 \
    --compact_latent_channels 32 --compact_latent_hw 64 64 \
    --budget_centroid 4 --budget_occupancy 0 --budget_shape 28 \
    --no_slot_redistribute --partition_mode morton --partition_block 32 \
    --slot_sort template --folding_decode --folding_frame_channels 6 \
    --folding_res_start 500 --folding_res_ramp_steps 500 \
    --model_dim 448 --heads 8 --decoder_layers 10 --residual_scale 0.6 \
    --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
    --decompress_intra_layers 3 --decompress_window_layers 2 \
    --patch_chunk 512 --encoder_residual --checkpoint_gen \
    --stage xyz --batch_size 1 --workers 4 --amp bf16 --seed 42 --max_steps "$STEPS" \
    --latent_end 100000 --late_decode_steps 99000 \
    --decoder_refine_start 100000 --geo_end 200000 --geo_weak_scale 0.15 \
    --gen_start 100000 --gen_end 100001 \
    --latent_decay_steps 6000 --diversity_decay_steps 8000 --encoder_residual_start 1000000 \
    --lr 2e-4 --lr_min 2e-4 --lr_warmup_steps 200 --grad_clip 1.0 \
    --w_z_raw 10 --pretrain_w_z_raw 30 \
    --w_z_residual 40 --pretrain_w_z_residual 80 \
    --w_z_intra_chamfer 25 --intra_chamfer_chunk 1024 \
    --w_xyz 0 --w_xyz_mse 0 --w_xyz_hard 0 --w_xyz_residual 0 --w_xyz_residual_hard 0 \
    --w_z_raw_mse 0 --w_z_hard 0 --w_z_token_var 0 --w_z_std_ratio 0 \
    --pretrain_w_z_raw_mse 0 --pretrain_w_z_hard 0 --pretrain_w_z_token_var 0 \
    --pretrain_w_z_std_ratio 0 --w_z_residual_hard 0 --pretrain_w_z_residual_hard 0 \
    --w_shape_direct 0 --pretrain_w_shape_direct 0 --w_res_ratio 0 \
    --w_chamfer 0 --w_coverage 0 --w_plane_chamfer 0 --w_proj_hist 0 --w_voxel_occ 0 \
    --w_dispersion 0 --w_presence 0 --w_equiv 0 --w_kl 0 \
    --score_metric intra_chamfer \
    --eval_val_indices 0,74 --log_every 100 --eval_every 100000 \
    --eval_milestones "$STEPS" --save_every 100000 "$@" > "/tmp/rank_$tag.log" 2>&1
  grep -oE "step [0-9]+/|lrank [0-9.]+|z_res [0-9.]+" "/tmp/rank_$tag.log" | paste - - - | sed -n '1~3p'
  grep -E "eval step" "/tmp/rank_$tag.log" | tail -1
}

case "${1:-A}" in
  A) arm regoff_mid32  --w_latent_std 0 --w_latent_decorr 0 ;;
  B) arm regoff_mid256 --w_latent_std 0 --w_latent_decorr 0 --compress_mid_channels 256 ;;
  C) arm current       --w_latent_std 2 --w_latent_decorr 0.1 ;;
esac
