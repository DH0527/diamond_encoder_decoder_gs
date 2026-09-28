#!/usr/bin/env bash
# Prefix packing kept. Compare shape_xyz regression vs folding shell decode.
set -u
cd "$(dirname "$0")/.."
ROOT="${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}"
STEPS="${STEPS:-2000}"
PY="${PY:-/home/super/anaconda3/bin/python}"

run() {
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u train.py \
    --root "$ROOT" --out_dir "runs/ab_fold_$tag" \
    --split_path assets/replay_seg_2700_300_seed42.json \
    --stats_path assets/stats_replay_seg.json \
    --drop_outside --sample_mode stratified --crop_prob 0.35 \
    --augment --aug_rot_deg 8 --aug_scale_jitter 0.03 --aug_shift 0.01 \
    --max_points 262144 --chunk_size 64 --group_size 64 --local_tokens_per_group 8 \
    --density_aware_sample --density_importance_weight 0.35 \
    --no_slot_redistribute --partition_mode morton --partition_block 32 --slot_sort morton \
    --latent_channels 32 --compact_latent_channels 32 --compact_latent_hw 64 64 \
    --budget_centroid 4 --budget_occupancy 0 --budget_shape 28 \
    --model_dim 448 --heads 8 --map_blocks 0 \
    --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
    --compress_mid_channels 32 --decompress_intra_layers 3 --decompress_window_layers 2 \
    --decoder_layers 10 --residual_scale 0.6 --patch_chunk 512 \
    --gen_offset_scale 0.35 --checkpoint_gen --encoder_residual \
    --shortcut_alpha_init 0 --shortcut_alpha_final 0 --shortcut_alpha_eval 0 \
    --stage xyz --batch_size 1 --workers 4 --amp bf16 --seed 42 \
    --max_steps "$STEPS" --latent_end 100000 --late_decode_steps 0 \
    --lr 2e-4 --lr_min 1e-5 --lr_warmup_steps 500 --weight_decay 1e-4 --grad_clip 1.0 \
    --pretrain_w_z_raw 30 --pretrain_w_z_raw_mse 3 --pretrain_w_z_hard 10 \
    --pretrain_w_z_token_var 6 --pretrain_w_z_std_ratio 3 \
    --z_hard_frac 0.12 --z_hard_min_tokens 256 --z_std_ratio_target 1 \
    --w_kl 0 --w_latent_std 8 --latent_std_floor 0.25 --latent_std_ceil 1.5 \
    --pretrain_w_z_residual 80 --pretrain_w_z_residual_hard 40 --residual_hard_frac 0.2 \
    --pretrain_w_learned_residual 0 --pretrain_w_shape_direct 30 \
    --w_learned_residual 0 --w_shape_direct 15 \
    --w_res_ratio 0.5 --res_ratio_floor 0.12 \
    --w_direct_ratio 4 --direct_ratio_floor 0.10 \
    --w_intra_chamfer 0 --w_group_cov 0 --w_dispersion 0 \
    "$@" \
    --log_every 50 --eval_every 1000000 --save_every 1000000 \
    > "runs/ab_fold_$tag.log" 2>&1 &
  echo "  gpu$gpu $tag (pid $!)"
}

mkdir -p runs
echo "folding A/B ($STEPS steps)"
run 0 baseline
run 1 folding --folding_decode --folding_frame_channels 3
wait
echo done
