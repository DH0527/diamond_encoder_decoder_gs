#!/usr/bin/env bash
# Tiny end-to-end run (real data, 8k points, CPU or 1 GPU) to validate wiring
# before burning GPU hours. ~2 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}"
OUT="runs/smoke_cpu"
export CHAMFER_PAIR_CHUNK="${CHAMFER_PAIR_CHUNK:-1024}"
export CUDA_VISIBLE_DEVICES=""

/home/super/anaconda3/envs/can3tok/bin/python train.py \
  --root "$ROOT" \
  --out_dir "$OUT" \
  --split_path assets/replay_seg_2700_300_seed42.json \
  --stats_path assets/stats_replay_seg.json \
  --drop_outside \
  --sample_mode stratified \
  --crop_prob 0.5 \
  --augment \
  --max_points 8192 \
  --group_size 32 \
  --latent_channels 32 \
  --compact_latent_channels 32 \
  --compact_latent_hw 16 8 \
  --model_dim 64 \
  --heads 4 \
  --compress_window 4 \
  --patch_chunk 32 \
  --gen_region_chunk 16 \
  --encoder_residual \
  --amp off \
  --no_checkpoint_decode \
  --workers 0 \
  --max_steps 12 \
  --latent_end 4 \
  --late_decode_steps 2 \
  --geo_end 8 \
  --gen_start 4 \
  --gen_ramp_steps 4 \
  --gen_end 10 \
  --encoder_residual_start 6 \
  --lr_warmup_steps 2 \
  --encoder_warmup_steps 3 \
  --log_every 2 \
  --eval_every 6 \
  --eval_milestones 6,12 \
  --eval_val_indices 0,1 \
  --eval_chamfer_samples 2048 \
  --save_every 12 \
  --chamfer_scales 256,1024 \
  --chamfer_scale_weights 0.4,0.6 \
  --plane_chamfer_samples 2048 \
  --proj_hist_samples 2048 \
  --hard_min_points 256 \
  --z_hard_min_tokens 32 \
  --balanced_chamfer

echo "smoke OK -> $OUT"
