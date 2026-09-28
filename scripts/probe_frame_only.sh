#!/usr/bin/env bash
# Isolation probe: with the folding residual gated off for good and the codec
# refine head never switched on, can the 6-output frame head alone drive the
# decoder to the measured envelope oracle (intra_chamfer ~0.294)?
#
# If yes, the design holds and opening the residual takes it to ~0.218.
# If no, the frame has to be anchored analytically (encoder-side eigenframe
# written into dedicated channels) the way centroid and scale already are.
#
# Both free paths that could otherwise overwrite the template are disabled here:
#   --folding_res_start 100000    folding residual gain stays 0
#   --decoder_refine_start 100000 refine head stays at alpha 0
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/home/super/anaconda3/bin/python}
ROOT=/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg
OUT=${OUT:-/tmp/smoke3}
rm -rf "$OUT"

CUDA_VISIBLE_DEVICES=${GPU:-0} "$PY" -u train.py \
  --root "$ROOT" --out_dir "$OUT" \
  --split_path assets/replay_seg_2700_300_seed42.json \
  --stats_path assets/stats_replay_seg.json \
  --drop_outside --sample_mode stratified --crop_prob 0.0 \
  --max_points 262144 --chunk_size 64 --density_aware_sample \
  --group_size 64 --local_tokens_per_group 8 --latent_channels 32 \
  --compact_latent_channels 32 --compact_latent_hw 64 64 \
  --budget_centroid 4 --budget_occupancy 0 --budget_shape 28 \
  --no_slot_redistribute --partition_mode morton --partition_block 32 \
  --slot_sort template --folding_decode --folding_frame_channels 6 \
  --model_dim 448 --heads 8 --decoder_layers 10 --residual_scale 0.6 \
  --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
  --compress_mid_channels 32 --decompress_intra_layers 3 --decompress_window_layers 2 \
  --patch_chunk 512 --gen_window_layers 4 --gen_group_layers 1 --gen_cross_layers 4 \
  --gen_region_chunk 256 --gen_offset_scale 0.35 --encoder_residual --checkpoint_gen \
  --stage xyz --batch_size 1 --workers 4 --amp bf16 --seed 42 --max_steps ${STEPS:-800} \
  --latent_end 100000 --late_decode_steps 99000 \
  --decoder_refine_start 100000 --decoder_refine_ramp_steps 1500 \
  --geo_end 200000 --geo_weak_scale 0.15 --gen_start 100000 --gen_end 100001 \
  --latent_decay_steps 6000 --diversity_decay_steps 3000 --encoder_residual_start 1000000 \
  --lr 2e-4 --lr_warmup_steps 200 --grad_clip 1.0 \
  --w_xyz 24 --w_xyz_mse 1 --w_xyz_hard 4 --xyz_beta 0.0005 \
  --w_chamfer 20 --w_coverage 12 --w_plane_chamfer 15 \
  --chamfer_scales 4096,16384 --chamfer_scale_weights 0.4,0.6 --balanced_chamfer \
  --coverage_samples 16384 --plane_chamfer_samples 32768 --spatial_weight_power 0 \
  --w_intra_chamfer 6 --intra_chamfer_chunk 1024 --w_dispersion 0 \
  --w_z_raw 10 --w_z_residual 40 --w_shape_direct 15 --w_latent_std 8 \
  --w_xyz_residual 30 --w_equiv 0 --score_metric intra_chamfer \
  --folding_res_start 100000 \
  --eval_val_indices 0,74 --log_every 200 --eval_every 100000 \
  --eval_milestones 400,800 --save_every 100000 "$@"
