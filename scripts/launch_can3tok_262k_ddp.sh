#!/usr/bin/env bash
# can3tok_fix: keep max_points=262144, kill the detail-killing terms.
#
# Applied (2026-08-06):
#   1) shape_direct loss — supervises the shape->offset MLP on its own, extent
#      normalised. Without it the deep context path served the whole residual,
#      the shape channels went dead (dfrac 0.24) and every group decoded the same
#      template, which is what showed up as repeated diagonal dashes in the PNGs.
#   2) w_latent_std 8 (was 0.5) — the 0.25 floor was never actually enforced
#   3) equivariance shape term OFF (centroid-only)
#   4) dispersion margin absolute 0.001 + 0.35 * GT group std
#   5) gen shape_xyz direct path (same unbroken offset path as decompressor)
#
# REVERTED after A/B (scripts/ab_layout.sh, 2600 steps x 4 on one GPU each):
#   slot_redistribute + kd partition. It flatlined within-group learning --
#   z_res 0.163 -> 0.155 (-5%, flat from step 350) versus prefix packing
#   0.171 -> 0.135 (-21%, still descending). The redistributed run's better
#   codec_rmse came purely from having 8192 dense centroids; the detail the
#   shape channels are supposed to carry never got learned.
#
#   bash scripts/launch_can3tok_262k_ddp.sh            # foreground
#   bash scripts/launch_can3tok_262k_ddp.sh --detached # nohup + setsid
#   EXTRA_ARGS="--resume path/to/ckpt.pt" bash scripts/launch_can3tok_262k_ddp.sh --detached
#   CHECKPOINT_GEN=0 bash ...   # 31GB급이면 gen checkpoint OFF 가능
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CHAMFER_PAIR_CHUNK="${CHAMFER_PAIR_CHUNK:-4096}"
export DATALOADER_PREFETCH="${DATALOADER_PREFETCH:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

ROOT="${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}"
# CAN3TOK_OUT is only set by the --detached re-exec so both invocations agree
OUT="${CAN3TOK_OUT:-runs/can3tok_262k_c32_fix_$(date +%Y%m%d_%H%M%S)}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
PATCH_CHUNK="${PATCH_CHUNK:-1024}"
GEN_REGION_CHUNK="${GEN_REGION_CHUNK:-512}"
WORKERS="${WORKERS:-6}"
MAX_STEPS="${MAX_STEPS:-60000}"
CROP_PROB="${CROP_PROB:-0.35}"
CHAMFER_SCALES="${CHAMFER_SCALES:-4096,16384,65536}"
PLANE_CHAMFER_SAMPLES="${PLANE_CHAMFER_SAMPLES:-65536}"
PROJ_HIST_SAMPLES="${PROJ_HIST_SAMPLES:-32768}"
COVERAGE_SAMPLES="${COVERAGE_SAMPLES:-32768}"
EVAL_CHAMFER_SAMPLES="${EVAL_CHAMFER_SAMPLES:-65536}"
CHECKPOINT_GEN="${CHECKPOINT_GEN:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG="${OUT}.log"

CKPT_GEN_ARGS=()
if [[ "${CHECKPOINT_GEN}" == "1" || "${CHECKPOINT_GEN}" == "true" ]]; then
  CKPT_GEN_ARGS=(--checkpoint_gen)
fi

if [[ "${1:-}" == "--detached" ]]; then
  mkdir -p runs
  CAN3TOK_OUT="$OUT" CAN3TOK_DETACHED=1 \
    CUDA_VISIBLE_DEVICES="$GPUS" NPROC="$NPROC" \
    PATCH_CHUNK="$PATCH_CHUNK" GEN_REGION_CHUNK="$GEN_REGION_CHUNK" \
    WORKERS="$WORKERS" MAX_STEPS="$MAX_STEPS" \
    CROP_PROB="$CROP_PROB" CHAMFER_SCALES="$CHAMFER_SCALES" \
    PLANE_CHAMFER_SAMPLES="$PLANE_CHAMFER_SAMPLES" \
    PROJ_HIST_SAMPLES="$PROJ_HIST_SAMPLES" \
    COVERAGE_SAMPLES="$COVERAGE_SAMPLES" \
    EVAL_CHAMFER_SAMPLES="$EVAL_CHAMFER_SAMPLES" \
    CHECKPOINT_GEN="$CHECKPOINT_GEN" EXTRA_ARGS="$EXTRA_ARGS" \
    CHAMFER_PAIR_CHUNK="$CHAMFER_PAIR_CHUNK" \
    nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  echo "Detached PID=$(cat "${OUT}.pid")"
  echo "Log: $LOG"
  echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT"
echo "launch can3tok_fix 262k | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  redistribute+kd | occ=1/shape=11 | equiv_shape=0 | dispersion relative"
echo "  crop_prob=$CROP_PROB chamfer=$CHAMFER_SCALES coverage=$COVERAGE_SAMPLES checkpoint_gen=$CHECKPOINT_GEN"
echo "  EXTRA_ARGS=${EXTRA_ARGS:-<none>}"

# detached mode already has stdout pointed at $LOG, so teeing there again would
# write every line twice
if [[ -n "${CAN3TOK_DETACHED:-}" ]]; then SINK=(cat); else SINK=(tee -a "$LOG"); fi

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" torchrun --standalone --nproc_per_node="$NPROC" train.py \
  --root "$ROOT" \
  --out_dir "$OUT" \
  --split_path assets/replay_seg_2700_300_seed42.json \
  --stats_path assets/stats_replay_seg.json \
  --drop_outside \
  --sample_mode stratified \
  --crop_prob "$CROP_PROB" \
  --augment \
  --aug_rot_deg 8 \
  --aug_scale_jitter 0.03 \
  --aug_shift 0.01 \
  --max_points 262144 \
  --chunk_size 64 \
  --group_size 32 \
  --no_slot_redistribute \
  --partition_mode morton \
  --latent_channels 32 \
  --compact_latent_channels 32 \
  --compact_latent_hw 64 64 \
  --budget_centroid 4 \
  --budget_occupancy 0 \
  --budget_shape 12 \
  --model_dim 448 \
  --heads 8 \
  --map_blocks 0 \
  --compress_intra_layers 3 \
  --compress_window_layers 2 \
  --compress_window 8 \
  --compress_mid_channels 32 \
  --compress_wide_channels 0 \
  --compress_wide_layers 0 \
  --decompress_intra_layers 3 \
  --decompress_window_layers 2 \
  --decoder_layers 10 \
  --residual_scale 0.6 \
  --patch_chunk "$PATCH_CHUNK" \
  --gen_window_layers 4 \
  --gen_group_layers 1 \
  --gen_cross_layers 4 \
  --gen_region_chunk "$GEN_REGION_CHUNK" \
  --gen_offset_scale 0.35 \
  --encoder_residual \
  "${CKPT_GEN_ARGS[@]}" \
  --shortcut_alpha_init 0.0 \
  --shortcut_alpha_final 0.0 \
  --shortcut_alpha_eval 0.0 \
  --decoder_refine_ramp_steps 1500 \
  --latent_mode deterministic \
  --denoise_std 0.02 \
  --stage xyz \
  --batch_size 1 \
  --workers "$WORKERS" \
  --amp bf16 \
  --seed 42 \
  --max_steps "$MAX_STEPS" \
  --latent_end 4000 \
  --late_decode_steps 1500 \
  --geo_end 10000 \
  --geo_weak_scale 0.15 \
  --gen_start 6000 \
  --gen_ramp_steps 3000 \
  --gen_end 15000 \
  --latent_decay_steps 6000 \
  --diversity_decay_steps 8000 \
  --encoder_residual_start 12000 \
  --lr 2e-4 \
  --lr_min 1e-5 \
  --lr_warmup_steps 500 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --encoder_warmup_steps 800 \
  --encoder_warmup_decoder_lr_scale 0.5 \
  --encoder_warmup_gen_lr_scale 0.25 \
  --late_codec_lr_scale 0.4 \
  --late_codec_start 15000 \
  --density_aware_sample \
  --density_importance_weight 0.35 \
  --w_xyz 40 --w_xyz_mse 4 --w_xyz_hard 14 \
  --w_chamfer 20 --w_coverage 12 --w_voxel_occ 2 \
  --w_plane_chamfer 15 --w_proj_hist 1.0 \
  --w_dispersion 3 --w_presence 0.2 \
  --xyz_beta 0.002 --hard_frac 0.05 --hard_min_points 2048 \
  --chamfer_scales "$CHAMFER_SCALES" \
  --chamfer_scale_weights 0.2,0.3,0.5 \
  --balanced_chamfer \
  --coverage_samples "$COVERAGE_SAMPLES" \
  --voxel_occ_bins 24 \
  --plane_chamfer_samples "$PLANE_CHAMFER_SAMPLES" \
  --proj_hist_samples "$PROJ_HIST_SAMPLES" --proj_hist_bins 64 --proj_hist_sigma 0.025 --proj_hist_scale 5 \
  --dispersion_group_size 32 --dispersion_margin 0.001 --dispersion_margin_frac 0.35 \
  --spatial_bins 32 --spatial_weight_power 0.5 --spatial_weight_max 8 \
  --w_z_raw 10 --w_z_raw_mse 1.5 --w_z_hard 3 --w_z_token_var 1.5 --w_z_std_ratio 1.5 \
  --pretrain_w_z_raw 30 --pretrain_w_z_raw_mse 3 --pretrain_w_z_hard 10 \
  --pretrain_w_z_token_var 6 --pretrain_w_z_std_ratio 3 \
  --z_hard_frac 0.12 --z_hard_min_tokens 256 --z_std_ratio_target 1 \
  --w_kl 0 --w_latent_std 8 --latent_std_floor 0.25 --latent_std_ceil 1.5 \
  --pretrain_w_z_residual 80 --pretrain_w_z_residual_hard 40 \
  --w_z_residual 40 --w_z_residual_hard 20 \
  --pretrain_w_learned_residual 0 --w_learned_residual 0 \
  --pretrain_w_shape_direct 30 --w_shape_direct 15 \
  --w_res_ratio 0.5 --res_ratio_floor 0.12 --residual_hard_frac 0.2 \
  --w_xyz_residual 30 --w_xyz_residual_hard 15 \
  --w_equiv 1.0 --equiv_every 4 --equiv_rot_deg 10 --equiv_shape_weight 0.0 \
  --w_gen_xyz 28 --w_gen_xyz_mse 3 --w_gen_xyz_hard 10 \
  --w_gen_chamfer 22 --w_gen_coverage 14 --w_gen_voxel_occ 2 \
  --w_gen_plane_chamfer 18 --w_gen_proj_hist 1.0 \
  --w_gen_dispersion 1.5 --w_gen_presence 0.2 \
  --w_gen_xyz_residual 24 --w_gen_xyz_residual_hard 12 --w_distill 3 \
  --log_every 50 \
  --eval_every 2000 \
  --eval_milestones 500,1000,2000,3000,5000,8000,9000,12000,15000,20000,30000,40000 \
  --eval_val_indices 0,4,7,63,74,138,183,288 \
  --eval_chamfer_samples "$EVAL_CHAMFER_SAMPLES" \
  --save_every 2000 \
  ${EXTRA_ARGS} \
  2>&1 | "${SINK[@]}"

echo "done: $OUT"
