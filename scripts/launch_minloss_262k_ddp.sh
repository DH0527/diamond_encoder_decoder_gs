#!/usr/bin/env bash
# can3tok_fix_2 — steps 0..3 of the measured plan. Constraints unchanged:
# max_points=262144, z_compact=32x64x64.
#
# Diagnosis this run acts on (all numbers measured on the _fix run at step 10000,
# 8 val scenes, index-aligned gt/pred):
#   * centroid error is 0.7% of the MSE, within-group offset error is 99.3%
#   * offset_rmse 0.01224 against a PCA-12 ceiling of 0.00907 -> only 1.35x of
#     headroom existed; every previous experiment was fighting inside that margin
#     without moving the ceiling
#   * offset_rmse / group radius = 0.64 at p50, and ~1.00 for the groups whose
#     Morton codes tie (their slot order is the npz file order, not geometry)
#
# P0  gen losses no longer backprop into z_compact.
#       hf_ratio fell 0.145 -> 0.027 exactly across the gen ramp while codec_rmse
#       went flat. Distillation still ties gen to the codec output.
# P1  canonical intra-group slot order + permutation-invariant within-group set
#     loss. Ceiling 0.00853 -> 0.00651 (-23.6%, measured through this loader).
# P2  group_size 64, merge 2 -> 1, shape 12 -> 28 channels. Paying the 4-channel
#     centroid overhead 4096 times instead of 8192 moves 16384 channels (12.5% of
#     the latent) from centroid to shape. Ceiling -> 0.00502 (-41.2%). merge=1
#     also removes the 16/16 cell split that produced the tile grid.
# P3  local kd inside runs of 32 Morton-consecutive groups. Global kd was measured
#     to HURT unless the slot order is fixed first (0.00699 -> 0.00784); local kd
#     with P1 gives 0.00453 (-46.8%) and keeps prefix packing, full groups and the
#     cell -> Z-order-grid locality that global kd + redistribute destroyed.
# P4  loss reweighting: the absolute-unit terms put 58% of the MSE gradient on the
#     ~1% of groups that are not representable at this budget at all.
# P5  instrumentation: shape-channel-only latent stats (the whole-tensor ones are
#     pinned by the centroid channels), group error decomposition, relq scoring.
#
# Deliberately NOT in this run (see the plan):
#   step 4  w_shape_direct ramp-down / direct-deep handover  -> ~3% gain, and it is
#           the same knob family that collapsed shape in the redistribute run
#   step 5  empty_frac recovery (slot_redistribute)          -> already failed once,
#           and it conflicts with P2/P3's stationary full groups
#
#   bash scripts/launch_fix2_262k_ddp.sh            # foreground
#   bash scripts/launch_fix2_262k_ddp.sh --detached # nohup + setsid
set -euo pipefail
cd "$(dirname "$0")/.."

# The _fix run used the base conda env (torch 2.4.1); the mcr2sc env on PATH is
# torch 2.1.2, where torch.amp.GradScaler does not exist. Pin it.
TORCHRUN=${TORCHRUN:-/home/super/anaconda3/bin/torchrun}

ROOT=${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
WORKERS=${WORKERS:-8}
MAX_STEPS=${MAX_STEPS:-20000}
TAG=${TAG:-v4_262k_g64_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}

# G=64 doubles the points per chunk, so halve the chunk counts to hold memory
PATCH_CHUNK=${PATCH_CHUNK:-512}
GEN_REGION_CHUNK=${GEN_REGION_CHUNK:-256}
INTRA_CHUNK=${INTRA_CHUNK:-1024}
CHAMFER_SCALES=${CHAMFER_SCALES:-4096,16384,65536}
COVERAGE_SAMPLES=${COVERAGE_SAMPLES:-32768}
PLANE_CHAMFER_SAMPLES=${PLANE_CHAMFER_SAMPLES:-65536}
PROJ_HIST_SAMPLES=${PROJ_HIST_SAMPLES:-32768}
EVAL_CHAMFER_SAMPLES=${EVAL_CHAMFER_SAMPLES:-65536}
CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-8192}
export CHAMFER_PAIR_CHUNK
# Measured on 4x GPU, decode+gen+refine all live (scripts/bench_ddp.sh):
#   current                        7.38 s/it  15.6 GB
#   + expandable_segments          7.37 s/it  15.5 GB
#   + patch_chunk 256/gen 128      6.96 s/it  10.3 GB
#   + chamfer max 32768, pair 16k  4.89 s/it  15.5 GB   <- only real lever, changes the loss
#   + patch_chunk 1024/gen 512     5.24 s/it  26.0 GB   <- more memory is SLOWER here
# CHAMFER_PAIR_CHUNK=16384 with the 65536 scale OOMs: cdist allocates
# 16384*65536*4B = 4.3 GB in one shot. The chunk and the sample count are coupled.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 ROOT="$ROOT" GPUS="$GPUS" NPROC="$NPROC" WORKERS="$WORKERS" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
    PATCH_CHUNK="$PATCH_CHUNK" GEN_REGION_CHUNK="$GEN_REGION_CHUNK" INTRA_CHUNK="$INTRA_CHUNK" \
    CHAMFER_SCALES="$CHAMFER_SCALES" COVERAGE_SAMPLES="$COVERAGE_SAMPLES" \
    PLANE_CHAMFER_SAMPLES="$PLANE_CHAMFER_SAMPLES" PROJ_HIST_SAMPLES="$PROJ_HIST_SAMPLES" \
    EVAL_CHAMFER_SAMPLES="$EVAL_CHAMFER_SAMPLES" CHAMFER_PAIR_CHUNK="$CHAMFER_PAIR_CHUNK" \
    TORCHRUN="$TORCHRUN" nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"; echo "Log: $LOG"; echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT"
echo "launch can3tok_fix2 | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  P0 gen_detach | P2 G=64/shape=28/merge=1 | P3 local kd(block=32)"
echo "  folding_decode ON | prefix packing (fullfill rejected by ab_utilization)"
echo "  folding+template-aligned slots | 19 loss terms -> 6 (see tools/audit_losses.py)"

if [[ -n "${CAN3TOK_DETACHED:-}" ]]; then SINK=(cat); else SINK=(tee -a "$LOG"); fi

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES="$GPUS" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" train.py \
  --root "$ROOT" \
  --out_dir "$OUT" \
  --split_path assets/replay_seg_2700_300_seed42.json \
  --stats_path assets/stats_replay_seg.json \
  --drop_outside \
  --sample_mode stratified \
  --crop_prob 0.35 \
  --augment --aug_rot_deg 8 --aug_scale_jitter 0.03 --aug_shift 0.01 \
  --max_points 262144 \
  --chunk_size 64 \
  --density_aware_sample --density_importance_weight 0.35 \
  `# ---- P2: group_size 64, merge 1, 28 shape channels ----` \
  --group_size 64 \
  --local_tokens_per_group 8 \
  --latent_channels 32 \
  --compact_latent_channels 32 --compact_latent_hw 64 64 \
  --budget_centroid 4 --budget_occupancy 0 --budget_shape 28 \
  `# ---- P3: local kd, prefix packing kept (full 4096-cell fill rejected: ----` \
  `#      oracle PCA↑ but train z_res flatlines; see VERIFY_UTILIZATION.md) ----` \
  --no_slot_redistribute \
  --partition_mode morton \
  --partition_block 32 \
  `# ---- folding: fixed shell + aniso frame + zero-init residual ----` \
  --folding_decode \
  --folding_frame_channels 6 \
  `# ---- P1a REVERTED: an axis-monotone slot order collapses every group to a` \
  `#      straight line (pred sqrt(lam2/lam1) 0.036 vs 0.544 in GT at step 2000)` \
  `#      while rmse and chamfer both keep improving. The rank-k PCA bound was` \
  `#      *rewarding* that degeneracy, which is why it read as -23.6%.` \
  --slot_sort template \
  --folding_res_start 500 --folding_res_ramp_steps 500 \
  `# ---- widths (unchanged except the chunking G=64 forces) ----` \
  --model_dim 448 --heads 8 --map_blocks 0 \
  --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
  `# mid must exceed the budget once the input actually carries rank ~27:` \
  `#      448 -> 32 -> 32 -> 28 measured effective rank 5.77 of 32 with 53% of` \
  `#      units in the LeakyReLU 0.1-slope region, i.e. a second bottleneck` \
  `#      tighter than the budget it feeds.` \
  --compress_mid_channels 256 --compress_wide_channels 0 --compress_wide_layers 0 \
  --decompress_intra_layers 3 --decompress_window_layers 2 \
  --decoder_layers 10 --residual_scale 0.6 \
  --patch_chunk "$PATCH_CHUNK" \
  --gen_window_layers 4 --gen_group_layers 1 --gen_cross_layers 4 \
  --gen_region_chunk "$GEN_REGION_CHUNK" --gen_offset_scale 0.35 \
  --encoder_residual --checkpoint_gen \
  --shortcut_alpha_init 0.0 --shortcut_alpha_final 0.0 --shortcut_alpha_eval 0.0 \
  --decoder_refine_ramp_steps 1500 \
  --latent_mode deterministic --denoise_std 0.02 \
  --stage xyz --batch_size 1 --workers "$WORKERS" --amp bf16 --seed 42 \
  `# ---- curriculum (unchanged) ----` \
  --max_steps "$MAX_STEPS" \
  --latent_end 4000 --late_decode_steps 1500 \
  --geo_end 10000 --geo_weak_scale 0.15 \
  --gen_start 5000 --gen_ramp_steps 1500 --gen_end 15000 \
  --latent_decay_steps 6000 --diversity_decay_steps 8000 \
  --encoder_residual_start 12000 \
  --lr 2e-4 --lr_min 1e-5 --lr_warmup_steps 500 --weight_decay 1e-4 --grad_clip 1.0 \
  --encoder_warmup_steps 800 --encoder_warmup_decoder_lr_scale 0.5 \
  --encoder_warmup_gen_lr_scale 0.25 \
  --late_codec_lr_scale 0.4 --late_codec_start 15000 \
  `# ---- P4: absolute-unit terms down, extent-normalised terms up ----` \
  `#      w_xyz 40->24, w_xyz_mse 4->1, w_xyz_hard 14->4 (its top-k is on absolute` \
  `#      error, so it selects the biggest groups by construction); the relative` \
  `#      hard mining already exists as w_xyz_residual_hard, so raise that instead.` \
  `#      xyz_beta 0.002->0.0005: the median error was 0.0011, i.e. over half the` \
  `#      points sat in smooth_l1's quadratic knee and the "L1" term acted as MSE.` \
  `#      spatial_weight_power 0.5->0: it up-weighted sparse regions = big groups.` \
  --w_xyz 0 --w_xyz_mse 0 --w_xyz_hard 0 \
  --xyz_beta 0.0005 --hard_frac 0.05 --hard_min_points 2048 \
  --w_chamfer 20 --w_coverage 0 --w_voxel_occ 0 \
  --w_plane_chamfer 0 --w_proj_hist 0 --w_presence 0.2 \
  --chamfer_scales "$CHAMFER_SCALES" --chamfer_scale_weights 0.2,0.3,0.5 --balanced_chamfer \
  --coverage_samples "$COVERAGE_SAMPLES" --voxel_occ_bins 24 \
  --plane_chamfer_samples "$PLANE_CHAMFER_SAMPLES" \
  --proj_hist_samples "$PROJ_HIST_SAMPLES" --proj_hist_bins 64 \
  --proj_hist_sigma 0.025 --proj_hist_scale 5 \
  --spatial_bins 32 --spatial_weight_power 0 --spatial_weight_max 8 \
  `# ---- P1b + 4C: permutation-invariant within-group supervision ----` \
  `#      group_cov is the two-sided full-tensor form of the dispersion hinge,` \
  `#      so the hinge itself goes to 0.` \
  `# group_cov removed: sprayed noise to match moments (thread 1.038 vs GT 0.576).` \
  --w_intra_chamfer 14 --intra_chamfer_chunk "$INTRA_CHUNK" \
  `# anti-collapse pressure that does NOT need the decoder, so it covers the whole` \
  `#      latent phase. Without it w_z_residual is alone from 500 to 2500 and the` \
  `#      groups collapse to lines: measured aniso_lam2 0.959 -> 0.082 (GT 0.579).` \
  --w_z_intra_chamfer ${W_Z_INTRA:-25} \
  --w_dispersion 0 --dispersion_group_size 64 \
  --dispersion_margin 0.001 --dispersion_margin_frac 0.35 \
  `# ---- latent ----` \
  --w_z_raw 10 --w_z_raw_mse 0 --w_z_hard 0 --w_z_token_var 0 --w_z_std_ratio 0 \
  --pretrain_w_z_raw 30 --pretrain_w_z_raw_mse 0 --pretrain_w_z_hard 0 \
  --pretrain_w_z_token_var 0 --pretrain_w_z_std_ratio 0 \
  --z_hard_frac 0.12 --z_hard_min_tokens 256 --z_std_ratio_target 1 \
  `# latent_std was 99.5% of the compressor's gradient at weight 8, and its cosine` \
  `#      against w_z_intra_chamfer is -0.38 -- the only genuinely opposing pair`      \
  `#      measured. It is also per-channel, so it is satisfied by a rank-1 code with`  \
  `#      28 scaled copies: every checkpoint has std 0.25-0.57 with effective rank ~4.`\
  --w_kl 0 --w_latent_std 2 --latent_std_floor 0.5 --latent_std_ceil 3.0 \
  `# the VICReg / Barlow-Twins covariance term: asks for rank, which std cannot.`      \
  --w_latent_decorr ${W_DECORR:-0.1} \
  `# ---- within-group residual: relative hard mining carries the weight now ----` \
  --pretrain_w_z_residual 80 --pretrain_w_z_residual_hard 0 \
  --w_z_residual 40 --w_z_residual_hard 0 \
  --pretrain_w_learned_residual 0 --w_learned_residual 0 \
  --pretrain_w_shape_direct 0 --w_shape_direct 0 \
  --w_res_ratio 0 --res_ratio_floor 0 --residual_hard_frac 0.2 \
  --w_xyz_residual 0 --w_xyz_residual_hard 0 \
  --w_equiv 1.0 --equiv_every 4 --equiv_rot_deg 10 --equiv_shape_weight 0.0 \
  `# ---- P0: gen detached from the latent (default; flag shown for the record) ----` \
  --w_gen_xyz 0 --w_gen_xyz_mse 0 --w_gen_xyz_hard 0 \
  --w_gen_chamfer 8 --w_gen_coverage 0 --w_gen_voxel_occ 0 \
  --w_gen_plane_chamfer 0 --w_gen_proj_hist 0 \
  --w_gen_dispersion 0 --w_gen_presence 0.2 \
  --w_gen_intra_chamfer 10 \
  --w_gen_xyz_residual 24 --w_gen_xyz_residual_hard 0 --w_distill 30 \
  `# ---- P5: score on the scale-free detail number, not the outlier-driven rmse ----` \
  --score_metric intra_chamfer \
  --ema_decay 0 \
  --log_every 50 \
  --eval_every 2000 \
  --eval_milestones 500,1000,2000,3000,4000,5000,6000,8000,9000,10000,12000,15000,20000,30000,40000 \
  --eval_val_indices 0,4,7,63,74,138,183,288 \
  --eval_chamfer_samples "$EVAL_CHAMFER_SAMPLES" \
  --save_every 2000 \
  $EXTRA_ARGS 2>&1 | "${SINK[@]}"
