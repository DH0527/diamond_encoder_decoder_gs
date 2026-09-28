#!/usr/bin/env bash
# ============================================================================
# ON HOLD -- DO NOT RUN. Its premise was measured wrong. See DIAGNOSIS.md.
#
# This run existed to raise nn_unique from 0.50 with --w_xyz_residual. Two
# oracles run afterwards showed that nn_unique is bound by the per-group
# information budget, not by the loss:
#
#   * tools/oracle_shape_rank.py -- 28 shape channels must describe 192 numbers
#     (64 points x 3). The linear ceiling at rank 28 is nn_unique 0.529 /
#     ich 0.236 / 17.4 dB, and the model already sits at 0.51 / 0.251 / 17.35.
#     So this run's lever is worth ~0.02 nn_unique and ~0 dB. Channels per point
#     is 131072/262144 = 0.5 and is fixed by the two constraints, independent of
#     group size.
#
#   * tools/oracle_attr_compensation.py -- freezing rank-28 positions and letting
#     only scale+opacity adapt to a multi-view render loss reaches 26.69 dB on
#     three HELD-OUT poses (SSIM 0.905), against 17.05 dB with GT attributes and
#     27.8 dB for the exact 262k GT subset. The budget limits reproduction of the
#     GT point *set*, not render quality.
#
# The file is kept for the measurements recorded below, which remain valid.
# The work moved to the attribute stage; see DIAGNOSIS.md section 7.
# ============================================================================
#
# Geometry coverage run: the duplication fix, and nothing else. The render loss is
# built and verified but deliberately OFF here -- measured ill-conditioned for
# positions (table below). Constraints unchanged: max_points=262144,
# z_compact=32x64x64.
#
# WHY THIS RUN EXISTS
# -------------------
# Until now every number in this project was point-space. Rendering the best
# checkpoint (v4 @2000, intra_chamfer 0.249) through the CUDA rasteriser gave, on
# 7 held-out frames:
#
#     original -> canonical (262k selection)   30.3 dB   SSIM 0.961
#     original -> reconstruction               17.2 dB   SSIM 0.518
#     original -> reconstruction, snapped      19.2 dB
#
# The reconstruction is recognisable -- the scene, its layout and its colours all
# survive -- but detail and dark regions collapse. The interesting number is the
# third: snapping every predicted point onto its nearest GT point, which removes
# *all* positional error, buys only 2 dB. So the loss is not "points are slightly
# off". Matching each prediction to its nearest GT point gives
#
#     distinct GT points hit / predictions =  0.540      (model)
#                                             0.991      (the GT subset itself)
#
# i.e. 46% of the 262k budget lands where another predicted point already is,
# and the corresponding surface goes uncovered. A symmetric chamfer does pay for
# the uncovered GT point through its g->p half, but far too weakly -- the
# duplicate itself scores a perfect p->g and the one missed point's cost is
# averaged over 64 slots -- which is why 1.23x spacing and 17.9 dB coexist.
#
# Calibration (isotropic noise on the GT points, same scene):
#
#     sigma/radius   intra_chamfer   nn_unique
#         0.05           0.070         0.828
#         0.20           0.208         0.636
#         0.30           0.280         0.587
#         0.45           0.383         0.538
#     model              0.246         0.475
#
# The model's error is *more clustered than random noise twice its size*. It puts
# points on the right surface (chamfer far better than the noise control) but
# arranges them wrongly. Consistent with template_erank_pred 7.2 against 22.5 in
# the GT: the shape->offset map produces ~7 distinct group shapes.
#
# WHAT THIS RUN CHANGES  (ONE lever)
# ----------------------------------
# The render loss is built, verified, and **deliberately off here (--w_render 0).**
# It was going to be lever 1. Measuring it first (tools/audit_render_direction.py)
# showed it is the wrong tool for positions. Cosine between -dL/dxyz and the known
# correction (target - pred), on one perturbed 262k cloud, and the fraction of
# points receiving any gradient at all:
#
#   error      render(1 view)   render(4 views)   chamfer     intra_ch    xyz_resid
#   0.6 px     0.134 @ 24%      0.155 @ 35%       0.394 @25%  0.682 @100% 0.866 @100%
#   2.4 px     0.114 @ 23%      0.129 @ 34%       0.449 @25%  0.670 @100% 0.866 @100%
#   6.0 px     0.054 @ 23%      0.064 @ 33%       0.405 @25%  0.662 @100% 0.866 @100%
#
# Three separate problems, all in one column:
#   * the direction is ~82 deg off the correction even at sub-pixel error, and it
#     gets *worse* as the error grows (0.134 -> 0.036 at 9 px);
#   * only ~24% of the 262k points receive any gradient (frustum + occlusion), so
#     three quarters of the cloud is unsupervised by it;
#   * the median Gaussian projects to 1.40 px while the model's actual error
#     (rel_offset_p50 0.499) is ~6 px, i.e. 4x its own footprint. Past that
#     distance the image gradient is sampled at the wrong place and cannot know
#     which way the point belongs. Multi-view raises the cosine by 0.02, nowhere
#     near the sqrt(2/3)=0.816 that depth-blindness alone would allow, so
#     single-view is not the cause -- transport is.
#
# This also corrects a claim made earlier in this file's history: the render loss
# *detects* duplication (a doubled point wastes opacity, the hole reads as
# background) but it cannot *fix* it, because fixing it means transporting a point
# across a gap far larger than its footprint, and no such gradient path exists.
#
# Vanilla 3DGS does optimise means by image loss alone -- while relying on adaptive
# density control (clone/split/prune) to repair exactly the coverage errors
# gradient descent cannot reach. With a fixed 262144-point budget and no
# densification, using the image loss here would be taking the half of 3DGS's
# machinery that depends on the half we do not have.
#
# Where the render loss IS the right tool, and where it is switched on:
#   * attributes (--stage full): opacity, colour, scale change the same pixels
#     they already occupy -- no transport, so the gradient is well conditioned;
#   * the canonicaliser: its decision variable is a per-Gaussian gate, and "is
#     this Gaussian visible / does removing it change the image" is precisely the
#     question an image loss answers;
#   * evaluation, always (tools/render_compare.py).
#
# So the single lever here is:
#
#    --w_xyz_residual: the extent-normalised one-to-one term on the decoder
#    *output*. Today the decoder output has no 1:1 supervision at all --
#    w_xyz/w_xyz_mse/w_xyz_hard are 0 and w_xyz_residual is 0, leaving only
#    set losses (chamfer 20, intra_chamfer 14), whose sensitivity to many-to-one
#    correspondence is far too weak to hold it.
#    w_z_residual (40) does supervise 1:1, but on the *packed* intermediate;
#    everything after the unpack is free to collapse, and rel_offset_p50 sits at
#    0.499, i.e. half the group radius. Across 45 evals rel_offset_p50 and
#    intra_chamfer correlate at r=+0.73, so this pushes the right way.
#
# GATES (pre-committed; a milestone that misses its gate does not advance)
# -----------------------------------------------------------------------
# Two different questions, two different bars. Conflating them is how a run that
# merely proves a term "works" gets promoted into a foundation.
#
# A. Does the lever work at all?  (this run's own gate)
#   step 4000   codec nn_unique       > 0.60   (from 0.499; the noise control at
#                                               this chamfer is 0.61)
#   step 8000   codec nn_unique       > 0.70
#               codec intra_chamfer   < 0.25   (must not be bought with chamfer)
#               render PSNR           > 20 dB  (from 17.2, via tools/render_compare.py)
#
# B. Is geometry good enough to hang attributes on?  (entry to --stage full)
#               codec nn_unique       > 0.85    <-- NOT 0.70
#
# At 0.70, 30% of the output still shares a nearest GT point with another
# prediction. Attributes are per-slot, so training them on top of that teaches
# the attribute decoder a correspondence that does not exist: slots A and B carry
# GT A's and GT B's attributes while their xyz sit on the same spot. Geometry
# coverage first, attributes second -- 0.70 clears this run, it does not clear
# the next one.
#
# The gen branch inherits the same requirement. Its old gate (ich < 0.45, radius
# < 1.2) cannot see duplication at all, so a student can hit both while piling
# points up: add gen nn_unique > 0.65 alongside them.
#
# If nn_unique does not move while render loss falls, the render term is being
# satisfied by opacity/scale compensation rather than by geometry, and the next
# test is w_render on geometry with attributes frozen at GT. Note that in the xyz
# stage that path is already largely blocked -- both sides get the *target's*
# attributes, detached -- so this diagnosis matters most once --stage full runs.
#
# If nn_unique still does not move with the slot term at full weight, the next
# candidate is a local one-to-one assignment (Sinkhorn / Hungarian on the 64x64
# cost matrix per group), NOT a repulsion term: repulsion pushes points apart
# regardless of where the GT density actually is, and would distort the very
# distribution being reconstructed.
#
#   bash scripts/launch_coverage_262k_ddp.sh            # foreground
#   bash scripts/launch_coverage_262k_ddp.sh --detached # nohup + setsid
set -euo pipefail
cd "$(dirname "$0")/.."

# The rasteriser lives in the can3tok env (torch 2.4.1+cu121); the base env does
# not have diff_gaussian_rasterization and the mcr2sc env on PATH is torch 2.1.2,
# where torch.amp.GradScaler does not exist.
TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}

ROOT=${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
WORKERS=${WORKERS:-8}
MAX_STEPS=${MAX_STEPS:-20000}
TAG=${TAG:-mG_262k_g64_$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-runs/$TAG}
LOG=${LOG:-runs/$TAG.log}
EXTRA_ARGS=${EXTRA_ARGS:-}

PATCH_CHUNK=${PATCH_CHUNK:-512}
GEN_REGION_CHUNK=${GEN_REGION_CHUNK:-256}
INTRA_CHUNK=${INTRA_CHUNK:-1024}
CHAMFER_SCALES=${CHAMFER_SCALES:-4096,16384,65536}
EVAL_CHAMFER_SAMPLES=${EVAL_CHAMFER_SAMPLES:-65536}
CHAMFER_PAIR_CHUNK=${CHAMFER_PAIR_CHUNK:-8192}
export CHAMFER_PAIR_CHUNK
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# W_RENDER defaults to 0 in this run -- see the header. The scale note below is
# kept because it applies whenever the term IS switched on (attributes,
# canonicaliser), and because it is the reason a weight cannot be picked by
# analogy.
#
# W_RENDER IS NOT ON THE SAME SCALE AS THE POINT-SPACE WEIGHTS.
# A rasteriser gradient has no reason to arrive at the size of a distance between
# points, and the first guess here (6, by analogy with w_chamfer 20) was wrong by
# three orders of magnitude. Measured with tools/audit_render_scale.py -- |dL/dxyz|
# on one perturbed 262k cloud, every other term zeroed:
#
#   term                weight   |w*dL/dxyz|   share
#   w_render              6.00    1.2340e+02   99.9%     <-- would own the step
#   w_chamfer            20.00    1.1823e-01    0.1%
#   w_intra_chamfer      14.00    3.5151e-02    0.0%
#   w_xyz_residual       12.00    1.3532e-02    0.0%
#
# The ratio holds across scenes (505x, 1027x, 1044x) and is insensitive to the
# perturbation size (999x at 0.1 group radii, 1044x at 0.5), so it is a property
# of the rasteriser, not of the error. Solving for a ~30% share of the combined
# gradient gives w_render ~= 0.0035; 0.004 is that, rounded.
# 0 here. When enabling it for the attribute stage, 0.004 is the measured
# ~30%-gradient-share value; do not raise it toward the point-space weights.
W_RENDER=${W_RENDER:-0}
RENDER_START=${RENDER_START:-4000}
RENDER_DOWNSCALE=${RENDER_DOWNSCALE:-2}
W_XYZ_RESIDUAL=${W_XYZ_RESIDUAL:-12}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 ROOT="$ROOT" GPUS="$GPUS" NPROC="$NPROC" WORKERS="$WORKERS" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
    PATCH_CHUNK="$PATCH_CHUNK" GEN_REGION_CHUNK="$GEN_REGION_CHUNK" INTRA_CHUNK="$INTRA_CHUNK" \
    CHAMFER_SCALES="$CHAMFER_SCALES" EVAL_CHAMFER_SAMPLES="$EVAL_CHAMFER_SAMPLES" \
    CHAMFER_PAIR_CHUNK="$CHAMFER_PAIR_CHUNK" TORCHRUN="$TORCHRUN" \
    W_RENDER="$W_RENDER" RENDER_START="$RENDER_START" RENDER_DOWNSCALE="$RENDER_DOWNSCALE" \
    W_XYZ_RESIDUAL="$W_XYZ_RESIDUAL" \
    nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"; echo "Log: $LOG"; echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT"
echo "launch can3tok mG | GPUs=$GPUS nproc=$NPROC out=$OUT"
echo "  render loss w=$W_RENDER (0 = off; ill-conditioned for positions, see header)"
echo "  SINGLE lever: 1:1 decoder-output term w_xyz_residual=$W_XYZ_RESIDUAL"
echo "  gate: nn_unique > 0.60 @4000, > 0.70 @8000 with intra_chamfer < 0.25"

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
  --group_size 64 \
  --local_tokens_per_group 8 \
  --latent_channels 32 \
  --compact_latent_channels 32 --compact_latent_hw 64 64 \
  --budget_centroid 4 --budget_occupancy 0 --budget_shape 28 \
  --no_slot_redistribute \
  --partition_mode morton \
  --partition_block 32 \
  --folding_decode \
  --folding_frame_channels 6 \
  --slot_sort template \
  --folding_res_start 500 --folding_res_ramp_steps 500 \
  --model_dim 448 --heads 8 --map_blocks 0 \
  --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
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
  --w_xyz 0 --w_xyz_mse 0 --w_xyz_hard 0 \
  --xyz_beta 0.0005 --hard_frac 0.05 --hard_min_points 2048 \
  --w_chamfer 20 --w_coverage 0 --w_voxel_occ 0 \
  --w_plane_chamfer 0 --w_proj_hist 0 --w_presence 0.2 \
  --chamfer_scales "$CHAMFER_SCALES" --chamfer_scale_weights 0.2,0.3,0.5 --balanced_chamfer \
  --spatial_bins 32 --spatial_weight_power 0 --spatial_weight_max 8 \
  --w_intra_chamfer 14 --intra_chamfer_chunk "$INTRA_CHUNK" \
  --w_z_intra_chamfer ${W_Z_INTRA:-25} \
  --w_dispersion 0 --dispersion_group_size 64 \
  --w_z_raw 10 --w_z_raw_mse 0 --w_z_hard 0 --w_z_token_var 0 --w_z_std_ratio 0 \
  --pretrain_w_z_raw 30 --pretrain_w_z_raw_mse 0 --pretrain_w_z_hard 0 \
  --pretrain_w_z_token_var 0 --pretrain_w_z_std_ratio 0 \
  --w_kl 0 --w_latent_std 2 --latent_std_floor 0.5 --latent_std_ceil 3.0 \
  --w_latent_decorr ${W_DECORR:-0.1} \
  --pretrain_w_z_residual 80 --pretrain_w_z_residual_hard 0 \
  --w_z_residual 40 --w_z_residual_hard 0 \
  --pretrain_w_learned_residual 0 --w_learned_residual 0 \
  --pretrain_w_shape_direct 0 --w_shape_direct 0 \
  --w_res_ratio 0 --res_ratio_floor 0 --residual_hard_frac 0.2 \
  `# ---- NEW 1: one-to-one on the decoder OUTPUT, extent-normalised ----` \
  --w_xyz_residual "$W_XYZ_RESIDUAL" --w_xyz_residual_hard 0 \
  --w_equiv 1.0 --equiv_every 4 --equiv_rot_deg 10 --equiv_shape_weight 0.0 \
  --w_gen_xyz 0 --w_gen_xyz_mse 0 --w_gen_xyz_hard 0 \
  --w_gen_chamfer 8 --w_gen_coverage 0 --w_gen_voxel_occ 0 \
  --w_gen_plane_chamfer 0 --w_gen_proj_hist 0 \
  --w_gen_dispersion 0 --w_gen_presence 0.2 \
  --w_gen_intra_chamfer 10 \
  --w_gen_xyz_residual 24 --w_gen_xyz_residual_hard 0 --w_distill 30 --w_gen_basis ${W_GEN_BASIS:-20} \
  --w_gen_p2g ${W_GEN_P2G:-8} --w_gen_radius ${W_GEN_RADIUS:-12} \
  --w_teacher_cycle ${W_TCYC:-6} \
  `# ---- NEW 2: photometric. gen gets it at half weight -- it is still on trial ----` \
  --w_render "$W_RENDER" --render_start "$RENDER_START" --render_ramp_steps 1500 \
  --render_downscale "$RENDER_DOWNSCALE" --render_lam_dssim 0.2 \
  --render_min_coverage 0.25 --render_gen_scale 0.5 \
  --score_metric intra_chamfer \
  --ema_decay 0 \
  --log_every 50 \
  --eval_every 2000 \
  --eval_milestones 500,1000,2000,3000,4000,5000,6000,8000,9000,10000,12000,15000,20000 \
  --eval_val_indices 0,4,7,63,74,138,183,288 \
  --eval_chamfer_samples "$EVAL_CHAMFER_SAMPLES" \
  --save_every 2000 \
  $EXTRA_ARGS 2>&1 | "${SINK[@]}"
