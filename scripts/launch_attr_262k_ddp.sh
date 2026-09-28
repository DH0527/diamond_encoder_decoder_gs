#!/usr/bin/env bash
# Milestone E -- the first run in which anything except xyz is learned.
#
# WHY THIS RUN EXISTS
# -------------------
# 22 runs at 262k have all been `--stage xyz`. The attribute heads were never
# even constructed (`cfg.target_dim == 3` skips them), so scale, rotation,
# opacity and colour have never been trained. Worse, the *encoder* packs
# `x[..., 0:3]` and a mask, so appearance never reached z_compact at all: two
# scenes differing only in colour receive the identical latent. Measured, the
# best any decoder could do from a geometry-only code is a held-out R^2 of
#   log_scale  0.227 | rot -0.248 | opacity -0.993 | colour -0.234
# i.e. worse than emitting the dataset mean for everything but scale.
#
# WHAT CHANGED (see APPEARANCE_PLAN.md for the 7 structural problems)
#   --attr_pack_dim 11      a learned per-group attribute encoder writes into the
#                           pack's aux block, where the mask used to sit. The mask
#                           is affordable to lose: prefix packing makes groups
#                           all-or-nothing (exactly ONE partially filled group per
#                           scene, measured), so it carried 0.40 bits per slot.
#   --budget_appearance 16  z_compact channels reserved for appearance, taken out
#                           of shape (28 -> 12). An earlier version used 8 on the
#                           grounds that 8 is where the appearance curve knees at
#                           *fixed* geometry -- true, but it measured one axis of a
#                           two-sided trade. Sweeping the split jointly says shape
#                           was the over-provisioned side: see the note at the
#                           --budget_shape flag below.
#   gen attribute heads     the student emitted `zeros(target_dim - 3)` before, so
#                           the deployment path could not produce a Gaussian.
#   --stage geometry        14 channels = everything a sh_degree=0 rasteriser
#                           reads. The 45 SH-rest channels are deferred: the
#                           render loss cannot see them yet.
#
# The geometry LOSSES are untouched -- it sat at 93% of its budget ceiling
# (nn_unique 0.51 against 0.55, ich 0.251 against 0.235), so there was ~0.3 dB
# left to win by training it harder and nothing worth spending on it.
# w_xyz_residual stays 0 for the same reason.
#
# Its BUDGET is not untouched: shape drops 20 -> 12. That is the same finding read
# the other way round. Geometry being nearly at its ceiling is exactly what makes
# its channels the ones to move -- more shape buys almost nothing, and the sweep
# below shows the channels are worth more than a dB on the appearance side.
# Expect the geometry numbers to get worse; the gate that guards this is
# codec intra_chamfer < 0.28.
#
# GATES (pre-committed)
#   step 4000   attribute held-out R^2 > 0 for opacity and colour
#               -- i.e. better than the dataset mean, which geometry-only cannot do
#   step 8000   render with the model's OWN attributes, held-out 3 views, must
#               beat 17.40 dB -- the best any correspondence rule can do on these
#               same predicted positions.
#
#               That is the honest bar, and it is a hard one, because 17.40 is
#               what you get by *looking up the GT attributes* of the points each
#               prediction covers. The model has no such access: it sees 16
#               appearance channels per group of 64 points -- 16 numbers standing
#               in for 64x11 = 704. Beating it means
#               the code genuinely carries appearance rather than the decoder
#               having memorised a palette.
#
#               Reference points, all rendered on identical predicted positions,
#               held-out 3 views (tools/oracle_attr_target.py, oracle_attr_ceiling.py):
#                 15.08  slot-to-slot target -- what the loss taught before
#                 16.78  nearest-GT target
#                 17.40  responsibility target                 <-- the gate
#                 17.62  even with PERFECT GT positions at the same effective
#                        point count, a correspondence target still only reaches
#                        this. Correspondence saturates; that is the whole reason
#                        the render loss has to be the primary signal.
#                 25.29  what the 12/16 split was measured to carry (3-scene mean
#                        of the joint sweep; 24.14 at the old 20/8)
#                 26.86  attributes fitted directly by the render (upper bound)
#               And 13.16 remains the rank-8 RECONSTRUCTION floor: at or below it
#               the objective change did nothing.
#
#               NOTE the 17.40 bar was measured on positions from a shape-20 run.
#               Shape is now 12, so those positions are slightly worse and the
#               correspondence target they support will render slightly lower --
#               keeping 17.40 makes the gate harder, not softer. Re-measure it with
#               tools/oracle_attr_target.py once this run has a checkpoint.
#
#               Read BOTH branches. codec is the diagnostic -- the strongest
#               decoder, so if it cannot use the appearance code nothing can --
#               but gen is what a world model actually decodes:
#                 codec own-attr > 17.40   <- the correspondence bar
#                 gen   own-attr > 15.11   <- ships
#               codec intra_chamfer < 0.28  -- geometry must not be traded away
#
# If R^2 stays negative the appearance path is not carrying information and the
# next check is tools/ tests on z_compact channel ablation, not more training.
#
#   bash scripts/launch_attr_262k_ddp.sh --detached
set -euo pipefail
cd "$(dirname "$0")/.."

TORCHRUN=${TORCHRUN:-/home/super/anaconda3/envs/can3tok/bin/torchrun}
ROOT=${ROOT:-/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg}
GPUS=${GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
WORKERS=${WORKERS:-8}
MAX_STEPS=${MAX_STEPS:-20000}
TAG=${TAG:-mE_attr262k_$(date +%Y%m%d_%H%M%S)}
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

# Render loss is ON here and OFF in the geometry run, on purpose: it is
# ill-conditioned for positions (cos 0.13 with the correction, 24% of points get
# any gradient) and well conditioned for appearance, which changes the pixels a
# Gaussian already occupies without needing transport.
# 0.004 = the measured ~30% gradient share. NOT comparable to the point-space
# weights: per unit weight the rasteriser gradient is ~1000x a chamfer gradient.
# Two render weights, because the two paths need weights 5000x apart and setting
# one from the other's gradient is what went wrong last time.
#   W_RENDER      -> the full render, position path live. 0.004 = the measured
#                    ~30% share of the *xyz* gradient. Left ON but small: it is
#                    ill-conditioned for positions (cos 0.13, 24% coverage).
#   W_RENDER_ATTR -> position path detached, so it drives only the attributes.
#                    Re-measured on the *separate* AttributeDecoder (3.325M params,
#                    its own trunk), which is where these gradients now land -- the
#                    old 22 was calibrated on heads hanging off the geometry
#                    decoder's refine token and does not carry over. At step 8000:
#                      render(w=22) 4.29e+1 43.2% | scale(w=6)  4.08e+1 41.1%
#                      color(w=6)   9.85e+0  9.9% | opacity(w=4) 3.16e+0  3.2%
#                      rot(w=3)     2.62e+0  2.6%
#                    68 puts the render at ~70%, and the parameter decay takes it
#                    past 90% after step 3000. That split is deliberate, not a
#                    compromise: rendered on identical predicted positions, the
#                    parameter anchor tops out at 17.40 dB (16.78 for a nearest-GT
#                    target, 15.08 for the slot-to-slot pairing it replaces) while
#                    attributes fitted by the render reach 26.86 dB on held-out
#                    poses. The anchor is 4.5 dB under what the 8-channel
#                    appearance code was measured to carry, so it cannot be the
#                    objective -- its job is the ~63% of points no camera ray
#                    reaches, where it is 100% of the gradient however small its
#                    global share.
# W_RENDER is 0 now, and that is not a demotion of the photometric term -- it is
# that the two render calls became the same call. Both route to `attr_pred` when
# the separate decoder exists, and once W_RENDER_ATTR stopped detaching xyz they
# differ in nothing but weight: 0.004 against 68. It was contributing 1/17000 of
# an identical gradient. The geometry decoder is trained by the point-space losses
# and is unreachable from either render term by construction (measured 0.0).
W_RENDER=${W_RENDER:-0}
W_RENDER_ATTR=${W_RENDER_ATTR:-68}
# gen had NO attribute signal at all in the previous run: branch_geometry_loss was
# called with with_attrs=False, w_distill covered xyz only, and the render term was
# applied to the codec's `pred`. Measured at step 20000 the student's emitted
# colour had 19% of the ground truth's spread and its scale 14% -- the heads sat
# near init and the render came out as a near-uniform plate. gen is the path a
# world model decodes, so it is the one that had to learn them.
# Weights from the gen attribute-head gradient (w=1): param 83.5, distill 30.5,
# render 0.533. After the 0.15 param decay that is 12.5, so distill 0.4 matches it
# and render at 22 lands near 32% -- lower than the codec's 60% on purpose, since
# gen's geometry is weaker (nn_unique 0.29 vs 0.48) and a point set that cannot
# form the image should not be asked to chase it with attributes alone.
W_DISTILL_ATTR=${W_DISTILL_ATTR:-0.4}
RENDER_START=${RENDER_START:-6000}
RENDER_DOWNSCALE=${RENDER_DOWNSCALE:-2}
# One camera cannot certify 3D equivalence. Measured: fitting attributes on 1 view
# gives 36.97 dB there and 21.02 dB on held-out poses; fitting on 4 gives 33.28 /
# 26.69. Multi-view is what makes the gain real rather than a per-view overfit.
RENDER_VIEWS=${RENDER_VIEWS:-4}
ATTR_START=${ATTR_START:-2500}
W_SCALE=${W_SCALE:-6}
W_ROT=${W_ROT:-3}
W_OPACITY=${W_OPACITY:-4}
W_COLOR=${W_COLOR:-6}

if [[ "${1:-}" == "--detached" && -z "${CAN3TOK_DETACHED:-}" ]]; then
  mkdir -p runs
  CAN3TOK_DETACHED=1 ROOT="$ROOT" GPUS="$GPUS" NPROC="$NPROC" WORKERS="$WORKERS" \
    MAX_STEPS="$MAX_STEPS" TAG="$TAG" OUT="$OUT" LOG="$LOG" EXTRA_ARGS="$EXTRA_ARGS" \
    PATCH_CHUNK="$PATCH_CHUNK" GEN_REGION_CHUNK="$GEN_REGION_CHUNK" INTRA_CHUNK="$INTRA_CHUNK" \
    CHAMFER_SCALES="$CHAMFER_SCALES" EVAL_CHAMFER_SAMPLES="$EVAL_CHAMFER_SAMPLES" \
    CHAMFER_PAIR_CHUNK="$CHAMFER_PAIR_CHUNK" TORCHRUN="$TORCHRUN" \
    W_RENDER="$W_RENDER" W_RENDER_ATTR="$W_RENDER_ATTR" W_DISTILL_ATTR="$W_DISTILL_ATTR" \
    RENDER_START="$RENDER_START" RENDER_DOWNSCALE="$RENDER_DOWNSCALE" \
    RENDER_VIEWS="$RENDER_VIEWS" ATTR_START="$ATTR_START" \
    W_SCALE="$W_SCALE" W_ROT="$W_ROT" W_OPACITY="$W_OPACITY" W_COLOR="$W_COLOR" \
    nohup setsid bash "$0" >>"$LOG" 2>&1 &
  echo "$!" >"${OUT}.pid"
  mkdir -p "$OUT"; echo "$!" >"${OUT}/run.pid"
  echo "Detached PID=$!"; echo "Log: $LOG"; echo "Out: $OUT"
  exit 0
fi

mkdir -p "$OUT"
echo "launch can3tok mE (attributes) | GPUs=$GPUS out=$OUT"
echo "  stage=geometry (14ch)  attr_pack_dim=11  budget: shape 20 + appearance 8"
echo "  render w=$W_RENDER (xyz) / $W_RENDER_ATTR (attr, xyz detached) from $RENDER_START, $RENDER_VIEWS views"
echo "  gate: attribute held-out R^2 > 0 @4000, held-out 3-view PSNR > 20 dB @8000"

if [[ -n "${CAN3TOK_DETACHED:-}" ]]; then SINK=(cat); else SINK=(tee -a "$LOG"); fi

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
  `# ---- 12/16, not 20/8: the split was measured, not inherited ----------------` \
  `#      Per group of 64 points, shape has to describe 64x3 = 192 numbers and` \
  `#      appearance 64x11 = 704. At 20/8 that is 10:1 against 88:1 -- appearance` \
  `#      was compressed nine times harder than geometry, and nothing had ever` \
  `#      checked whether that was the right place to sit.` \
  `#` \
  `#      tools/oracle_budget_split.py sweeps the split jointly (rank-k positions,` \
  `#      best rank-k appearance code fitted to the image), 3 scenes, held-out` \
  `#      poses. Mean held-out PSNR / within-group position error over group radius:` \
  `#        24/ 4   23.04   rel 1.23` \
  `#        20/ 8   24.14   rel 1.28   <- the old split` \
  `#        16/12   24.90   rel 1.32` \
  `#        12/16   25.29   rel 1.42   <- here, best on 2 of the 3 scenes` \
  `#         8/20   25.23   rel 1.54   (no render gain, more geometry lost)` \
  `#` \
  `#      Shape was over-provisioned: even shape 8 renders better than shape 20.` \
  `#      The cost is real and shows in rel, which is the column that carries npz` \
  `#      point similarity and the 3D structure a world model consumes -- the` \
  `#      render barely feels a Gaussian displaced inside its own 1.40 px splat,` \
  `#      but it feels a wrong colour immediately, so a render-only optimum would` \
  `#      keep trading structure away. 12/16 takes the gain while the marginal` \
  `#      trade is still cheap (+1.15 dB for rel +0.14); 8/20 pays rel +0.26 for` \
  `#      nothing.` \
  --budget_centroid 4 --budget_occupancy 0 --budget_shape 12 --budget_appearance 16 \
  --attr_pack_dim 11 \
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
  --stage geometry --batch_size 1 --workers "$WORKERS" --amp bf16 --seed 42 \
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
  --w_xyz_residual 0 --w_xyz_residual_hard 0 \
  `# ---- attribute curriculum: teacher forcing -> scheduled sampling -> inference ----` \
  --attr_start "$ATTR_START" --attr_force_steps 1500 --attr_anneal_steps 3000 \
  --w_scale "$W_SCALE" --w_rot "$W_ROT" --w_opacity "$W_OPACITY" --w_color "$W_COLOR" \
  --w_equiv 1.0 --equiv_every 4 --equiv_rot_deg 10 --equiv_shape_weight 0.0 \
  --w_gen_xyz 0 --w_gen_xyz_mse 0 --w_gen_xyz_hard 0 \
  --w_gen_chamfer 8 --w_gen_coverage 0 --w_gen_voxel_occ 0 \
  --w_gen_plane_chamfer 0 --w_gen_proj_hist 0 \
  --w_gen_dispersion 0 --w_gen_presence 0.2 \
  --w_gen_intra_chamfer 9 \
  `# ---- gen weights set from MEASURED contributions, not by analogy ----` \
  `#  Decomposing the previous run at ramp 1.0 showed w_gen_basis 20 = 57% of the` \
  `#  whole objective while gen_intra_chamfer was 5% and w_distill 30 was 0.1%:` \
  `#  the student was being trained to copy the teacher's basis parameters, not to` \
  `#  produce good points. distill was dead because it was ABSOLUTE-unit smooth_l1` \
  `#  with beta 0.02 while the median group radius is 0.0027 -- deep in the` \
  `#  quadratic regime. It is extent-normalised now, so its scale matches` \
  `#  gen_xyz_residual and the weight below is meaningful.` \
  `#  Target allocation of a 30-unit gen side (codec side measures ~12):` \
  `#    xyz_residual 35% | intra_chamfer 20% | distill 20% | basis 10% | p2g 10% | radius 5%` \
  --w_gen_xyz_residual 12 --w_gen_xyz_residual_hard 0 --w_distill 7 --w_gen_basis ${W_GEN_BASIS:-0.8} \
  --w_gen_p2g ${W_GEN_P2G:-3.9} --w_gen_radius ${W_GEN_RADIUS:-7.5} \
  `# ---- teacher_cycle OFF: measured 0.0% of the objective, and the cause of two ----` \
  `#      crashes. With gradient checkpointing on it corrupts the backward (the` \
  `#      functional_call pass saves a different tensor set than the trainable one,` \
  `#      -> "Recomputed values have different metadata"); with it off the extra` \
  `#      un-checkpointed decode OOMs at 262k (31.39 of 31.47 GiB). Both failures` \
  `#      landed on the exact step the decoder turns on. Its measured contribution` \
  `#      was value 0.0004 x weight 3.02 = 0.00, so removing it costs nothing.` \
  --w_teacher_cycle ${W_TCYC:-0} \
  `# ---- NEW 2: photometric. gen gets it at half weight -- it is still on trial ----` \
  --w_render "$W_RENDER" --render_start "$RENDER_START" --render_ramp_steps 1500 \
  --render_views "$RENDER_VIEWS" --render_view_jitter_deg 8 \
  --w_render_attr "$W_RENDER_ATTR" --w_distill_attr "$W_DISTILL_ATTR" \
  `# ---- the attribute decoder proper: separate module, detached geometry in ----` \
  `#      Its own trunk, so no render gradient can reach the ten attention layers` \
  `#      of the geometry stack -- verified, every geometry module reads exactly` \
  `#      0.0 under an attribute-render backward. Chunks are checkpointed: without` \
  `#      that, four attention layers over 4096 groups need 31 GB and do not fit.` \
  --attr_decoder_layers ${ATTR_LAYERS:-4} --attr_decoder_dim ${ATTR_DIM:-256} \
  `# ---- the decoder's only per-point input has to resolve per-point detail ----` \
  `#      xyz arrives scene-normalised and a group spans ~0.5% of the scene, so` \
  `#      the Fourier encoding was mostly a group address: within-group variation` \
  `#      0.27 of between-group, while 52-78% of the attribute variance is` \
  `#      within-group. Group-local coordinates alone give 3.52; both concatenated` \
  `#      give 1.00, and the output's response to a 0.1-radius jitter goes from` \
  `#      0.018 to 0.436. Nothing else in the module can supply per-point detail --` \
  `#      the code is per-group and the slot basis is shared across scenes.` \
  --attr_local_pe 1 \
  `# ---- cross-attention, not FiLM, for the group code -------------------------` \
  `#      FiLM broadcasts one scale/shift across all 64 slots, so the code cannot` \
  `#      say "slot 5 is red, slot 6 is blue" -- and 60% of the attribute variance` \
  `#      is exactly that within-group part. Measured as an autodecoder over 8` \
  `#      scenes sharing one network (tools/oracle_conditioning.py), attribute` \
  `#      nrmse / within-group nrmse:` \
  `#        none     0.3021 / 0.3830     code ignored entirely` \
  `#        film     0.3055 / 0.3876     WORSE than ignoring it` \
  `#        concat   0.2983 / 0.3759` \
  `#        xattn    0.2783 / 0.3547     <- 8.9% better than film` \
  `#        film+cat 0.2969 / 0.3741` \
  `#      Same ordering on a single scene. FiLM landing below "none" is what` \
  `#      settled it: with appearance now holding 16 of the 32 channels, a` \
  `#      conditioning path that cannot use them makes that budget dead weight.` \
  --attr_cond xattn \
  `# ---- positions may move, but only within their own footprint ----` \
  `#      The image gradient points the right way only inside a Gaussian's own` \
  `#      splat (cos 0.134 at 0.6 px, 0.054 at 6 px). Median footprint 1.40 px` \
  `#      against a 12.0 px group radius = 0.117 group extents, so the tanh cap` \
  `#      sits just above one footprint: sub-footprint corrections are allowed,` \
  `#      transport is not.` \
  --attr_nudge_cap ${ATTR_NUDGE_CAP:-0.15} --attr_scale_log_cap 3.0 \
  `# ---- the anchor targets the ground each prediction covers, not its slot ----` \
  `#      Slot i of the prediction has no reason to be slot i of the target: the` \
  `#      slot partner is the nearest GT point only 7% of the time, and the two` \
  `#      disagree by 58-80% of each attribute's own std. Instead every GT point` \
  `#      picks its closest prediction and that prediction inherits the set it` \
  `#      won -- union extent as a floor under scale, opacity composited as` \
  `#      1-prod(1-a), colour averaged. This is also the answer to holding fewer` \
  `#      Gaussians than the npz, which is the normal case here: only 37% of` \
  `#      predictions win any ground at all, and those carry 2.7 GT points each.` \
  --attr_responsibility 1 \
  `# ---- reconstruction weights decay once equivalence is live ----` \
  `#      They ask for the GT parameters, which need ~8 channels/point against a` \
  `#      budget of 0.125, so at full strength they hold the heads on a target` \
  `#      they cannot reach and pull against the render loss. Floor, not zero:` \
  `#      they keep the heads in range while the render loss reshapes them.` \
  --attr_param_decay_steps 3000 --attr_param_floor 0.15 \
  --render_downscale "$RENDER_DOWNSCALE" --render_lam_dssim 0.2 \
  --render_min_coverage 0.25 --render_gen_scale 1.0 \
  --score_metric intra_chamfer \
  --ema_decay 0 \
  --log_every 50 \
  --eval_every 2000 \
  --eval_milestones 500,1000,2000,3000,4000,5000,6000,8000,9000,10000,12000,15000,20000 \
  --eval_val_indices 0,4,7,63,74,138,183,288 \
  --eval_chamfer_samples "$EVAL_CHAMFER_SAMPLES" \
  --save_every 2000 \
  $EXTRA_ARGS 2>&1 | "${SINK[@]}"
