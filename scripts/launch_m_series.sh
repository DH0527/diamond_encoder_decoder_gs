#!/usr/bin/env bash
# M series: fix the measured cause of the blur and the over-sized fine Gaussians.
#
# Measured on R8I step 5000 (held-out, 5 val snapshots):
#   linear scale median  pred 1.55e-4  GT 1.79e-4   -> 0.86x  (NOT larger on average)
#   linear scale mean    pred 2.89e-4  GT 5.74e-4   -> 0.50x
#   log_scale std        pred 0.96     GT 1.63
#   p0.1   pred -11.03   GT -16.62  -> +5.59   the SMALLEST Gaussians are ~270x too big
#   p95    pred  -7.25   GT  -6.16  -> -1.09   the LARGEST are too small
# So the distribution is compressed toward its own mean. Fine structure gets a
# medium Gaussian where GT uses a tiny one -> exactly the blur, and locally the
# splats do look too large.
#
# Within-group residual (group median removed):
#   std ratio pred/GT = 0.444 ; pred p0.1 -1.47 against GT -10.12
# and NOT because the band clips: 0.00% of predictions sit at the lower cap. The
# module has no information to vary within a group -- slot_emb is identical for
# all 4096 groups, attr_nbr_window is 0, and 8 appearance channels condition
# 64 points x 11 attributes.
# On the upper side the band DOES clip: 13.13% of GT's within-group residual
# exceeds cap_up 1.5, which matches the -1.09 shortfall at p95.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VARIANT:?VARIANT required}"
# INIT_CK=none runs from scratch: the variant then supplies its own curriculum and
# no --init_from is emitted. Needed because one of the fix_4 experiments asks
# whether the attribute collapse is baked in by the training history rather than
# by the objective, and that cannot be answered from a warm start.
CK="${INIT_CK:?INIT_CK required}"
if [ "$CK" = "none" ]; then CK=""; fi

# (1) the band's upper half is below what the data needs (measured p99.9 = +2.66)
BASE="--attr_scale_cap_up 3.0 --attr_scale_cap_down 3.0"
# (2) sources of within-group variation, all at zero latent cost:
#     neighbour codes, and the cell's own SHAPE channels through a zero-init
#     additive branch (function-preserving: step 0 reproduces the checkpoint)
INFO="--attr_nbr_window 1 --attr_read_shape 1"

case "$VARIANT" in
  M1) EXTRA="$BASE $INFO" ;;
  # M2 adds the 3D covariance term. Rendering only sees R diag(s^2) R^T, so
  # supervising scale and rot apart picks a factorisation the image cannot
  # observe -- measured attr_rot_nrmse 1.089, no better than the dataset mean.
  M2) EXTRA="$BASE $INFO --w_cov3d 2.0 --w_rot 0.0" ;;
  # M3 gives the render loss ownership of the position gradient. Every earlier
  # attempt at this was confounded: the point-space geometry terms contribute
  # ~190 against render_attr's ~2.31 (98.8% vs 1.2%), so the render was outvoted
  # ~80:1 on the same parameters. Here the geometry terms are cut ~8x, the render
  # is raised, the displacement stays trust-region bounded, and the resolution
  # starts coarse so a 6px error sits inside its own footprint.
  M3) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --render_downscale_start 8 --render_downscale_steps 4000 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1" ;;
  # M4 = M3 plus opening the geometry decoder itself part-way through. M3 only
  # lets the render move a point by +-attr_nudge_cap (0.15 group extents), but the
  # position error to be repaid is codec_snap_psnr - codec_psnr = +1.87 dB, and
  # whether that fits inside 0.15 extents is unmeasured. Releasing the detach at
  # 6000 covers the case where it does not. This is NOT the T3 setup that failed:
  # there the point-space terms outweighed the render ~80:1 on the same
  # parameters; here they are cut ~8x and the render is raised 3x.
  M4) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --render_downscale_start 8 --render_downscale_steps 4000 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1 \
            --attr_detach_geometry 1 --attr_detach_release 6000" ;;
  # M5 attacks the one thing 12000 steps of M1/M3/M4 did not move: the latent's
  # effective rank, which sat at 8.6 -> 8.7 of 32 while held-out PSNR rose a full
  # dB. Measured on M4's step-12000 latent over 23754 held-out groups, the 27
  # non-anchor channels carry effective rank 9.14, median |corr| 0.227, max 0.842,
  # and the decorrelation term that is supposed to attack exactly this evaluates
  # to 0.1005 -- so at w_latent_decorr 0.05 it contributed 0.024% of the
  # objective. It was not losing the argument, it was never in it. w 5.0 puts it
  # at 2.4%, about twice render_attr's measured share.
  #
  # The wide bottleneck is the second, independent lever: a DC-AE-style staged
  # residual on the SHAPE channels only, zero-initialised at `to_compact` so the
  # model is bit-identical to its checkpoint at step 0. It gives the latent a
  # nonlinear path to spread information the budget heads emitted into 9 of 27
  # directions across the rest.
  #
  # Deliberately NOT included: compress_merge_attn and compress_head_stages.
  # Both replace modules at the measured collapse points (intra 34.26 -> merge
  # 13.83; mid 23.63 -> z_compact 10.91), so neither can be warm-started -- the
  # budget heads emit z_compact directly and reinitialising them is the same
  # mistake that took held-out PSNR 15.39 -> 9.55 when to_code_tok was widened.
  # They need a from-scratch run, which this result decides whether to fund.
  #
  # Initialised from M4 step 12000, not R8I, and every absolute-step gate is set
  # to the value M4 had reached there (detach already released, downscale already
  # at 2, attr param multiplier already at its floor, lr continuing the cosine
  # from 2.08e-5). So M5 at +8000 steps and M4 at 20000 share a start, a step
  # budget and a schedule, and differ only in the two rank terms.
  M5) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1 \
            --w_latent_decorr 5.0 \
            --compress_wide_channels 64 --compress_wide_layers 2 \
            --attr_detach_geometry 0 \
            --render_downscale_start 2 --render_downscale 2 \
            --attr_param_floor 0.5 --attr_param_decay_steps 1 \
            --lr 2.08e-5 --lr_min 5e-6 --lr_warmup_steps 100" ;;
  # M6 attacks the cause of the residual blur, which a render ablation finally
  # pinned down and which is NEITHER position accuracy nor attributes.
  #
  # Rendering a RANDOM half of the GT Gaussians with their own GT positions and
  # GT attributes scores 19.93 dB against 48.39 for the full selected set. So
  # covering half the surface costs 28.5 dB -- and this model covers half:
  # nn_unique 0.493. Its own render with GT attributes copied is 19.05, i.e.
  # within 0.9 dB of that random half. Fixing positions perfectly buys 1.75 dB
  # (snap_uniq 19.11 against M4's 17.36) and nothing more; the other ~29 dB is
  # coverage.
  #
  # The duplication is INSIDE the groups: global unique 49.3%, within-group
  # unique 50.9%, and 64.5% of duplicate pairs share a group against the 0.02%
  # a uniform collapse would give. Median within-group NN spacing is 0.261 group
  # radii where 64 well-spread points sit near 0.5. Each group emits 64 points
  # onto ~32 places.
  #
  # M5 already ruled out the latent as the cause: it raised latent effective rank
  # 8.7 -> 9.5 and coverage did not move (49.3% -> 49.4%). Measured along the
  # chain, shape19 erank 6.94 -> 7.36 but fold_local held 17.17 -> 17.26. The
  # choke is the folding head, not the code feeding it.
  #
  # Two terms, chosen to be mutually protective rather than redundant:
  #   w_intra_spacing 6.0   a one-sided hinge demanding each group's mean NN
  #                         spacing reach its own GT group's. Measures 0.363 on
  #                         M4 step 12000, so this is ~10% of the objective.
  #                         Full coverage of the failure: all 64 points of all
  #                         4096 groups, every step.
  #   w_coverage    500     the GT->pred one-sided term, units-corrected. It is
  #                         written in absolute normalised-scene units while every
  #                         within-group term divides by the group extent (median
  #                         0.0027), so at w=2.0 it evaluated to 0.0023 and
  #                         contributed 0.02%. 500 puts it at ~5%. This is the
  #                         same units mistake the code already documents for
  #                         w_distill and w_render.
  # Why both: the cheapest way to satisfy a spacing hinge is to inflate the
  # group, which is exactly how the gen branch once reached 1.83x the GT radius.
  # Coverage cannot be satisfied by inflation -- it demands a prediction near
  # each GT point -- so it closes that loophole. Watch relq / ich / rad999: if
  # spacing is being paid for by inflation they move together.
  #
  # Same warm start and schedule as M5, so M6 / M5 / M4-continued share a branch
  # point and differ only in which axis they attack.
  M6) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1 \
            --w_intra_spacing 6.0 --intra_spacing_ratio 1.0 \
            --w_coverage 500.0 \
            --attr_detach_geometry 0 \
            --render_downscale_start 2 --render_downscale 2 \
            --attr_param_floor 0.5 --attr_param_decay_steps 1 \
            --lr 2.08e-5 --lr_min 5e-6 --lr_warmup_steps 100" ;;
  # M7. Every item traces to a measurement on M4 step 12000, and three items
  # that earlier plans contained were dropped because the measurement refuted
  # them. Recorded here so they are not retried.
  #
  # WHAT THE BLUR IS. Rendering a random half of the GT Gaussians with GT
  # positions and GT attributes scores 19.93 dB against 48.39 for the full set, so
  # covering half the surface costs 28.5 dB, and this model's own render with GT
  # attributes copied is 19.05 -- within 0.9 dB of that random half. Perfect
  # positions buy 1.75 dB and nothing more. The rest is coverage.
  #
  # WHY COVERAGE IS LOW. Two defects, both invisible to the active objective.
  #   group radius pred/GT = 0.670        the group is shrunk toward its centroid
  #   off-surface distance = 0.315 radii  the median point is not on the GT surface
  # intra_chamfer is the symmetric mean of precision and coverage, so it is
  # stationary under trades between them and reports neither. p2g and radius were
  # computed all along and wired only for the gen branch.
  #
  # WHY THE GAUSSIANS LOOK BIG AND THE DETAIL IS GONE. They are not big: the
  # projected footprint is 0.40x the GT's at p50, p90, p99 and p99.9 alike. What is
  # missing is VARIETY. Within a group, pred/GT diversity is 9.7% for scale and
  # 2.9% for orientation -- 64 essentially identical Gaussians at 64 places, where
  # the GT puts Gaussians of very different size and orientation at nearly the same
  # place (its median within-group NN distance is 0.007 group radii). Uniform
  # medium splats cannot render both fine structure and flat wall, which is the
  # blur, and identical Gaussians are interchangeable, which is why the top 10% by
  # contribution reproduces the model's own image at 21.62 dB against the GT's
  # 12.68.
  #
  # WHERE THE VARIETY IS LOST. Traced through the attribute decoder, the
  # within-group / between-group std ratio of the slot representation is 1.44 at
  # in_proj's output and 0.27 once the group code is added -- the within-group part
  # survives, the between-group part explodes 0.364 -> 1.194 because the code is
  # identical for all 64 slots. Knocking out slot_emb and collapsing the position
  # input takes within-group diversity to exactly 0.0, so those two are the only
  # sources, and slot_emb supplies 11% of it while sitting at std 0.029 against a
  # code branch at |w| 0.18. It moved 1.47x in 14000 steps.
  #
  # DROPPED, with the measurement that killed each:
  #   head_scale rescale -- the head already uses the band (residual std 0.911,
  #     15.8% saturating a +-3 band). Its variation is between groups, not within.
  #   n_code_tok 4 -> 16 -- more tokens do not differentiate slots; all 64 slots
  #     read the same tokens, and the ablation shows the code contributes nothing
  #     to within-group diversity.
  #   aniso cap 1.5 -> 3.0 -- unbounded aniso is not untried, it is the older
  #     regime: every run with cap None tops out at 15.54 dB, every run with cap
  #     1.5 reaches 16.35-17.62.
  #   dropping the spacing hinge -- an earlier reading said the model was already
  #     more spread than the GT. That measurement normalised each set by its OWN
  #     group radius, which hid the 0.670x shrinkage. In a common unit the median
  #     pairwise distance is 0.699 against the GT's 1.032. The hinge stays.
  #
  # w_splat_area goes to 0: it penalises the top 20% of splat area at quantile
  # 0.80 (~6 px) while the whole distribution is 0.40x too small.
  M7) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1 \
            --w_p2g 0.5 --w_radius 8.0 \
            --w_intra_spacing 6.0 --intra_spacing_ratio 1.0 \
            --w_coverage 500.0 --w_splat_area 0.0 \
            --w_cov3d 2.0 --w_rot 0.0 \
            --init_rescale attr_decoder.slot_emb.weight:5.0 \
            --polish_attr_slot_scale 30.0 \
            --attr_detach_geometry 0 \
            --render_downscale_start 2 --render_downscale 2 \
            --attr_param_floor 0.5 --attr_param_decay_steps 1 \
            --lr 2.08e-5 --lr_min 5e-6 --lr_warmup_steps 100" ;;
  # ---------------------------------------------------------------- fix_5 ---
  # All three are "N1 + one change". N1 (fix_4) is the current best: adding a
  # permutation-invariant attribute SET distance took held-out PSNR 17.80 -> 18.58
  # and within-group orientation diversity from 2.3% of the ground truth's to
  # 15.8%. Every variant here keeps w_attr_set 3.0 and changes exactly one thing,
  # so each is attributable against N1.
  #
  # CAVEAT carried forward: N1 also turned w_splat_area 4.0 -> 0, so the +0.78 dB
  # against M4 is not cleanly attributable to the set distance alone. N1 vs N2
  # (+0.21) is. These runs inherit splat_area 0 and do not resolve that; a control
  # with only splat_area changed is still owed.

  # P1. The encoder never sees a dense scene. group_in = max_input_points/4096 is
  # 144, and the cell assignment is closest-first, so an over-full cell keeps its
  # innermost 144 points and drops the periphery. Measured over all 3000
  # snapshots (max N 565,017, max cell count 768):
  #     cap  144   drops 14.4-23.3% of the densest snapshots' points
  #     cap  256   drops  2.2-5.2%
  #     cap  512   drops  0.0-0.2%
  #     cap 1024   drops  0.0%
  # so 512 recovers essentially all of it. pool_chunk is now derived from group_in
  # (product held at 73,728) so this costs no extra attention memory; the loader's
  # enc_input tensor goes 35 MB -> 126 MB per sample. Warm start survives because
  # the pooler's 16 queries and its output layer do not depend on the key count.
  # This is also the prerequisite for variable-N scenes: at N=4M a 144 cap would
  # drop 85% before the encoder ever ran.
  P1) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1 \
            --w_splat_area 0.0 --w_intra_spacing 0.0 \
            --w_attr_set 3.0 \
            --attr_detach_geometry 0 \
            --render_downscale_start 2 --render_downscale 2 \
            --attr_param_floor 0.5 --attr_param_decay_steps 1 \
            --lr 3e-5 --lr_min 5e-6 --lr_warmup_steps 200 \
            --max_input_points 2097152 --pool_chunk -1" ;;

  # P2. Lower the share held by terms tied to the SAMPLED SUBSET, raise coverage.
  # Measured shares in N1's objective: intra_chamfer 24% + group_centroid 23% +
  # intra_sinkhorn 16% = 63% is "match the per-anchor statistics of the 262k the
  # sampler happened to pick", against coverage at 5%. Those statistics move when
  # N moves, so the encoder is partly fitting the sampler. The slot ORDERING is
  # kept -- it is what lets 19 shape channels generate 64x3 offsets (measured
  # rank-22 bound 0.679 against Morton's 1.176) and it is not what ties the
  # objective to the subset.
  P2) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1 \
            --w_splat_area 0.0 --w_intra_spacing 0.0 \
            --w_attr_set 3.0 \
            --attr_detach_geometry 0 \
            --render_downscale_start 2 --render_downscale 2 \
            --attr_param_floor 0.5 --attr_param_decay_steps 1 \
            --lr 3e-5 --lr_min 5e-6 --lr_warmup_steps 200 \
            --w_group_centroid 0.1 --w_intra_chamfer 0.2 --w_intra_sinkhorn 0.3 \
            --w_coverage 1500.0" ;;

  # P3. Strengthen the coverage instrument itself rather than its weight.
  # coverage_onesided_loss samples 16384 of 262144 points, so it sees 6% of the
  # scene per step -- a blunt signal for the defect that dominates everything:
  # rendering a random half of the GT Gaussians with GT positions and GT
  # attributes scores 19.74 dB against canon's 48.39, and N1 reaches only 48.1%
  # distinct points. Raised to 49152 (19%), and paired with the two within-group
  # position terms M7 verified: w_radius fixed the group radius 0.670x -> 0.911x,
  # and w_p2g needs to be ~3.0 because at 0.5 its value did not move at all over
  # 500 steps. N1's own position state is worse than M4's (radius 0.405x,
  # off-surface 0.323) because N1 improved attributes without touching position.
  P3) EXTRA="$BASE $INFO --w_group_centroid 0.5 --w_intra_chamfer 0.8 \
            --w_intra_sinkhorn 1.0 --w_shape_sinkhorn 1.5 \
            --w_render_attr 60.0 --attr_nudge_cap 0.15 \
            --polish_decoder_scale 0.5 --polish_compressor_scale 0.3 \
            --polish_decompressor_scale 0.3 --polish_encoder_scale 0.1 \
            --w_splat_area 0.0 --w_intra_spacing 0.0 \
            --w_attr_set 3.0 \
            --attr_detach_geometry 0 \
            --render_downscale_start 2 --render_downscale 2 \
            --attr_param_floor 0.5 --attr_param_decay_steps 1 \
            --lr 3e-5 --lr_min 5e-6 --lr_warmup_steps 200 \
            --coverage_samples 49152 --w_coverage 1000.0 \
            --w_radius 4.0 --w_p2g 3.0" ;;

  *) echo "unknown VARIANT $VARIANT" >&2; exit 1 ;;
esac

# The base launcher also carries `--init_skip attr_decoder.head_scale`. That was
# correct for R8I, which warm-started from R8H whose head_scale had been trained
# under the OLD single-global-bias band, so those weights meant something else.
# It is WRONG here: the M series warm-starts from R8I itself, where head_scale is
# already trained under this band (bias -2.51, scale_base_a 0.719), so skipping it
# throws that away. Measured on the first launch, held-out PSNR at step 100 was
# 11.03 dB against R8I's 16.35 -- 5.3 dB of warm start discarded on all three
# variants. EXTRA_ARGS is a word-split string so an empty value cannot be passed;
# a sentinel that is no parameter's prefix disables the skip instead.
# The base launcher keys off TAG / GPUS / NPROC, not CAN3TOK_OUT. One GPU each so
# the three variants run in parallel and stay separately attributable.
TAG="${VARIANT}_$(date +%Y%m%d_%H%M%S)" \
GPUS="${GPUS:?GPUS required}" NPROC=1 MAX_STEPS=20000 \
  EXTRA_ARGS="${CK:+--init_from $CK --init_skip __none__} $EXTRA --max_steps 20000 \
    --eval_every 1000 --eval_milestones 100,250,500,1000,2000,4000,6000,8000,12000,16000,20000 \
    --save_every 2000 ${USER_EXTRA:-}" \
  exec bash scripts/launch_r8i_scale_band.sh "${1:---detached}"
