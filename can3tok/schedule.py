"""Training curriculum.

Four overlapping stages, every transition ramped rather than switched, because
the previous run lost most of its latent diversity the moment it crossed from
pretraining into joint training (``w_z_raw`` 30 -> 8 and the diversity terms to
exactly 0 in a single step).

    [0, latent_end)          latent only. Compressor/decompressor learn to
                             reproduce z_raw. Decoder refine frozen, no decode.
    [latent_end, geo_end)    decode switched on with *weak* geometry weights, so
                             the latent is optimised for reconstructability and
                             not just for L1 on itself. Refine unfreezes here.
    [geo_end, gen_end)       full codec geometry; latent/diversity weights ramp
                             down smoothly; generative branch ramps in.
    [gen_end, max_steps)     both branches at full strength + distillation.
    [polish_start, ...]      optional DC-AE style final phase: encoder and
                             compressor frozen, only decoder heads keep learning.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple


def _ramp(step: int, start: int, length: int, lo: float = 0.0, hi: float = 1.0) -> float:
    if length <= 0:
        return hi if step >= start else lo
    t = (float(step) - float(start)) / float(length)
    t = min(max(t, 0.0), 1.0)
    return lo + (hi - lo) * t


def phase_of(step: int, args) -> str:
    if step < args.latent_end:
        return "latent"
    if step < args.geo_end:
        return "geo"
    # Polish is an explicit fine-tuning override, not a stage that must wait for
    # the generative curriculum to finish.  Codec-only runs conventionally set
    # gen_end to a very large sentinel; checking gen_end first made polish
    # unreachable in exactly those runs, even when --polish_start was set.
    # polish_start=0 remains the backwards-compatible "disabled" value.
    if args.polish_start > 0 and step >= args.polish_start:
        return "polish"
    if step < args.gen_end:
        return "joint"
    return "gen"


def shortcut_alpha_at(step: int, args) -> float:
    """Centroid-shortcut mix for the decompressor.

    Stays at ``shortcut_alpha_init`` (default 0) until ``shortcut_alpha_start``,
    then ramps to ``shortcut_alpha_final``. Keeping early alpha at 0 is what
    removes the free "all points at centroid" solution during the phase where
    shape channels have to learn.
    """
    init = float(getattr(args, "shortcut_alpha_init", 0.0))
    final = float(getattr(args, "shortcut_alpha_final", 0.0))
    start = int(getattr(args, "shortcut_alpha_start", 10**9))
    ramp = int(getattr(args, "shortcut_alpha_ramp_steps", 1))
    return _ramp(step, start, max(ramp, 1), init, final)


def apply_eval_schedule(model, step: int, args) -> Dict:
    """Put decode gates on the values training would have used at ``step``.

    Tools that force ``decoder_refine_alpha=1`` silently open a head that E1
    still has closed at early checkpoints.
    """
    flags = schedule_flags(int(step), args)
    cfg = model.cfg if hasattr(model, "cfg") else model
    cfg.decoder_refine_alpha = float(flags["decoder_refine_alpha"])
    cfg.folding_res_gain = float(flags["folding_res_gain"])
    cfg.shortcut_alpha = float(flags["shortcut_alpha"])
    cfg.shortcut_alpha_eval = float(getattr(args, "shortcut_alpha_eval", 0.0))
    cfg.attr_teacher_prob = 0.0
    cfg.attr_detach_geometry = bool(flags["attr_detach_geometry"])
    return flags


def decoder_refine_alpha_at(step: int, args) -> float:
    """Gate on codec decoder residual refine.

    Stays at 0 until ``decoder_refine_start`` (default = latent_end) so early
    geometry loss cannot be satisfied by the refine head alone while shape is
    still empty. Then ramps to 1.
    """
    start = int(getattr(args, "decoder_refine_start", getattr(args, "latent_end", 0)))
    ramp = int(getattr(args, "decoder_refine_ramp_steps", 1500))
    return _ramp(step, start, max(ramp, 1), 0.0, 1.0)


def render_downscale_at(step: int, args) -> int:
    """Coarse-to-fine rasterisation.

    The reason the render loss was previously kept away from geometry is that an
    image gradient is badly conditioned for position transport: measured cosine
    0.13 with the true correction, because the median Gaussian projects to
    1.40 px while the position error is ~6 px, so the gradient is computed from
    pixels the point does not even touch. Detaching removes the symptom and the
    signal together.

    Lowering the resolution fixes the conditioning instead. At downscale 8 a
    pixel spans ~4 group radii, so a point that is 6 px out at full resolution is
    inside its own footprint and the gradient points the right way; the schedule
    then steps back up as the geometry converges, which is the standard
    coarse-to-fine used by every photometric-alignment method.

    Returns a power-of-two-friendly integer because ``Camera`` divides the
    intrinsics by it.
    """
    hi = int(getattr(args, "render_downscale_start", 0) or 0)
    lo = int(getattr(args, "render_downscale", 2))
    if hi <= lo:
        return lo
    n = max(int(getattr(args, "render_downscale_steps", 2000)), 1)
    # Geometric interpolation in log2, so the transition is uniform in "pixels
    # per world unit" rather than in pixel count.
    # Do not consume the coarse-to-fine window while rendering is still off.
    # R6 started the interpolation at global step 0, so a delayed render_start
    # could enter at almost the final resolution -- exactly when geometry still
    # needs the wide-footprint, well-conditioned image gradient.
    t = _ramp(step, int(getattr(args, "render_start", 0)), n, 0.0, 1.0)
    import math

    v = math.exp(math.log(hi) + t * (math.log(lo) - math.log(hi)))
    out = max(lo, min(hi, int(round(v))))
    # SECOND LEG: late detail phase. The first leg stops at `render_downscale`,
    # which on this data is 488x272 out of 977x544 -- a quarter of the pixels, so
    # anything under two source pixels is below Nyquist and cannot be supervised
    # at all. The handrails, steps and sleepers that the renders lose are exactly
    # that size. Held until `detail_start` because a sparse high-resolution image
    # gradient is the wrong signal while the geometry is still moving: R6 entered
    # near the final resolution and lost the wide-footprint gradient the geometry
    # needed.
    ds = int(getattr(args, "detail_start", 0) or 0)
    fin = int(getattr(args, "render_downscale_final", 0) or 0)
    if ds > 0 and fin > 0 and fin < out and step >= ds:
        n2 = max(int(getattr(args, "detail_phase_ramp", 2000)), 1)
        t2 = _ramp(step, ds, n2, 0.0, 1.0)
        v2 = math.exp(math.log(out) + t2 * (math.log(fin) - math.log(out)))
        out = max(fin, min(out, int(round(v2))))
    return out


def detail_gains(step: int, args):
    """Late-phase detail knobs: (edge_gain, w_render_perc, w_sobel). 0 before detail_start.

    `lam_dssim` is deliberately NOT ramped here. It was already tried: raising it
    0.2 -> 0.45 did not move the within-group colour variation (14.9% of the GT's
    74.7%), because L1 and SSIM are both distortion metrics minimised by the same
    blurred conditional mean. The two knobs that are not are a weight that says
    WHERE to look (edge_gain) and a term that scores texture statistics instead
    of pixel distance (the VGG feature loss).

    edge_gain turned out not to be enough on its own. T16k ran 23000 steps with it
    fully on and the handrail and lettering stayed blurred, because deciding where
    to spend the L1 never asks for contrast -- a blur that keeps the local mean is
    still cheap. w_sobel does ask, by matching the two images' gradients, and it is
    what moved the crops on views that were never supervised (see photometric_loss).
    """
    ds = int(getattr(args, "detail_start", 0) or 0)
    if ds <= 0 or step < ds:
        return 0.0, 0.0, 0.0
    n = max(int(getattr(args, "detail_phase_ramp", 2000)), 1)
    t = _ramp(step, ds, n, 0.0, 1.0)
    return (float(getattr(args, "edge_gain", 0.0)) * t,
            float(getattr(args, "w_render_perc", 0.0)) * t,
            float(getattr(args, "w_sobel", 0.0)) * t)


def schedule_flags(step: int, args) -> Dict:
    phase = phase_of(step, args)
    run_decode = phase != "latent" or step >= args.latent_end - args.late_decode_steps
    encoder_residual = bool(args.encoder_residual) and step >= args.encoder_residual_start
    run_gen = (not bool(getattr(args, "no_gen_branch", False))) and step >= args.gen_start
    with_attrs = args.stage in ("geometry", "full") and step >= args.attr_start
    # `refine_trainable` and `freeze_encoder_compressor` used to be returned here
    # and were read by nobody -- grep found zero consumers in train.py, and
    # `Can3Tok.set_decoder_refine_trainable` was never called from the loop. They
    # are gone rather than wired up, because wiring them means flipping
    # requires_grad mid-run, which invalidates DDP's gradient buckets under
    # find_unused_parameters=False and would silently corrupt the all-reduce.
    #
    # Freezing is expressed as a learning-rate multiplier instead (see the
    # `phase == "polish"` branch of `lr_scales`), which reaches the same place
    # without changing the graph. Anything that needs a new freeze should go
    # there, not here.
    return {
        "phase": phase,
        "run_decode": bool(run_decode),
        "run_gen": bool(run_gen),
        "encoder_residual": bool(encoder_residual),
        "with_attrs": bool(with_attrs),
        "render_downscale": int(render_downscale_at(step, args)),
        # Geometry-first handover. Detaching the attribute decoder's position input
        # is the right call EARLY -- the image gradient is badly conditioned for
        # position transport (measured cosine 0.13 with the true correction) and
        # the point set is still coarse, so letting the render pull on geometry
        # then is worse than useless. It is the wrong call FOREVER: it leaves the
        # only objective aligned with held-out PSNR training 3.6M of 84M
        # parameters. `attr_detach_release` is the step at which geometry stops
        # being protected and starts being refined. -1 keeps it protected for the
        # whole run, which is the historical behaviour.
        "attr_detach_geometry": bool(
            int(getattr(args, "attr_detach_geometry", 1))
            and (int(getattr(args, "attr_detach_release", -1)) < 0
                 or step < int(getattr(args, "attr_detach_release", -1)))
        ),
        "shortcut_alpha": float(shortcut_alpha_at(step, args)),
        # Hold the folding residual at zero until the 6-output frame head has had
        # a clean run at the envelope. Otherwise the 192-output residual head wins
        # the race and reproduces the anisotropy itself: measured, the decoder went
        # from an exact ball (lam2 0.999) to lam2 0.243 by step 500 while the frame
        # head sat at its zero init and intra_chamfer got *worse* (0.499 -> 0.557).
        "folding_res_gain": _ramp(
            step,
            int(getattr(args, "folding_res_start", 800)),
            max(int(getattr(args, "folding_res_ramp_steps", 700)), 1),
            0.0,
            1.0,
        ),
        "decoder_refine_alpha": float(decoder_refine_alpha_at(step, args)),
        # Teacher forcing -> scheduled sampling -> inference conditioning for the
        # xyz the attribute heads read. Held at 1 for the first window so the
        # heads learn A(x) on true positions, then annealed to 0 so they end up
        # conditioned on exactly what they will see at inference. Annealing rather
        # than switching matters: a hard switch moves the head's input
        # distribution by more than a point spacing in one step.
        "attr_teacher_prob": _ramp(
            step,
            int(getattr(args, "attr_start", 0)) + int(getattr(args, "attr_force_steps", 2000)),
            max(int(getattr(args, "attr_anneal_steps", 4000)), 1),
            1.0,
            0.0,
        ),
    }


def _attr_param_mult(step: int, args) -> float:
    """Multiplier on the attribute *reconstruction* weights."""
    n = int(getattr(args, "attr_param_decay_steps", 0))
    if n <= 0:
        return 1.0
    floor = float(getattr(args, "attr_param_floor", 0.15))
    return _ramp(step, int(getattr(args, "render_start", 0)), n, 1.0, floor)


def sinkhorn_epsilon_at(step: int, args) -> float:
    """Geometric anneal of the entropic regularisation, eps0 -> eps_final.

    Geometric because the plan's sharpness depends on cost/eps, so equal ratios
    are equal steps in sharpness. Holds at eps0 until ``sinkhorn_anneal_start``
    so the anneal begins only once the decoder is actually producing groups.
    """
    e0 = float(getattr(args, "sinkhorn_epsilon", 0.08))
    e1 = float(getattr(args, "sinkhorn_epsilon_final", 0.0) or 0.0)
    if e1 <= 0.0 or e1 >= e0:
        return e0
    start = int(getattr(args, "sinkhorn_anneal_start", 0))
    length = max(int(getattr(args, "sinkhorn_anneal_steps", 1)), 1)
    t = _ramp(step, start, length, 0.0, 1.0)
    return float(math.exp(math.log(e0) + (math.log(e1) - math.log(e0)) * t))


def effective_weights(step: int, args) -> Dict[str, float]:
    """Blend the configured base weights with the phase multipliers."""
    phase = phase_of(step, args)

    # geometry ramps in over the "geo" window, weak at first
    if phase == "latent":
        m_geo = args.geo_weak_scale if step >= args.latent_end - args.late_decode_steps else 0.0
    elif phase == "geo":
        m_geo = _ramp(step, args.latent_end, max(args.geo_end - args.latent_end, 1), args.geo_weak_scale, 1.0)
    else:
        m_geo = 1.0

    # latent weights start at their pretrain value and glide to steady state
    t_latent = _ramp(step, args.latent_end, max(args.latent_decay_steps, 1), 0.0, 1.0)
    t_div = _ramp(step, args.latent_end, max(args.diversity_decay_steps, 1), 0.0, 1.0)

    def lerp(pre: float, post: float, t: float) -> float:
        return float(pre) + (float(post) - float(pre)) * float(t)

    m_gen = _ramp(step, args.gen_start, max(args.gen_ramp_steps, 1), 0.0, 1.0)
    m_distill = _ramp(step, args.gen_start, max(args.gen_ramp_steps, 1), 0.0, 1.0)
    # Geometry first: attribute heads stay entirely out of the shared
    # representation until attr_start, then enter smoothly.  A hard switch here
    # caused the same shared-27 trunk that emits xyz to receive a large,
    # unrelated scale/opacity gradient before positions had converged.
    m_attr = _ramp(
        step, int(getattr(args, "attr_start", 0)),
        max(int(getattr(args, "attr_loss_ramp_steps", 1)), 1), 0.0, 1.0,
    )
    # Permutation-invariant detail terms: full strength as soon as the decoder is
    # actually running, not on the weak geometry ramp. See their entry below.
    decode_on = int(args.latent_end) - int(args.late_decode_steps)
    m_detail = _ramp(step, decode_on, max(int(getattr(args, "detail_ramp_steps", 500)), 1), 0.0, 1.0)

    w: Dict[str, float] = {
        # ---- codec geometry ----
        "w_xyz": args.w_xyz * m_geo,
        "w_xyz_mse": args.w_xyz_mse * m_geo,
        "w_xyz_hard": args.w_xyz_hard * m_geo,
        "w_chamfer": args.w_chamfer * m_geo,
        "w_coverage": getattr(args, "w_coverage", 0.0) * m_geo,
        "w_voxel_occ": getattr(args, "w_voxel_occ", 0.0) * m_geo,
        "w_plane_chamfer": args.w_plane_chamfer * m_geo,
        "w_proj_hist": args.w_proj_hist * m_geo,
        "w_dispersion": args.w_dispersion * m_geo,
        "w_presence": args.w_presence * m_geo,
        # Permutation-invariant within-group detail. These get their own ramp
        # rather than riding m_geo: geo_weak_scale exists to stop the absolute-unit
        # terms from wrecking pose while the shape channels are still empty, and
        # neither of these is absolute-unit (both divide by the group extent) nor
        # order-dependent. They are the terms that fight the shared shape->offset
        # map collapsing to a handful of templates, so holding them at 15% until
        # geo_end wastes the window where that collapse actually happens.
        "w_intra_chamfer": getattr(args, "w_intra_chamfer", 0.0) * m_detail,
        "w_intra_spacing": getattr(args, "w_intra_spacing", 0.0) * m_detail,
        "w_attr_set": getattr(args, "w_attr_set", 0.0) * m_detail,
        "w_attr_local_set": getattr(args, "w_attr_local_set", 0.0) * m_detail,
        "attr_local_set_k": int(getattr(args, "attr_local_set_k", 16)),
        "attr_local_set_chunk": int(getattr(args, "attr_local_set_chunk", 32)),
        # 셀 내부 속성 모멘트. attr_set 과 같은 결함을 겨냥하되 붕괴의 정의를 직접
        # 맞춘다. m_attr 로 attr_start 를 기다리지만 _attr_param_mult 의 감쇠는
        # 걸지 않는다 -- 그 감쇠가 허용하는 붕괴가 바로 이 항이 막으려는 것이다.
        "w_attr_spread": getattr(args, "w_attr_spread", 0.0) * m_attr * m_detail,
        "w_attr_slope": getattr(args, "w_attr_slope", 0.0) * m_attr * m_detail,
        "w_attr_hung_scale": getattr(args, "w_attr_hung_scale", 0.0) * m_attr * m_detail,
        "w_attr_hung_opacity": getattr(args, "w_attr_hung_opacity", 0.0) * m_attr * m_detail,
        "w_attr_hung_rot": getattr(args, "w_attr_hung_rot", 0.0) * m_attr * m_detail,
        "attr_set_chunk": int(getattr(args, "attr_set_chunk", 512)),
        "sinkhorn_attr_weight": float(getattr(args, "sinkhorn_attr_weight", 0.0)),
        "w_p2g": getattr(args, "w_p2g", 0.0) * m_detail,
        "w_radius": getattr(args, "w_radius", 0.0) * m_detail,
        "intra_spacing_ratio": float(getattr(args, "intra_spacing_ratio", 1.0)),
        "w_intra_sinkhorn": getattr(args, "w_intra_sinkhorn", 0.0) * m_detail,
        "w_group_centroid": getattr(args, "w_group_centroid", 0.0) * m_detail,
        "geometry_target_scale": bool(int(getattr(args, "geometry_target_scale", 0))),
        "geometry_center_local": bool(int(getattr(args, "geometry_center_local", 0))),
        "geometry_centroid_absolute": bool(int(getattr(args, "geometry_centroid_absolute", 0))),
        # Annealed, not fixed. Measured on M_speedy_both step 10000: at the
        # historical eps=0.08 the balanced plan is NOT a permutation -- each
        # prediction spreads over exp(H)=4.7-6.1 target slots (peak p50 0.34-0.41
        # where a hard match is 1.0), so the "duplicates must cover distinct
        # points" gradient is averaged into a pull toward the local target mean.
        # That is why w_intra_sinkhorn=4.0 was on for the whole run and nn_unique
        # never left 0.45-0.49. eps=0.005 gives exp(H)=1.04-1.13, an actual
        # assignment. Annealed rather than set low from step 0 because a hard
        # assignment against a still-random point set locks in an arbitrary
        # permutation before the group pose exists.
        "sinkhorn_epsilon": sinkhorn_epsilon_at(step, args),
        "sinkhorn_iterations": int(getattr(args, "sinkhorn_iterations", 6)),
        "sinkhorn_chunk": int(getattr(args, "sinkhorn_chunk", 256)),
        # No ramp and no decode gate: this is the only anti-collapse pressure the
        # latent phase has, and the collapse happens inside it.
        "w_z_intra_chamfer": getattr(args, "w_z_intra_chamfer", 0.0),
        "w_latent_decorr": getattr(args, "w_latent_decorr", 0.0),
        "intra_chamfer_chunk": int(getattr(args, "intra_chamfer_chunk", 1024)),
        # Photometric. Ramped over its own window: at render_start the geometry is
        # still coarse enough that the image gradient is mostly noise, and a term
        # this strong arriving at full weight destabilises the point set before it
        # can guide it.
        "w_render": float(getattr(args, "w_render", 0.0))
        * _ramp(step, int(getattr(args, "render_start", 0)),
                max(int(getattr(args, "render_ramp_steps", 1000)), 1), 0.0, 1.0),
        # ---- attributes ----
        # Reconstruction weights, decayed once the equivalence objective is live.
        # These are an anchor, not the objective, and the measurements are blunt
        # about the difference. Rendered on the same predicted positions:
        #
        #   slot-matched target (what these used to teach)   15.08 dB
        #   nearest-GT target                                16.78 dB
        #   responsibility target (now used)                 16.91 dB
        #   attributes fitted by the render loss itself      26.86 dB
        #
        # Every correspondence-based target saturates near 17; even *perfect* GT
        # positions at the same effective point count only reach 17.62. So the
        # anchor is 4.5 dB below what the 8-channel appearance code was measured to
        # carry (21.43) and 10 dB below what the render reaches. Held at strength it
        # would cap the decoder well under its own capacity. It decays to a floor
        # whose only job is the ~63% of points no camera ray gives gradient to.
        **{
            k: getattr(args, k, 0.0) * m_attr * _attr_param_mult(step, args)
            for k in ("w_scale", "w_rot", "w_opacity", "w_color", "w_sh", "w_cov3d",
                      "w_aniso")
        },
        "w_distill_attr": getattr(args, "w_distill_attr", 0.0) * m_distill,
        "attr_responsibility": float(getattr(args, "attr_responsibility", 1.0)),
        "attr_match_mode": str(getattr(args, "attr_match_mode", "responsibility")),
        "attr_anchor_covered": float(getattr(args, "attr_anchor_covered", 1.0)),
        "w_render_attr": float(getattr(args, "w_render_attr", 0.0))
        * _ramp(step, int(getattr(args, "render_start", 0)),
                max(int(getattr(args, "render_ramp_steps", 1000)), 1), 0.0, 1.0),
        # ---- latent ----
        "w_z_raw": lerp(args.pretrain_w_z_raw, args.w_z_raw, t_latent),
        "w_z_raw_mse": lerp(args.pretrain_w_z_raw_mse, args.w_z_raw_mse, t_latent),
        "w_z_hard": lerp(args.pretrain_w_z_hard, args.w_z_hard, t_div),
        "w_z_token_var": lerp(args.pretrain_w_z_token_var, args.w_z_token_var, t_div),
        "w_z_std_ratio": lerp(args.pretrain_w_z_std_ratio, args.w_z_std_ratio, t_div),
        "w_kl": args.w_kl,
        "w_latent_std": args.w_latent_std,
        "latent_std_floor": args.latent_std_floor,
        "latent_std_ceil": args.latent_std_ceil,
        # ---- within-group detail (always on; this is what centroid shortcuts skip) ----
        "w_z_residual": lerp(args.pretrain_w_z_residual, args.w_z_residual, t_latent),
        "w_z_residual_hard": lerp(args.pretrain_w_z_residual_hard, args.w_z_residual_hard, t_latent),
        "w_learned_residual": lerp(args.pretrain_w_learned_residual, args.w_learned_residual, t_latent),
        "w_shape_direct": lerp(args.pretrain_w_shape_direct, args.w_shape_direct, t_latent),
        "w_shape_sinkhorn": float(getattr(args, "w_shape_sinkhorn", 0.0)),
        "w_res_ratio": args.w_res_ratio,
        "res_ratio_floor": args.res_ratio_floor,
        "residual_hard_frac": args.residual_hard_frac,
        "w_xyz_residual": args.w_xyz_residual * m_geo,
        "w_xyz_residual_hard": args.w_xyz_residual_hard * m_geo,
        # ---- generative ----
        "w_gen_scale": m_gen,
        "w_gen_xyz": args.w_gen_xyz,
        "w_gen_xyz_mse": args.w_gen_xyz_mse,
        "w_gen_xyz_hard": args.w_gen_xyz_hard,
        "w_gen_chamfer": args.w_gen_chamfer,
        "w_gen_coverage": getattr(args, "w_gen_coverage", 0.0),
        "w_gen_voxel_occ": getattr(args, "w_gen_voxel_occ", 0.0),
        "w_gen_plane_chamfer": args.w_gen_plane_chamfer,
        "w_gen_proj_hist": args.w_gen_proj_hist,
        "w_gen_dispersion": args.w_gen_dispersion,
        "w_gen_presence": args.w_gen_presence,
        "w_gen_intra_chamfer": getattr(args, "w_gen_intra_chamfer", 0.0),
        "w_gen_xyz_residual": args.w_gen_xyz_residual,
        "w_gen_xyz_residual_hard": args.w_gen_xyz_residual_hard,
        "w_distill": args.w_distill * m_distill,
        "w_gen_basis": getattr(args, "w_gen_basis", 0.0) * m_distill,
        "w_gen_p2g": getattr(args, "w_gen_p2g", 0.0) * m_gen,
        "w_gen_radius": getattr(args, "w_gen_radius", 0.0) * m_gen,
        "w_teacher_cycle": getattr(args, "w_teacher_cycle", 0.0) * m_geo,
        # Equivariance is useful only after anchor geometry has acquired meaning.
        # Ramping it also avoids an early constant-latent solution.  The actual
        # rotated forward is required to use the same encoder tensor/grouping in
        # train.py; this schedule is the second half of that safety contract.
        "w_equiv": float(args.w_equiv)
        * _ramp(
            step,
            int(getattr(args, "equiv_start", 0)),
            max(int(getattr(args, "equiv_ramp_steps", 1)), 1),
            0.0,
            1.0,
        ),
        # ---- shared hyper-parameters ----
        "xyz_beta": args.xyz_beta,
        "hard_frac": args.hard_frac,
        "hard_min_points": args.hard_min_points,
        "chamfer_scales": tuple(int(s) for s in str(args.chamfer_scales).split(",")),
        "chamfer_scale_weights": tuple(float(s) for s in str(args.chamfer_scale_weights).split(",")),
        "balanced_chamfer": args.balanced_chamfer,
        "coverage_samples": int(getattr(args, "coverage_samples", 16384)),
        "voxel_occ_bins": int(getattr(args, "voxel_occ_bins", 24)),
        "plane_chamfer_samples": args.plane_chamfer_samples,
        "proj_hist_samples": args.proj_hist_samples,
        "proj_hist_bins": args.proj_hist_bins,
        "proj_hist_sigma": args.proj_hist_sigma,
        "proj_hist_scale": args.proj_hist_scale,
        "dispersion_group_size": args.dispersion_group_size,
        "dispersion_margin": args.dispersion_margin,
        "dispersion_margin_frac": float(getattr(args, "dispersion_margin_frac", 0.35)),
        "spatial_bins": args.spatial_bins,
        "spatial_weight_power": args.spatial_weight_power,
        "spatial_weight_max": args.spatial_weight_max,
        "z_hard_frac": args.z_hard_frac,
        "z_hard_min_tokens": args.z_hard_min_tokens,
        "z_std_ratio_target": args.z_std_ratio_target,
        "z_target_detach": True,
    }
    return w


def lr_scales(step: int, args) -> Dict[str, float]:
    """Per-module learning-rate multipliers, including the encoder warmup."""
    # "attr" is deliberately absent from every cut below and stays at 1.0. The
    # polish phase freezes the encoder and compressor to stop the latent moving
    # under the decoders; the attribute decoder is downstream of a detached input,
    # so nothing it learns can disturb the latent and it has no reason to stop.
    # "attr_enc" is the encoder's appearance path. It is a separate group from
    # "encoder" on purpose: train.py zeroes "encoder" whenever the residual branch
    # is off, and that must not freeze the module that learns what appearance to
    # put in the latent. It still follows encoder's late-codec and polish cuts,
    # because those exist to stop the latent moving under the decoders.
    scales = {"encoder": 1.0, "attr_enc": 1.0, "compressor": 1.0, "decompressor": 1.0,
              "decoder": 1.0, "gen": 1.0, "attr": 1.0, "attr_slot": 1.0, "joint": 1.0}
    if step < args.encoder_warmup_steps:
        # let the latent codec settle before the decoders start chasing it
        scales["decoder"] = args.encoder_warmup_decoder_lr_scale
        scales["gen"] = args.encoder_warmup_gen_lr_scale
    phase = phase_of(step, args)
    # Delay late codec LR cut until shape/residual has had time to settle.
    # Default late_codec_start=-1 means gen_end (not gen_start).
    late_start = int(getattr(args, "late_codec_start", -1))
    if late_start < 0:
        late_start = int(getattr(args, "gen_end", args.gen_start))
    if step >= late_start and phase in ("gen", "polish"):
        scales["compressor"] = args.late_codec_lr_scale
        scales["encoder"] = args.late_codec_lr_scale
        scales["attr_enc"] = args.late_codec_lr_scale
    if phase == "polish":
        # Fine-grained polish controls.  The old hard-coded freeze also stopped
        # attr_enc, even though improving the 8-channel appearance code is the
        # main purpose of a render fine-tune. Defaults preserve the old policy;
        # R9 opens only the appearance path and tiny geometry corrections.
        scales["encoder"] = float(getattr(args, "polish_encoder_scale", 0.0))
        scales["attr_enc"] = float(getattr(args, "polish_attr_encoder_scale", 0.0))
        scales["compressor"] = float(getattr(args, "polish_compressor_scale", 0.0))
        scales["decompressor"] = float(getattr(args, "polish_decompressor_scale", 0.1))
        scales["decoder"] = float(getattr(args, "polish_decoder_scale", 1.0))
        scales["attr"] = float(getattr(args, "polish_attr_scale", 1.0))
        _as = float(getattr(args, "polish_attr_slot_scale", -1.0))
        scales["attr_slot"] = (float(getattr(args, "polish_attr_scale", 1.0))
                               if _as < 0 else _as)
    return scales


def global_lr(step: int, args) -> float:
    """Linear warmup then cosine decay of the base learning rate."""
    import math

    if step < args.lr_warmup_steps:
        return args.lr * float(step + 1) / float(max(args.lr_warmup_steps, 1))
    t = (step - args.lr_warmup_steps) / max(args.max_steps - args.lr_warmup_steps, 1)
    t = min(max(t, 0.0), 1.0)
    return args.lr_min + 0.5 * (args.lr - args.lr_min) * (1.0 + math.cos(math.pi * t))
