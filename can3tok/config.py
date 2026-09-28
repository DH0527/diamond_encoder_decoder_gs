"""Model configuration and the latent-budget arithmetic.

Everything that decides "how many channels does one point get" lives here, and
``describe_layout`` prints it at startup so a misconfigured run is obvious within
the first second instead of after 8k steps.

Key invariant of this code base
-------------------------------
The identity patch pack writes ``group_size * 4`` numbers per group
(``group_size`` xyz triplets + ``group_size`` mask flags). If
``local_tokens_per_group * latent_channels`` is larger than that, the extra
z_raw channels are *exactly zero* and the compressor spends part of the compact
budget encoding constants. The previous code base ran with

    patch_dim = 128, token_dim = 8 * 32 = 256   ->  half of z_raw was zero
    -> 16 of the 32 compact channels per cell encoded nothing

so we require ``tokens_per_group == patch_dim / latent_channels`` unless the user
explicitly opts out with ``allow_padded_tokens``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple


@dataclass
class Can3TokConfig:
    # ---------------- data / target ----------------
    max_points: int = 262144
    target_dim: int = 59
    sh_dim: int = 45
    input_dim: int = 63

    # ---------------- patch / latent geometry ----------------
    group_size: int = 32
    local_tokens_per_group: int = 4
    latent_channels: int = 32
    latent_hw: Tuple[int, int] = (256, 128)
    compact_latent_channels: int = 32
    compact_latent_hw: Tuple[int, int] = (64, 64)

    # ---------------- channel budget per group inside a compact cell ----------
    # centroid + occupancy + shape must sum to compact_latent_channels / merge
    # After dense truncate, per-group occupancy is nearly constant (+1), so the
    # default reclaims that channel for shape and conditions on global N/fill
    # instead (see StagedCompressor.global_to_mid).
    budget_centroid: int = 4
    # Occupancy would carry the per-group live count, but under prefix packing
    # every live group is full, so the channel is redundant and better spent on
    # shape. Only set it to 1 if groups can be partially filled.
    budget_occupancy: int = 0
    budget_shape: int = 12
    # Compact channels reserved for appearance (scale / rotation / opacity / colour).
    # 0 keeps the historical geometry-only latent bit-identical. The measured knee
    # is 8 per group: an 8-dimensional appearance code lifts a rank-28 geometry
    # from 17.05 to 21.43 dB on held-out poses, and 64 buys only 2.9 dB more
    # (MODEL.md 7.7). Taken out of budget_shape, whose cost is ~0.5 dB (7.1).
    attr_cond: str = "xattn"
    # How the per-group appearance code enters the attribute decoder:
    # "xattn" | "film" | "concat" | "film+cat". See AttributeDecoder.cond.
    attr_local_pe: bool = True
    # Feed the attribute decoder group-local coordinates alongside the
    # scene-normalised ones. See AttributeDecoder.loc_pe.
    budget_appearance: int = 0
    # Opt out of the shape-budget floor in `validate_layout`. Set it for a
    # plumbing test that builds a deliberately tiny config and only checks that
    # tensors flow, never for a training run: the floor exists because the L16k
    # run passed every other check at 0.012 shape channels per point and spent
    # 64000 steps on a bottleneck whose intra-group geometry measured
    # uncorrelated with its target.
    allow_low_shape_budget: bool = False
    # Attribute channels per point entering the pack. 0 = the historical
    # xyz-and-mask pack. 11 = log_scale 3 + quaternion 4 + logit opacity 1 +
    # SH DC 3, i.e. everything a sh_degree=0 rasteriser reads.
    attr_pack_dim: int = 0
    attr_pack_hidden: int = 64
    # Half-width of the neighbouring-group window the attribute decoder may read.
    # 0 reproduces the old behaviour (each group sees only its own code).
    attr_nbr_window: int = 0
    # Widths inserted between `mid` and the per-group budget heads, e.g. (128, 64).
    # Empty reproduces the old single Linear(mid -> c).
    compress_head_stages: tuple = ()
    # Extra width in the compressor's token_merge (T*d -> ... -> d).
    compress_merge_stages: int = 0
    # Pool a group's T tokens with a learned query instead of concatenating them.
    compress_merge_attn: bool = False
    # Emit all non-anchor channels through one head.  The old 19/8 heads made
    # geometry and rendering optimise disjoint compact tensors.
    shared_free_head: bool = False
    # Preserve local-token identity inside the fixed non-anchor budget. F3D uses
    # global[3] | local[4, 6] = 27 learned channels per group.
    structured_local_code: bool = False
    compact_global_dim: int = 3
    compact_local_tokens: int = 4
    compact_local_dim: int = 6
    # Points the ENCODER may read, as opposed to `max_points` which is how many
    # Gaussians the DECODER emits. Equal values reproduce the old behaviour.
    #
    # Why they must differ: the loader currently subsamples to max_points BEFORE
    # the encoder sees anything, so whatever is dropped is gone before the
    # bottleneck and a fixed hand-written rule decides what to drop. Measured on
    # eight over-budget snapshots, the discarded points are worth 2.50 dB on
    # average and up to 5.32 dB, and the loss does not track the keep ratio -- one
    # frame keeps 92% and still loses 5.17 dB while another keeps 56% and loses
    # 2.92 dB. What is dropped matters more than how much, which is exactly the
    # decision a learned encoder should be making.
    max_input_points: int = 0          # 0 -> same as max_points
    # Width and query count of the per-group learned pooling that replaces the
    # identity pack once the encoder reads more points than it emits.
    pool_dim: int = 128
    # Internal set queries, independent of the fixed z_raw storage token count.
    pool_queries: int = 16
    # Progressive compression: how many cross-attention reductions the pooler
    # stages instead of collapsing 2048 points into 16 queries in one shot.
    # 1 = the original single collapse. >1 inserts query self-attention between
    # reductions and (pool_feedback) lets the points re-read the running summary.
    # Width of the shared shape -> offset dictionary in StagedDecompressor.
    # 0 = the historical max(128, 4 * group_size).
    shape_xyz_hidden: int = 0
    pool_blocks: int = 1
    pool_feedback: bool = True
    # Use the transformed fixed scene anchors as the canonical group centres.
    # The encoder and output have different per-anchor capacities, so their
    # sample centroids cannot safely define the same latent coordinate frame.
    use_fixed_anchor_center: bool = False
    # Divide pooler local xyz by the same analytic extent stored in compact.
    # Off keeps the historical (xyz - center) input, whose RMS is ~1000x smaller
    # than the standardized attribute channels on thin cells.
    normalize_pooler_xyz: bool = False
    # Mean-center the folding template on the predicted live prefix, not all 256
    # slots. Partial cells otherwise start from a biased Fibonacci prefix.
    count_aware_template: bool = False
    # AttributeDecoder local PE / self-attention ignore inactive slots.
    attr_slot_mask: bool = False
    # -1 = derive from group_in so pool_chunk * group_in stays constant. A fixed
    # 512 made the pooler's attention memory scale with the per-cell cap, which
    # is the one number that has to be free to grow for variable-N scenes.
    pool_chunk: int = -1
    uniform_budget: bool = False  # ablation: plain n_tok x (C / n_tok) split

    # ---------------- widths ----------------
    model_dim: int = 384
    heads: int = 8
    dropout: float = 0.0
    num_freqs_xyz: int = 6
    num_freqs_slot: int = 4

    # ---------------- encoder ----------------
    encoder_residual: bool = True
    encoder_residual_hidden: int = 256
    map_blocks: int = 0  # ResBlock2d stem on z_raw before compress (analysis capacity)

    # ---------------- compressor ----------------
    compress_intra_layers: int = 3
    compress_window_layers: int = 2
    compress_window: int = 8
    compress_mid_channels: int = 16  # staged bottleneck: d -> mid -> budget
    # Optional wide mid stage on the compact grid (DC-AE-style). After semantic
    # pack to compact_latent_channels, expand to wide, refine with ResBlocks,
    # then project back to compact_latent_channels. 0 disables (identity).
    # DIAMOND still sees only the final compact tensor.
    compress_wide_channels: int = 0
    compress_wide_layers: int = 0
    decompress_intra_layers: int = 3
    decompress_window_layers: int = 2

    # ---------------- codec decoder ----------------
    decoder_layers: int = 4
    decoder_neighbor_context: bool = True
    # If True, cross memory sees neighbour *group tokens* (3x3 x merge) instead
    # of only neighbour cell summaries.
    decoder_neighbor_group_tokens: bool = True
    residual_scale: float = 0.6  # fraction of the group extent
    patch_chunk: int = 512
    use_slot_embedding: bool = True

    # ---------------- generative decoder ----------------
    use_gen_branch: bool = True
    # Cut the gen losses out of the latent's gradient. The gen decoder is strictly
    # weaker than the codec (its own rmse 0.0154 vs 0.0124, and gen_offset_scale
    # clamps its template), so co-optimising z_compact for it drives the latent
    # smooth: hf_ratio fell 0.145 -> 0.027 exactly across the gen ramp
    # (gen_start=6000, gen_ramp_steps=3000) while codec_rmse stopped improving.
    # Distillation still keeps gen tied to the codec output.
    gen_detach_latent: bool = True
    gen_window_layers: int = 3
    gen_group_layers: int = 1
    gen_cross_layers: int = 3
    gen_region_chunk: int = 512
    gen_offset_scale: float = 1.0  # in units of the group extent
    gen_use_slot_embedding: bool = True
    gen_neighbor_context: bool = True
    gen_neighbor_group_tokens: bool = True

    # ---------------- latent regularisation ----------------
    latent_mode: str = "deterministic"  # deterministic | vae
    latent_zorder: bool = True
    latent_scale_momentum: float = 0.01
    denoise_std: float = 0.02

    # ---------------- free-shortcut removal ----------------
    # If True, the identity pack stores (xyz - group_centroid) instead of absolute
    # xyz. Then plain z_l1 cannot be satisfied by broadcasting centroids alone.
    residual_pack: bool = True
    # tok = shortcut_alpha * shortcut + learned. Training sets this each step;
    # 0 means the decompressor must reconstruct from shape (no free centroid xyz).
    shortcut_alpha: float = 0.0
    # Eval may restore a little centroid assist for numerical stability; training
    # should stay near 0 until shape actually carries geometry.
    shortcut_alpha_eval: float = 0.0
    # FoldingNet / Scaffold-GS style decode: fixed unit-shell pattern, deformed by
    # an anisotropic frame from the first 3 shape channels, plus a *zero-init*
    # residual from the remaining channels. Avoids L2 collapse of shape_xyz into
    # a rank-~5 blob (measured effective rank 4.67 on the learned map).
    folding_decode: bool = False
    folding_frame_channels: int = 6  # 3 per-axis scales + 3 axis-angle rotation
    folding_res_cap: float = 1.0     # tanh bound on the residual, in template units
    # tanh bound on the folding frame's per-axis log-scale ratio. 0 keeps the
    # unbounded softplus form, which measured aniso 0.027..1411.5 and cost the
    # group shapes 1.6 of their 5.8 effective rank. See _shape_to_offsets.
    folding_aniso_log_cap: float = 1.5
    folding_res_gain: float = 1.0    # set per step by the training loop (ramp)
    teacher_cycle: bool = False      # also decode the true pack, for w_teacher_cycle
    # Detach the decoder's parameters in the teacher-cycle branch so the term
    # trains the compressor, not the decoder's sensitivity. See model.forward.
    teacher_cycle_freeze: bool = True
    # Codec decoder residual refine gate. 0 = coarse-only (identity unpack /
    # residual-pack + centroid). Prevents the refine head from hiding a dead
    # shape path. Set each step by the trainer; eval uses 1.0.
    decoder_refine_alpha: float = 1.0
    # Probability that an attribute head reads the *true* xyz instead of the
    # decoder's own. Set each step by the trainer (1 -> 0); eval always uses 0,
    # which is the inference condition.
    attr_teacher_prob: float = 1.0
    # tanh band on the emitted log-scale, in log units around the head's bias.
    # 3.0 = ~20x either way. 0 disables. See CodecDecoder._bounded_log_scale.
    attr_scale_log_cap: float = 3.0
    # Asymmetric band around a group-relative base. Both > 0 activates it and
    # supersedes attr_scale_log_cap. See AttributeDecoder._bounded_log_scale.
    attr_scale_cap_down: float = 0.0
    attr_scale_cap_up: float = 0.0
    attr_scale_group_base: bool = True
    # Let the attribute decoder read the cell's SHAPE channels alongside its
    # appearance channels. Measured motivation: with the position input detached,
    # the render gradient reaches only the 8 appearance channels of every 32, so
    # `shape` is trained by the point-space reconstruction terms and `appearance`
    # by the render -- two disjoint halves of each cell under two different
    # objectives. The channel-coalition ladder says {xyz, scale, opacity, colour}
    # only pay off when they move TOGETHER, and xyz comes from shape. This is the
    # one path that carries render gradient into shape WITHOUT going through
    # position transport, which failed four separate ways (nudge x4 = +0.02 dB,
    # detach released at step 0 = -1.0 dB, staged release worse than joint, and
    # always-open worse than released-at-4000). Costs zero latent channels.
    attr_read_shape: bool = False
    # Function-preserving shared refiner.  It reads every non-anchor compact
    # channel and predicts bounded residuals for all Gaussian parameters from a
    # single slot representation; see joint_decoder.py.
    joint_shared_decoder: bool = False
    # Replacement, rather than residual, shared Gaussian decoder.
    joint_direct_decoder: bool = False
    # Hierarchical direct decoder: reconstruct T local tokens from each compact
    # cell and let the G Gaussian queries cross-attend to them.  The old direct
    # path broadcast one group vector to every query, which converged to one
    # low-rank template repeated across the scene.
    joint_local_memory: bool = False
    joint_memory_layers: int = 1
    joint_direct_xyz_cap: float = 1.5
    # Absolute scene-normalised group translation around the fixed anchor.
    joint_translation_cap: float = 0.25
    joint_decoder_dim: int = 192
    joint_decoder_layers: int = 2
    joint_decoder_chunk: int = 256
    joint_nbr_window: int = 1
    joint_xyz_cap: float = 0.10
    joint_scale_delta_cap: float = 1.0
    joint_opacity_delta_cap: float = 2.0
    joint_color_delta_cap: float = 0.5
    joint_rot_delta_cap: float = 0.25
    joint_detach_render_xyz: bool = True
    # Separate attribute decoder. `attr_decoder_layers = 0` keeps the old
    # in-geometry-decoder heads, which share the refine token and therefore let
    # the render loss reach the geometry stack. >0 builds AttributeDecoder, whose
    # inputs from the geometry side are all detached.
    attr_decoder_layers: int = 0
    attr_decoder_dim: int = 256
    # tanh cap on the render-driven position nudge, in units of the group extent.
    # 0.15 is a little over one projected Gaussian footprint (measured: 1.40 px
    # against a 12.0 px group radius), i.e. the range where the image gradient
    # still points at the right place -- at 6 px its cosine with the true
    # correction is 0.054, at 0.6 px it is 0.134.
    attr_nudge_cap: float = 0.15
    # Detach the geometry decoder's positions before the attribute decoder reads
    # them. True reproduces the historical behaviour, in which the render loss
    # could not reach geometry at all. See Can3TokAE.forward.
    attr_detach_geometry: bool = True
    # AttributeDecoder emits scale along the folding cell-frame axes, and
    # rotation as a residual around that frame (identity residual at init).
    # Without this, scale/rot are free per slot and photometric's cheap
    # answer is a mid-size sphere (C1 40000 aniso p50 3.88 vs GT 16.9;
    # Sobel 2000 steps pushed it to 3.48).
    attr_frame_needles: bool = False
    # Photometric / VGG / Sobel do not flow to scale or rotation. Colour and
    # opacity still do. Measured: raising photometric weight rounded Gaussians
    # further; stretching them without a tied orientation smeared the image (H2).
    attr_detach_scale_rot: bool = False
    # Let the patch pack learn. False keeps the frozen identity matrix, which is
    # correct only while a z_raw reconstruction loss defines what the pack means.
    pack_trainable: bool = False

    # ---------------- misc ----------------
    checkpoint_decode: bool = True
    checkpoint_gen: bool = False

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# layout arithmetic
# ---------------------------------------------------------------------------


def patch_layout(cfg: Can3TokConfig) -> Dict[str, int]:
    g = int(cfg.group_size)
    tpg = int(cfg.local_tokens_per_group)
    num_groups = int(math.ceil(float(cfg.max_points) / float(g)))
    raw_cells = int(cfg.latent_hw[0]) * int(cfg.latent_hw[1])
    raw_cells_used = num_groups * tpg
    compact_cells = int(cfg.compact_latent_hw[0]) * int(cfg.compact_latent_hw[1])
    merge = int(math.ceil(num_groups / float(compact_cells)))
    return {
        "group_size": g,
        "tokens_per_group": tpg,
        "num_groups": num_groups,
        "patch_dim": g * 4,
        "token_dim": tpg * int(cfg.latent_channels),
        "raw_cells": raw_cells,
        "raw_cells_used": raw_cells_used,
        "compact_cells": compact_cells,
        "merge": merge,
        "tokens_per_cell": merge * tpg,
        "points_per_cell": merge * g,
    }


def channel_budget(cfg: Can3TokConfig) -> Dict[str, int]:
    """Split ``compact_latent_channels`` into per-group semantic slots."""
    lay = patch_layout(cfg)
    merge = lay["merge"]
    total = int(cfg.compact_latent_channels)
    if total % merge != 0:
        raise ValueError(
            f"compact_latent_channels={total} not divisible by merge={merge}; "
            "adjust compact_latent_hw or group_size"
        )
    per_group = total // merge

    if cfg.uniform_budget:
        n_tok = lay["tokens_per_cell"]
        if total % n_tok != 0:
            raise ValueError(f"uniform budget needs {total} % {n_tok} == 0")
        cpt = total // n_tok
        return {
            "merge": merge,
            "per_group": per_group,
            "centroid": 0,
            "occupancy": 0,
            "shape": per_group,
            "appearance": 0,          # uniform budget has no semantic split at all
            "uniform_tokens": n_tok,
            "uniform_channels": cpt,
        }

    c_cen = int(cfg.budget_centroid)
    c_occ = int(cfg.budget_occupancy)
    c_shp = int(cfg.budget_shape)
    c_app = int(getattr(cfg, "budget_appearance", 0))
    if c_cen < 4:
        raise ValueError(
            "budget_centroid must be >= 4: xyz of the group centroid plus its log-extent, "
            "which every within-group position head needs as its unit of length"
        )
    if c_cen + c_occ + c_shp + c_app != per_group:
        raise ValueError(
            f"budget_centroid+budget_occupancy+budget_shape+budget_appearance="
            f"{c_cen + c_occ + c_shp + c_app} "
            f"must equal compact_latent_channels/merge={per_group}"
        )
    # Order matters: every consumer slices by prefix, so appearance goes last and
    # c_app = 0 leaves the historical layout byte-identical.
    return {
        "merge": merge,
        "per_group": per_group,
        "centroid": c_cen,
        "occupancy": c_occ,
        "shape": c_shp,
        "appearance": c_app,
        "uniform_tokens": 0,
        "uniform_channels": 0,
    }


def validate_layout(cfg: Can3TokConfig, allow_padded_tokens: bool = False) -> Dict:
    lay = patch_layout(cfg)
    if lay["raw_cells"] < lay["raw_cells_used"]:
        raise ValueError(
            f"latent_hw={tuple(cfg.latent_hw)} gives {lay['raw_cells']} cells but "
            f"{lay['raw_cells_used']} are needed (groups={lay['num_groups']} x "
            f"tokens_per_group={lay['tokens_per_group']})"
        )
    if lay["token_dim"] < lay["patch_dim"]:
        raise ValueError(
            f"token_dim={lay['token_dim']} < patch_dim={lay['patch_dim']}: the identity "
            "pack would be lossy. Increase local_tokens_per_group or latent_channels."
        )
    if lay["token_dim"] > lay["patch_dim"] and not allow_padded_tokens:
        waste = lay["token_dim"] - lay["patch_dim"]
        raise ValueError(
            f"token_dim={lay['token_dim']} > patch_dim={lay['patch_dim']}: {waste} of every "
            f"{lay['token_dim']} z_raw channels are provably zero and would eat "
            f"{waste * int(cfg.compact_latent_channels) // lay['token_dim']} compact channels "
            f"per cell. Set local_tokens_per_group={lay['patch_dim'] // int(cfg.latent_channels)} "
            "or pass allow_padded_tokens=True on purpose."
        )
    budget = channel_budget(cfg)
    # Shape budget against the points each cell has to reconstruct.
    #
    # The checks above catch arithmetic mistakes and provably-zero z_raw channels.
    # They do NOT catch the configuration that actually shipped: L16k ran
    # group_size=256 with budget_shape=3, i.e. 256 points and 768 intra-group
    # coordinates reconstructed from three free scalars. Everything validated,
    # `describe_layout` printed "0.062 compact channels per point", and the run
    # spent 64000 steps on a bottleneck that could not represent its target.
    #
    # It was measurable at step 8000 and never moved after: codec_xyz_rmse_norm
    # 0.03316 -> 0.03274 over the next 51000 steps, and rel_offset_p50 settled at
    # 1.414 == sqrt(2), which is exactly the value uncorrelated offsets of matched
    # magnitude produce. The intra-group geometry was noise.
    #
    # Thresholds are per free shape channel per point. The last configuration this
    # codebase reached its stated targets on had shape=19 over 64 points (0.297);
    # the failure above was 0.012. The floor is set two orders of magnitude below
    # the working point and the warning an order below, so neither fires on any
    # layout that has previously worked.
    ppc = int(lay["points_per_cell"]) * max(int(lay["merge"]), 1)
    shape_per_point = float(budget["shape"]) / float(max(ppc, 1))
    lax = bool(getattr(cfg, "allow_low_shape_budget", False))
    if budget["shape"] > 0 and shape_per_point < 0.02 and not lax:
        raise ValueError(
            f"budget_shape={budget['shape']} for {ppc} points per group is "
            f"{shape_per_point:.4f} free shape channels per point. Intra-group "
            f"geometry is not representable at this rate -- the L16k run measured "
            f"rel_offset_p50=1.414 (uncorrelated with the target) at 0.012. "
            f"Raise --budget_shape, lower --group_size, or set "
            f"allow_low_shape_budget=True to override deliberately."
        )
    if budget["shape"] > 0 and shape_per_point < 0.05 and not lax:
        print(
            f"[layout] WARNING: budget_shape={budget['shape']} over {ppc} points "
            f"per group = {shape_per_point:.4f} shape channels per point. Expect "
            f"intra-group geometry to saturate early; watch rel_offset_p50 "
            f"(1.0 = no better than the centroid).",
            flush=True,
        )
    if budget.get("appearance", 0) > 2 * max(budget["shape"], 1):
        print(
            f"[layout] WARNING: appearance={budget['appearance']} is more than "
            f"twice shape={budget['shape']}. Geometry is usually the binding "
            f"constraint; check codec_xyz_rmse_norm against the attribute NRMSEs "
            f"before accepting this split.",
            flush=True,
        )
    return {"layout": lay, "budget": budget}


def describe_layout(cfg: Can3TokConfig, allow_padded_tokens: bool = False) -> str:
    info = validate_layout(cfg, allow_padded_tokens=allow_padded_tokens)
    lay = info["layout"]
    bud = info["budget"]
    eff = float(cfg.compact_latent_channels) / float(lay["points_per_cell"])
    lines = [
        "----- can3tok latent layout -----",
        f"points          : max_points={cfg.max_points} group_size={lay['group_size']} "
        f"num_groups={lay['num_groups']}",
        f"z_raw           : {cfg.latent_channels}x{cfg.latent_hw[0]}x{cfg.latent_hw[1]} "
        f"(cells used {lay['raw_cells_used']}/{lay['raw_cells']}) "
        f"tokens_per_group={lay['tokens_per_group']} token_dim={lay['token_dim']} "
        f"patch_dim={lay['patch_dim']}",
        f"z_compact       : {cfg.compact_latent_channels}x{cfg.compact_latent_hw[0]}x"
        f"{cfg.compact_latent_hw[1]} cells={lay['compact_cells']} merge={lay['merge']} "
        f"points/cell={lay['points_per_cell']}",
        f"budget/group    : centroid={bud['centroid']} occupancy={bud['occupancy']} "
        f"shape={bud['shape']} appearance={bud.get('appearance', 0)} (sum={bud['per_group']})",
        (f"structured code : global={int(getattr(cfg, 'compact_global_dim', 3))} "
         f"local={int(getattr(cfg, 'compact_local_tokens', 4))}x"
         f"{int(getattr(cfg, 'compact_local_dim', 6))} (direct encoder-token pooling)"
         if bool(getattr(cfg, "structured_local_code", False))
         else "structured code : disabled"),
        f"attribute path  : attr_pack_dim={int(getattr(cfg, 'attr_pack_dim', 0))} "
        f"({'appearance reaches z_compact' if int(getattr(cfg, 'attr_pack_dim', 0)) > 0 else 'XYZ-ONLY -- attributes never reach the latent'})",
        f"effective       : {eff:.3f} compact channels per point"
        f"  [old 262k run: 0.250, old 131k run: 0.500]",
        f"latent layout   : {'z-order 2D' if cfg.latent_zorder else 'row-major'} | "
        f"mode={cfg.latent_mode}",
        f"free-path fix   : residual_pack={cfg.residual_pack} "
        f"shortcut_alpha={cfg.shortcut_alpha} (eval={cfg.shortcut_alpha_eval})",
        f"contracts       : pooler_xyz_norm={int(getattr(cfg, 'normalize_pooler_xyz', False))} "
        f"count_template={int(getattr(cfg, 'count_aware_template', False))} "
        f"attr_slot_mask={int(getattr(cfg, 'attr_slot_mask', False))}",
        f"mid capacity    : model_dim={cfg.model_dim} compress_mid={cfg.compress_mid_channels} "
        f"map_blocks={cfg.map_blocks} decoder_layers={cfg.decoder_layers} "
        f"z_mid={cfg.compress_wide_channels}x{cfg.compact_latent_hw[0]}x{cfg.compact_latent_hw[1]} "
        f"(layers={cfg.compress_wide_layers})",
        "---------------------------------",
    ]
    return "\n".join(lines)
