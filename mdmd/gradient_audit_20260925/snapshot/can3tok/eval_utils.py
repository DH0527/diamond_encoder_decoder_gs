"""Evaluation metrics, latent diagnostics and side-by-side visualisation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .io_utils import denormalize_xyz, save_json, write_ply
from .losses import symmetric_chamfer

EVAL_CONTRACT_VERSION = "scene_tagged_v1"


@dataclass
class EvalView:
    """One held-out photograph, tagged with the scene it belongs to."""
    scene_id: str
    image_id: str
    camera: np.ndarray
    photo: Optional[torch.Tensor]
    pool_index: int = -1


def scene_ids_match(a, b) -> bool:
    if a is None or b is None:
        return False
    sa, sb = str(a).rstrip("/"), str(b).rstrip("/")
    if sa == sb:
        return True
    return os.path.basename(sa) == os.path.basename(sb)


def filter_eval_views(views, scene_id: Optional[str]) -> List:
    """Keep only views whose scene tag matches ``scene_id``.

    Untagged ``(cam, photo)`` tuples are returned unchanged so older single-scene
    callers keep working. A tagged list with no match is empty -- the caller
    must treat that as an error, not fall back to every scene's cameras.
    """
    if not views:
        return []
    tagged = [v for v in views if isinstance(v, EvalView)]
    if not tagged:
        return list(views)
    if scene_id is None:
        return tagged
    return [v for v in tagged if scene_ids_match(v.scene_id, scene_id)]


def eval_fingerprint(args, held_view_idx) -> Dict:
    """Identity of the score that ``ckpt_best`` is selected on.

    Changing the metric, the held-out split, or the eval contract without
    resetting ``best_score`` is how T kept S's ``-PSNR`` after switching to
    ``psnr_gap_worst``.
    """
    return {
        "contract": EVAL_CONTRACT_VERSION,
        "score_metric": str(getattr(args, "score_metric", "")),
        "held_views": sorted(int(i) for i in (held_view_idx or [])),
        "eval_val_indices": str(getattr(args, "eval_val_indices", "")),
        "eval_pred_mask": int(getattr(args, "eval_pred_mask", 0)),
        "eval_view_force": str(getattr(args, "eval_view_force", "")),
        "eval_view_count": int(getattr(args, "eval_view_count", 0)),
        "eval_view_seed": int(getattr(args, "eval_view_seed", 0)),
    }


def unpack_eval_view(view) -> tuple:
    if isinstance(view, EvalView):
        return view.camera, view.photo, view.scene_id, view.image_id, int(view.pool_index)
    cam, photo = view
    return cam, photo, "", "", -1


@torch.no_grad()
def compute_eval_metrics(
    pred: torch.Tensor,
    presence: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    chamfer_samples: int = 32768,
    attr_pred: Optional[torch.Tensor] = None,
    attr_target: Optional[torch.Tensor] = None,
    group_size: int = 0,
) -> Dict[str, float]:
    m = mask > 0.5
    p = pred[..., 0:3][m]
    g = target[..., 0:3][m]
    if p.numel() == 0:
        return {"xyz_rmse_norm": float("inf")}
    diff = p - g
    rmse = float(torch.sqrt((diff ** 2).sum(dim=-1).mean()))
    out = {
        "xyz_rmse_norm": rmse,
        "xyz_rmse_metric": rmse * float(scale),
        "xyz_l1_norm": float(diff.abs().mean()),
        "xyz_max_norm": float(diff.abs().max()),
    }
    # Attribute error, relative to the target's own spread. Without this the eval
    # is silent about the only thing an attribute run is testing -- the first
    # attempt shipped with `with_attrs` on and produced no attribute number at all,
    # so "are attributes being learned" could not be answered from any output.
    # Normalised per channel group because a quaternion component and a log scale
    # are not comparable in absolute terms; 1.0 means "no better than predicting
    # the target's mean".
    # Attributes come from `attr_pred` when the separate decoder is running. They
    # have to: nothing supervises the geometry decoder's own attribute channels
    # any more, so reading them here would report the error of an untrained head
    # and the step-4000 gate would be measured on noise.
    ap = pred if attr_pred is None else attr_pred
    if ap.shape[-1] > 3 and target.shape[-1] >= 14:
        for name, sl in (("scale", slice(3, 6)), ("rot", slice(6, 10)),
                         ("opacity", slice(10, 11)), ("color", slice(11, 14))):
            if ap.shape[-1] < sl.stop:
                continue
            pv, gv = ap[..., sl][m], target[..., sl][m]
            sd = gv.std(dim=0, keepdim=True).clamp(min=1e-6)
            if name == "rot":
                # q and -q are the same rotation, so a raw component difference is
                # not an error measure. Untrained heads emit a constant (1,0,0,0)
                # and score better than a model that has learned the rotations but
                # picked the other hemisphere -- this metric read 5.94 at init and
                # 28.05 after the head started learning, while the sign-invariant
                # training loss went the right way the whole time.
                pv = pv * torch.sign((pv * gv).sum(-1, keepdim=True)).clamp(min=-1)
            out[f"attr_{name}_nrmse"] = float(
                ((pv - gv) / sd).pow(2).mean().clamp(min=0).sqrt()
            )
            # ...and again against the target the decoder is actually trained on.
            # Slot i of the prediction is not slot i of the npz -- the pairing is
            # the nearest GT point only 7% of the time -- so the number above
            # measures an arbitrary correspondence and cannot go to zero even for
            # a perfect model. This one asks the answerable question: does each
            # Gaussian match what the ground it covers calls for?
            if attr_target is not None:
                tv = attr_target[..., sl][m]
                if name == "rot":
                    tv = tv * torch.sign((tv * gv).sum(-1, keepdim=True)).clamp(min=-1)
                out[f"attr_{name}_nrmse_resp"] = float(
                    ((pv - tv) / sd).pow(2).mean().clamp(min=0).sqrt()
                )
            # Spread, not just error: an L1 loss under uncertainty is minimised by
            # the conditional mean, so the cheapest way to cut the error above is
            # to stop varying. That failure is invisible in an nrmse -- predicting
            # the mean everywhere scores 1.0, which reads as "mediocre" rather than
            # "collapsed" -- and it is what a group budget of 8 appearance channels
            # against 64 slots invites. A previous run's student emitted colour at
            # 19% of the ground truth's spread and rendered as a near-uniform plate.
            #
            # Reported against the target's own spread, so 1.0 means "as varied as
            # it should be". The within-group half is the one at risk: 52-78% of
            # the attribute variance lives inside a group, and the only per-point
            # inputs the decoder has there are the point's own position and the
            # slot basis.
            if group_size > 1:
                ref = gv if attr_target is None else attr_target[..., sl][m]
                n = (pv.shape[0] // group_size) * group_size
                if n >= group_size:
                    pg = pv[:n].reshape(-1, group_size, pv.shape[-1])
                    rg = ref[:n].reshape(-1, group_size, ref.shape[-1])
                    out[f"attr_{name}_spread"] = float(
                        pv.std() / ref.std().clamp(min=1e-9))
                    out[f"attr_{name}_spread_within"] = float(
                        (pg - pg.mean(1, keepdim=True)).std()
                        / (rg - rg.mean(1, keepdim=True)).std().clamp(min=1e-9))
    if presence is not None and presence.numel() == mask.numel():
        acc = ((presence > 0).float() == mask).float().mean()
        out["presence_acc"] = float(acc)
    n = min(int(chamfer_samples), p.shape[0])
    if n >= 64:
        idx = torch.randperm(p.shape[0], device=p.device)[:n]
        out["chamfer_norm"] = float(symmetric_chamfer(p[idx], g[idx]))
    return out


@torch.no_grad()
def render_eval_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    center,
    scale: float,
    views,
    layout: Dict[str, int],
    downscale: int = 2,
    scene_id: Optional[str] = None,
    pred_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Held-out photographic PSNR -- the metric this model is actually judged on.

    Every other number in this file is a proxy, and the proxies have now been
    measured to disagree with the target: over steps 2000-6000 of the Q1 run
    ``intra_chamfer`` (which ``ckpt_best`` was selected on) improved 0.280 ->
    0.254 while ``xyz_rmse`` went 0.0143 -> 0.0163 and the attribute errors
    diverged. A run cannot be steered by a quantity it never computes.

    Three numbers, all on views the optimiser never sees:

    ``psnr_photo``     prediction against the real photograph. The objective.
    ``psnr_teacher``   prediction against a render of the target Gaussians.
                       Separates "the autoencoder lost it" from "the snapshot
                       never had it" -- these are mid-optimisation 3DGS states.
    ``psnr_gt_photo``  the target Gaussians against the photograph, i.e. the
                       ceiling this snapshot allows. Logged every eval because
                       it moves with the snapshot (18.6 dB early in the 3DGS
                       run, 21.3 dB late) and a prediction PSNR is unreadable
                       without it.

    Returns ``{}`` rather than raising when the CUDA rasteriser is missing, so a
    CPU smoke run still exercises everything else.
    """
    try:
        from .render import Camera, psnr, render_gaussians, sh_dc_to_rgb, ssim
        from .io_utils import camera_from_vector
    except Exception:
        return {}
    views = filter_eval_views(views, scene_id)
    if not views:
        return {}
    a = layout
    dev = pred.device
    m_gt = mask > 0.5
    m_pr = (pred_mask > 0.5) if pred_mask is not None else m_gt
    if int(m_gt.sum()) < 64 or int(m_pr.sum()) < 64:
        return {}
    idx_gt = torch.nonzero(m_gt, as_tuple=False).reshape(-1)
    idx_pr = torch.nonzero(m_pr, as_tuple=False).reshape(-1)
    c = (center.to(dev) if torch.is_tensor(center)
         else torch.as_tensor(center, device=dev)).reshape(1, 3).float()
    s = float(scale)

    def unpack(t: torch.Tensor, idx: torch.Tensor):
        t = t[idx].float()
        return (
            t[:, a["xyz"]: a["xyz"] + 3] * s + c,
            torch.exp(t[:, a["scale"]: a["scale"] + 3].clamp(-15.0, 5.0)) * s,
            t[:, a["rot"]: a["rot"] + 4],
            torch.sigmoid(t[:, a["opacity"]: a["opacity"] + 1]),
            sh_dc_to_rgb(t[:, a["color"]: a["color"] + 3]),
        )

    gp, gt = unpack(pred, idx_pr), unpack(target, idx_gt)
    acc: Dict[str, list] = {}
    used_ids: List[str] = []
    for view in views:
        cam_vec, photo, view_scene, image_id, _pool_i = unpack_eval_view(view)
        if scene_id is not None and view_scene and not scene_ids_match(view_scene, scene_id):
            raise RuntimeError(
                f"cross-scene eval view {view_scene!r} used for scene {scene_id!r}"
            )
        cv = np.asarray(cam_vec, np.float32).reshape(-1)
        if float(cv[0]) <= 0.0:
            continue
        cam = Camera(camera_from_vector(cv), device=dev, downscale=int(downscale))
        img_p = render_gaussians(*gp, cam)
        img_g = render_gaussians(*gt, cam)
        acc.setdefault("psnr_teacher", []).append(psnr(img_p, img_g))
        acc.setdefault("ssim_teacher", []).append(ssim(img_p, img_g))
        if photo is None:
            continue
        ref = photo.to(dev).float()
        if ref.dim() != 3 or ref.shape[0] != 3:
            continue
        if ref.shape[-2:] != img_p.shape[-2:]:
            ref = torch.nn.functional.interpolate(
                ref[None], size=img_p.shape[-2:], mode="bilinear", align_corners=False)[0]
        acc.setdefault("psnr_photo", []).append(psnr(img_p, ref))
        acc.setdefault("ssim_photo", []).append(ssim(img_p, ref))
        acc.setdefault("psnr_gt_photo", []).append(psnr(img_g, ref))
        # The GT Gaussians' own SSIM against the photograph. Without it a model
        # whose psnr_gt_photo gap has gone NEGATIVE cannot be read: beating the
        # target Gaussians on PSNR is what an over-smoothed render does when the
        # target has sharp detail that is slightly misaligned with the photo,
        # because PSNR punishes a small offset more than it punishes blur. SSIM
        # separates the two, and there was no baseline to compare against.
        acc.setdefault("ssim_gt_photo", []).append(ssim(img_g, ref))
        used_ids.append(image_id or f"pool{_pool_i}")
    out = {k: float(np.mean(v)) for k, v in acc.items() if v}
    out["n_eval_views"] = float(len(used_ids))
    if used_ids:
        out["_eval_view_ids"] = used_ids
    # What the autoencoder itself costs, with the snapshot's own quality divided
    # out. The eval set spans 3DGS states from step_000010 (ceiling 12.08 dB, the
    # scene barely exists yet) to step_005070 (20.47 dB), so a raw mean PSNR mixes
    # "how good is the model" with "which snapshots were drawn". This is the half
    # the model is responsible for.
    if "psnr_gt_photo" in out and "psnr_photo" in out:
        out["psnr_gap"] = out["psnr_gt_photo"] - out["psnr_photo"]
    return out


@torch.no_grad()
def latent_diagnostics(
    z: torch.Tensor, per_group: int = 0, c_anchor: int = 0
) -> Dict[str, float]:
    """Statistics that predict how hard the latent will be for a diffusion model.

    ``hf_ratio`` is the share of 2D spectral energy above half the Nyquist
    frequency; latents dominated by high frequencies are known to hurt
    diffusability (arXiv 2502.14831).

    The whole-tensor versions of these are dominated by the centroid channels
    (std ~0.81) and the log-extent channel (std ~0.13), which are anchored to
    analytic quantities: ``latent_chan_std_min/max`` came out bit-identical across
    consecutive evals of a run whose codec error was still moving, i.e. they said
    nothing at all about the 12 shape channels that carry 99.3% of the error. Pass
    ``per_group``/``c_anchor`` to also get the shape-channel-only versions, which
    are the ones worth watching.
    """
    zf = z.detach().float()
    b, c, h, w = zf.shape

    def _spectral(t: torch.Tensor) -> float:
        spec = torch.fft.rfft2(t, norm="ortho").abs() ** 2
        fy = torch.fft.fftfreq(h, device=t.device).abs().view(1, 1, h, 1)
        fx = torch.fft.rfftfreq(w, device=t.device).abs().view(1, 1, 1, -1)
        radial = torch.sqrt(fy ** 2 + fx ** 2)
        return float(spec[(radial > 0.25).expand_as(spec)].sum() / spec.sum().clamp(min=1e-12))

    per_channel_std = zf.flatten(2).std(dim=-1).mean(dim=0)
    out = {
        "latent_std": float(zf.std()),
        "latent_absmean": float(zf.abs().mean()),
        "latent_max": float(zf.abs().max()),
        "latent_chan_std_min": float(per_channel_std.min()),
        "latent_chan_std_max": float(per_channel_std.max()),
        "latent_hf_ratio": _spectral(zf),
    }
    if per_group > 0:
        sel = (torch.arange(c, device=zf.device) % per_group) >= int(c_anchor)
        if bool(sel.any()):
            zs = zf[:, sel]
            ss = per_channel_std[sel]
            out.update(
                shape_hf_ratio=_spectral(zs),
                shape_chan_std_min=float(ss.min()),
                shape_chan_std_max=float(ss.max()),
                shape_chan_std_mean=float(ss.mean()),
            )
    return out


@torch.no_grad()
def group_error_breakdown(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, group_size: int
) -> Dict[str, float]:
    """Split the point error the way the latent splits the signal.

        point error = group centroid error + within-group offset error

    ``rel_offset_*`` is ``offset_rmse / gt group radius`` -- the scale-free "how
    mushy is each group" number. It tracks what the renders look like, whereas the
    plain rmse is dominated by the ~1% of Morton-jump groups that hold most of the
    squared error. 1.0 means the group is no better than putting every one of its
    points on the centroid.
    """
    # Group by the SLOT LAYOUT, not by compacting the mask first.
    #
    # Compacting concatenates every live point and then cuts the result into runs
    # of `group_size`. That is only equivalent to the real grouping when the live
    # slots are a prefix, which is true for Morton chunks and false for fixed
    # anchors, where each group is filled to its own count. Under anchors the
    # compacted runs straddle anchor boundaries, and every statistic below --
    # radius, anisotropy, template rank, thread, intra-chamfer -- is then computed
    # on groups that do not exist. Measured on an anchor run it reported a GT
    # group radius p99.9 of 0.955 against a true 0.087, i.e. groups spanning the
    # whole scene.
    #
    # Every statistic here is a fixed-size shape descriptor, so groups have to
    # enter at a common width. Under prefix packing every live group is full and
    # the width is `group_size`; under anchors the fill runs 8-42% full, so
    # requiring full groups would silently restrict the diagnosis to the densest
    # regions. Take the widest prefix that still admits a representative sample,
    # and report which one was used so the number is readable.
    gs = int(group_size)
    b, n, _ = pred.shape
    cnt = (mask.reshape(b, -1, gs) > 0.5).sum(dim=-1).reshape(-1)
    q = gs
    for cand in (gs, (3 * gs) // 4, gs // 2, gs // 4):
        if cand >= 8 and int((cnt >= cand).sum()) >= max(64, int(0.15 * cnt.numel())):
            q = cand
            break
    sel = cnt >= q
    if int(sel.sum()) < 8:
        return {}
    # Slot order inside a group is distance-to-anchor rank (or the template
    # assignment), so the leading q slots are a consistent subset on both sides.
    p = pred[..., 0:3].reshape(-1, gs, 3)[sel][:, :q]
    g = target[..., 0:3].reshape(-1, gs, 3)[sel][:, :q]
    ng = p.shape[0]
    gs = q
    pg = p.float()
    gg = g.float()
    cp, cg = pg.mean(1), gg.mean(1)
    op, og = pg - cp[:, None], gg - cg[:, None]
    cen_sq = ((cp - cg) ** 2).sum(-1)
    off_sq = ((op - og) ** 2).sum(-1).mean(-1)
    rad = og.pow(2).sum(-1).mean(-1).clamp(min=1e-18).sqrt()
    rel = off_sq.clamp(min=0).sqrt() / rad
    tot = float(cen_sq.mean() + off_sq.mean())

    # Per-group anisotropy. A group collapsed onto a straight line still scores
    # well on rmse and chamfer -- the line runs along the group's principal axis,
    # which carries most of the variance -- but has no lateral detail at all. This
    # is the failure an axis-monotone slot order produces, and rmse cannot see it.
    def _lam2(o: torch.Tensor) -> float:
        ev = torch.linalg.eigvalsh(torch.einsum("gpi,gpj->gij", o, o)).clamp(min=0)
        return float((ev[:, 1] / ev[:, 2].clamp(min=1e-20)).sqrt().median())

    # How many *distinct* group shapes the decoder actually produces, as the
    # exponential of the spectral entropy of the scale-normalised offset vectors.
    # This is the number behind the "same stroke repeated everywhere" look, and no
    # other metric here sees it: the _fix run sat at ~11 against ~44 in the ground
    # truth while its rmse and chamfer both kept improving. The shape channels can
    # be perfectly diverse (latent std healthy, hf_ratio healthy) and this still
    # collapses, because the collapse is in the shared shape -> offset map.
    def _erank(o: torch.Tensor) -> float:
        s = o.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-9)
        x = (o / s).reshape(o.shape[0], -1)
        x = x - x.mean(0, keepdim=True)
        ev = torch.linalg.eigvalsh(x.T @ x / max(x.shape[0] - 1, 1)).clamp(min=0)
        p = ev / ev.sum().clamp(min=1e-20)
        p = p[p > 0]
        return float(torch.exp(-(p * p.log()).sum()))

    # THE gate. Everything else here is slot-ordered, so it moves when the decoder
    # merely permutes its output: at step 3000 of the group_cov run rel_offset went
    # 0.758 -> 1.011 and rmse 0.01571 -> 0.01453 while this stayed at 0.393 -> 0.396,
    # i.e. nothing about the reconstructed point *set* had changed. Normalised by
    # the mean (not rms) group radius so it is comparable to the measured ladder:
    #   0.345  best this codebase ever reached (_fix @6000, plain shape_xyz MLP)
    #   0.299  folding envelope alone, zero training
    #   0.225  folding + rank-22 residual on a template-aligned target
    #   0.191  the GT cloud's own nearest-neighbour spacing, i.e. 1.0x
    rad_mean = og.norm(dim=-1).mean(-1).clamp(min=1e-9)

    def _intra_chamfer(a: torch.Tensor, b: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        vals = []
        for i in range(0, a.shape[0], chunk):
            d = torch.cdist(a[i : i + chunk], b[i : i + chunk])
            vals.append(0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)))
        return torch.cat(vals) / rad_mean

    ich = _intra_chamfer(op, og)

    # How many *distinct* GT points the prediction reaches, per group. A symmetric
    # chamfer does see duplication -- the uncovered GT point is paid for by the
    # g->p half -- but far too weakly: the duplicate itself scores a perfect p->g,
    # and the single uncovered point's cost is averaged over the whole group. So
    # the correct statement is that chamfer is *insufficiently sensitive to
    # many-to-one correspondence*, not that it is blind to it. That weakness is
    # not hypothetical -- rendering the step-2000 v4 checkpoint and matching
    # each predicted point to its nearest GT point gave 0.540 distinct against
    # 0.991 for the selected GT subset, i.e. 46% of the point budget was spent
    # twice while the corresponding surface went uncovered. It is the single
    # largest gap between the point-space score (1.23x spacing, respectable) and
    # the render (17.9 dB, poor), so it needs to be visible every eval.
    def _uniq(a: torch.Tensor, b: torch.Tensor, chunk: int = 512) -> float:
        tot = 0.0
        for i in range(0, a.shape[0], chunk):
            nn = torch.cdist(a[i : i + chunk], b[i : i + chunk]).argmin(dim=2)
            hit = torch.zeros_like(nn, dtype=torch.bool).scatter_(1, nn, True)
            tot += float(hit.sum())
        return tot / float(a.shape[0] * a.shape[1])
    # Consecutive-slot distance over radius. Near 0 means the group decoded as one
    # continuous thread (the streak artifact); far above the GT value means the
    # slots were scrambled, which is what moment-matching losses buy.
    thr = lambda o: float(((o[:, 1:] - o[:, :-1]).norm(dim=-1).mean(1) / rad_mean).median())

    return {
        # Which groups these shape statistics were computed on. Without it a
        # comparison between a Morton run (64 slots, every group) and an anchor run
        # (a narrower prefix, a subset of groups) reads as a difference in the
        # model when it is a difference in what was measured.
        "group_stat_slots": float(gs),
        "group_stat_groups": float(ng),
        "intra_chamfer_rel": float(ich.median()),
        "intra_chamfer_rel_p90": float(ich.quantile(0.9)),
        "nn_unique": _uniq(op, og),
        "thread_pred": thr(op),
        "thread_gt": thr(og),
        "aniso_lam2_pred": _lam2(op),
        "aniso_lam2_gt": _lam2(og),
        "template_erank_pred": _erank(op),
        "template_erank_gt": _erank(og),
        "centroid_rmse_norm": float(cen_sq.mean().sqrt()),
        "offset_rmse_norm": float(off_sq.mean().sqrt()),
        "centroid_mse_share": float(cen_sq.mean() / max(tot, 1e-18)),
        "centroid_only_rmse_norm": float(og.pow(2).sum(-1).mean().sqrt()),
        "rel_offset_p50": float(rel.median()),
        "rel_offset_p90": float(rel.quantile(0.9)),
        "group_radius_p50": float(rad.median()),
        "group_radius_p999": float(rad.quantile(0.999)),
    }


def _scatter_panels(ax, xyz: np.ndarray, axes, color, size: float, label: str):
    """One projection panel, with readable axes.

    Ticks, numeric labels and a light grid are kept on purpose: without them a
    reconstruction that is subtly *scaled* or *shifted* looks identical to a
    correct one, because every panel auto-fits its own data range. The numbers are
    what make the GT and PRED rows comparable at a glance.
    """
    a0, a1 = axes
    ax.scatter(xyz[:, a0], xyz[:, a1], s=size, c=color, linewidths=0, marker=".")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.tick_params(labelsize=7)
    ax.set_title(label, fontsize=9)


def save_comparison_figure(
    path: str,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    max_points: int = 60000,
    title: str = "",
) -> None:
    """GT on the top row, prediction on the bottom, three projections across.

    Layout matches the CoordFirstGSAE figures so runs from the two code bases can
    be put side by side. Rows are GT / PRED rather than projections, which is what
    makes a difference in *shape* jump out -- the eye compares the two rows of the
    same column. The third column is YZ, which the previous 2-projection version
    omitted: XY and XZ share the x axis, so a fault purely in the y-z plane was
    invisible in both.

    Both rows are drawn on the *same* axis limits, taken from the GT. A panel that
    auto-fits its own data hides exactly the failure worth seeing, because a
    collapsed or inflated prediction is rescaled back to fill the frame.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Predicted presence and the GT mask are different sets once
    # eval_pred_mask=1, so the two clouds can have different lengths. Indexing
    # pred with a GT subsample was valid only when both used the same mask.
    rng = np.random.default_rng(0)

    def _take(xyz: np.ndarray) -> np.ndarray:
        n = int(xyz.shape[0])
        if n == 0 or n <= max_points:
            return xyz
        return xyz[rng.choice(n, max_points, replace=False)]

    gt_s, pr_s = _take(gt_xyz), _take(pred_xyz)
    if gt_s.shape[0] == 0:
        return

    planes = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=110)
    for c, (a0, a1, nm) in enumerate(planes):
        _scatter_panels(axes[0][c], gt_s, (a0, a1), "#1f77b4", 0.35, f"GT {nm}")
        _scatter_panels(axes[1][c], pr_s, (a0, a1), "#ff7f0e", 0.35, f"PRED {nm}")
        lo = [gt_s[:, a0].min(), gt_s[:, a1].min()]
        hi = [gt_s[:, a0].max(), gt_s[:, a1].max()]
        pad = [max((hi[i] - lo[i]) * 0.03, 1e-6) for i in range(2)]
        for r in (0, 1):
            axes[r][c].set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            axes[r][c].set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def save_eval_outputs(
    out_dir: str,
    name: str,
    branch: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    center,
    scale: float,
    metrics: Optional[Dict[str, float]] = None,
    write_ply_files: bool = True,
    pred_mask: Optional[torch.Tensor] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    m = (mask > 0.5).cpu().numpy()
    pm = (pred_mask > 0.5).cpu().numpy() if pred_mask is not None else m
    g = target[..., 0:3].cpu().numpy()[m]
    p = pred[..., 0:3].cpu().numpy()[pm]
    c = np.asarray(center.cpu() if torch.is_tensor(center) else center, np.float32)
    stem = f"{name}_{branch}"
    if write_ply_files:
        write_ply(os.path.join(out_dir, f"{stem}_gt.ply"), denormalize_xyz(g, c, scale))
        write_ply(os.path.join(out_dir, f"{stem}_pred.ply"), denormalize_xyz(p, c, scale))
    title = stem
    if metrics:
        title += f" | rmse_norm={metrics.get('xyz_rmse_norm', float('nan')):.5f}"
        if "chamfer_norm" in metrics:
            title += f" chamfer={metrics['chamfer_norm']:.5f}"
    save_comparison_figure(os.path.join(out_dir, f"{stem}_gt_pred_2x3.png"), g, p, title=title)
    if metrics:
        save_json(os.path.join(out_dir, f"{stem}_metrics.json"), metrics)
