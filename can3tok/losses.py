"""Losses for the codec teacher and the generative student.

Detail / coverage fixes (minimal; architecture unchanged)
---------------------------------------------------------
* multi-scale chamfer samples GT and pred **independently** (set distance).
  Same-index sampling was effectively pointwise and under-punished holes.
* GT→pred onesided coverage + soft 3D voxel occupancy for sparse arms.
* latent / residual / equivariance terms unchanged from the working baseline.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Fixed per-channel dataset statistics, shared with the encoder so the attribute
# set distance and the encoder's standardisation cannot drift apart.
from .encoder import ATTR_STD


# ---------------------------------------------------------------------------
# basic masked reductions
# ---------------------------------------------------------------------------


def masked_reduce(loss: torch.Tensor, mask: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-8):
    m = mask
    if weight is not None:
        m = m * weight
    if loss.dim() > m.dim():
        m = m.unsqueeze(-1).expand_as(loss)
    return (loss * m).sum() / (m.sum() + eps)


def masked_smooth_l1(pred, target, mask, beta: float = 0.02, weight=None):
    return masked_reduce(F.smooth_l1_loss(pred, target, reduction="none", beta=beta), mask, weight)


def masked_mse(pred, target, mask, weight=None):
    return masked_reduce((pred - target) ** 2, mask, weight)


def masked_hard_smooth_l1(pred, target, mask, frac: float = 0.05, min_points: int = 2048, beta: float = 0.02):
    """Average error over the worst ``frac * max_points`` points (hard mining).

    ``frac`` is relative to the padded length rather than the valid count on
    purpose: reading the valid count would force a host sync on every step.
    Invalid slots have zero error and therefore never enter the top-k as long as
    ``frac`` stays below the smallest occupancy in the dataset (~0.43 here).
    """
    err = F.smooth_l1_loss(pred, target, reduction="none", beta=beta).mean(dim=-1) * mask
    n = err.shape[1]
    k = max(1, min(int(max(min_points, int(frac * n))), n))
    top, _ = torch.topk(err, k, dim=1)
    return top.mean()


def quat_loss(pred, target, mask):
    dot = (pred * target).sum(dim=-1).abs().clamp(max=1.0)
    return masked_reduce(1.0 - dot, mask)


def balanced_presence_loss(logits, mask, eps: float = 1e-8):
    loss = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    pos = (loss * mask).sum() / (mask.sum() + eps)
    neg = (loss * (1.0 - mask)).sum() / ((1.0 - mask).sum() + eps)
    return 0.5 * (pos + neg)


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------


def _sample_valid_indices(mask_1d: torch.Tensor, max_samples: int) -> torch.Tensor:
    idx = torch.nonzero(mask_1d > 0.5, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return idx
    if idx.numel() <= max_samples:
        return idx
    sel = torch.randperm(idx.numel(), device=idx.device)[:max_samples]
    return idx[sel]


def _sample_spatial_indices(xyz_1d: torch.Tensor, mask_1d: torch.Tensor, max_samples: int, bins: int = 32):
    """Stratified over a coarse voxel grid so empty regions are not oversampled.

    The previous ``sorted[::stride][:K]`` dropped the tail of the sorted volume
    whenever N was not a multiple of K. Split the sorted range into K bins and
    take one representative from each bin so the whole span is covered.
    """
    idx = torch.nonzero(mask_1d > 0.5, as_tuple=False).squeeze(-1)
    if idx.numel() <= max_samples:
        return idx
    p = xyz_1d[idx]
    q = ((p.clamp(-1, 1) + 1.0) * 0.5 * (bins - 1)).long()
    key = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
    order = torch.argsort(key)
    idx = idx[order]
    n = int(idx.numel())
    k = int(max_samples)
    # Inclusive edges so the last point of the sorted volume is eligible.
    edges = torch.linspace(0, n, k + 1, device=idx.device)
    starts = edges[:-1].long().clamp(0, n - 1)
    ends = edges[1:].long().clamp(1, n)
    mid = ((starts + ends - 1) // 2).clamp(0, n - 1)
    return idx[mid]


def _even_subset(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Take n points spaced across the whole index list, not a prefix."""
    if idx.numel() <= n:
        return idx
    pos = torch.linspace(0, idx.numel() - 1, int(n), device=idx.device).long()
    return idx[pos]


def spatial_balance_weights(xyz, mask, bins: int = 32, power: float = 0.5, max_weight: float = 8.0):
    """Down-weight dense regions so sparse structure still contributes."""
    with torch.no_grad():
        q = ((xyz.clamp(-1, 1) + 1.0) * 0.5 * (bins - 1)).long()
        key = (q[..., 0] * bins + q[..., 1]) * bins + q[..., 2]
        w = torch.ones_like(mask)
        for b in range(xyz.shape[0]):
            k = key[b][mask[b] > 0.5]
            if k.numel() == 0:
                continue
            counts = torch.bincount(k, minlength=bins ** 3).float()
            dens = counts[key[b]].clamp(min=1.0)
            wb = (dens.mean() / dens) ** power
            w[b] = wb.clamp(max=max_weight) * mask[b]
        return w


# ---------------------------------------------------------------------------
# chamfer
# ---------------------------------------------------------------------------


def _pair_chunk() -> int:
    return int(os.environ.get("CHAMFER_PAIR_CHUNK", "4096"))


@torch.no_grad()
def _nearest_indices(p: torch.Tensor, g: torch.Tensor, chunk: int) -> torch.Tensor:
    """Chunked pairwise argmin. ``cdist`` is the fast path on this GPU class."""
    chunk = max(int(chunk), 1)
    out = []
    for i0 in range(0, p.shape[0], chunk):
        d = torch.cdist(p[i0 : i0 + chunk].unsqueeze(0), g.unsqueeze(0)).squeeze(0)
        out.append(d.argmin(dim=1))
    return torch.cat(out, dim=0)


def symmetric_chamfer(p: torch.Tensor, g: torch.Tensor, chunk: Optional[int] = None) -> torch.Tensor:
    """Symmetric chamfer with O(N) autograd memory.

    ``cdist`` saves its full N x M output for backward, which at 16k samples is
    ~1 GB per direction and was the single biggest memory consumer in the old
    loss. The nearest-neighbour search does not need gradients at all: only the
    distance to the (fixed) arg-min does, and that is exactly the chamfer
    gradient almost everywhere. So we find the indices under ``no_grad`` and
    then evaluate one distance per point.
    """
    chunk = chunk or _pair_chunk()
    i_pg = _nearest_indices(p.detach(), g.detach(), chunk)
    i_gp = _nearest_indices(g.detach(), p.detach(), chunk)
    a = torch.sqrt(((p - g[i_pg]) ** 2).sum(dim=-1) + 1e-12).mean()
    b = torch.sqrt(((g - p[i_gp]) ** 2).sum(dim=-1) + 1e-12).mean()
    return 0.5 * (a + b)


def onesided_chamfer(src: torch.Tensor, dst: torch.Tensor, chunk: Optional[int] = None) -> torch.Tensor:
    """단방향 Chamfer: src→dst 최근접 거리 평균.

    GT→pred 로 쓰면 GT에만 있고 pred 근처에 없는 구멍을 직접 처벌한다.
    """
    chunk = chunk or _pair_chunk()
    idx = _nearest_indices(src.detach(), dst.detach(), chunk)
    return torch.sqrt(((src - dst[idx]) ** 2).sum(dim=-1) + 1e-12).mean()


def multiscale_chamfer(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    mask: torch.Tensor,
    scales: Sequence[int] = (2048, 8192, 32768),
    scale_weights: Optional[Sequence[float]] = None,
    balanced: bool = True,
    bins: int = 32,
    pred_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """멀티스케일 set Chamfer.

    중요: 예전엔 ``pred[idx], target[idx]`` 로 **같은 슬롯 인덱스**를 써서
    사실상 대응점 비교에 가까웠다. GT·pred 를 각각 샘플해 진짜 집합 거리로 본다.
    """
    if scale_weights is None:
        scale_weights = [1.0 / len(scales)] * len(scales)
    scales_i = [int(n) for n in scales]
    n_max = max(scales_i)
    total = pred_xyz.new_tensor(0.0)
    wsum = 0.0
    pm = pred_mask if pred_mask is not None else torch.ones_like(mask)

    gt_idx, pr_idx = [], []
    for b in range(pred_xyz.shape[0]):
        if balanced:
            gt_idx.append(_sample_spatial_indices(target_xyz[b], mask[b], n_max, bins=bins))
            pr_idx.append(_sample_spatial_indices(pred_xyz[b].detach(), pm[b], n_max, bins=bins))
        else:
            gt_idx.append(_sample_valid_indices(mask[b], n_max))
            pr_idx.append(_sample_valid_indices(pm[b], n_max))

    for n, w in zip(scales_i, scale_weights):
        acc = pred_xyz.new_tensor(0.0)
        used = 0
        for b in range(pred_xyz.shape[0]):
            # Evenly spaced subset of the full-span sample, not a spatial prefix.
            gi = _even_subset(gt_idx[b], int(n))
            pi = _even_subset(pr_idx[b], int(n))
            if gi.numel() < 8 or pi.numel() < 8:
                continue
            acc = acc + symmetric_chamfer(pred_xyz[b][pi], target_xyz[b][gi])
            used += 1
        if used:
            total = total + float(w) * acc / used
            wsum += float(w)
    return total / max(wsum, 1e-8)


def coverage_onesided_loss(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    mask: torch.Tensor,
    samples: int = 16384,
    bins: int = 32,
    pred_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GT→pred onesided Chamfer (구멍 패널티)."""
    pm = pred_mask if pred_mask is not None else torch.ones_like(mask)
    total = pred_xyz.new_tensor(0.0)
    used = 0
    for b in range(pred_xyz.shape[0]):
        gi = _sample_spatial_indices(target_xyz[b], mask[b], samples, bins=bins)
        pi = _sample_spatial_indices(pred_xyz[b].detach(), pm[b], samples, bins=bins)
        if gi.numel() < 8 or pi.numel() < 8:
            continue
        total = total + onesided_chamfer(target_xyz[b][gi], pred_xyz[b][pi])
        used += 1
    return total / max(used, 1)


def soft_voxel_occupancy_loss(
    pred_xyz: torch.Tensor,
    target_xyz: torch.Tensor,
    mask: torch.Tensor,
    bins: int = 24,
    samples: int = 16384,
    pred_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """미분 가능한 soft 3D occupancy (삼선형 누적).

    hard ``.long()`` binning 은 xyz 그래디언트가 끊기므로 쓰지 않는다.
    GT가 차지한 복셀에서 pred soft mass 가 부족하면 패널티.
    """
    pm = pred_mask if pred_mask is not None else torch.ones_like(mask)
    total = pred_xyz.new_tensor(0.0)
    used = 0
    nbin = bins ** 3

    def _trilinear_hist(xyz: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # xyz ∈ [-1,1] → continuous bin coords ∈ [0, bins-1]
        u = (xyz.clamp(-1.0, 1.0) + 1.0) * 0.5 * float(bins - 1)
        u0 = u.floor().clamp(0, bins - 2)
        u1 = u0 + 1.0
        f = (u - u0).clamp(0.0, 1.0)
        o0 = (1.0 - f) * w.unsqueeze(-1)
        o1 = f * w.unsqueeze(-1)
        # 8 corners
        corners = []
        for ix, wx in ((u0[..., 0], o0[..., 0]), (u1[..., 0], o1[..., 0])):
            for iy, wy in ((u0[..., 1], o0[..., 1]), (u1[..., 1], o1[..., 1])):
                for iz, wz in ((u0[..., 2], o0[..., 2]), (u1[..., 2], o1[..., 2])):
                    key = (ix.long() * bins + iy.long()) * bins + iz.long()
                    mass = wx * wy * wz
                    corners.append((key.clamp(0, nbin - 1), mass))
        hist = xyz.new_zeros(nbin)
        for key, mass in corners:
            hist.scatter_add_(0, key, mass)
        return hist

    for b in range(pred_xyz.shape[0]):
        gi = _sample_spatial_indices(target_xyz[b], mask[b], samples, bins=bins)
        pi = _sample_spatial_indices(pred_xyz[b].detach(), pm[b], samples, bins=bins)
        if gi.numel() < 8 or pi.numel() < 8:
            continue
        gt_w = torch.ones(gi.numel(), device=pred_xyz.device, dtype=pred_xyz.dtype)
        pr_w = pm[b][pi].clamp(0.0, 1.0)
        with torch.no_grad():
            gt_hist = _trilinear_hist(target_xyz[b][gi], gt_w)
            gt_occ = (gt_hist > 1e-3).float()
        pr_hist = _trilinear_hist(pred_xyz[b][pi], pr_w)
        # occupied GT voxels should carry comparable soft mass
        target_mass = gt_hist[gt_occ > 0.5].clamp(min=1e-3)
        pred_mass = pr_hist[gt_occ > 0.5]
        miss = F.relu(0.5 * target_mass - pred_mass) / (target_mass + 1e-6)
        total = total + miss.mean()
        used += 1
    return total / max(used, 1)


def projected_chamfer(pred_xyz, target_xyz, mask, samples: int = 16384,
                      pred_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Chamfer on the three axis-aligned 2D projections: cheap silhouette match."""
    total = pred_xyz.new_tensor(0.0)
    used = 0
    planes = ((0, 1), (0, 2), (1, 2))
    for b in range(pred_xyz.shape[0]):
        gi = _sample_spatial_indices(target_xyz[b], mask[b], samples)
        pm = mask[b] if pred_mask is None else pred_mask[b]
        pi = _sample_spatial_indices(pred_xyz[b].detach(), pm, samples)
        if gi.numel() < 8 or pi.numel() < 8:
            continue
        for a0, a1 in planes:
            p = pred_xyz[b][pi][:, [a0, a1]]
            g = target_xyz[b][gi][:, [a0, a1]]
            total = total + symmetric_chamfer(p, g)
        used += 1
    return total / max(used * len(planes), 1)


def _soft_hist2d(points_2d: torch.Tensor, bins: int = 64, sigma: float = 1.0) -> torch.Tensor:
    centers = torch.linspace(-1.0, 1.0, bins, device=points_2d.device, dtype=points_2d.dtype)
    dx = (points_2d[:, 0:1] - centers[None, :]) / sigma
    dy = (points_2d[:, 1:2] - centers[None, :]) / sigma
    wx = torch.exp(-0.5 * dx * dx)
    wy = torch.exp(-0.5 * dy * dy)
    h = wx.transpose(0, 1) @ wy
    return h / (h.sum() + 1e-8)


def projected_histogram_loss(pred_xyz, target_xyz, mask, samples=8192, bins=64, sigma=0.025, scale=100.0,
                             pred_mask: Optional[torch.Tensor] = None):
    total = pred_xyz.new_tensor(0.0)
    used = 0
    planes = ((0, 1), (0, 2), (1, 2))
    for b in range(pred_xyz.shape[0]):
        gi = _sample_spatial_indices(target_xyz[b], mask[b], samples)
        pm = mask[b] if pred_mask is None else pred_mask[b]
        pi = _sample_spatial_indices(pred_xyz[b].detach(), pm, samples)
        if gi.numel() < 8 or pi.numel() < 8:
            continue
        for a0, a1 in planes:
            hp = _soft_hist2d(pred_xyz[b][pi][:, [a0, a1]], bins, sigma)
            hg = _soft_hist2d(target_xyz[b][gi][:, [a0, a1]], bins, sigma)
            total = total + (hp - hg).abs().sum()
        used += 1
    return scale * total / max(used * len(planes), 1)


def intra_group_chamfer(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    scale: Optional[torch.Tensor] = None,
    chunk_groups: int = 1024,
    return_parts: bool = False,
    center: bool = False,
):
    """Permutation-invariant within-group set distance, in units of group extent.

    Every other within-group term (``z_residual``, ``xyz_residual``,
    ``shape_direct``, plain ``xyz``) compares slot *i* of the prediction against
    slot *i* of the target. That is only well posed if the slot order is a
    function of the geometry; where it is not, the ordered target is noise and its
    Bayes-optimal answer is the per-slot mean, i.e. the group collapses onto its
    centroid. This term asks only that the two 32/64-point *sets* agree, so it
    stays well posed no matter how the slots were ordered.

    Normalising by the group extent before measuring is what keeps the ~1% of
    Morton-jump groups (which hold >55% of the raw MSE and are not representable
    at this channel budget anyway) from consuming the whole gradient.

    Cost is far below the global chamfer this already runs: 4096 groups x 64 x 64
    pairwise, and the nearest-neighbour search itself needs no autograd graph.
    """
    b, n, _ = pred.shape
    gsz = int(group_size)
    g = n // gsz
    p = pred[..., 0:3].reshape(b * g, gsz, 3)
    t = target[..., 0:3].reshape(b * g, gsz, 3)
    m = mask[:, : g * gsz].reshape(b * g, gsz).to(p.dtype)
    if center:
        den = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        p = p - (p * m.unsqueeze(-1)).sum(dim=1, keepdim=True) / den.unsqueeze(-1)
        t = t - (t * m.unsqueeze(-1)).sum(dim=1, keepdim=True) / den.unsqueeze(-1)
    if scale is not None:
        # Floor relative to the batch, not an absolute 1e-4. Group radii run 0.042
        # (p10) to 0.250 (p999) under this normalisation, so a fixed 1e-4 lets one
        # near-degenerate group amplify its own chamfer ~400x and dominate the term
        # -- and the right constant would move arbitrarily on a dataset normalised
        # to a different scale. A fraction of the batch median does not.
        sc = scale[:, :g].to(p.dtype).detach().abs().reshape(b * g, 1, 1)
        s = sc.clamp(min=0.05 * sc.median().clamp(min=1e-6))
        p = p / s
        t = t / s

    idx_pt, idx_tp = [], []
    with torch.no_grad():
        for i0 in range(0, b * g, max(int(chunk_groups), 1)):
            i1 = min(i0 + max(int(chunk_groups), 1), b * g)
            d = torch.cdist(p[i0:i1].detach().float(), t[i0:i1].detach().float())
            mm = m[i0:i1]
            # One masked matrix serves both directions: padding the invalid rows
            # *and* columns leaves every valid row's argmin over valid columns and
            # every valid column's argmin over valid rows unchanged.
            d = d + (1.0 - mm).unsqueeze(1) * 1e4 + (1.0 - mm).unsqueeze(2) * 1e4
            idx_pt.append(d.argmin(dim=2))
            idx_tp.append(d.argmin(dim=1))
    i_pt = torch.cat(idx_pt, 0).unsqueeze(-1).expand(-1, -1, 3)
    i_tp = torch.cat(idx_tp, 0).unsqueeze(-1).expand(-1, -1, 3)

    a = torch.sqrt(((p - torch.gather(t, 1, i_pt)) ** 2).sum(-1) + 1e-12)
    c = torch.sqrt(((t - torch.gather(p, 1, i_tp)) ** 2).sum(-1) + 1e-12)
    denom = m.sum(-1).clamp(min=1.0)
    gv = (m.sum(-1) > 0.5).to(p.dtype)
    pg_p2g = (a * m).sum(-1) / denom          # precision: is a predicted point near GT
    pg_g2p = (c * m).sum(-1) / denom          # coverage:  is a GT point near a prediction
    per_group = 0.5 * (pg_p2g + pg_g2p)
    out = masked_reduce(per_group.reshape(b, g), gv.reshape(b, g))
    if not return_parts:
        return out
    # Group radius ratio. gen inflated to 1.83x GT while precision fell to 1.294
    # against coverage 0.521 -- the signature of satisfying a *symmetric* chamfer
    # by spreading. A symmetric term alone cannot see that; these can.
    pm = m.unsqueeze(-1)
    pc = (p * pm).sum(1) / denom.unsqueeze(-1)
    tc = (t * pm).sum(1) / denom.unsqueeze(-1)
    rp = (((p - pc.unsqueeze(1)) * pm).norm(dim=-1) * m).sum(-1) / denom
    rt = (((t - tc.unsqueeze(1)) * pm).norm(dim=-1) * m).sum(-1) / denom
    # Batch-relative floor, matching the extent floor above rather than the
    # absolute 1e-6 this used to carry. A GT group whose points are coincident
    # (Morton-jump groups and near-empty anchors both produce them) drove the
    # ratio to 1e6; at w_radius=4 that is a 4e6 single-batch contribution, and
    # `radius` is not in the printed breakdown, which is why the L16k run showed
    # 300x jumps in the logged total with every visible term flat.
    #
    # The degenerate groups are also dropped from the reduction, not merely
    # clamped: a group with no spatial extent has no radius to match, so the
    # ratio is undefined there rather than just large, and clamping alone would
    # still hand it a full-weight gradient pointing at an arbitrary target.
    rt_floor = 0.05 * rt.detach().median().clamp(min=1e-6)
    ratio = (rp / rt.clamp(min=rt_floor) - 1.0).abs()
    rad_valid = gv * (rt.detach() > rt_floor).to(gv.dtype)
    return out, {
        "p2g": masked_reduce(pg_p2g.reshape(b, g), gv.reshape(b, g)),
        "g2p": masked_reduce(pg_g2p.reshape(b, g), gv.reshape(b, g)),
        "radius": masked_reduce(ratio.reshape(b, g), rad_valid.reshape(b, g)),
    }


def intra_group_sinkhorn_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    scale: Optional[torch.Tensor] = None,
    epsilon: float = 0.08,
    iterations: int = 6,
    chunk_groups: int = 256,
    center: bool = False,
) -> torch.Tensor:
    """Balanced within-group transport that penalises duplicate predictions.

    Symmetric Chamfer independently chooses a nearest neighbour in each direction,
    so several predicted slots may use the same target point. Sinkhorn assigns
    equal mass to every live prediction and target slot, making a duplicate pay
    for the different target point it failed to cover.

    The plan is computed from detached distances and then treated as a matching
    target. This keeps the one-to-one gradient without retaining every Sinkhorn
    iteration in the autograd graph. Fixed-anchor groups are partially occupied,
    so the target mask is applied on both axes.
    """
    b, n, _ = pred.shape
    gsz = int(group_size)
    g = n // gsz
    p = pred[..., :3].reshape(b * g, gsz, 3)
    t = target[..., :3].reshape(b * g, gsz, 3)
    m = mask[:, : g * gsz].reshape(b * g, gsz) > 0.5
    mf = m.to(p.dtype).unsqueeze(-1)
    denom = mf.sum(dim=1, keepdim=True).clamp(min=1.0)

    if center:
        p = p - (p * mf).sum(dim=1, keepdim=True) / denom
        t = t - (t * mf).sum(dim=1, keepdim=True) / denom
    if scale is not None:
        sc = scale[:, :g].to(p.dtype).detach().abs().reshape(b * g, 1, 1)
        sc = sc.clamp(min=0.05 * sc.median().clamp(min=1e-6))
        p, t = p / sc, t / sc

    eps = max(float(epsilon), 1e-4)
    n_iter = max(int(iterations), 1)
    chunk = max(int(chunk_groups), 1)
    total = p.new_tensor(0.0, dtype=torch.float32)
    used = p.new_tensor(0.0, dtype=torch.float32)
    neg = -1e4
    for i0 in range(0, b * g, chunk):
        i1 = min(i0 + chunk, b * g)
        pc, tc, mc = p[i0:i1].float(), t[i0:i1].float(), m[i0:i1]
        valid_group = mc.sum(dim=-1) > 0
        cost = torch.cdist(pc, tc)
        with torch.no_grad():
            pair = mc.unsqueeze(2) & mc.unsqueeze(1)
            # Shift each row by its own minimum valid cost before dividing by eps.
            # The Sinkhorn plan is invariant to a per-row shift (it is absorbed
            # into log_u), but the sentinel is not: at eps=0.005 a cost of 50
            # gives -cost/eps = -1e4, exactly `neg`, so a far-but-valid row
            # becomes indistinguishable from a masked one and, if every entry in
            # the row crosses it, the row sums to zero and the logsumexp returns
            # -inf. Subtracting the row minimum pins at least one valid entry per
            # row at exp(0)=1, which makes the anneal to small eps safe without
            # changing the plan it converges to.
            cd = cost.detach()
            far = cd.new_full((), float("inf"))
            row_min = torch.where(pair, cd, far).amin(dim=2, keepdim=True)
            row_min = torch.where(torch.isfinite(row_min), row_min,
                                  torch.zeros_like(row_min))
            log_k = torch.where(pair, -(cd - row_min) / eps, cd.new_full((), neg))
            count = mc.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
            log_mass = -count.log()
            log_u = torch.where(mc, log_mass.expand_as(cost[..., 0]), cost.new_full((), neg))
            log_v = log_u.clone()
            for _ in range(n_iter):
                row = torch.logsumexp(log_k + log_v.unsqueeze(1), dim=2)
                log_u = torch.where(mc, log_mass - row, cost.new_full((), neg))
                col = torch.logsumexp(log_k + log_u.unsqueeze(2), dim=1)
                log_v = torch.where(mc, log_mass - col, cost.new_full((), neg))
            plan = torch.exp(log_k + log_u.unsqueeze(2) + log_v.unsqueeze(1))
            plan = torch.where(pair, plan, torch.zeros_like(plan))
        per_group = (plan * cost).sum(dim=(1, 2))
        total = total + per_group[valid_group].sum()
        used = used + valid_group.sum().to(used.dtype)
    return total / used.clamp(min=1.0)


def intra_group_spacing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    scale: Optional[torch.Tensor] = None,
    chunk_groups: int = 1024,
    ratio: float = 1.0,
) -> torch.Tensor:
    """Hinge on the WITHIN-group nearest-neighbour SPACING.

    This is the term that was missing, and the measurement that says so is a
    render ablation rather than another point-space number. Rendering a random
    half of the GT Gaussians with their own GT positions and GT attributes scores
    19.93 dB against 48.39 for the full selected set -- so covering only half the
    surface costs 28.5 dB, which is essentially the whole of this model's
    remaining error (its own end-to-end render is 19.05 with GT attributes
    copied). And it covers only half: ``nn_unique`` is 0.493, i.e. of the 262144
    Gaussians it emits, about 129000 land on a GT point some other prediction has
    already claimed.

    The duplication is inside the groups, not between them. Measured over five
    held-out scenes: global unique 49.3%, unique *within a group* 50.9% (so the
    groups are not redrawing each other), and 64.5% of duplicate pairs come from
    the same group against the 0.02% a uniform-across-groups collapse would give.
    The median within-group nearest-neighbour spacing is 0.261 of the group
    radius where 64 well-spread points would sit near 0.5. So each group emits 64
    points onto roughly 32 distinct places.

    Nothing in the objective forbids that. ``intra_chamfer`` is symmetric, and a
    symmetric chamfer scores two predictions sharing one GT point as perfect --
    duplication is free. The one-sided ``coverage`` term does penalise it, but it
    is written in absolute normalised-scene units while every within-group term is
    divided by the group extent (median 0.0027), so it evaluates to 0.0023 and
    contributed 0.02% of the objective at w=2.0. ``patch_dispersion_loss`` does
    not see this either: it hinges on the group's standard deviation, and the
    groups already spread to 1.29 group radii. Std is blind to 64 points forming
    32 pairs inside that spread.

    So the hinge is on the spacing itself. Target is the group's OWN ground truth
    median nearest-neighbour spacing, detached -- not a constant, because a group
    covering a dense surface patch genuinely should pack tighter than one on a
    sparse edge, and a fixed margin is what made ``patch_dispersion_loss`` push
    blur when its margin was 5.7x the median group radius.

    Extent-normalised for the same reason ``intra_chamfer`` is: without it the
    ~1% of huge groups own the gradient, and the term inherits the units mistake
    that took ``coverage`` out of the objective.
    """
    b, n, _ = pred.shape
    gsz = int(group_size)
    g = n // gsz
    p = pred[..., 0:3].reshape(b * g, gsz, 3)
    t = target[..., 0:3].reshape(b * g, gsz, 3)
    m = mask[:, : g * gsz].reshape(b * g, gsz).to(p.dtype)
    if scale is not None:
        sc = scale[:, :g].to(p.dtype).detach().abs().reshape(b * g, 1)
        s = sc.clamp(min=0.05 * sc.median().clamp(min=1e-6))
    else:
        s = p.new_ones(b * g, 1)

    BIG = 1e9
    eye = torch.eye(gsz, device=p.device, dtype=torch.bool).unsqueeze(0)
    step = max(int(chunk_groups), 1)
    hinge_sum = p.new_tensor(0.0)
    live_sum = p.new_tensor(0.0)
    for i0 in range(0, b * g, step):
        i1 = min(i0 + step, b * g)
        pp, tt, mm, ss = p[i0:i1], t[i0:i1], m[i0:i1], s[i0:i1]
        pad = (mm <= 0.5).unsqueeze(1)                       # (c, 1, G) columns to ignore
        # Target spacing: the GT group's own median NN distance. no_grad -- it is
        # a property of the data, and letting gradient through it would let the
        # model satisfy the hinge by moving the target.
        # One hinge per GROUP, on the MEAN spacing, not one per point against the
        # group median. A per-point hinge against a median penalises half the
        # points by construction -- pred = GT scored 0.0885 instead of 0 in the
        # unit test -- so it is a standing demand to be sparser than the data,
        # which is exactly the bug that made patch_dispersion_loss push blur.
        # Mean against mean is 0 iff the prediction is at least as spread as the
        # target, and unlike a median it still carries gradient to every point.
        with torch.no_grad():
            dt = torch.cdist(tt.float(), tt.float())
            dt = dt.masked_fill(eye, BIG).masked_fill(pad, BIG)
            nn_t = dt.min(dim=-1).values                     # (c, G)
            ok_t = ((mm > 0.5) & (nn_t < BIG * 0.5)).to(p.dtype)
            mean_t = (nn_t * ok_t).sum(-1) / ok_t.sum(-1).clamp(min=1.0)
        # Prediction spacing keeps its graph: only the arg-min's distance needs
        # one, the same trick symmetric_chamfer uses.
        dp = torch.cdist(pp, pp)
        dp = dp.masked_fill(eye, BIG).masked_fill(pad, BIG)
        nn_p = dp.min(dim=-1).values                         # (c, G)
        ok_p = ((mm > 0.5) & (nn_p < BIG * 0.5)).to(p.dtype)
        mean_p = (nn_p * ok_p).sum(-1) / ok_p.sum(-1).clamp(min=1.0)
        # A group needs at least two live points before "spacing" means anything.
        live = (ok_p.sum(-1) > 1.5).to(p.dtype)
        hinge = F.relu(float(ratio) * mean_t - mean_p) / ss.squeeze(-1).clamp(min=1e-8)
        hinge_sum = hinge_sum + (hinge * live).sum()
        live_sum = live_sum + live.sum()
    return hinge_sum / live_sum.clamp(min=1.0)


def intra_group_attr_moment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    layout: Dict[str, int],
):
    """셀 내부 속성 '변동'을 직접 맞춘다: 퍼짐(std) 과 위치의존 기울기.

    측정된 결함 (T16k step 70000, step_028630, 셀당 >=16점):

        채널        GT 셀내 std   pred 셀내 std   비율
        logscale      1.5481        0.6031       0.39
        opacity       4.4448        2.2199       0.50
        rot           0.2027        0.1070       0.53
        sh_dc         1.0066        0.6603       0.66
        xyz           0.0051        0.0050       0.98

        셀 내부 위치 -> 속성 선형 설명력 (R^2)
        logscale   GT 0.234   pred 0.045
        sh_dc      GT 0.226   pred 0.050

    위치는 정상적으로 퍼지는데(0.98) 속성만 셀 평균으로 뭉친다. 난간·모서리는
    "이웃과 크기가 다른 길쭉한 가우시안 몇 개"라서 이 붕괴가 곧 디테일 소실이다.

    ``intra_group_attr_set_loss`` 가 이미 같은 붕괴를 겨냥하지만, 그것은 속성공간
    chamfer 라 부분집합 덮기로도 만족될 수 있고 기울기도 간접적이다. 여기서는
    붕괴의 정의 그 자체 -- 2차 모멘트 -- 를 직접 맞춘다. std 와 최소제곱 기울기는
    둘 다 집합 함수라 슬롯 순서에 무관하고, 같은 자리에 놓인 서로 다른 GT 두 개를
    평균내는 문제(그 docstring 이 지적한 것)도 겪지 않는다.

    반환: (spread, slope)
      spread  셀별 채널별 |std_pred - std_gt|,  표준화 채널 단위
      slope   셀 로컬 좌표 -> 속성의 11x3 최소제곱 계수 차이
    """
    b, n, _ = pred.shape
    gsz = int(group_size)
    g = n // gsz
    a_s, a_r = int(layout["scale"]), int(layout["rot"])
    a_o, a_c = int(layout["opacity"]), int(layout["color"])
    ncol = a_c + 3

    def prep(x):
        v = x[:, : g * gsz, :ncol].reshape(b * g, gsz, ncol).float()
        q = v[..., a_r : a_r + 4]
        q = q * torch.sign(q[..., :1].detach() + 1e-12)
        return torch.cat([v[..., a_s : a_s + 3], q,
                          v[..., a_o : a_o + 1], v[..., a_c : a_c + 3]], dim=-1)

    # 2차 모멘트와 3x3 정규방정식은 fp32 에서만 의미가 있다. bf16 autocast 아래서
    # torch.linalg.solve 는 아예 구현이 없고 ("lu_factor_cublas not implemented for
    # BFloat16"), 분산 자체도 bf16 의 8비트 가수로는 신뢰할 수 없다.
    _ac = torch.autocast("cuda", enabled=False)
    _ac.__enter__()
    sd = torch.tensor(ATTR_STD[:11], device=pred.device, dtype=torch.float32).clamp(min=1e-3)
    P = prep(pred) / sd
    T = prep(target) / sd
    m = mask[:, : g * gsz].reshape(b * g, gsz).to(P.dtype).unsqueeze(-1)
    cnt = m.sum(dim=1, keepdim=True)
    live = (cnt.squeeze(-1).squeeze(-1) > 7.5)
    if not bool(live.any()):
        _ac.__exit__(None, None, None)
        z = pred.new_tensor(0.0)
        return z, z
    den = cnt.clamp(min=1.0)

    def moments(A, xyz):
        mu = (A * m).sum(1, keepdim=True) / den
        Ac = (A - mu) * m
        var = (Ac * Ac).sum(1) / den.squeeze(1).clamp(min=1.0)
        std = var.clamp(min=1e-12).sqrt()
        # 셀 로컬 좌표 (중심 제거, 셀 반경으로 정규화) -> 속성의 최소제곱 기울기
        cmu = (xyz * m).sum(1, keepdim=True) / den
        Q = (xyz - cmu) * m
        rad = (Q.pow(2).sum(-1, keepdim=True).sum(1) / den.squeeze(1).clamp(min=1.0)).clamp(min=1e-10).sqrt()
        Q = Q / rad.unsqueeze(1).clamp(min=1e-6)
        # normal equations, 3x3 만 풀면 된다
        G = torch.einsum('bni,bnj->bij', Q, Q).float()
        G = G + 1e-3 * torch.eye(3, device=G.device, dtype=G.dtype).unsqueeze(0)
        R = torch.einsum('bni,bnk->bik', Q, Ac).float()
        B = torch.linalg.solve(G, R)          # (b*g, 3, 11)
        return std, B

    xp = pred[:, : g * gsz, 0:3].reshape(b * g, gsz, 3).float()
    xt = target[:, : g * gsz, 0:3].reshape(b * g, gsz, 3).float()
    sp, Bp = moments(P, xp)
    with torch.no_grad():
        st, Bt = moments(T, xt)
    lv = live.to(P.dtype)
    nl = lv.sum().clamp(min=1.0)
    spread = ((sp - st).abs().mean(-1) * lv).sum() / nl
    slope = ((Bp - Bt).abs().mean(dim=(1, 2)) * lv).sum() / nl
    _ac.__exit__(None, None, None)
    return spread, slope


_HUNG_POOL = None
_HUNG_BUF: Dict[tuple, torch.Tensor] = {}


def _hung_pool():
    """짝짓기 전용 스레드 풀. 스텝마다 만들면 풀 생성 비용이 이득을 먹는다."""
    global _HUNG_POOL
    if _HUNG_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        n = int(os.environ.get("CAN3TOK_HUNG_WORKERS", "16"))
        _HUNG_POOL = ThreadPoolExecutor(max_workers=max(1, n),
                                        thread_name_prefix="hung")
    return _HUNG_POOL


def _hung_host_buf(shape, dtype) -> torch.Tensor:
    """모양별로 하나씩 재사용하는 고정 페이지 호스트 버퍼.

    매번 할당하면 pinned 할당 자체가 전송보다 비싸다. 여기서 넘기는 것은 비용 행렬
    하나뿐이고 모양이 (groups, slots, slots) 로 고정이므로 캐시가 한 칸이면 된다.
    """
    key = (tuple(int(s) for s in shape), dtype)
    buf = _HUNG_BUF.get(key)
    if buf is None:
        buf = torch.empty(key[0], dtype=dtype, pin_memory=True)
        _HUNG_BUF.clear()
        _HUNG_BUF[key] = buf
    return buf


def intra_group_hungarian_attr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    layout: Dict[str, int],
    min_live: int = 8,
):
    """셀 안 xyz Hungarian 짝 뒤 표준화 L1 (scale / opacity / rot).

    sinkhorn/responsibility 타깃은 위치만으로 짝을 정해서 같은 자리의 얇은 점과
    두꺼운 점을 평균낸다. 그 평균은 둥근 중간 크기이고, 난간이 죽는 이유다.
    여기서는 매칭을 detach 한 뒤 *살아 있는* 점끼리 hard 배정하고, 그 짝의
    속성을 직접 맞춘다. 슬롯 평균이 아니라 '이 GT 점은 이만큼 얇다'.

    반환: (scale, opacity, rot)  — 매칭된 live 점 평균, 표준화 채널 단위.
    """
    from scipy.optimize import linear_sum_assignment

    b, n, c = pred.shape
    gsz = int(group_size)
    ng = n // gsz
    bg = b * ng
    a_s, a_r = int(layout["scale"]), int(layout["rot"])
    a_o = int(layout["opacity"])
    z = pred.new_tensor(0.0)
    if c <= a_o:
        return z, z, z

    _ac = torch.autocast("cuda", enabled=False)
    _ac.__enter__()
    try:
        p = pred[:, : ng * gsz].reshape(bg, gsz, c).float()
        t = target[:, : ng * gsz].reshape(bg, gsz, c).float()
        m = mask[:, : ng * gsz].reshape(bg, gsz) > 0.5
        if int(m.sum()) < int(min_live):
            return z, z, z

        with torch.no_grad():
            cost = torch.cdist(p[..., 0:3], t[..., 0:3])
            big = cost.new_tensor(1.0e6)
            inv = ~m
            cost = cost.masked_fill(inv.unsqueeze(2), big)
            cost = cost.masked_fill(inv.unsqueeze(1), big)
            # 고정 페이지 버퍼로 옮긴다. 같은 268MB (1024x256x256) 를 pageable .cpu() 로
            # 보내면 149ms, pinned + non_blocking 이면 20ms 다 (측정, bench_hung.py).
            host = _hung_host_buf(cost.shape, cost.dtype)
            host.copy_(cost, non_blocking=True)
            torch.cuda.synchronize()
            cost_np = host.numpy()
            m_np = m.cpu().numpy()

        # 그룹 1024개를 스레드로 나눈다. scipy 의 linear_sum_assignment 는 C++ 이라 GIL 을
        # 놓으므로 코어가 남는 만큼 그대로 빨라지고, 배정 결과는 비트 단위로 같다:
        # 단일 스레드 515ms -> 16 스레드 85ms (측정). 이 손실은 스텝마다 한 번 불리고
        # 그 사이 GPU 는 놀기 때문에, 여기서 줄인 시간은 스텝 시간에서 그대로 빠진다.
        def _solve(i):
            li = np.flatnonzero(m_np[i])
            if li.size < int(min_live):
                return None
            r, col = linear_sum_assignment(cost_np[i][np.ix_(li, li)])
            return np.full(r.shape[0], i, dtype=np.int64), li[r], li[col]

        groups, rows, cols = [], [], []
        for got in _hung_pool().map(_solve, range(bg)):
            if got is None:
                continue
            groups.append(got[0]); rows.append(got[1]); cols.append(got[2])
        if not groups:
            return z, z, z

        gi = torch.from_numpy(np.concatenate(groups)).to(device=p.device)
        ri = torch.from_numpy(np.concatenate(rows)).to(device=p.device)
        ci = torch.from_numpy(np.concatenate(cols)).to(device=p.device)
        ps, ts = p[gi, ri], t[gi, ci]
        pq = ps[..., a_r: a_r + 4]
        tq = ts[..., a_r: a_r + 4]
        pq = pq * torch.sign(pq[..., :1].detach() + 1e-12)
        tq = tq * torch.sign(tq[..., :1].detach() + 1e-12)
        sd = torch.tensor(ATTR_STD[:11], device=p.device, dtype=p.dtype).clamp(min=1e-3)
        sc = ((ps[..., a_s: a_s + 3] - ts[..., a_s: a_s + 3]).abs() / sd[0:3]).mean()
        op = ((ps[..., a_o: a_o + 1] - ts[..., a_o: a_o + 1]).abs() / sd[7:8]).mean()
        rt = ((pq - tq).abs() / sd[3:7]).mean()
        return sc, op, rt
    finally:
        _ac.__exit__(None, None, None)


def intra_group_attr_set_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    layout: Dict[str, int],
    chunk_groups: int = 512,
    quat_sign: bool = True,
) -> torch.Tensor:
    """Permutation-invariant set distance between a group's ATTRIBUTE vectors.

    This is the term the objective was missing, and the reason it was missing is
    that every attribute path here is keyed on POSITION:

      slot-ordered target   data.py assigns slot i by an optimal match to a fixed
                            template in position space; attributes never enter
                            the cost, so between two GT Gaussians at nearly the
                            same place, which one gets the lower slot index is a
                            near-tie. Measured on GT co-located pairs, the lower
                            slot holds the larger Gaussian 47.0% of the time --
                            a coin flip.
      sinkhorn target       cost = cdist(pred_xyz, target_xyz). Position only, so
                            a co-located pair of very different GT Gaussians
                            produces a BLENDED target.
      responsibility        same, position only.
      render                sees the composite image, not individual Gaussians.

    So when the ground truth puts a thin-vertical and a thin-horizontal Gaussian
    at one place, every point-space target asks for the average of the two -- and
    the average is a round Gaussian. The collapse is the correct answer to the
    question being asked, not a failure to optimise.

    Measured, per group, on standardised attribute channels (so a quaternion
    component and a log scale are commensurable):

        GT against itself                    0.0017
        the model                            2.0628
        the model collapsed to its group mean 2.1868
        GT collapsed to its group mean        2.0390

    The model sits 5.7% away from a total collapse and right on top of "the GT's
    group mean", which is exactly what it was asked for. Nothing in the objective
    distinguishes those.

    This term asks the different question: do the 64 attribute vectors, AS A SET,
    match the ground truth's 64? Order-free, so it does not care that the slot
    assignment is uninformative -- it sidesteps the 47% coin flip rather than
    trying to fix it. 64 identical vectors cannot match a diverse set, so the
    collapse is penalised directly.

    Why this should also revive the per-slot basis without being forced: M7
    rescaled slot_emb 5x and gave it 30x the learning rate, and it SHRANK
    (0.147 -> 0.136) because no term rewarded within-group variety. With a reward
    present the optimiser has a reason to keep it.

    Channels are standardised by fixed dataset statistics, not per-batch, so the
    term means the same thing at every step. Quaternions get their sign fixed
    first: q and -q are the same rotation, and without that the set distance would
    charge for a sign flip.
    """
    b, n, _ = pred.shape
    gsz = int(group_size)
    g = n // gsz
    a_s, a_r = int(layout["scale"]), int(layout["rot"])
    a_o, a_c = int(layout["opacity"]), int(layout["color"])
    ncol = a_c + 3

    def prep(x):
        # Built by concatenation, never by in-place assignment: writing the
        # sign-fixed quaternion back into a slice of the same tensor makes that
        # slice a stale autograd input and the backward pass fails outright
        # ("modified by an inplace operation"). Caught by the unit test.
        v = x[:, : g * gsz, :ncol].reshape(b * g, gsz, ncol).float()
        q = v[..., a_r : a_r + 4]
        if quat_sign:
            # q and -q are the same rotation. Detached because sign() is
            # piecewise constant -- it carries no useful gradient and leaving it
            # attached only adds a zero-derivative path.
            q = q * torch.sign(q[..., :1].detach() + 1e-12)
        # scale / rot / opacity / colour only. Position is already supervised by
        # four separate terms, and including it here would let a position error
        # masquerade as an attribute one.
        return torch.cat([v[..., a_s : a_s + 3], q,
                          v[..., a_o : a_o + 1], v[..., a_c : a_c + 3]], dim=-1)

    sd = torch.tensor(ATTR_STD[:11], device=pred.device, dtype=torch.float32).clamp(min=1e-3)
    P = prep(pred) / sd
    T = prep(target) / sd
    m = mask[:, : g * gsz].reshape(b * g, gsz)
    live_pt = m > 0.5
    live_g = (live_pt.sum(-1) > 1.5)
    BIG = 1e9
    step = max(int(chunk_groups), 1)
    num = P.new_tensor(0.0)
    den = P.new_tensor(0.0)
    for i0 in range(0, b * g, step):
        i1 = min(i0 + step, b * g)
        pp, tt, mm = P[i0:i1], T[i0:i1], live_pt[i0:i1]
        lg = live_g[i0:i1]
        if not bool(lg.any()):
            continue
        d = torch.cdist(pp, tt)
        pad_t = (~mm).unsqueeze(1)          # target columns to ignore
        pad_p = (~mm).unsqueeze(2)          # prediction rows to ignore
        dm = d.masked_fill(pad_t, BIG).masked_fill(pad_p, BIG)
        # Symmetric: every prediction needs a target near it AND every target
        # needs a prediction near it. One direction alone is satisfiable by
        # emitting a single vector that happens to sit in the middle.
        p2t = dm.min(dim=-1).values
        t2p = dm.min(dim=-2).values
        w = mm.to(P.dtype)
        cnt = w.sum(-1).clamp(min=1.0)
        per = 0.5 * ((p2t * w).sum(-1) / cnt + (t2p * w).sum(-1) / cnt)
        gw = lg.to(P.dtype)
        num = num + (per * gw).sum()
        den = den + gw.sum()
    return num / den.clamp(min=1.0)


def intra_group_local_attr_set_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    layout: Dict[str, int],
    k: int = 16,
    chunk_groups: int = 32,
    quat_sign: bool = True,
) -> torch.Tensor:
    """attr_set on a *spatial neighbourhood*, not the whole 256-slot cell.

    `intra_group_attr_set_loss` compares the bag of every live slot in the cell.
    At group 64 that bag was the object (N1). At group 256 with ~70 live points
    a bag of medium spheres can match the cell-wide set, so rails in one corner
    are optional. This term builds a k-NN bag around each predicted point,
    inside the same cell, and compares that bag to the k nearest GT attributes
    at the same place. It is still a SET distance: it does not copy a GT
    quaternion onto a predicted slot (the pairing that smeared H2 / F2 / GT
    scale swap). xyz is detached so the term cannot move positions.

    k must stay well below typical occupancy (~71) or this collapses back to
    the cell-wide set. k=1 is the closed nearest-neighbour copy. Default 16.
    """
    b, n, _ = pred.shape
    gsz = int(group_size)
    g = n // gsz
    a_s, a_r = int(layout["scale"]), int(layout["rot"])
    a_o, a_c = int(layout["opacity"]), int(layout["color"])
    ncol = a_c + 3
    k = int(max(2, min(int(k), gsz)))

    def prep(x):
        v = x[:, : g * gsz, :ncol].reshape(b * g, gsz, ncol).float()
        q = v[..., a_r : a_r + 4]
        if quat_sign:
            q = q * torch.sign(q[..., :1].detach() + 1e-12)
        return torch.cat([v[..., a_s : a_s + 3], q,
                          v[..., a_o : a_o + 1], v[..., a_c : a_c + 3]], dim=-1)

    sd = torch.tensor(ATTR_STD[:11], device=pred.device, dtype=torch.float32).clamp(min=1e-3)
    P = prep(pred) / sd
    T = prep(target) / sd
    xyz_p = pred[:, : g * gsz, 0:3].reshape(b * g, gsz, 3).float().detach()
    xyz_t = target[:, : g * gsz, 0:3].reshape(b * g, gsz, 3).float().detach()
    live = mask[:, : g * gsz].reshape(b * g, gsz) > 0.5
    C = P.shape[-1]
    BIG = 1.0e4
    step = max(int(chunk_groups), 1)
    num = P.new_tensor(0.0)
    den = P.new_tensor(0.0)
    for i0 in range(0, b * g, step):
        i1 = min(i0 + step, b * g)
        mm = live[i0:i1]
        if not bool((mm.sum(-1) > 1.5).any()):
            continue
        pp, tt = P[i0:i1], T[i0:i1]
        xp, xt = xyz_p[i0:i1], xyz_t[i0:i1]
        s, nn = pp.shape[0], pp.shape[1]
        dx = torch.cdist(xp, xp)
        dx = dx.masked_fill(~mm.unsqueeze(1), BIG).masked_fill(~mm.unsqueeze(2), BIG)
        _, nbp = dx.topk(k, largest=False, dim=-1)
        dg = torch.cdist(xp, xt)
        dg = dg.masked_fill(~mm.unsqueeze(1), BIG)
        _, nbt = dg.topk(k, largest=False, dim=-1)
        bix = torch.arange(s, device=pp.device)[:, None, None]
        okp = mm[bix, nbp]
        okt = mm[bix, nbt]
        pa = pp[bix, nbp].reshape(s * nn, k, C)
        ta = tt[bix, nbt].reshape(s * nn, k, C)
        vp = okp.reshape(s * nn, k)
        vt = okt.reshape(s * nn, k)
        d = torch.cdist(pa, ta)
        d = d.masked_fill(~vt.unsqueeze(1), BIG).masked_fill(~vp.unsqueeze(2), BIG)
        p2t = d.min(dim=-1).values
        t2p = d.min(dim=-2).values
        wp = vp.to(P.dtype)
        wt = vt.to(P.dtype)
        per = 0.5 * ((p2t * wp).sum(-1) / wp.sum(-1).clamp(min=1.0)
                     + (t2p * wt).sum(-1) / wt.sum(-1).clamp(min=1.0))
        center = (mm.reshape(s * nn)
                  & (vp.sum(-1) > 1.5)
                  & (vt.sum(-1) > 1.5)
                  & (per < (BIG * 0.5)))
        gw = center.to(P.dtype)
        num = num + (per * gw).sum()
        den = den + gw.sum()
    return num / den.clamp(min=1.0)


def target_group_extent(target_xyz: torch.Tensor, mask: torch.Tensor,
                        group_size: int) -> torch.Tensor:
    """Detached RMS extent of each target group, shaped ``(B,G,1)``.

    Training normalisation must be defined by the target.  Using the encoder's
    extent made independently sampled 144-point input groups decide how strongly
    errors on the 64-point output groups were amplified.
    """
    b, n, _ = target_xyz.shape
    gsz = int(group_size)
    g = n // gsz
    t = target_xyz[:, :g * gsz].reshape(b, g, gsz, 3)
    m = mask[:, :g * gsz].reshape(b, g, gsz).to(t.dtype)
    den = m.sum(dim=2, keepdim=True).clamp(min=1.0)
    cen = (t * m.unsqueeze(-1)).sum(dim=2) / den
    var = (((t - cen.unsqueeze(2)) ** 2) * m.unsqueeze(-1)).sum(dim=2) / den
    return var.clamp(min=0.0).sqrt().amax(dim=-1, keepdim=True).detach()


@torch.no_grad()
def sinkhorn_parameter_target(
    pred_xyz: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    scale: Optional[torch.Tensor] = None,
    epsilon: float = 0.08,
    iterations: int = 6,
    chunk_groups: int = 256,
    pred_attr: Optional[torch.Tensor] = None,
    attr_weight: float = 0.0,
    layout: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """Transfer all GT parameters with the geometry loss's balanced OT plan.

    Unlike ``responsibility_target``, this is balanced and cannot merge several
    GT Gaussians into one enlarged splat.  xyz and attributes consequently refer
    to the same soft one-to-one correspondence.
    """
    b, n, _ = pred_xyz.shape
    gsz = int(group_size)
    ng = n // gsz
    n_used = ng * gsz
    p = pred_xyz[:, :n_used, :3].reshape(b * ng, gsz, 3).float()
    t = target[:, :n_used].reshape(b * ng, gsz, -1).float()
    m = mask[:, :n_used].reshape(b * ng, gsz) > 0.5
    if scale is not None:
        sc = scale[:, :ng].float().detach().abs().reshape(b * ng, 1, 1)
        sc = sc.clamp(min=0.05 * sc.median().clamp(min=1e-6))
        p_cost, t_cost = p / sc, t[..., :3] / sc
    else:
        p_cost, t_cost = p, t[..., :3]
    # Optionally let ATTRIBUTES enter the transport cost. Position-only matching
    # is why a co-located pair of very different GT Gaussians yields a blended
    # target: the plan cannot tell them apart, so it splits mass between them and
    # the prediction is asked for the average. Appending standardised attribute
    # channels makes the plan prefer the GT Gaussian this prediction already
    # resembles, which turns the target from an average into a choice.
    #
    # `pred_attr` is detached by the caller for the same reason pred_xyz is: the
    # target must not be steerable by the prediction it supervises.
    if pred_attr is not None and float(attr_weight) > 0.0:
        a_s, a_r = int(layout["scale"]), int(layout["rot"])
        a_o, a_c = int(layout["opacity"]), int(layout["color"])
        sd = torch.tensor(ATTR_STD[:11], device=p.device, dtype=torch.float32).clamp(min=1e-3)
        def _sel(x):
            v = x.reshape(b * ng, gsz, -1).float()
            q = v[..., a_r:a_r + 4] * torch.sign(v[..., a_r:a_r + 1] + 1e-12)
            return torch.cat([v[..., a_s:a_s + 3], q, v[..., a_o:a_o + 1],
                              v[..., a_c:a_c + 3]], dim=-1) / sd
        pa = _sel(pred_attr[:, :n_used]) * float(attr_weight)
        ta = _sel(target[:, :n_used]) * float(attr_weight)
        p_cost = torch.cat([p_cost, pa], dim=-1)
        t_cost = torch.cat([t_cost, ta], dim=-1)

    out = t.clone()
    eps = max(float(epsilon), 1e-4)
    neg = -1e4
    for i0 in range(0, b * ng, max(int(chunk_groups), 1)):
        i1 = min(i0 + max(int(chunk_groups), 1), b * ng)
        pc, tc, mc = p_cost[i0:i1], t_cost[i0:i1], m[i0:i1]
        cost = torch.cdist(pc, tc)
        pair = mc.unsqueeze(2) & mc.unsqueeze(1)
        log_k = torch.where(pair, -cost / eps, cost.new_full((), neg))
        count = mc.sum(dim=-1, keepdim=True).float().clamp(min=1.0)
        log_mass = -count.log()
        log_u = torch.where(mc, log_mass.expand_as(cost[..., 0]), cost.new_full((), neg))
        log_v = log_u.clone()
        for _ in range(max(int(iterations), 1)):
            log_u = torch.where(
                mc, log_mass - torch.logsumexp(log_k + log_v.unsqueeze(1), dim=2),
                cost.new_full((), neg))
            log_v = torch.where(
                mc, log_mass - torch.logsumexp(log_k + log_u.unsqueeze(2), dim=1),
                cost.new_full((), neg))
        plan = torch.exp(log_k + log_u.unsqueeze(2) + log_v.unsqueeze(1))
        plan = torch.where(pair, plan, torch.zeros_like(plan))
        row = plan.sum(dim=2, keepdim=True)
        transferred = torch.bmm(plan, t[i0:i1]) / row.clamp(min=1e-12)
        if transferred.shape[-1] >= 10:
            # q and -q are the same rotation. Align every transported quaternion
            # to the row's strongest match before averaging, then return to S3.
            ref_i = plan.argmax(dim=2)
            q = t[i0:i1, :, 6:10]
            ref_q = torch.gather(q, 1, ref_i.unsqueeze(-1).expand(-1, -1, 4))
            sign = torch.sign((q.unsqueeze(1) * ref_q.unsqueeze(2)).sum(dim=-1, keepdim=True))
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            q_mean = (plan.unsqueeze(-1) * q.unsqueeze(1) * sign).sum(dim=2)
            q_mean = F.normalize(q_mean, dim=-1)
            transferred[..., 6:10] = q_mean
        out[i0:i1] = torch.where(mc.unsqueeze(-1), transferred, t[i0:i1])

    full = target.clone()
    full[:, :n_used] = out.reshape(b, n_used, -1).to(target.dtype)
    return full


def group_centroid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Directly supervise the pose of each fixed-anchor group."""
    b, n, _ = pred.shape
    gsz = int(group_size)
    g = n // gsz
    p = pred[..., :3].reshape(b, g, gsz, 3)
    t = target[..., :3].reshape(b, g, gsz, 3)
    m = mask[:, : g * gsz].reshape(b, g, gsz).to(p.dtype)
    denom = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
    pc = (p * m.unsqueeze(-1)).sum(dim=2) / denom
    tc = (t * m.unsqueeze(-1)).sum(dim=2) / denom
    d = (pc - tc).norm(dim=-1)
    if scale is not None:
        sc = scale[:, :g].to(d.dtype).detach().squeeze(-1).abs()
        sc = sc.clamp(min=0.05 * sc.median().clamp(min=1e-6))
        d = d / sc
    valid = (m.sum(dim=-1) > 0).to(d.dtype)
    return masked_reduce(d, valid)


def patch_dispersion_loss(
    pred_xyz,
    mask,
    group_size: int = 32,
    margin: float = 0.001,
    target_xyz: Optional[torch.Tensor] = None,
    margin_frac: float = 0.35,
):
    """Hinge that stops a group from collapsing all its points onto one spot.

    ``margin`` alone used to be 0.01 — ~5.7× the median GT group radius — so the
    hinge was always active on dense groups and pushed blur. Prefer a small
    absolute floor plus a fraction of the GT group std.
    """
    b, n, _ = pred_xyz.shape
    ng = n // group_size
    xyz = pred_xyz[:, : ng * group_size].reshape(b, ng, group_size, 3)
    m = mask[:, : ng * group_size].reshape(b, ng, group_size, 1)
    cnt = m.sum(dim=2).clamp(min=1.0)
    mean = (xyz * m).sum(dim=2) / cnt
    var = ((xyz - mean.unsqueeze(2)) ** 2 * m).sum(dim=2) / cnt
    std = var.clamp(min=1e-12).sqrt().mean(dim=-1)
    thr = pred_xyz.new_full(std.shape, float(margin))
    if target_xyz is not None and float(margin_frac) > 0.0:
        t = target_xyz[:, : ng * group_size].reshape(b, ng, group_size, 3)
        t_mean = (t * m).sum(dim=2) / cnt
        t_var = ((t - t_mean.unsqueeze(2)) ** 2 * m).sum(dim=2) / cnt
        t_std = t_var.clamp(min=1e-12).sqrt().mean(dim=-1).detach()
        thr = torch.maximum(thr, float(margin_frac) * t_std)
    active = (m.sum(dim=2).squeeze(-1) > 1.5).float()
    return masked_reduce(F.relu(thr - std), active)


# ---------------------------------------------------------------------------
# latent (z_raw) losses
# ---------------------------------------------------------------------------


def shape_channel_mask(channels: int, per_group: int, c_centroid: int, c_occupancy: int,
                       c_shape: Optional[int] = None) -> torch.Tensor:
    """1.0 on the learned shape channels of every group, 0.0 on the anchors.

    ``c_shape`` bounds the block from above. Without it the mask was
    ``offsets >= c_centroid + c_occupancy``, i.e. everything after the anchors --
    which silently swept the *appearance* channels in too. On the L16k layout
    (centroid 4 | occupancy 1 | shape 3 | appearance 8) that made the "shape"
    block 11 channels wide instead of 3, so ``w_latent_decorr`` was decorrelating
    colour against geometry and ``latent_std``'s band was being applied to an
    appearance code that has no reason to sit in [floor, ceil]. Measured on that
    run, ``latent_erank`` pinned at exactly 11.0 out of 11 -- the regulariser had
    won outright against reconstruction on channels it was never meant to touch.

    Left optional so a caller that genuinely wants "everything past the anchors"
    keeps the old behaviour, but every in-tree caller now passes the width.
    """
    offsets = torch.arange(channels) % max(per_group, 1)
    lo = c_centroid + c_occupancy
    m = offsets >= lo
    if c_shape is not None:
        m = m & (offsets < lo + int(c_shape))
    return m.float()


def z_map_tokens(z_map: torch.Tensor, num_groups: int, tpg: int) -> torch.Tensor:
    b, c, h, w = z_map.shape
    cells = z_map.reshape(b, c, h * w).transpose(1, 2)
    return cells[:, : num_groups * tpg].reshape(b, num_groups, tpg, c)


def tokens_to_xyz_mask(
    z_map: torch.Tensor, num_groups: int, tpg: int, group_size: int,
    group_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Undo the pack layout: xyz, then the aux block.

    The aux block is the per-point mask in the geometry-only pack, but it becomes
    a *learned appearance code* once attributes are packed -- and clamping an
    appearance code to [0, 1] and using it as an occupancy weight would silently
    zero out points and corrupt every loss built on it.

    So pass ``group_valid`` whenever appearance is active. Prefix packing makes
    groups all-or-nothing (measured: exactly one partially-filled group per
    scene), which is what makes the group-level flag an accurate stand-in.
    """
    tok = z_map_tokens(z_map, num_groups, tpg)
    flat = tok.reshape(tok.shape[0], num_groups, -1)
    xyz = flat[..., : group_size * 3].reshape(tok.shape[0], num_groups, group_size, 3)
    if group_valid is not None:
        m = group_valid[:, :num_groups].to(xyz.dtype).unsqueeze(-1).expand(-1, -1, group_size)
        return xyz, m
    m = flat[..., group_size * 3 : group_size * 4].reshape(tok.shape[0], num_groups, group_size)
    return xyz, m.clamp(0.0, 1.0)


def group_residual_z_loss(
    z_hat: torch.Tensor,
    z_raw: torch.Tensor,
    group_valid: torch.Tensor,
    num_groups: int,
    tpg: int,
    group_size: int,
    scale: Optional[torch.Tensor] = None,
    hard_frac: float = 0.2,
    packed_mask: bool = True,
) -> Dict[str, torch.Tensor]:
    """Supervise *within-group* geometry, ignoring the shared centroid.

    Plain ``z_l1`` is dominated by the centroid/mask which the non-parametric
    shortcut already gets right. This loss subtracts the group centroid from both
    sides (and divides by the *target* group extent) so the only way to drive it
    down is to put the point offsets into the compact shape channels.
    """
    gv_arg = None if packed_mask else group_valid
    xyz_h, _ = tokens_to_xyz_mask(z_hat, num_groups, tpg, group_size, gv_arg)
    xyz_t, m = tokens_to_xyz_mask(z_raw, num_groups, tpg, group_size, gv_arg)
    gv = group_valid[:, :num_groups].to(xyz_t.dtype)
    # mask-aware centroid of the target pack (empty slots are xyz=0 and must not
    # pull the mean). Detached so the compressor cannot cheat by moving centroids.
    m_sum = m.sum(dim=2).clamp(min=1.0)  # (B, G)
    cen = ((xyz_t * m.unsqueeze(-1)).sum(dim=2) / m_sum.unsqueeze(-1)).detach()
    res_h = (xyz_h - cen.unsqueeze(2)) * m.unsqueeze(-1)
    res_t = (xyz_t - cen.unsqueeze(2)) * m.unsqueeze(-1)
    # extent from the target itself — never from the decoded scale channel, which
    # starts near zero and would blow the normalised residual up by 100x.
    ext = (xyz_t.amax(dim=2) - xyz_t.amin(dim=2)).amax(dim=-1, keepdim=True).clamp(min=1e-3)
    if scale is not None:
        # prefer the (detached) analytic scale when provided, still clamped
        s = scale[:, :num_groups].to(xyz_t.dtype).detach()
        if s.dim() == 2:
            s = s.unsqueeze(-1)
        ext = torch.maximum(ext, s.clamp(min=1e-3))
    res_h = res_h / ext.unsqueeze(-1)
    res_t = res_t / ext.unsqueeze(-1)
    per_pt = (res_h - res_t).abs().mean(dim=-1)
    per_group = (per_pt * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
    l1 = masked_reduce(per_group, gv)
    flat = (per_group * gv).reshape(per_group.shape[0], -1)
    k = max(1, min(flat.shape[1], int(hard_frac * flat.shape[1])))
    hard = torch.topk(flat, k, dim=1).values.mean()
    return {"z_residual": l1, "z_residual_hard": hard}


def point_group_residual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_size: int,
    scale: Optional[torch.Tensor] = None,
    hard_frac: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """Same residual supervision in point space (codec / gen decoder outputs)."""
    b, n, _ = pred.shape
    g = n // group_size
    p = pred[..., 0:3].reshape(b, g, group_size, 3)
    t = target[..., 0:3].reshape(b, g, group_size, 3)
    m = mask.reshape(b, g, group_size).to(p.dtype)
    gv = (m.sum(dim=-1) > 0.5).to(p.dtype)
    cen = (t * m.unsqueeze(-1)).sum(dim=2) / m.sum(dim=2, keepdim=True).clamp(min=1.0)
    cen = cen.detach()
    res_p = (p - cen.unsqueeze(2)) * m.unsqueeze(-1)
    res_t = (t - cen.unsqueeze(2)) * m.unsqueeze(-1)
    if scale is not None:
        s = scale[:, :g].to(p.dtype).clamp(min=1e-4)
        if s.dim() == 2:
            s = s.unsqueeze(-1)
        res_p = res_p / s.unsqueeze(-1)
        res_t = res_t / s.unsqueeze(-1)
    per_pt = (res_p - res_t).abs().mean(dim=-1)  # (B,G,S)
    per_group = (per_pt * m).sum(-1) / m.sum(-1).clamp(min=1.0)
    l1 = masked_reduce(per_group, gv)
    flat = (per_group * gv).reshape(b, -1)
    k = max(1, min(flat.shape[1], int(hard_frac * flat.shape[1])))
    hard = torch.topk(flat, k, dim=1).values.mean()
    return {"xyz_residual": l1, "xyz_residual_hard": hard}


def z_raw_losses(
    z_hat: torch.Tensor,
    z_raw: torch.Tensor,
    group_valid: torch.Tensor,
    num_groups: int,
    tpg: int,
    hard_frac: float = 0.12,
    hard_min_tokens: int = 256,
    std_target: float = 1.0,
    supervise_empty: bool = True,
    geom_dim: int = 0,
) -> Dict[str, torch.Tensor]:
    a = z_map_tokens(z_hat, num_groups, tpg)
    t = z_map_tokens(z_raw, num_groups, tpg)
    if geom_dim > 0:
        # Once the pack's aux block is a *learned* appearance code, comparing the
        # whole token compares against a target that moves every step -- and this
        # is the strongest term in the objective (51.8% of the within-group
        # gradient). Restrict it to the deterministic xyz prefix, which is exactly
        # as exact as it always was. Appearance is supervised where it is actually
        # decoded: the attribute losses and the render loss.
        b, ng, tp, c = a.shape
        a = a.reshape(b, ng, tp * c)[..., :geom_dim].reshape(b, ng, tp, -1)
        t = t.reshape(b, ng, tp * c)[..., :geom_dim].reshape(b, ng, tp, -1)
    gv = group_valid[:, :num_groups]
    # Empty groups have zero pack targets. Supervising them forces compact pad
    # slots toward a quiet code instead of leaving them unconstrained.
    if supervise_empty:
        supervise = torch.ones_like(gv)
    else:
        supervise = gv
    tok_valid = supervise[:, :, None].expand(-1, -1, tpg)

    # SCALE-FREE. Every term below used to be in the raw units of `t`, and `t` is
    # `out["z_raw"].detach()` -- which, whenever the encoder's GroupPointPooler is
    # active, is an unbounded LEARNED code, not the metric xyz pack the docstring
    # above assumes. Nothing in the objective bounds its magnitude, so it drifted
    # freely: measured on L17k, |z_raw| went 20.8 -> 323.3 over 32000 steps while
    # z_raw_hat sat at 0.0094, making `mse` simply E[t^2] = 109,062. At
    # w_z_raw_mse=1.5 that single term was 97.2% of the objective (163,594 of
    # 168,369) and the whole loss tracked 2.2 * z_l1^2 with R^2 = 0.997. The
    # target being detached is what closes the loop: the encoder gets no gradient
    # to shrink the code, so the term grows quadratically for free.
    #
    # Dividing by the target's own RMS makes each term dimensionless, so a weight
    # means the same thing regardless of what scale the pooler happens to settle
    # at. Detached: this is a normalisation, not a path the gradient may exploit
    # by shrinking the denominator.
    t_rms = (t.detach() ** 2).mean().clamp(min=1e-8).sqrt()
    a = a / t_rms
    t = t / t_rms

    diff = (a - t).abs()
    l1 = masked_reduce(diff.mean(dim=-1), tok_valid)
    mse = masked_reduce(((a - t) ** 2).mean(dim=-1), tok_valid)

    # per-token error, mined over the worst tokens (see masked_hard_smooth_l1 on
    # why the fraction is relative to the padded token count)
    tok_err = diff.mean(dim=-1).reshape(a.shape[0], -1)
    tok_mask = tok_valid.reshape(a.shape[0], -1)
    tok_err = tok_err * tok_mask
    k = max(int(hard_min_tokens), int(hard_frac * tok_err.shape[1]))
    k = max(1, min(k, tok_err.shape[1]))
    hard = torch.topk(tok_err, k, dim=1).values.mean()

    # Diversity / std terms stay on live groups only (empty std is trivially 0).
    std_a = a.std(dim=2)
    std_t = t.std(dim=2)
    token_var = masked_reduce((std_a - std_t).abs().mean(dim=-1), gv)
    ratio = std_a.mean(dim=-1) / (std_t.mean(dim=-1) + 1e-6)
    std_ratio = masked_reduce((ratio - std_target).abs(), gv)
    return {"z_l1": l1, "z_mse": mse, "z_hard": hard, "z_token_var": token_var, "z_std_ratio": std_ratio}


# ---------------------------------------------------------------------------
# equivariance (EQ-VAE flavoured)
# ---------------------------------------------------------------------------


def equivariance_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    rot: torch.Tensor,
    merge: int,
    per_group: int,
    c_cen: int,
    shape_weight: float = 0.0,
) -> torch.Tensor:
    """Centroid channels must rotate with the cloud.

    Shape channels are *not* rotation-invariant (within-group offsets rotate), so
    the old ``loss_shape`` term forced an isotropic code and killed detail. Keep
    ``shape_weight=0`` unless you deliberately want that ablation.
    """
    b, c, h, w = z_a.shape
    a = z_a.reshape(b, merge, per_group, h, w)
    d = z_b.reshape(b, merge, per_group, h, w)
    cen_a = a[:, :, 0:3].permute(0, 1, 3, 4, 2)
    cen_b = d[:, :, 0:3].permute(0, 1, 3, 4, 2)
    cen_rot = torch.einsum("ij,bmhwj->bmhwi", rot, cen_a)
    loss_cen = (cen_b - cen_rot).abs().mean()
    if float(shape_weight) == 0.0:
        return loss_cen
    loss_shape = (d[:, :, c_cen:] - a[:, :, c_cen:]).abs().mean()
    return loss_cen + float(shape_weight) * loss_shape


def frame_distance(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                   lam_rot: float = 1.0) -> torch.Tensor:
    """Distance between two folding frames, in the space each part lives in.

    A frame is ``[3 pre-softplus anisotropy, 3 axis-angle]`` and an L1 on the raw
    6 numbers is wrong on both halves:

    * the scales are compared **before** ``softplus`` and before the geometric-mean
      normalisation, so the same L1 means a different scale ratio depending on
      where on the softplus curve the pair sits;
    * axis-angle is not a metric space for rotations. ``v`` and
      ``v (1 - 2*pi/|v|)`` are the *same* rotation but are far apart in L1, and
      the representation is discontinuous at ``|v| = pi``. A student that has the
      teacher's rotation can still be punished for it.

    So compare the anisotropy in log space after the same normalisation the
    decoder applies (which makes the loss proportional to the scale *ratio* and
    invariant to overall size, matching how the frame is actually used), and the
    rotation by the Frobenius distance between the rotation matrices. Frobenius
    rather than the geodesic angle: it is monotone in the geodesic angle, bounded,
    and its gradient is finite at zero, where ``arccos`` blows up.
    """
    from .compressor import axis_angle_to_matrix

    a_p = F.softplus(pred[..., :3]) + 1e-3
    a_t = F.softplus(target[..., :3]) + 1e-3
    a_p = a_p / a_p.prod(dim=-1, keepdim=True).clamp(min=1e-6).pow(1.0 / 3.0)
    a_t = a_t / a_t.prod(dim=-1, keepdim=True).clamp(min=1e-6).pow(1.0 / 3.0)
    d_scale = (a_p.log() - a_t.log()).abs().mean(dim=-1)
    if pred.shape[-1] < 6 or float(lam_rot) == 0.0:
        return masked_reduce(d_scale, valid)
    r_p = axis_angle_to_matrix(pred[..., 3:6])
    r_t = axis_angle_to_matrix(target[..., 3:6])
    d_rot = (r_p - r_t).flatten(-2).pow(2).sum(-1).clamp(min=1e-12).sqrt()
    return masked_reduce(d_scale + float(lam_rot) * d_rot, valid)


def _detach_near_gaussians(gp, cam, frac: float):
    """Keep close points in the image, and stop their render gradient.

    Screen radius is ``f * scale / z``. A point at a small fraction of the
    view's median depth makes that gradient enormous while the scalar loss
    barely moves, and after grad clipping the whole update follows it. The
    cutoff is relative to this camera, so it does not depend on the dataset's
    scale. ``frac <= 0`` leaves the tuple unchanged.
    """
    if frac <= 0.0:
        return gp
    xyz, scale = gp[0], gp[1]
    W = cam.world_view_transform
    z = (xyz @ W[:3, :3])[:, 2] + W[3, 2]
    z_min = (float(frac) * z.median().clamp(min=1e-2)).clamp(min=1e-2)
    # Depth is not the only 1/z blow-up. A moderate depth with a large scale
    # has the same screen radius. Compare to this view's own median so the
    # cutoff does not depend on the dataset.
    f = 0.5 * float(cam.image_width) / max(float(cam.tanfovx), 1e-6)
    radius = f * scale.detach().amax(dim=-1) / z.detach().clamp(min=1e-3)
    fat = radius > (4.0 * radius.median().clamp(min=1e-3))
    near = ((z < z_min) | fat).view(-1, 1)
    if not bool(near.any()):
        return gp

    def _mix(t):
        return torch.where(near, t.detach(), t)

    return tuple(_mix(t) for t in gp)


def render_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    camera_vec: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    layout: Dict[str, int],
    lam_dssim: float = 0.2,
    edge_gain: float = 0.0,
    w_sobel: float = 0.0,
    downscale: int = 2,
    max_points: int = 0,
    min_coverage: float = 0.25,
    views: int = 1,
    view_jitter_deg: float = 8.0,
    seed: int = 0,
    detach_xyz: bool = False,
    detach_scale_rot: bool = False,
    w_perc: float = 0.0,
    w_splat: float = 0.0,
    splat_quantile: float = 0.80,
    near_detach_frac: float = 0.1,
    pred_presence: Optional[torch.Tensor] = None,
    ref_image: Optional[torch.Tensor] = None,
    extra_cams: Optional[torch.Tensor] = None,
    extra_imgs: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Photometric agreement between the predicted and the target Gaussians.

    This is the only term in the objective that is not a proxy. Every point-space
    metric here has turned out to be satisfiable by a reconstruction that renders
    badly -- the clearest case being duplication, where two predicted points share
    one nearest GT point, which a symmetric chamfer scores as perfect while half
    the surface goes uncovered.

    The reference is a render of the *target* Gaussians rather than a photograph
    (the npz stores no image), so this asks for view equivalence, not pixel truth.
    That is the weaker requirement, and it is the one that makes attributes
    affordable at all: storing them costs 8-112x the compact latent, matching
    their appearance from the group code costs nothing extra.

    Both sides are rendered from the *same* camera with the *same* rasteriser, so
    the difference is entirely due to the Gaussians. Gradients reach ``pred``
    only; the target branch is built under ``no_grad``.
    """
    from .render import (Camera, render_gaussians, photometric_loss, sh_dc_to_rgb,
                         perturb_camera_vector, vgg_perceptual, _sobel)
    from .io_utils import camera_from_vector

    dev = pred.device
    b = pred.shape[0]
    a = layout
    has_attrs = pred.shape[-1] > 11
    zero = pred.sum() * 0.0
    # Per-point visibility, accumulated over batch and views. The anchor needs it
    # to answer "did the render have anything to say about this point?" -- on the
    # points it did, the anchor is the weaker signal by a measured 10 dB and only
    # pulls them back towards the group mean.
    vis = torch.zeros(pred.shape[0], pred.shape[1], dtype=torch.uint8, device=dev)
    out = {"render": zero, "render_l1": zero.detach(), "render_dssim": zero.detach(),
           "render_sobel": zero.detach(), "render_used": zero.detach(), "splat_area": zero}
    tot, n_ok = zero, 0

    for i in range(b):
        cv = camera_vec[i].detach().float().cpu().numpy()
        if float(cv[0]) <= 0.0:                       # frame carried no camera
            continue
        m = mask[i] > 0.5
        if int(m.sum()) < 64:
            continue
        idx = torch.nonzero(m, as_tuple=False).reshape(-1)
        # Which slots the PREDICTION contributes. Until now this was the GT mask
        # for both sides, so `presence` never entered the render objective at all
        # and the model was scored with information it does not have at inference
        # (measured on R8: GT mask 14.89 dB vs predicted presence 14.76, and
        # 11.85 with every slot present -- so the head is doing real work, it just
        # was never asked to). Gating the prediction here is also the precondition
        # for a variable-count decoder: without it, emitting a Gaussian in an empty
        # region costs nothing.
        idx_p = idx
        if pred_presence is not None:
            pm = (pred_presence[i] > 0)
            if int(pm.sum()) >= 64:
                idx_p = torch.nonzero(pm, as_tuple=False).reshape(-1)
        if max_points and idx.numel() > max_points:
            # Subsampling both sides identically keeps the comparison fair: the
            # same slots are dropped from prediction and target.
            idx = idx[torch.randperm(idx.numel(), device=dev)[:max_points]]
        cam = Camera(camera_from_vector(cv), device=dev, downscale=int(downscale))
        c, s = center[i].reshape(1, 3).to(dev), float(scale[i])

        def unpack(t, grad: bool, sel=None):
            sel = idx if sel is None else sel
            xyz = t[sel, 0:3] * s + c
            # `detach_xyz` splits this term in two. The rasteriser gradient is
            # well conditioned for attributes -- opacity, colour and scale change
            # the pixels a Gaussian already covers -- and badly conditioned for
            # positions, which need transport: measured cosine 0.13 with the true
            # correction, and only 24% of points receive any gradient at all,
            # because the median Gaussian projects to 1.4 px while the position
            # error is ~6 px. Cutting the position path lets the weight be set
            # from the attribute gradient alone instead of being held down by the
            # one it must not drive.
            if detach_xyz:
                xyz = xyz.detach()
            if has_attrs:
                sc = torch.exp(t[sel, a["scale"] : a["scale"] + 3].clamp(-15.0, 5.0)) * s
                rq = t[sel, a["rot"] : a["rot"] + 4]
                op = torch.sigmoid(t[sel, a["opacity"] : a["opacity"] + 1])
                co = sh_dc_to_rgb(t[sel, a["color"] : a["color"] + 3])
                if detach_scale_rot and grad:
                    sc = sc.detach()
                    rq = rq.detach()
            else:
                # xyz-only stage: give both sides the *target's* attributes so the
                # image difference isolates geometry. Detached on the pred side --
                # they are not what is being learned here.
                sc = torch.exp(target[i][sel, a["scale"] : a["scale"] + 3].clamp(-15.0, 5.0)) * s
                rq = target[i][sel, a["rot"] : a["rot"] + 4]
                op = torch.sigmoid(target[i][sel, a["opacity"] : a["opacity"] + 1])
                co = sh_dc_to_rgb(target[i][sel, a["color"] : a["color"] + 3])
                sc, rq, op, co = sc.detach(), rq.detach(), op.detach(), co.detach()
            if not grad:
                xyz, sc, rq, op, co = (x.detach() for x in (xyz, sc, rq, op, co))
            return xyz, sc, rq, op, co

        with torch.no_grad():
            ref = render_gaussians(*unpack(target[i].float(), grad=False, sel=idx), cam)
            # The real photograph, when the loader supplied one and its shape
            # matches the rasteriser's output. Only the frame's own view has a
            # photograph; the jittered views keep rendering the target Gaussians,
            # which is what they are for -- preventing view overfitting, not
            # defining the appearance target.
            if ref_image is not None:
                ri = ref_image[i]
                # The photograph is full resolution (544x977 here) while the
                # rasteriser runs at `downscale`, so it has to be resampled. An
                # equality check on shapes would have silently dropped it and left
                # the proxy reference in place with no error anywhere -- the exact
                # failure mode this file has already been bitten by several times.
                if ri.dim() == 3 and ri.shape[0] == ref.shape[0] and ri.numel() > 3:
                    if ri.shape[-2:] != ref.shape[-2:]:
                        ri = F.interpolate(ri[None].float(), size=ref.shape[-2:],
                                           mode="bilinear", align_corners=False)[0]
                    ref = ri.to(ref.dtype)
            # Random crops keep a spatial sub-region, so the camera still frames
            # the whole scene while only part of it is populated. Both sides then
            # render nearly black: L1 goes to 0 and SSIM to 1 while nothing has
            # been learned. Measured on the smoke config, 2 of 6 samples covered
            # 11% and 26% of the frame. Skipping them is right, but the count has
            # to be reported -- silently dropping a third of the steps would read
            # as "the render loss converged".
            covered = (ref.abs().sum(0) > 1e-3).float().mean()
        if float(covered) < min_coverage:
            continue

        gp = unpack(pred[i].float(), grad=True, sel=idx_p)
        gt_ = unpack(target[i].float(), grad=False, sel=idx)
        # (camera, reference) pairs. A reference of None means "rasterise the
        # target Gaussians", which is the old behaviour and carries no information
        # the target does not already have. Real photographs are what actually
        # constrain appearance, so when the loader supplies them they replace the
        # synthetic jitter entirely rather than being added alongside it.
        views_pr = [(cam, ref)]
        n_extra = 0
        if extra_cams is not None and extra_imgs is not None and extra_cams.dim() == 3:
            ec, ei = extra_cams[i], extra_imgs[i]
            for j in range(ec.shape[0]):
                if float(ec[j][0]) == 0.0:            # fx == 0 marks an empty slot
                    continue
                c_j = Camera(camera_from_vector(ec[j].detach().cpu().numpy()),
                             device=dev, downscale=int(downscale))
                im = ei[j].float()
                if im.shape[-2:] != (c_j.image_height, c_j.image_width):
                    im = F.interpolate(im[None], size=(c_j.image_height, c_j.image_width),
                                       mode="bilinear", align_corners=False)[0]
                views_pr.append((c_j, im))
                n_extra += 1
        cams = [cam]
        if views > 1:
            # Extra synthetic viewpoints. One camera cannot certify that two
            # Gaussian sets are the same 3D object -- what it cannot see is
            # unconstrained, and scale/opacity can absorb geometric error that a
            # single projection would expose. There is no second photograph, but
            # the target Gaussians can be rasterised from any pose, so this
            # becomes view equivalence rather than one-view agreement.
            rng = np.random.default_rng((seed * 1000003 + i) & 0x7FFFFFFF)
            for _ in range(views - 1):
                cams.append(Camera(camera_from_vector(perturb_camera_vector(cv, rng, view_jitter_deg)),
                                   device=dev, downscale=int(downscale)))

        if n_extra > 0:
            cams = [c for c, _ in views_pr]          # real views replace the jitter
        s_loss, s_l1, s_d, n_v = zero, zero.detach(), zero.detach(), 0
        s_p = zero.detach()
        s_sb = zero.detach()
        s_area = zero
        for k, c_k in enumerate(cams):
            with torch.no_grad():
                if n_extra > 0:
                    r_k = views_pr[k][1]
                else:
                    r_k = ref if k == 0 else render_gaussians(*gt_, c_k)
                if k > 0 and float((r_k.abs().sum(0) > 1e-3).float().mean()) < min_coverage:
                    continue
            gp_k = _detach_near_gaussians(gp, c_k, near_detach_frac)
            i_k, radii_k = render_gaussians(*gp_k, c_k, return_radii=True)
            # Screen-space size hinge. Widening the log-scale band gives the
            # attribute decoder the range the data needs, and simultaneously opens
            # the one shortcut the band existed to block: covering a hole with a
            # single huge opaque splat is always cheaper than getting the geometry
            # right. `radii` from the rasteriser is detached, so the penalty is
            # computed analytically -- projected radius ~ f * s_max / z -- which is
            # differentiable in both scale and opacity.
            #
            # A hinge against the TARGET's own projected-radius quantile in the
            # same view, not an absolute cap: it costs nothing for a splat the
            # size of ones that actually occur, and grows quadratically past it.
            # Weighted by opacity so a large transparent Gaussian (which barely
            # marks the image) is not punished like a large opaque one.
            #
            # p99 was a no-op on R8H (logged splat=0.00000 every step). The
            # scene-wide 99th percentile is the legitimate giant tail --
            # hole-covering blobs that look huge in the photo still sit under it.
            # p80 is past typical structure and before that tail.
            if w_splat > 0.0:
                def _pr(xyz_w, sc_w):
                    W = c_k.world_view_transform
                    z = (xyz_w @ W[:3, :3])[:, 2] + W[3, 2]
                    f = 0.5 * float(c_k.image_width) / max(float(c_k.tanfovx), 1e-6)
                    return f * sc_w.amax(dim=-1) / z.clamp(min=1e-3)
                q = min(max(float(splat_quantile), 0.5), 0.99)
                def _z(xyz_w):
                    W = c_k.world_view_transform
                    return (xyz_w @ W[:3, :3])[:, 2] + W[3, 2]
                with torch.no_grad():
                    z_gt = _z(gt_[0])
                    # Drop near/behind-camera points from the reference: they
                    # make p99 a no-op (r_ref explodes with them) and p80 a
                    # bomb (r_ref is sane, then the same points in pred explode
                    # the mean). R8H logged splat=0.00000 every step on p99.
                    z_min = (0.1 * z_gt.median().clamp(min=1e-2)).clamp(min=1e-2)
                    r_gt = _pr(gt_[0], gt_[1])
                    ok = r_gt[z_gt > z_min]
                    r_ref = torch.quantile(
                        (ok if ok.numel() else r_gt).float(), q
                    ).clamp(min=1e-3)
                z_p = _z(gp[0])
                r_p = _pr(gp[0], gp[1])
                # Per-point cap: one splat at z=1e-3 must not contribute 1e7.
                hinge = F.relu(r_p / r_ref - 1.0).clamp(max=4.0) ** 2
                valid = (z_p > z_min).to(hinge.dtype)
                denom = valid.sum().clamp(min=1.0)
                s_area = s_area + ((gp[3].reshape(-1) * hinge * valid).sum() / denom)
            # Which points this view can say anything about. The rasteriser
            # already computed it; accumulated across views it becomes the mask
            # the parameter anchor uses to stay off the points the render covers.
            # Scatter through `idx`, not into the first N slots: the render works
            # on the masked (and possibly subsampled) point set, so radii is
            # ordered by position within `idx` and a positional write would mark
            # entirely different Gaussians as visible.
            vis[i].index_put_((idx_p,), (radii_k.detach() > 0).to(vis.dtype), accumulate=True)
            l_k, l1_k, d_k = photometric_loss(i_k, r_k, lam_dssim, edge_gain=edge_gain,
                                              w_sobel=w_sobel)
            s_sb = s_sb + (_sobel(i_k) - _sobel(r_k.detach())).abs().mean().detach()
            if w_perc > 0:
                # Added to the same per-view term so it shares the coverage gating
                # and the view averaging; both would otherwise be silently wrong.
                p_k = vgg_perceptual(i_k, r_k)
                l_k = l_k + w_perc * p_k
                s_p = s_p + p_k.detach()
            s_loss, s_l1, s_d, n_v = s_loss + l_k, s_l1 + l1_k.detach(), s_d + d_k.detach(), n_v + 1
        if n_v == 0:
            continue
        tot = tot + s_loss / n_v
        out["splat_area"] = out.get("splat_area", zero) + s_area / n_v
        out["render_l1"] = out["render_l1"] + s_l1 / n_v
        out["render_dssim"] = out["render_dssim"] + s_d / n_v
        out["render_perc"] = out.get("render_perc", zero.detach()) + s_p / n_v
        out["render_sobel"] = out["render_sobel"] + s_sb / n_v
        n_ok += 1

    out["visible"] = (vis > 0)
    out["render_covered"] = (vis > 0).float().mean().detach()
    out["render_used"] = out["render_used"] + float(n_ok) / max(b, 1)
    if n_ok:
        out["render"] = tot / n_ok
        out["splat_area"] = out["splat_area"] / n_ok
        out["render_l1"] = out["render_l1"] / n_ok
        out["render_dssim"] = out["render_dssim"] / n_ok
        out["render_sobel"] = out["render_sobel"] / n_ok
        if "render_perc" in out:
            out["render_perc"] = out["render_perc"] / n_ok
    return out


# ---------------------------------------------------------------------------
# branch losses
# ---------------------------------------------------------------------------


@torch.no_grad()
def responsibility_target(pred_xyz, target, mask, layout, group_size: int):
    """Re-aggregate the target onto the slots the prediction actually occupies.

    Slot *i* of the prediction has no reason to correspond to slot *i* of the
    target -- measured, the slot partner is the nearest GT point only 7.0% of the
    time, and the two disagree by 58-80% of each attribute's own standard
    deviation. Supervising slot-to-slot therefore teaches the colour and opacity
    of some other Gaussian for 93% of the points, and it renders 1.8 dB worse than
    the target built here.

    What a prediction *should* look like is decided by the ground it covers: every
    GT point picks its closest prediction, and that prediction inherits the set it
    won. This is also the answer to standing in for more Gaussians than you are --
    which is the normal case here, since only 37% of predictions win any territory
    at all and those win 2.7 GT points each:

      scale     must span the ground the owned points covered, so the union extent
                is a floor under their mean -- averaging their scales leaves holes
      opacity   composites as 1 - prod(1 - a) rather than averaging, because that
                is what the rasteriser will do to them
      colour    the mean of the owned points
      rotation  the nearest point's, since quaternions do not average

    Predictions that win nothing fall back to their nearest GT point. Recomputed
    each step from the current positions, and detached: this is a target, not a
    path back into the geometry decoder.
    """
    b, n, _ = pred_xyz.shape
    g = int(group_size)
    ng = n // g
    if ng == 0:
        return target
    n_used = ng * g
    p = pred_xyz[:, :n_used].reshape(b * ng, g, 3).float()
    t = target[:, :n_used].reshape(b * ng, g, -1).float()
    v = mask[:, :n_used].reshape(b * ng, g) > 0.5

    d = torch.cdist(p, t[..., 0:3])
    d = d.masked_fill(~v.unsqueeze(1), float("inf"))        # never own an empty slot
    near = d.argmin(2)
    res = torch.gather(t, 1, near.unsqueeze(-1).expand(-1, -1, t.shape[-1])).clone()

    own = torch.zeros(b * ng, g, g, device=p.device, dtype=p.dtype)
    own.scatter_(1, d.argmin(1).unsqueeze(1), 1.0)
    own = own * v.unsqueeze(1).to(own.dtype)
    cnt = own.sum(2, keepdim=True)
    w = own / cnt.clamp(min=1.0)

    sl, op, co, sh = layout["scale"], layout["opacity"], layout["color"], layout["sh"]
    agg = torch.bmm(w, t)
    res[..., sl : sl + 3] = agg[..., sl : sl + 3]
    if t.shape[-1] > co:
        res[..., co:] = agg[..., co:]

    # Opacity is averaged along with everything else, NOT composited. An earlier
    # version used 1 - prod(1 - a) on the reasoning that this is what alpha does.
    # It is -- but only for Gaussians stacked along a view ray, and a Voronoi cell
    # collects points lying side by side on a surface. Those tile the image, cover
    # different pixels, and never multiply their alphas; the ground they cover is
    # already accounted for by the union-extent floor on scale below.
    #
    # Measured both ways on the same predicted positions, held-out 3 views:
    # composited 16.91 dB / SSIM 0.486, averaged 17.40 dB / SSIM 0.505. It was
    # also unstable to train against -- compositing saturates towards alpha 1 while
    # the geometry is still coarse, so the anchor demanded an opacity that occurs
    # nowhere in the data (the channel spans -2.89..-1.54, alpha 0.05..0.18) and
    # the measured error went from 1.32 to 3.61 standard deviations in 500 steps.
    res[..., op : op + 1] = agg[..., op : op + 1]

    ctr = torch.bmm(w, t[..., 0:3])
    spread = (torch.bmm(w, t[..., 0:3].pow(2)) - ctr.pow(2)).clamp(min=0).sqrt().mean(-1, keepdim=True)
    grown = torch.maximum(res[..., sl : sl + 3], torch.log(spread.clamp(min=1e-6)))
    # Never ask for a Gaussian bigger than the biggest one in the group, for the
    # same reason the opacity composite is capped: a prediction that has drifted
    # across the group owns points whose union extent is the whole group, and the
    # anchor would demand a splat that does not occur anywhere in the data.
    hi = t[..., sl : sl + 3].amax(dim=1, keepdim=True)
    res[..., sl : sl + 3] = torch.minimum(grown, hi)

    orphan = (cnt.squeeze(-1) < 0.5).unsqueeze(-1).expand_as(res)
    nearest = torch.gather(t, 1, near.unsqueeze(-1).expand(-1, -1, t.shape[-1]))
    res = torch.where(orphan, nearest, res)

    full = target.clone()
    full[:, :n_used] = res.reshape(b, n_used, -1).to(target.dtype)
    return full


def _quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) w-first quaternion -> (..., 3, 3). Sign-agnostic by construction."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def covariance3d_loss(pred, target, mask, layout, weights, log_space: bool = True):
    """Supervise the 3D covariance R diag(s^2) R^T instead of scale and rotation apart.

    Rendering only ever sees the covariance. Supervising `scale` and `rot`
    separately picks ONE (R, s) factorisation out of the many that produce the
    same ellipsoid -- the axis permutations and sign flips are all equivalent --
    so the rotation term spends its gradient on a choice the image cannot observe.
    Measured symptom: attr_rot_nrmse sits at 1.089, i.e. no better than emitting
    the dataset mean, while the render keeps improving.

    Compared in log-Frobenius form by default: covariance entries span
    exp(-16)..exp(0), so a plain squared difference is dominated entirely by the
    largest Gaussians. Normalising each pair by the target's own Frobenius norm
    makes the term scale-free, which is what lets a single weight cover both the
    fine structures and the background splats.
    """
    a = layout
    sl, rq = a["scale"], a["rot"]
    sp = torch.exp(pred[..., sl : sl + 3].clamp(-15.0, 5.0))
    st = torch.exp(target[..., sl : sl + 3].clamp(-15.0, 5.0))
    Rp = _quat_to_R(pred[..., rq : rq + 4])
    Rt = _quat_to_R(target[..., rq : rq + 4])
    Cp = torch.einsum("...ij,...j,...kj->...ik", Rp, sp * sp, Rp)
    Ct = torch.einsum("...ij,...j,...kj->...ik", Rt, st * st, Rt)
    if log_space:
        # Scale-free: divide both by the target's Frobenius norm before comparing.
        #
        # Floored against the batch, not at 1e-20. A needle Gaussian whose target
        # covariance norm underflows turns `Cp / n` into ~1e20 and the squared
        # difference into ~1e40, which is finite in fp32 and therefore passes every
        # isfinite guard while owning the entire step. 1e-20 is also below bf16's
        # smallest normal (~1e-38 in fp32 terms, but the einsum runs in the autocast
        # dtype), so the clamp was not even reliably reached.
        nrm = Ct.flatten(-2).norm(dim=-1)
        n_floor = 0.01 * nrm.detach()[mask > 0.5].median().clamp(min=1e-12) if bool(
            (mask > 0.5).any()) else torch.as_tensor(1e-12, device=nrm.device, dtype=nrm.dtype)
        n = nrm.clamp(min=n_floor)[..., None, None]
        d = ((Cp / n - Ct / n) ** 2).flatten(-2).sum(-1)
    else:
        d = ((Cp - Ct) ** 2).flatten(-2).sum(-1)
    return masked_reduce(d, mask)


def aniso_shape_loss(pred, target, mask, layout, beta: float = 0.1):
    """A Gaussian's shape with its size and its orientation both divided out.

    Why this exists as its own term. `covariance3d_loss` was supposed to teach
    scale and rotation together, and measured on T16k step 70000 it teaches only
    scale: handing the model the ground-truth ROTATION changes it by 0.0% while
    handing it the ground-truth SCALE removes 99.1%. The predicted covariance is
    off by 7.7x the target's own norm, so the size mismatch swamps orientation.

    Removing size from that loss does not fix it either -- trace-normalised, the
    rotation sensitivity is 1.6% -- because the failure is in the prediction, not
    the metric: `R S S^T R^T` loses `R` entirely when `S` is spherical, and the
    model emits near-spheres (anisotropy p50 3.73 against the ground truth's
    15.51, and 52.4% of predictions under 4x where the data has 13.2%). Rotation
    is not a learnable quantity until the ellipsoids are actually elongated, so
    anisotropy has to be driven by a term that still has a gradient on a sphere.

    Sorted descending so the axis permutation that makes (R, s) non-unique cannot
    change the value -- that ambiguity is why supervising `rot` directly was
    dropped -- and mean-removed in log space so overall size stays the covariance
    term's job and this one is purely "how elongated, in what proportions".
    """
    sl = layout["scale"]
    lp = pred[..., sl : sl + 3].clamp(-15.0, 5.0).sort(dim=-1, descending=True).values
    lt = target[..., sl : sl + 3].clamp(-15.0, 5.0).sort(dim=-1, descending=True).values
    lp = lp - lp.mean(dim=-1, keepdim=True)
    lt = lt - lt.mean(dim=-1, keepdim=True)
    return masked_smooth_l1(lp, lt, mask, beta=beta)


def _attr_losses(pred, target, mask, layout, weights, sh_dim: int) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    sl, rq, op, co, sh = layout["scale"], layout["rot"], layout["opacity"], layout["color"], layout["sh"]
    if pred.shape[-1] > 3:
        out["scale"] = masked_smooth_l1(pred[..., sl : sl + 3], target[..., sl : sl + 3], mask, beta=0.1)
        out["aniso"] = aniso_shape_loss(pred, target, mask, layout)
        out["rot"] = quat_loss(pred[..., rq : rq + 4], target[..., rq : rq + 4], mask)
        out["opacity"] = masked_smooth_l1(pred[..., op : op + 1], target[..., op : op + 1], mask, beta=0.1)
    if pred.shape[-1] > 11:
        out["color"] = masked_smooth_l1(pred[..., co : co + 3], target[..., co : co + 3], mask, beta=0.1)
    if sh_dim > 0 and pred.shape[-1] >= sh + sh_dim:
        out["sh"] = masked_smooth_l1(pred[..., sh : sh + sh_dim], target[..., sh : sh + sh_dim], mask, beta=0.1)
    return out


def branch_geometry_loss(
    pred: torch.Tensor,
    presence: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: Dict[str, float],
    prefix: str,
    layout: Dict[str, int],
    sh_dim: int,
    group_size: int,
    balance_weight: Optional[torch.Tensor] = None,
    with_attrs: bool = False,
    group_scale: Optional[torch.Tensor] = None,
):
    """Geometry (+optional attribute) terms for one decoder branch."""
    pxyz = pred[..., 0:3]
    txyz = target[..., 0:3]
    local_scale = group_scale
    if bool(weights.get("geometry_target_scale", False)):
        local_scale = target_group_extent(txyz, mask, group_size)
    center_local = bool(weights.get("geometry_center_local", False))
    parts: Dict[str, torch.Tensor] = {}
    total = pred.new_tensor(0.0)

    def add(key: str, value: torch.Tensor, w_key: str):
        nonlocal total
        w = float(weights.get(w_key, 0.0))
        parts[key] = value.detach()
        if w != 0.0:
            total = total + w * value

    if float(weights.get(f"w_{prefix}xyz", 0.0)) != 0.0:
        add("xyz", masked_smooth_l1(pxyz, txyz, mask, beta=float(weights.get("xyz_beta", 0.02)), weight=balance_weight), f"w_{prefix}xyz")
    if float(weights.get(f"w_{prefix}xyz_mse", 0.0)) != 0.0:
        add("xyz_mse", masked_mse(pxyz, txyz, mask, weight=balance_weight), f"w_{prefix}xyz_mse")
    if float(weights.get(f"w_{prefix}xyz_hard", 0.0)) != 0.0:
        add(
            "xyz_hard",
            masked_hard_smooth_l1(
                pxyz, txyz, mask,
                frac=float(weights.get("hard_frac", 0.08)),
                min_points=int(weights.get("hard_min_points", 2048)),
                beta=float(weights.get("xyz_beta", 0.02)),
            ),
            f"w_{prefix}xyz_hard",
        )
    if float(weights.get(f"w_{prefix}chamfer", 0.0)) != 0.0:
        add(
            "chamfer",
            multiscale_chamfer(
                pxyz, txyz, mask,
                scales=weights.get("chamfer_scales", (2048, 8192, 32768)),
                scale_weights=weights.get("chamfer_scale_weights", None),
                balanced=bool(weights.get("balanced_chamfer", True)),
                # Prefix-packed target slots define the deployable point count.
                # Treating padding as real predictions was inconsistent with
                # evaluation/rendering and rewarded duplicates in empty slots.
                pred_mask=mask,
            ),
            f"w_{prefix}chamfer",
        )
    # GT→pred 구멍 패널티 (set chamfer 와 별도로 onesided 강화)
    if float(weights.get(f"w_{prefix}coverage", 0.0)) != 0.0:
        add(
            "coverage",
            coverage_onesided_loss(
                pxyz, txyz, mask,
                samples=int(weights.get("coverage_samples", 16384)),
                bins=int(weights.get("spatial_bins", 32)),
                pred_mask=mask,
            ),
            f"w_{prefix}coverage",
        )
    # soft voxel occupancy (미분 가능; hard binning 금지)
    if float(weights.get(f"w_{prefix}voxel_occ", 0.0)) != 0.0:
        add(
            "voxel_occ",
            soft_voxel_occupancy_loss(
                pxyz, txyz, mask,
                bins=int(weights.get("voxel_occ_bins", 24)),
                samples=int(weights.get("coverage_samples", 16384)),
                pred_mask=mask,
            ),
            f"w_{prefix}voxel_occ",
        )
    if float(weights.get(f"w_{prefix}plane_chamfer", 0.0)) != 0.0:
        add(
            "plane_chamfer",
            projected_chamfer(
                pxyz, txyz, mask,
                samples=int(weights.get("plane_chamfer_samples", 16384)),
                pred_mask=mask,
            ),
            f"w_{prefix}plane_chamfer",
        )
    if float(weights.get(f"w_{prefix}proj_hist", 0.0)) != 0.0:
        add(
            "proj_hist",
            projected_histogram_loss(
                pxyz, txyz, mask,
                samples=int(weights.get("proj_hist_samples", 8192)),
                bins=int(weights.get("proj_hist_bins", 64)),
                sigma=float(weights.get("proj_hist_sigma", 0.025)),
                scale=float(weights.get("proj_hist_scale", 100.0)),
                pred_mask=mask,
            ),
            f"w_{prefix}proj_hist",
        )
    # Permutation-invariant within-group terms. These are the ones that survive a
    # slot order that is not a function of the geometry.
    # intra_chamfer and its two halves come from one call. p2g / radius were only
    # ever wired for the gen branch, and the codec branch is the one that ships --
    # so the symmetric average was the only thing supervising within-group
    # geometry here, and it cannot see either of the two defects that were
    # measured. Predicted group radius is 0.67x the ground truth's, i.e. the group
    # is shrunk toward its own centroid, and the median predicted point sits 0.315
    # group radii off the GT surface. A symmetric mean of precision and coverage
    # is stationary under trades between them and reports neither.
    _w_ich = float(weights.get(f"w_{prefix}intra_chamfer", 0.0))
    _w_p2g = float(weights.get(f"w_{prefix}p2g", 0.0))
    _w_rad = float(weights.get(f"w_{prefix}radius", 0.0))
    if _w_ich != 0.0 or _w_p2g != 0.0 or _w_rad != 0.0:
        _ich, _parts = intra_group_chamfer(
            pxyz, txyz, mask, group_size, scale=local_scale,
            chunk_groups=int(weights.get("intra_chamfer_chunk", 1024)),
            center=center_local, return_parts=True,
        )
        add("intra_chamfer", _ich, f"w_{prefix}intra_chamfer")
        # Precision: pull each predicted point onto the GT surface. This is the
        # direction a spacing hinge cannot express and a symmetric chamfer averages
        # away.
        add("p2g", _parts["p2g"], f"w_{prefix}p2g")
        parts["g2p"] = _parts["g2p"].detach()
        # Radius: make the group span what the GT group spans. Without it the
        # spacing hinge is ambiguous -- it can be satisfied by expanding the group
        # (wanted) or by pushing apart the near-coincident pairs the GT actually
        # uses (not wanted, GT median within-group NN distance is 0.007 group
        # radii). This term fixes the direction.
        add("radius", _parts["radius"], f"w_{prefix}radius")
    if float(weights.get(f"w_{prefix}intra_sinkhorn", 0.0)) != 0.0:
        add(
            "intra_sinkhorn",
            intra_group_sinkhorn_loss(
                pxyz, txyz, mask, group_size, scale=local_scale,
                epsilon=float(weights.get("sinkhorn_epsilon", 0.08)),
                iterations=int(weights.get("sinkhorn_iterations", 6)),
                chunk_groups=int(weights.get("sinkhorn_chunk", 256)),
                center=center_local,
            ),
            f"w_{prefix}intra_sinkhorn",
        )
    if float(weights.get(f"w_{prefix}intra_spacing", 0.0)) != 0.0:
        add(
            "intra_spacing",
            intra_group_spacing_loss(
                pxyz, txyz, mask, group_size, scale=local_scale,
                chunk_groups=int(weights.get("intra_chamfer_chunk", 1024)),
                ratio=float(weights.get("intra_spacing_ratio", 1.0)),
            ),
            f"w_{prefix}intra_spacing",
        )
    if float(weights.get(f"w_{prefix}group_centroid", 0.0)) != 0.0:
        add(
            "group_centroid",
            group_centroid_loss(
                pxyz, txyz, mask, group_size,
                scale=None if bool(weights.get("geometry_centroid_absolute", False)) else local_scale,
            ),
            f"w_{prefix}group_centroid",
        )
    if float(weights.get(f"w_{prefix}dispersion", 0.0)) != 0.0:
        add(
            "dispersion",
            patch_dispersion_loss(
                pxyz, mask,
                group_size=int(weights.get("dispersion_group_size", group_size)),
                margin=float(weights.get("dispersion_margin", 0.001)),
                target_xyz=txyz,
                margin_frac=float(weights.get("dispersion_margin_frac", 0.35)),
            ),
            f"w_{prefix}dispersion",
        )
    if float(weights.get(f"w_{prefix}presence", 0.0)) != 0.0:
        add("presence", balanced_presence_loss(presence, mask), f"w_{prefix}presence")
    if with_attrs and pred.shape[-1] > 3:
        for key, value in _attr_losses(pred, target, mask, layout, weights, sh_dim).items():
            add(key, value, f"w_{key}")
    return total, parts


def total_loss(
    out: Dict[str, torch.Tensor],
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: Dict[str, float],
    layout: Dict[str, int],
    model_layout: Dict[str, int],
    sh_dim: int,
    with_attrs: bool = False,
    run_decode: bool = True,
    run_gen: bool = True,
):
    # The pack's aux block is a per-point mask only while appearance is off. Once
    # `geom_pack_dim` is set, that block holds a learned appearance code and every
    # consumer must stop reading it as occupancy.
    geom_pack_dim = int(model_layout.get("geom_pack_dim", 0))
    packed_mask = geom_pack_dim <= 0
    # log values stay as detached tensors; converting them here would force a
    # host sync on every step
    logs: Dict[str, torch.Tensor] = {}
    total = out["z_raw"].new_tensor(0.0)

    balance = None
    if float(weights.get("spatial_weight_power", 0.0)) > 0:
        balance = spatial_balance_weights(
            target[..., 0:3], mask,
            bins=int(weights.get("spatial_bins", 32)),
            power=float(weights.get("spatial_weight_power", 0.5)),
            max_weight=float(weights.get("spatial_weight_max", 8.0)),
        )

    zl = z_raw_losses(
        out["z_raw_hat"], out["z_raw"].detach() if weights.get("z_target_detach", True) else out["z_raw"],
        out["group_valid"], model_layout["num_groups"], model_layout["tokens_per_group"],
        geom_dim=int(model_layout.get("geom_pack_dim", 0)),
        hard_frac=float(weights.get("z_hard_frac", 0.12)),
        hard_min_tokens=int(weights.get("z_hard_min_tokens", 256)),
        std_target=float(weights.get("z_std_ratio_target", 1.0)),
        supervise_empty=bool(weights.get("supervise_empty_groups", True)),
    )
    for key, w_key in (
        ("z_l1", "w_z_raw"),
        ("z_mse", "w_z_raw_mse"),
        ("z_hard", "w_z_hard"),
        ("z_token_var", "w_z_token_var"),
        ("z_std_ratio", "w_z_std_ratio"),
    ):
        w = float(weights.get(w_key, 0.0))
        logs[key] = zl[key].detach()
        if w != 0.0:
            total = total + w * zl[key]

    if "n_used" in out:
        logs["n_used"] = out["n_used"].float().mean().detach()
        logs["empty_frac"] = out["empty_frac"].float().mean().detach()

    if "res_ratio" in out:
        logs["res_ratio"] = out["res_ratio"].detach()
        # Floor on the learned/shortcut ratio. Without it the compressor can keep
        # shape channels "alive" (unit std) while the decompressor ignores them
        # and reconstructs every point at the group centroid.
        floor = float(weights.get("res_ratio_floor", 0.0))
        w_rr = float(weights.get("w_res_ratio", 0.0))
        if w_rr > 0.0 and floor > 0.0:
            rr_pen = F.relu(floor - out["res_ratio"])
            total = total + w_rr * rr_pen
            logs["res_ratio_pen"] = rr_pen.detach()

    # Within-group residual on the z_raw path: the signal that was previously
    # drowned out by the centroid/mask terms of plain z_l1.
    w_zr = float(weights.get("w_z_residual", 0.0))
    w_zrh = float(weights.get("w_z_residual_hard", 0.0))
    if w_zr > 0.0 or w_zrh > 0.0:
        zr = group_residual_z_loss(
            out["z_raw_hat"],
            out["z_raw"].detach() if weights.get("z_target_detach", True) else out["z_raw"],
            out["group_valid"],
            model_layout["num_groups"],
            model_layout["tokens_per_group"],
            model_layout["group_size"],
            scale=out.get("decoded_scale"),
            hard_frac=float(weights.get("residual_hard_frac", 0.2)),
            packed_mask=packed_mask,
        )
        logs["z_residual"] = zr["z_residual"].detach()
        logs["z_residual_hard"] = zr["z_residual_hard"].detach()
        if w_zr > 0.0:
            total = total + w_zr * zr["z_residual"]
        if w_zrh > 0.0:
            total = total + w_zrh * zr["z_residual_hard"]

    # Permutation-invariant twin of the above, on the *packed* reconstruction so it
    # does not need the decoder to be running.
    #
    # ``w_intra_chamfer`` acts on the decoder output, so it is necessarily zero until
    # decode switches on at ``latent_end - late_decode_steps``. That leaves
    # ``w_z_residual`` -- slot-ordered and L1/L2 -- as the *only* geometry term for
    # the whole latent phase, and its optimum on a high-entropy target is the
    # conditional mean: measured, the folding decoder went from an exact ball
    # (aniso_lam2 0.959) to aniso_lam2 0.082 against 0.579 in the GT between steps
    # 500 and 1000, i.e. every group collapsed to a line while the anti-collapse
    # term was still scheduled off. Under residual_pack the xyz slice of z_raw_hat
    # *is* the predicted offsets, so the set distance is available from step 0.
    w_zic = float(weights.get("w_z_intra_chamfer", 0.0))
    if w_zic > 0.0:
        gsz = int(model_layout["group_size"])
        ng = int(model_layout["num_groups"])
        tpg = int(model_layout["tokens_per_group"])
        p_xyz, _ = tokens_to_xyz_mask(out["z_raw_hat"], ng, tpg, gsz)
        t_src = out["z_raw"].detach() if weights.get("z_target_detach", True) else out["z_raw"]
        t_xyz, _ = tokens_to_xyz_mask(t_src, ng, tpg, gsz)
        gv = out["group_valid"][:, :ng]
        b = p_xyz.shape[0]
        zic = intra_group_chamfer(
            p_xyz.reshape(b, ng * gsz, 3),
            t_xyz.reshape(b, ng * gsz, 3),
            gv[:, :, None].expand(-1, -1, gsz).reshape(b, ng * gsz),
            gsz,
            scale=out.get("decoded_scale"),
            chunk_groups=int(weights.get("intra_chamfer_chunk", 1024)),
        )
        logs["z_intra_chamfer"] = zic.detach()
        total = total + w_zic * zic

    # Direct match of the decompressor's learned xyz residual against the pack
    # residual. This backprops into shape channels even when the decoder is off.
    #
    # ``w_shape_direct`` supervises the shape->offset MLP *on its own*. Without it
    # the deep path (group token + window attention over neighbouring centroids)
    # can serve the whole residual from spatial context, the shape channels get no
    # gradient and collapse to ~0 std, and every group is then decoded from the
    # same near-constant code -- which is exactly the repeated-template artifact.
    w_learn = float(weights.get("w_learned_residual", 0.0))
    w_direct = float(weights.get("w_shape_direct", 0.0))
    if (w_learn > 0.0 and "learned_xyz" in out) or (w_direct > 0.0 and "direct_xyz" in out):
        gsz = model_layout["group_size"]
        ng = model_layout["num_groups"]
        tpg = model_layout["tokens_per_group"]
        xyz_t, m = tokens_to_xyz_mask(out["z_raw"].detach(), ng, tpg, gsz,
                                      None if packed_mask else out["group_valid"])
        if bool(weights.get("residual_pack", out.get("residual_pack", False))):
            # Pack already stores (xyz - centroid); do not subtract again.
            tgt_pts = xyz_t
        else:
            cen = (xyz_t * m.unsqueeze(-1)).sum(dim=2) / m.sum(dim=2, keepdim=True).clamp(min=1.0)
            tgt_pts = xyz_t - cen.unsqueeze(2)
        gv = out["group_valid"][:, :ng].to(tgt_pts.dtype)
        # Same extent normalisation as ``group_residual_z_loss``. Without it this
        # term is a raw world-unit L1 (~1e-3) against an extent-normalised
        # ``z_residual`` (~1e-1), i.e. ~400x smaller, so it never actually steered
        # anything regardless of its weight.
        ext = (xyz_t.amax(dim=2) - xyz_t.amin(dim=2)).amax(dim=-1, keepdim=True).clamp(min=1e-3)
        if "decoded_scale" in out:
            s = out["decoded_scale"][:, :ng].to(tgt_pts.dtype).detach()
            if s.dim() == 2:
                s = s.unsqueeze(-1)
            ext = torch.maximum(ext, s.clamp(min=1e-3))
        inv_ext = (1.0 / ext).unsqueeze(-1)

        def _res_l1(key: str) -> torch.Tensor:
            pred_pts = out[key].reshape(m.shape[0], ng, gsz, 3)
            per_pt = ((pred_pts - tgt_pts) * inv_ext).abs().mean(dim=-1) * m
            per_group = per_pt.sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
            return masked_reduce(per_group, gv)

        if w_learn > 0.0 and "learned_xyz" in out:
            learn_l1 = _res_l1("learned_xyz")
            total = total + w_learn * learn_l1
            logs["learned_residual"] = learn_l1.detach()
        if w_direct > 0.0 and "direct_xyz" in out:
            direct_l1 = _res_l1("direct_xyz")
            total = total + w_direct * direct_l1
            logs["shape_direct"] = direct_l1.detach()
    if "direct_frac" in out:
        logs["direct_frac"] = out["direct_frac"].detach()
    if "direct_ratio" in out:
        logs["direct_ratio"] = out["direct_ratio"].detach()
    if "joint_xyz_delta_abs" in out:
        logs["joint_dxyz"] = out["joint_xyz_delta_abs"].detach()

    # Fixed-anchor targets are distance ordered, not folding-template ordered.
    # A slot-wise shape_direct L1 is therefore ill posed in this layout. This
    # set-valued twin directly trains shape->offset without assuming a
    # permutation and without letting decoder refinement hide a collapsed code.
    w_shape_sh = float(weights.get("w_shape_sinkhorn", 0.0))
    if w_shape_sh > 0.0 and "direct_xyz" in out:
        gsz = int(model_layout["group_size"])
        ng = int(model_layout["num_groups"])
        n = ng * gsz
        tp = target[:, :n, :3]
        tm = mask[:, :n]
        direct = out["direct_xyz"].reshape(tp.shape[0], n, 3)
        shape_sh = intra_group_sinkhorn_loss(
            direct, tp, tm, gsz, scale=out.get("decoded_scale"),
            epsilon=float(weights.get("sinkhorn_epsilon", 0.08)),
            iterations=int(weights.get("sinkhorn_iterations", 6)),
            chunk_groups=int(weights.get("sinkhorn_chunk", 256)),
            center=True,
        )
        total = total + w_shape_sh * shape_sh
        logs["shape_sinkhorn"] = shape_sh.detach()

    if float(weights.get("w_kl", 0.0)) != 0.0:
        total = total + float(weights["w_kl"]) * out["kl"]
    logs["kl"] = out["kl"].detach()

    # Decorrelate the shape channels across groups.
    #
    # ``w_latent_std`` is a *per-channel* statistic and is therefore blind to
    # correlation: measured, every checkpoint has healthy per-channel std (0.25 to
    # 0.57) while the shape block's effective rank sits at ~4 no matter how wide
    # the budget is or how long it trains --
    #
    #     _fix @10000   12 channels   effective rank 4.38   (36.5% of budget)
    #     fix2 @2000    28 channels   effective rank 4.33   (15.5%)
    #     folding @1000 28 channels   effective rank 2.61   ( 9.3%)
    #
    # so widening 12 -> 28 bought nothing. A rank-4 code caps the reachable
    # within-group chamfer near 0.34 whatever the decoder does, which is exactly
    # where the best run of this codebase stopped (0.345). Penalising the
    # off-diagonal of the channel correlation matrix is the VICReg / Barlow Twins
    # covariance term and attacks the rank directly.
    # Teacher cycle: decode the *true* pack with the same decoder and require the
    # bottleneck's decode to match it. z_l1 / z_residual only ask that z_raw_hat
    # resemble the pack; this asks the stronger question -- does z_compact retain
    # what the teacher decoder actually needs? Cheap, and diagnostic in a way the
    # pack-space losses are not.
    w_cyc = float(weights.get("w_teacher_cycle", 0.0))
    if w_cyc != 0.0 and "pred_from_true_pack" in out and run_decode:
        # `pred_cycle` is the frozen-parameter decode of z_raw_hat, so this
        # gradient reaches the compressor/decompressor only. Falling back to
        # `pred` would let the decoder satisfy the term by going deaf to its
        # input; that path is kept only for cfg.teacher_cycle_freeze = False,
        # which exists to A/B the freeze rather than as a default.
        src = out.get("pred_cycle", out["pred"])
        cyc = masked_smooth_l1(
            src[..., 0:3], out["pred_from_true_pack"][..., 0:3].detach(), mask, beta=0.02
        )
        total = total + w_cyc * cyc
        logs["teacher_cycle"] = cyc.detach()

    w_dec = float(weights.get("w_latent_decorr", 0.0))
    if w_dec != 0.0 or "z_compact" in out:
        z = out["z_compact"]
        per_g = int(model_layout["per_group"])
        c0 = int(model_layout["c_centroid"]) + int(model_layout["c_occupancy"])
        cells = z.flatten(2).transpose(1, 2)                       # (B, cells, C)
        b, n_cells, C = cells.shape
        merge = max(1, C // per_g)
        # Shape block only. This used to be `[..., c0:]`, i.e. shape AND appearance,
        # so a 3-channel geometry code was being decorrelated against an 8-channel
        # colour code. Nothing about the two being independent is desirable -- a
        # group's colour and its geometry are correlated in the data -- and with
        # w_latent_decorr=5.0 the penalty comfortably outweighed reconstruction on
        # the 3 channels that carry all the intra-group shape.
        c1 = c0 + int(model_layout.get("c_shape", per_g - c0))
        g = cells.reshape(b, n_cells * merge, per_g)[..., c0:c1]     # (B, slots, c_shape)
        gv = out["group_valid"][:, : g.shape[1]].reshape(-1) > 0.5
        f = g.reshape(-1, g.shape[-1])[gv]
        if f.shape[0] > f.shape[1] + 1:
            f = f - f.mean(0, keepdim=True)
            # Batch-relative floor. At 1e-4 a channel that has collapsed gets its
            # residual noise amplified 1e4x, the correlation matrix fills with
            # ~1e8 entries and `dec` -- weighted 5.0 and absent from the printed
            # breakdown -- becomes one of the spike terms. A fraction of the
            # largest channel std keeps a dead channel contributing ~0 instead.
            sd = f.std(0, keepdim=True)
            f = f / sd.clamp(min=0.01 * sd.detach().max().clamp(min=1e-6))
            corr = (f.T @ f) / float(f.shape[0] - 1)
            d = corr.shape[0]
            off = corr - torch.diag_embed(torch.diagonal(corr))
            dec = (off ** 2).sum() / float(max(d * (d - 1), 1))
            if w_dec != 0.0:
                total = total + w_dec * dec
            logs["latent_decorr"] = dec.detach()
            with torch.no_grad():
                ev = torch.linalg.eigvalsh(corr.float()).clamp(min=0)
                p = ev / ev.sum().clamp(min=1e-20)
                p = p[p > 1e-12]
                logs["latent_erank"] = torch.exp(-(p * p.log()).sum())

    if float(weights.get("w_latent_std", 0.0)) != 0.0:
        # Only the learned shape channels are pushed towards unit variance: the
        # centroid and occupancy channels carry physical quantities and must keep
        # their natural scale. Roughly isotropic channels make the latent easier
        # for a diffusion model to fit (Hunyuan3D normalises its latent the same
        # way before handing it to the DiT).
        z = out["z_compact"]
        chan_std = z.flatten(2).std(dim=-1).mean(dim=0)
        shape_mask = shape_channel_mask(
            chan_std.shape[0], model_layout["per_group"], model_layout["c_centroid"], model_layout["c_occupancy"],
            c_shape=model_layout.get("c_shape"),
        ).to(chan_std.device)
        if float(shape_mask.sum()) > 0:
            # Two-sided band. The floor stops the channels collapsing; the ceiling
            # stops the compressor inflating its output while the decompressor
            # divides it back out again, which costs nothing to train but leaves
            # the latent scale drifting and non-stationary -- bad for a diffusion
            # prior that has to model this distribution.
            floor = float(weights.get("latent_std_floor", 0.25))
            ceil = float(weights.get("latent_std_ceil", 0.0))
            band = F.relu(floor - chan_std)
            if ceil > floor:
                band = band + F.relu(chan_std - ceil)
            lat = (band * shape_mask).sum() / shape_mask.sum()
            total = total + float(weights["w_latent_std"]) * lat
        logs["latent_std"] = chan_std.mean().detach()
        logs["latent_shape_std"] = (chan_std * shape_mask).sum().detach() / shape_mask.sum().clamp(min=1)
        logs["latent_std_max"] = (chan_std * shape_mask).max().detach()

    # Attribute parameter losses on the separate decoder's output. Kept apart from
    # branch_geometry_loss so the geometry branch keeps supervising the geometry
    # decoder's own xyz and nothing else -- the two modules never share a term.
    if run_decode and with_attrs and "attr_pred" in out:
        ap = out["attr_pred"]
        gsz = model_layout["group_size"]
        # Down-weight the anchor exactly where the render already has a say.
        # On a covered point the two disagree and the anchor is the worse of the
        # two by a measured 10 dB (16.74 vs 26.77 on the same positions), so
        # letting it pull there only drags the point back towards its group mean.
        # Off the covered set the render gives literally nothing, and the anchor
        # is 100% of the gradient however small its global share.
        # `attr_anchor_covered` is the fraction kept on covered points.
        a_pw = None
        vis = out.get("render_visible")
        if vis is not None:
            keep = float(weights.get("attr_anchor_covered", 1.0))
            a_pw = mask * torch.where(vis[:, : mask.shape[1]],
                                      mask.new_full((), keep), mask.new_ones(()))
        mode = str(weights.get("attr_match_mode", "responsibility"))
        if mode == "sinkhorn":
            _aw = float(weights.get("sinkhorn_attr_weight", 0.0))
            a_tgt = sinkhorn_parameter_target(
                ap[..., 0:3].detach(), target, mask, gsz,
                scale=out.get("decoded_scale"),
                epsilon=float(weights.get("sinkhorn_epsilon", 0.08)),
                iterations=int(weights.get("sinkhorn_iterations", 6)),
                chunk_groups=int(weights.get("sinkhorn_chunk", 256)),
                pred_attr=(ap.detach() if _aw > 0.0 else None),
                attr_weight=_aw, layout=layout,
            )
        elif mode == "responsibility" and float(weights.get("attr_responsibility", 1.0)) != 0.0:
            a_tgt = responsibility_target(ap[..., 0:3].detach(), target, mask, layout, gsz)
        else:
            a_tgt = target
        # Set distance in attribute space. Deliberately placed BEFORE the matched
        # per-slot terms and computed against `target`, not `a_tgt`: a_tgt is the
        # position-matched (and therefore blended) target, which is the very thing
        # this term exists to bypass.
        w_as = float(weights.get("w_attr_set", 0.0))
        if w_as != 0.0 and ap.shape[-1] > 10:
            aset = intra_group_attr_set_loss(
                ap, target, mask, gsz, layout,
                chunk_groups=int(weights.get("attr_set_chunk", 512)),
            )
            total = total + w_as * aset
            logs["codec_attr_set"] = aset.detach()
        w_al = float(weights.get("w_attr_local_set", 0.0))
        if w_al != 0.0 and ap.shape[-1] > 10:
            aloc = intra_group_local_attr_set_loss(
                ap, target, mask, gsz, layout,
                k=int(weights.get("attr_local_set_k", 16)),
                chunk_groups=int(weights.get("attr_local_set_chunk", 32)),
            )
            total = total + w_al * aloc
            logs["codec_attr_local_set"] = aloc.detach()
        # 셀 내부 속성 2차 모멘트. attr_set 이 집합거리로 붕괴를 벌한다면 이쪽은
        # 붕괴의 정의(퍼짐과 위치의존성)를 직접 맞춘다. 측정된 결함은 std 비율
        # 0.39(logscale) / 0.50(opacity) 와 위치->속성 R^2 0.045 vs GT 0.234.
        w_sp = float(weights.get("w_attr_spread", 0.0))
        w_sl = float(weights.get("w_attr_slope", 0.0))
        if (w_sp != 0.0 or w_sl != 0.0) and ap.shape[-1] > 10:
            spread, slope = intra_group_attr_moment_loss(ap, target, mask, gsz, layout)
            logs["codec_attr_spread"] = spread.detach()
            logs["codec_attr_slope"] = slope.detach()
            if w_sp != 0.0:
                total = total + w_sp * spread
            if w_sl != 0.0:
                total = total + w_sl * slope
        w_hs = float(weights.get("w_attr_hung_scale", 0.0))
        w_ho = float(weights.get("w_attr_hung_opacity", 0.0))
        w_hr = float(weights.get("w_attr_hung_rot", 0.0))
        if (w_hs != 0.0 or w_ho != 0.0 or w_hr != 0.0) and ap.shape[-1] > 10:
            hsc, hop, hrt = intra_group_hungarian_attr_loss(ap, target, mask, gsz, layout)
            logs["codec_attr_hung_scale"] = hsc.detach()
            logs["codec_attr_hung_opacity"] = hop.detach()
            logs["codec_attr_hung_rot"] = hrt.detach()
            if w_hs != 0.0:
                total = total + w_hs * hsc
            if w_ho != 0.0:
                total = total + w_ho * hop
            if w_hr != 0.0:
                total = total + w_hr * hrt
        w_c3 = float(weights.get("w_cov3d", 0.0))
        if w_c3 != 0.0 and ap.shape[-1] > 10:
            c3 = covariance3d_loss(ap, a_tgt, mask if a_pw is None else a_pw, layout, weights)
            total = total + w_c3 * c3
            logs["codec_cov3d"] = c3.detach()
        for k, v in _attr_losses(ap, a_tgt, mask if a_pw is None else a_pw,
                                 layout, weights, sh_dim).items():
            w_k = float(weights.get(f"w_{k}", 0.0))
            logs[f"codec_{k}"] = v.detach()
            if w_k != 0.0:
                total = total + w_k * v
        if "gen_attr_pred" in out and run_gen:
            gp = out["gen_attr_pred"]
            if mode == "sinkhorn":
                g_tgt = sinkhorn_parameter_target(
                    gp[..., 0:3].detach(), target, mask, gsz,
                    scale=out.get("decoded_scale"),
                    epsilon=float(weights.get("sinkhorn_epsilon", 0.08)),
                    iterations=int(weights.get("sinkhorn_iterations", 6)),
                    chunk_groups=int(weights.get("sinkhorn_chunk", 256)),
                )
            elif mode == "responsibility" and float(weights.get("attr_responsibility", 1.0)) != 0.0:
                g_tgt = responsibility_target(gp[..., 0:3].detach(), target, mask, layout, gsz)
            else:
                g_tgt = target
            for k, v in _attr_losses(gp, g_tgt, mask if a_pw is None else a_pw,
                                     layout, weights, sh_dim).items():
                w_k = float(weights.get(f"w_{k}", 0.0))
                logs[f"gen_{k}"] = v.detach()
                if w_k != 0.0:
                    total = total + w_k * v

    if run_decode:
        codec_total, codec_parts = branch_geometry_loss(
            out["pred"], out["presence"], target, mask, weights, "", layout, sh_dim,
            model_layout["group_size"], balance_weight=balance,
            with_attrs=with_attrs and "attr_pred" not in out,
            group_scale=out.get("decoded_scale"),
        )
        total = total + codec_total
        for k, v in codec_parts.items():
            logs[f"codec_{k}"] = v
        w_xr = float(weights.get("w_xyz_residual", 0.0))
        w_xrh = float(weights.get("w_xyz_residual_hard", 0.0))
        if w_xr > 0.0 or w_xrh > 0.0:
            xr = point_group_residual_loss(
                out["pred"], target, mask, model_layout["group_size"],
                scale=out.get("decoded_scale"),
                hard_frac=float(weights.get("residual_hard_frac", 0.2)),
            )
            logs["codec_xyz_residual"] = xr["xyz_residual"].detach()
            logs["codec_xyz_residual_hard"] = xr["xyz_residual_hard"].detach()
            if w_xr > 0.0:
                total = total + w_xr * xr["xyz_residual"]
            if w_xrh > 0.0:
                total = total + w_xrh * xr["xyz_residual_hard"]

    if run_gen and "gen_pred" in out:
        gen_scale = float(weights.get("w_gen_scale", 1.0))
        gen_total, gen_parts = branch_geometry_loss(
            out["gen_pred"], out["gen_presence"], target, mask, weights, "gen_", layout, sh_dim,
            # with_attrs was False here, so nothing ever reached the student's
            # attribute heads: not the parameter losses, not distillation (xyz
            # only), not the render term (codec's `pred` only). Measured at step
            # 20000 the student's emitted colour had 19% of the ground truth's
            # point-to-point spread and its scale 14% -- the heads had barely
            # moved from init, and the render came out as a near-uniform plate.
            # gen is the deployment path, so this was the one branch that had to
            # learn them.
            model_layout["group_size"], balance_weight=balance,
            with_attrs=with_attrs and "attr_pred" not in out,
            group_scale=out.get("decoded_scale"),
        )
        total = total + gen_scale * gen_total
        for k, v in gen_parts.items():
            logs[f"gen_{k}"] = v
        w_gxr = float(weights.get("w_gen_xyz_residual", 0.0))
        w_gxrh = float(weights.get("w_gen_xyz_residual_hard", 0.0))
        if (w_gxr > 0.0 or w_gxrh > 0.0) and gen_scale > 0.0:
            gxr = point_group_residual_loss(
                out["gen_pred"], target, mask, model_layout["group_size"],
                scale=out.get("decoded_scale"),
                hard_frac=float(weights.get("residual_hard_frac", 0.2)),
            )
            logs["gen_xyz_residual"] = gxr["xyz_residual"].detach()
            logs["gen_xyz_residual_hard"] = gxr["xyz_residual_hard"].detach()
            if w_gxr > 0.0:
                total = total + gen_scale * w_gxr * gxr["xyz_residual"]
            if w_gxrh > 0.0:
                total = total + gen_scale * w_gxrh * gxr["xyz_residual_hard"]
        w_distill = float(weights.get("w_distill", 0.0))
        if w_distill != 0.0 and run_decode:
            teacher = out["pred"][..., 0:3].detach()
            # Extent-normalised, not absolute. The median group radius is 0.0027 in
            # normalised scene units, so an absolute smooth_l1 with beta=0.02 sits
            # deep in its quadratic regime and evaluates to ~0.0025 -- measured at
            # 0.1% of the objective even at w_distill=30, i.e. the one term that
            # implements "the student learns from the teacher" was effectively off.
            # This is the same units mistake that made w_render wrong by 1000x.
            # The centroid is identical on both paths (both read it from the same
            # z_compact anchor channels), so the within-group residual is the whole
            # of what distillation has to transfer.
            dg = point_group_residual_loss(
                out["gen_pred"][..., 0:3], teacher, mask, model_layout["group_size"],
                scale=out.get("decoded_scale"),
            )
            distill = dg["xyz_residual"]
            total = total + w_distill * distill
            logs["gen_distill"] = distill.detach()

        # Distil the *decoding basis*, not only the final points.
        #
        # Both decoders build a group as R(a) diag(s) (fixed_template + residual),
        # so the teacher's frame and its pre-scale offsets are directly meaningful
        # targets for the student. Matching only the final xyz leaves the student
        # free to reach a similar point set through a different -- and measurably
        # worse -- basis: gen's groups came out 1.83x the GT radius with
        # precision 1.294 against coverage 0.521, i.e. it spread points to satisfy
        # a symmetric chamfer instead of reproducing the shape. ``fold_local`` is
        # exactly the quantity whose magnitude sets that radius.
        # Anti-inflation terms for the student. Symmetric chamfer is satisfiable by
        # spreading: measured gen radius 1.83x GT with precision (p->g) 1.294
        # against coverage (g->p) 0.521. A one-sided precision term and an explicit
        # radius-ratio term are the two things a symmetric term cannot express.
        w_p2g = float(weights.get("w_gen_p2g", 0.0))
        w_rad = float(weights.get("w_gen_radius", 0.0))
        if (w_p2g != 0.0 or w_rad != 0.0) and "gen_pred" in out:
            _, parts = intra_group_chamfer(
                out["gen_pred"][..., 0:3], target[..., 0:3], mask,
                model_layout["group_size"], scale=out.get("decoded_scale"),
                chunk_groups=int(weights.get("intra_chamfer_chunk", 1024)),
                return_parts=True,
            )
            if w_p2g != 0.0:
                total = total + w_p2g * parts["p2g"]
                logs["gen_p2g"] = parts["p2g"].detach()
            if w_rad != 0.0:
                total = total + w_rad * parts["radius"]
                logs["gen_radius"] = parts["radius"].detach()
            logs["gen_g2p"] = parts["g2p"].detach()

        # Attribute distillation. Deliberately *not* inside the `w_distill` block:
        # that one is gated on the xyz distillation weight, so nesting this under
        # it would silently switch the attribute path off whenever xyz
        # distillation is off -- which is how the student ended up with no
        # attribute signal at all in the first place.
        # Normalised per channel by the teacher's own spread, because a quaternion
        # component and a log scale are not comparable in absolute terms.
        w_da = float(weights.get("w_distill_attr", 0.0))
        if w_da != 0.0 and run_decode and out["gen_pred"].shape[-1] > 3:
            t_a = out["pred"][..., 3:].detach()
            sd = t_a.reshape(-1, t_a.shape[-1]).std(0).clamp(min=1e-3)
            d_attr = masked_reduce(
                ((out["gen_pred"][..., 3:] - t_a) / sd).abs().mean(-1), mask
            )
            total = total + w_da * d_attr
            logs["gen_distill_attr"] = d_attr.detach()

        w_basis = float(weights.get("w_gen_basis", 0.0))
        if w_basis != 0.0 and "gen_fold_local" in out and "teacher_fold_local" in out:
            gv = out["group_valid"][:, : out["gen_fold_local"].shape[1]]
            m3 = gv[..., None, None]
            tl = out["teacher_fold_local"].detach()
            gl = out["gen_fold_local"]
            n = min(tl.shape[1], gl.shape[1])
            b_local = masked_reduce(
                (gl[:, :n] - tl[:, :n]).abs().mean(dim=-1), m3[:, :n].squeeze(-1).expand(-1, n, gl.shape[2])
            )
            tf = out["teacher_fold_frame"].detach()[:, :n]
            gf = out["gen_fold_frame"][:, :n]
            b_frame = frame_distance(gf, tf, gv[:, :n])
            basis = b_local + b_frame
            total = total + w_basis * basis
            logs["gen_basis"] = basis.detach()
            logs["gen_basis_local"] = b_local.detach()
            logs["gen_basis_frame"] = b_frame.detach()

    logs["total"] = total.detach()
    return total, logs
