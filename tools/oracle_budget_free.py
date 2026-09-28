"""Ways to buy shape budget without throwing away points.

max_points stays at 262,144 and z_compact stays at 32x64x64, so the number of
groups (8192) and the channels per group (16 = 4 anchor + 12 shape) are fixed.
What is *not* fixed:

* how many real points land in a group. The loader packs every point into the
  first ceil(used/32) groups and leaves the rest completely empty, so ~24% of
  the latent codes nothing. Spreading the same points over all 8192 groups puts
  ~24 points under each 12-channel code instead of 32.
* how the groups are cut. Fixed-size Morton chunks produce a small number of
  groups that span half the scene.
* what frame the offsets are expressed in. If the frame can be rebuilt by the
  decoder from information it already has (neighbouring group centroids), then
  canonicalising into it is free.

Each variant is measured with the same oracle autoencoder so the numbers are
comparable to oracle_bottleneck.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_detail import read_ply, pca_bound
from diag_grouping import kd_partition
from oracle_bottleneck import GroupAE


def load_scene(eval_dir: str, scene: str) -> np.ndarray:
    with open(os.path.join(eval_dir, f"{scene}_codec_metrics.json")) as f:
        met = json.load(f)
    s = met["xyz_rmse_metric"] / met["xyz_rmse_norm"]
    return read_ply(os.path.join(eval_dir, f"{scene}_codec_gt.ply")) / s


def neighbour_frames(cent: np.ndarray, k: int = 9) -> np.ndarray:
    """Local frame per group, built only from the *group centroids*.

    The decompressor already reads every neighbouring cell's centroid channels,
    so it can rebuild this frame with no extra latent channels.
    """
    _, nn = cKDTree(cent).query(cent, k=k, workers=-1)
    nb = cent[nn] - cent[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    _, V = np.linalg.eigh(cov)  # ascending eigenvalues
    V = V[:, :, ::-1]  # major axis first
    # sign disambiguation so the frame is a deterministic function of the input
    sgn = np.sign(V[:, 0, :] + 1e-12)[:, None, :]
    return V * sgn


def own_frames(off: np.ndarray) -> np.ndarray:
    cov = np.einsum("gki,gkj->gij", off, off) / off.shape[1]
    _, V = np.linalg.eigh(cov)
    V = V[:, :, ::-1]
    sgn = np.sign(V[:, 0, :] + 1e-12)[:, None, :]
    return V * sgn


def run_oracle(off: np.ndarray, code: int, steps: int, dev, tag: str) -> None:
    """off: (num_groups, group_size, 3) already centroid-subtracted."""
    ng, gsz, _ = off.shape
    scale = np.maximum(np.abs(off).max(axis=(1, 2), keepdims=True), 1e-6)
    x = (off / scale).reshape(ng, -1).astype(np.float32)
    sc = scale.reshape(-1, 1).astype(np.float32)
    X, S = torch.from_numpy(x).to(dev), torch.from_numpy(sc).to(dev)
    LS = torch.log(S / 0.02) / 3.0
    perm = np.random.default_rng(0).permutation(ng)
    tr = torch.from_numpy(perm[: int(0.9 * ng)]).to(dev)
    te = torch.from_numpy(perm[int(0.9 * ng) :]).to(dev)

    torch.manual_seed(0)
    net = GroupAE(gsz, code).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = tr[torch.randint(0, len(tr), (2048,), device=dev)]
        rec, _ = net(X[i], LS[i])
        loss = ((rec - X[i]) * S[i]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        rec, _ = net(X[te], LS[te])
        d = ((rec - X[te]) * S[te]).reshape(-1, gsz, 3)
        e = d.pow(2).sum(-1).sqrt()
        rmse = float(d.pow(2).sum(-1).mean().sqrt())
        p50, p90 = float(e.median()), float(e.quantile(0.9))
    lin = pca_bound(off.reshape(ng, -1)[:20000], code)
    print(f"{tag:<48}{ng:>8}{gsz:>6}{code:>6}{rmse:>12.6f}{p50:>11.6f}{p90:>11.6f}{lin:>12.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--groups", type=int, default=8192, help="fixed by z_compact: 4096 cells x merge 2")
    ap.add_argument("--slots", type=int, default=32, help="architectural slots per group")
    ap.add_argument("--code", type=int, default=12)
    ap.add_argument("--scenes", type=str, default="")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if a.scenes:
        scenes = a.scenes.split(",")
    else:
        scenes = sorted({f.split("_codec_gt.ply")[0] for f in os.listdir(a.eval_dir) if f.endswith("_codec_gt.ply")})
    dev = torch.device(a.device)

    for sc in scenes:
        gt = load_scene(a.eval_dir, sc)
        n = len(gt)
        k = max(n // a.groups, 4)
        used_groups = n // a.slots
        print(f"\n### {sc}  N={n}  현재 사용 그룹 {used_groups}/{a.groups} "
              f"(유휴 {100*(1-used_groups/a.groups):.0f}%)  → 재분배 시 그룹당 {k}점")
        print(f"{'variant':<48}{'groups':>8}{'pts/g':>6}{'code':>6}{'rmse':>12}{'p50':>11}"
              f"{'p90':>11}{'PCA':>12}")

        def morton_fixed():
            m = n // a.slots * a.slots
            return gt[:m].reshape(-1, a.slots, 3)

        def morton_spread():
            return gt[: k * a.groups].reshape(a.groups, k, 3)

        def kd_spread():
            return gt[kd_partition(gt[: k * a.groups], k)]

        variants = [
            ("A  morton chunk, 32 pts/group  (current)", morton_fixed, None),
            ("B  morton chunk, spread over all groups", morton_spread, None),
            ("C  kd-tree,      spread over all groups", kd_spread, None),
            ("D  C + frame from neighbour centroids (free)", kd_spread, "nb"),
            ("E  C + frame from own points (upper bound)", kd_spread, "own"),
        ]
        for tag, part, frame in variants:
            g = part()
            cent = g.mean(1)
            off = g - cent[:, None]
            if frame == "nb":
                off = np.einsum("gki,gij->gkj", off, neighbour_frames(cent))
            elif frame == "own":
                off = np.einsum("gki,gij->gkj", off, own_frames(off))
            run_oracle(off, a.code, a.steps, dev, tag)


if __name__ == "__main__":
    main()
