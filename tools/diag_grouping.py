"""Two questions the first diagnostic left open.

1. How much of the pointwise error is *ordering* rather than shape? Compare the
   identity assignment the loss uses against the optimal assignment inside each
   group, and against a full-cloud nearest-neighbour distance.
2. Is the fixed-size Morton chunking responsible for the fat tail? Re-partition
   the same cloud with a balanced kd-tree (equal-count, spatially compact) and
   recompute the group radii and the PCA rate-distortion bound.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from diag_detail import read_ply, pca_bound


def kd_partition(xyz: np.ndarray, group_size: int) -> np.ndarray:
    """Equal-count recursive median split -> (num_groups, group_size) indices."""
    n = len(xyz) // group_size * group_size
    idx = np.arange(n)
    stack = [idx]
    out = []
    while stack:
        cur = stack.pop()
        if len(cur) <= group_size:
            out.append(cur)
            continue
        pts = xyz[cur]
        ax = int(np.argmax(pts.max(0) - pts.min(0)))
        half = (len(cur) // group_size // 2) * group_size
        if half == 0:
            half = group_size
        order = np.argpartition(pts[:, ax], half)[: len(cur)]
        cur = cur[order[np.argsort(pts[order, ax])]]
        stack.append(cur[:half])
        stack.append(cur[half:])
    return np.stack([g for g in out if len(g) == group_size])


def group_stats(g_gt: np.ndarray, name: str, ks=(12, 28)) -> None:
    c = g_gt.mean(1)
    off = g_gt - c[:, None]
    rad = np.sqrt((off ** 2).sum(2).mean(1))
    gsz = g_gt.shape[1]
    print(f"\n-- {name}: {len(g_gt)} groups of {gsz}")
    print("   radius p50/90/99/max: " + " ".join(f"{v:.5f}" for v in np.percentile(rad, [50, 90, 99, 100])))
    print(f"   centroid-only rmse   : {np.sqrt((off**2).sum(2).mean()):.6f}")
    top = np.argsort(-rad)[: max(len(rad) // 100, 1)]
    share = (off[top] ** 2).sum() / (off ** 2).sum()
    print(f"   worst 1% of groups hold {share*100:.1f}% of the offset energy")
    flat = off.reshape(len(off), gsz * 3)
    sub = flat[np.random.default_rng(0).choice(len(flat), min(len(flat), 20000), replace=False)]
    for k in ks:
        print(f"   PCA-{k} bound        : {pca_bound(sub, k):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--scene", default="step_012730")
    ap.add_argument("--branch", default="codec")
    a = ap.parse_args()

    with open(os.path.join(a.eval_dir, f"{a.scene}_{a.branch}_metrics.json")) as f:
        met = json.load(f)
    s = met["xyz_rmse_metric"] / met["xyz_rmse_norm"]
    gt = read_ply(os.path.join(a.eval_dir, f"{a.scene}_{a.branch}_gt.ply")) / s
    pr = read_ply(os.path.join(a.eval_dir, f"{a.scene}_{a.branch}_pred.ply")) / s
    n = len(gt) // 32 * 32
    gt, pr = gt[:n], pr[:n]

    print(f"===== {a.scene} / {a.branch}  N={n} =====")
    print(f"pointwise rmse (what the loss optimises): {np.sqrt(((pr-gt)**2).sum(1).mean()):.6f}")

    tg, tp = cKDTree(gt), cKDTree(pr)
    d_pg, _ = tg.query(pr, k=1, workers=-1)
    d_gp, _ = tp.query(gt, k=1, workers=-1)
    print(f"full-cloud chamfer pred->gt {d_pg.mean():.6f}   gt->pred {d_gp.mean():.6f}")
    print(f"  pred->gt p50/90/99 " + " ".join(f"{v:.5f}" for v in np.percentile(d_pg, [50, 90, 99])))
    print(f"  gt->pred p50/90/99 " + " ".join(f"{v:.5f}" for v in np.percentile(d_gp, [50, 90, 99])))
    dnn, _ = tg.query(gt, k=2, workers=-1)
    print(f"  gt self nn spacing mean {dnn[:, 1].mean():.6f} (scale of 'perfect')")

    g_gt = gt.reshape(-1, 32, 3)
    g_pr = pr.reshape(-1, 32, 3)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(g_gt), min(600, len(g_gt)), replace=False)
    ident, opt = [], []
    for i in pick:
        cost = np.linalg.norm(g_pr[i][:, None] - g_gt[i][None], axis=-1)
        ident.append(np.trace(cost) / 32)
        r, c = linear_sum_assignment(cost)
        opt.append(cost[r, c].mean())
    ident, opt = np.array(ident), np.array(opt)
    print(f"within-group mean distance: identity order {ident.mean():.6f}  "
          f"optimal matching {opt.mean():.6f}  -> {(1-opt.mean()/ident.mean())*100:.1f}% "
          "of the pointwise error is pure ordering")

    for gsz in (32, 64):
        m = len(gt) // gsz * gsz
        group_stats(gt[:m].reshape(-1, gsz, 3), f"morton contiguous, group_size={gsz}",
                    ks=(12, 28) if gsz == 64 else (12,))
        kd = kd_partition(gt[:m], gsz)
        group_stats(gt[kd], f"balanced kd-tree,   group_size={gsz}",
                    ks=(12, 28) if gsz == 64 else (12,))


if __name__ == "__main__":
    main()
