"""Where does the xyz error actually live?

Reads an index-aligned (gt, pred) ply pair written by save_eval_outputs and
decomposes the error the same way the architecture decomposes the latent:

    point error = group centroid error  +  within-group offset error

then compares the observed offset error against the best a 12-channel linear
code could possibly do (PCA on the same offsets). If the model is already at
the PCA bound the bottleneck is the channel budget; if it is far above it the
bottleneck is the optimisation / decoder.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def read_ply(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header, n = [], 0
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            header.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line == "end_header":
                break
        fmt = [h for h in header if h.startswith("format")][0]
        if "ascii" in fmt:
            data = np.loadtxt(f, max_rows=n, dtype=np.float32)
            return np.asarray(data[:, :3], np.float64)
        arr = np.frombuffer(f.read(n * 12), dtype="<f4", count=n * 3)
        return np.asarray(arr.reshape(n, 3), np.float64)


def pca_bound(off: np.ndarray, k: int) -> float:
    """Best rank-k linear reconstruction rmse of the (G, 96) offset matrix."""
    mu = off.mean(0, keepdims=True)
    x = off - mu
    # economy SVD on the covariance is enough (96 dims)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    tail = ev[k:].sum()
    # rmse over points = sqrt(residual variance summed over the 96 dims / 32 pts)
    return float(np.sqrt(max(tail, 0.0) / (off.shape[1] // 3)))


def summarise(gt: np.ndarray, pr: np.ndarray, group_size: int, tag: str) -> None:
    n = len(gt)
    d = pr - gt
    sq = (d ** 2).sum(1)
    rmse = np.sqrt(sq.mean())
    print(f"\n===== {tag}  N={n} =====")
    print(f"rmse           {rmse:.6f}     l1 {np.abs(d).mean():.6f}   max {np.abs(d).max():.4f}")

    order = np.argsort(-sq)
    cum = np.cumsum(sq[order]) / sq.sum()
    for frac in (0.001, 0.01, 0.05, 0.10):
        k = max(int(frac * n), 1)
        print(f"  worst {frac*100:5.1f}% of points carry {cum[k-1]*100:5.1f}% of the MSE")
    q = np.sqrt(np.percentile(sq, [50, 75, 90, 95, 99, 99.9]))
    print("  per-point error percentiles p50/75/90/95/99/99.9: "
          + " ".join(f"{v:.4f}" for v in q))

    ng = n // group_size
    g_gt = gt[: ng * group_size].reshape(ng, group_size, 3)
    g_pr = pr[: ng * group_size].reshape(ng, group_size, 3)
    c_gt, c_pr = g_gt.mean(1), g_pr.mean(1)
    off_gt, off_pr = g_gt - c_gt[:, None], g_pr - c_pr[:, None]

    cen_mse = ((c_pr - c_gt) ** 2).sum(1).mean()
    off_mse = ((off_pr - off_gt) ** 2).sum(2).mean()
    tot = cen_mse + off_mse
    print(f"  centroid rmse {np.sqrt(cen_mse):.6f}  ({cen_mse/tot*100:4.1f}% of MSE)")
    print(f"  offset   rmse {np.sqrt(off_mse):.6f}  ({off_mse/tot*100:4.1f}% of MSE)")

    ext = np.sqrt((off_gt ** 2).sum(2).mean(1))  # rms radius of each gt group
    print("  gt group rms-radius p10/50/90/99: "
          + " ".join(f"{v:.5f}" for v in np.percentile(ext, [10, 50, 90, 99])))
    print(f"  'predict centroid only' rmse would be {np.sqrt((off_gt**2).sum(2).mean()):.6f}")

    # error as a function of group size, in extent bins
    bins = np.percentile(ext, [0, 50, 90, 99, 100])
    print("  offset rmse by gt group radius bin:")
    for i in range(len(bins) - 1):
        sel = (ext >= bins[i]) & (ext <= bins[i + 1])
        if sel.sum() < 4:
            continue
        e = np.sqrt(((off_pr[sel] - off_gt[sel]) ** 2).sum(2).mean())
        base = np.sqrt((off_gt[sel] ** 2).sum(2).mean())
        print(f"    radius[{bins[i]:.5f},{bins[i+1]:.5f}] n={sel.sum():6d}  "
              f"offset_rmse {e:.6f}  centroid-only {base:.6f}  ratio {e/max(base,1e-9):.3f}")

    # information bound: 12 shape channels per group of 32 points
    flat = off_gt.reshape(ng, group_size * 3)
    sub = flat[np.random.default_rng(0).choice(ng, min(ng, 20000), replace=False)]
    print("  PCA bound on raw offsets (rmse if k dims were coded perfectly):")
    for k in (4, 8, 12, 16, 24, 32):
        print(f"    k={k:3d}  {pca_bound(sub, k):.6f}")
    # scale-normalised, the way the net actually codes them
    s = np.maximum(np.abs(off_gt).max(axis=(1, 2), keepdims=True), 1e-6)
    nf = (off_gt / s).reshape(ng, group_size * 3)
    nsub = nf[np.random.default_rng(0).choice(ng, min(ng, 20000), replace=False)]
    ssub = s.reshape(-1)[np.random.default_rng(0).choice(ng, min(ng, 20000), replace=False)]
    print("  PCA bound on scale-normalised offsets, re-multiplied by mean scale "
          f"({ssub.mean():.5f}):")
    for k in (4, 8, 12, 16, 24, 32):
        print(f"    k={k:3d}  {pca_bound(nsub, k) * float(np.sqrt((ssub**2).mean())):.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--scene", default="step_012730")
    ap.add_argument("--group_size", type=int, default=32)
    a = ap.parse_args()
    for branch in ("codec", "gen"):
        gp = os.path.join(a.eval_dir, f"{a.scene}_{branch}_gt.ply")
        pp = os.path.join(a.eval_dir, f"{a.scene}_{branch}_pred.ply")
        if not (os.path.exists(gp) and os.path.exists(pp)):
            continue
        gt, pr = read_ply(gp), read_ply(pp)
        # ply is written in world units; recover the normalisation scale from the
        # metrics json so the numbers line up with xyz_rmse_norm
        import json

        with open(os.path.join(a.eval_dir, f"{a.scene}_{branch}_metrics.json")) as f:
            met = json.load(f)
        scale = met["xyz_rmse_metric"] / met["xyz_rmse_norm"]
        summarise(gt / scale, pr / scale, a.group_size,
                  f"{a.scene} / {branch}  (norm scale={scale:.3f} world units)")


if __name__ == "__main__":
    main()
