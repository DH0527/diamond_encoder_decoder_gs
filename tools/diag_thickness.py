"""Is the prediction blurred, shrunk, or just jittered?

Three measurements on an index-aligned (gt, pred) pair:

* local thickness  - smallest local-PCA eigenvalue over a k-NN ball. A thin
  filament that turned into a fuzzy tube shows up here and nowhere else.
* group spread     - std of the predicted within-group offsets over the GT std.
  < 1 means the decoder is regressing to the conditional mean (classic
  bottleneck blur); ~1 means it keeps the energy but puts it in the wrong place.
* nn spacing       - how the point-to-point spacing changed, i.e. whether the
  prediction clumps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_detail import read_ply


def local_shape(xyz: np.ndarray, k: int = 24, sample: int = 30000, seed: int = 0):
    tree = cKDTree(xyz)
    idx = np.random.default_rng(seed).choice(len(xyz), min(sample, len(xyz)), replace=False)
    _, nn = tree.query(xyz[idx], k=k, workers=-1)
    nb = xyz[nn]
    nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    ev = np.linalg.eigvalsh(cov)  # ascending
    return np.sqrt(np.clip(ev, 0, None)), idx


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
    print(f"===== {a.scene} / {a.branch} =====")

    for name, c in (("gt  ", gt), ("pred", pr)):
        ev, _ = local_shape(c)
        d, _ = cKDTree(c).query(c, k=2, workers=-1)
        print(f"{name}: local sigma_min p50 {np.percentile(ev[:,0],50):.6f}  "
              f"sigma_mid p50 {np.percentile(ev[:,1],50):.6f}  "
              f"sigma_max p50 {np.percentile(ev[:,2],50):.6f}  |  "
              f"nn spacing p50 {np.percentile(d[:,1],50):.6f}")

    g = gt.reshape(-1, 32, 3)
    p = pr.reshape(-1, 32, 3)
    sg = (g - g.mean(1, keepdims=True)).std(axis=(1, 2))
    sp = (p - p.mean(1, keepdims=True)).std(axis=(1, 2))
    ok = sg > 1e-6
    r = sp[ok] / sg[ok]
    print(f"within-group spread pred/gt: p10 {np.percentile(r,10):.3f}  p50 {np.percentile(r,50):.3f}  "
          f"p90 {np.percentile(r,90):.3f}   (1.0 = no shrinkage)")

    # how much of the pointwise error is along vs across the local surface
    ev, idx = local_shape(gt)
    tree = cKDTree(gt)
    _, nn = tree.query(gt[idx], k=24, workers=-1)
    nb = gt[nn] - gt[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / 24
    w, V = np.linalg.eigh(cov)
    d = (pr[idx] - gt[idx])[:, None, :]
    comp = np.abs((d @ V)[:, 0, :])  # projection onto [min, mid, max] axes
    print(f"error projected on local frame p50: across {np.percentile(comp[:,0],50):.6f}  "
          f"mid {np.percentile(comp[:,1],50):.6f}  along {np.percentile(comp[:,2],50):.6f}")


if __name__ == "__main__":
    main()
