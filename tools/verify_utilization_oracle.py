#!/usr/bin/env python3
"""Oracle: prefix packing vs full 4096-cell utilization.

Reports live-cell fraction, pts/group, rank-k PCA residual (ordered — for
reference only), and a permutation-invariant proxy:

  fixed anisotropic ellipsoid sampler chamfer / group radius

Does NOT train. Decision still requires the short train A/B.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from can3tok.data import ReplayGaussianDataset  # noqa: E402


def _pca_residual(res: np.ndarray, k: int) -> float:
    """Mean L2 residual after rank-k PCA on flattened centered offsets."""
    x = res.reshape(res.shape[0], -1)
    if x.shape[0] < 2:
        return float("nan")
    x = x - x.mean(0, keepdims=True)
    # economy SVD on covariance via eig of Gram when n < d
    n, d = x.shape
    if n >= d:
        c = (x.T @ x) / n
        w, v = np.linalg.eigh(c)
        p = v[:, -k:]
        rec = (x @ p) @ p.T
    else:
        g = (x @ x.T) / n
        w, u = np.linalg.eigh(g)
        # x ≈ u_k @ diag @ ...
        uk = u[:, -k:]
        rec = uk @ (uk.T @ x)
    err = x - rec
    return float(np.sqrt((err ** 2).mean()))


def _aniso_shell_chamfer(gt: np.ndarray, n_pts: int, rng: np.random.Generator) -> float:
    """GT points vs anisotropic Gaussian samples matching GT cov (no learning)."""
    if gt.shape[0] < 3:
        return float("nan")
    c = gt.mean(0)
    x = gt - c
    cov = (x.T @ x) / max(len(gt) - 1, 1) + 1e-8 * np.eye(3)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 1e-12, None)
    samp = (rng.normal(size=(n_pts, 3)) * np.sqrt(w)) @ v.T + c
    # chamfer (symmetric mean NN L2)
    from scipy.spatial import cKDTree

    t1, t2 = cKDTree(gt), cKDTree(samp)
    d12, _ = t1.query(samp, k=1)
    d21, _ = t2.query(gt, k=1)
    return float(0.5 * (d12.mean() + d21.mean()))


def _group_radius(pts: np.ndarray) -> float:
    c = pts.mean(0)
    return float(np.linalg.norm(pts - c, axis=1).mean())


def collect(ds: ReplayGaussianDataset, gsz: int, ng: int):
    it = ds[0]
    xyz = it["target"].numpy()[:, :3]
    m = it["mask"].numpy().astype(bool)
    g = xyz.reshape(ng, gsz, 3)
    gm = m.reshape(ng, gsz)
    live = gm.sum(1) > 0
    groups = []
    for i in np.where(live)[0]:
        k = int(gm[i].sum())
        groups.append(g[i, :k])
    return groups, int(live.sum()), int(m.sum())


def run_arm(name: str, common: dict, gsz: int, ng: int, scenes: list, k_rank: int, seed: int):
    rows = []
    rng = np.random.default_rng(seed)
    for sid in scenes:
        ds = ReplayGaussianDataset(**common, file_indices=[sid])
        groups, n_live, n_pts = collect(ds, gsz, ng)
        # centered residuals padded to gsz with zeros for ordered PCA (fair across arms)
        res_list = []
        radii = []
        shells = []
        for pts in groups:
            c = pts.mean(0)
            off = pts - c
            pad = np.zeros((gsz, 3), dtype=np.float64)
            pad[: len(pts)] = off
            res_list.append(pad)
            r = _group_radius(pts)
            radii.append(r)
            shells.append(_aniso_shell_chamfer(pts, len(pts), rng) / max(r, 1e-8))
        res = np.stack(res_list, 0)
        pca = _pca_residual(res, k_rank)
        # also relative to mean radius
        mean_r = float(np.mean(radii)) if radii else float("nan")
        rows.append(
            dict(
                scene=sid,
                n_pts=n_pts,
                n_live=n_live,
                util=n_live / ng,
                pts_per_g=n_pts / max(n_live, 1),
                pca_abs=pca,
                pca_rel=pca / max(mean_r, 1e-8),
                shell_chamfer_rel=float(np.median(shells)) if shells else float("nan"),
                radius_p50=float(np.median(radii)) if radii else float("nan"),
            )
        )
    print(f"\n=== {name} ===")
    print(
        f"{'sc':>4} {'n_pts':>8} {'live':>5} {'util':>6} {'pts/g':>6} "
        f"{'pca_abs':>8} {'pca_rel':>8} {'shell_rel':>9}"
    )
    for r in rows:
        print(
            f"{r['scene']:4d} {r['n_pts']:8d} {r['n_live']:5d} {r['util']:6.2f} "
            f"{r['pts_per_g']:6.1f} {r['pca_abs']:8.5f} {r['pca_rel']:8.4f} "
            f"{r['shell_chamfer_rel']:9.4f}"
        )
    util = np.mean([r["util"] for r in rows])
    pca = np.mean([r["pca_abs"] for r in rows])
    shell = np.mean([r["shell_chamfer_rel"] for r in rows])
    print(f"mean util={util:.3f}  mean pca_abs={pca:.5f}  mean shell_rel={shell:.4f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/data/daeho/aaaa_proj/seondo/speedy-splat/output/train_colmap_seg_prune_scores/replay_seg",
    )
    p.add_argument("--stats", default=str(ROOT / "assets/stats_replay_seg.json"))
    p.add_argument("--scenes", default="0,4,7,63,74,138,183,288")
    p.add_argument("--group_size", type=int, default=64)
    p.add_argument("--max_points", type=int, default=262144)
    p.add_argument("--rank", type=int, default=28)
    p.add_argument("--out", default=str(ROOT / "runs/verify_utilization_oracle.json"))
    args = p.parse_args()

    gsz = args.group_size
    ng = args.max_points // gsz
    scenes = [int(x) for x in args.scenes.split(",") if x.strip() != ""]
    base = dict(
        root=args.root,
        max_points=args.max_points,
        group_size=gsz,
        drop_outside=True,
        sample_mode="stratified",
        crop_prob=0.0,
        stats_path=args.stats,
        seed=0,
        density_aware_sample=True,
    )

    arms = {
        "A_prefix_localkd": dict(
            slot_redistribute=False, partition_mode="morton", partition_block=32, slot_sort="morton"
        ),
        "B_fullfill_morton": dict(
            slot_redistribute=True, partition_mode="morton", partition_block=0, slot_sort="morton"
        ),
        "C_fullfill_kd": dict(
            slot_redistribute=True, partition_mode="kd", partition_block=0, slot_sort="morton"
        ),
    }
    out = {}
    for name, kw in arms.items():
        out[name] = run_arm(name, {**base, **kw}, gsz, ng, scenes, args.rank, seed=0)

    # deltas vs A
    a = out["A_prefix_localkd"]
    print("\n=== deltas vs A (negative pca = better ceiling) ===")
    for name in ("B_fullfill_morton", "C_fullfill_kd"):
        b = out[name]
        for i in range(len(a)):
            da = (b[i]["pca_abs"] - a[i]["pca_abs"]) / max(a[i]["pca_abs"], 1e-12) * 100
            print(
                f"{name} scene {a[i]['scene']}: util {a[i]['util']:.2f}->{b[i]['util']:.2f}  "
                f"pca {da:+.1f}%  pts/g {a[i]['pts_per_g']:.1f}->{b[i]['pts_per_g']:.1f}"
            )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
