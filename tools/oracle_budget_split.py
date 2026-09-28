"""How should the 28 free channels per group be split between shape and appearance?

`z_compact` is 32x64x64 and that is fixed. Four channels per group go to the
centroid, so shape and appearance divide the remaining 28. Today it is 20/8, and
the obvious question is whether shape needs all 20 -- if geometry saturates
earlier, the surplus is worth more spent on appearance.

Neither existing oracle answers it. `oracle_shape_rank.py` varies geometry with
appearance absent; `oracle_attr_code_size.py` varies appearance at a fixed
geometry rank of 28. Both measure one axis of a trade-off that only makes sense
jointly, because a channel given to one side is taken from the other.

This sweeps the split itself. For each (shape, appearance) pair summing to the
budget, positions are the rank-`shape` reconstruction of the true groups and the
attributes are the best rank-`appearance` code that can be fitted to the image --
i.e. the most any decoder could extract from that split in the linear limit. The
score is the render, on held-out poses, which is the only place the two sides
become comparable: a channel of shape and a channel of appearance have no common
unit until they are both turned into pixels.

Upper bounds, not forecasts -- every code here is fitted per scene with direct
access to the target. The comparison between splits is what carries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.io_utils import camera_from_vector  # noqa: E402
from can3tok.render import (Camera, perturb_camera_vector, photometric_loss,  # noqa: E402
                            psnr, render_gaussians, sh_dc_to_rgb, ssim)
from can3tok.train import make_datasets  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oracle_attr_code_size import A_DIM, rank_k_positions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--budget", type=int, default=28,
                    help="channels per group left after the centroid takes 4")
    ap.add_argument("--splits", type=str, default="24,20,16,12,8,4",
                    help="shape ranks to try; appearance gets budget - shape")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fit_views", type=int, default=4)
    ap.add_argument("--test_views", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    item = val_ds[a.scene]
    g, dev = int(targs.group_size), "cuda"

    m = item["mask"].to(dev) > 0.5
    t = item["target"].to(dev).float()[m]
    ng = t.shape[0] // g
    t = t[: ng * g]
    sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
    gt_grp = t[:, 0:3].reshape(ng, g, 3)
    A0 = t[:, 3:14].contiguous()

    cv = item["camera"].numpy()
    rf, rt = np.random.default_rng(0), np.random.default_rng(9999)
    fit = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
    fit += [Camera(camera_from_vector(perturb_camera_vector(cv, rf)), device=dev,
                   downscale=a.downscale) for _ in range(a.fit_views - 1)]
    test = [Camera(camera_from_vector(perturb_camera_vector(cv, rt)), device=dev,
                   downscale=a.downscale) for _ in range(a.test_views)]

    def gaussians(xyz, attr):
        return (xyz, torch.exp(attr[:, 0:3].clamp(-15, 5)) * sc, attr[:, 3:7],
                torch.sigmoid(attr[:, 7:8]), sh_dc_to_rgb(attr[:, 8:11]))

    gt_xyz = gt_grp.reshape(-1, 3) * sc + cen
    with torch.no_grad():
        fit_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in fit]
        test_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in test]
    sd = A0.std(0, keepdim=True).clamp(min=1e-6)
    sd_g = sd.repeat(1, g).reshape(1, g * A_DIM)

    def fit_appearance(xyz_k, k):
        """Best rank-k per-group appearance code, fitted to the image."""
        base = A0.reshape(ng, g, A_DIM).mean(1, keepdim=True).expand(ng, g, A_DIM)
        base = base.reshape(ng, g * A_DIM).clone()
        if k <= 0:
            return base.reshape(-1, A_DIM)          # group mean only, no code
        B = torch.empty(k, g * A_DIM, device=dev)
        torch.nn.init.normal_(B, std=1.0 / (g * A_DIM) ** 0.5)
        C = torch.zeros(ng, k, device=dev)
        B.requires_grad_(True)
        C.requires_grad_(True)
        opt = torch.optim.Adam([B, C], lr=a.lr)
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            attr = (base + (C @ B) * sd_g).reshape(-1, A_DIM)
            loss = 0.0
            for r, c in zip(fit_ref, fit):
                l, _, _ = photometric_loss(render_gaussians(*gaussians(xyz_k, attr), c), r, 0.2)
                loss = loss + l / len(fit)
            loss.backward()
            opt.step()
        with torch.no_grad():
            return (base + (C @ B) * sd_g).reshape(-1, A_DIM)

    print(f"scene {item['name']}  groups {ng}  budget {a.budget} ch/group "
          f"(+ centroid 4 = {a.budget + 4})")
    print(f"  fit {len(fit)} views, held-out {len(test)} views. "
          f"모든 코드는 장면별 직접 최적화 = 상한\n")
    # Geometry has to be reported in its own units too. The render is the only
    # place a shape channel and an appearance channel become comparable, but it is
    # not the only thing the latent is for: the point cloud itself has to resemble
    # the npz, and a world model consumes this geometry. A split chosen on PSNR
    # alone would happily trade away 3D structure that the image barely notices.
    print(f"{'shape':>6} {'appear':>7} {'held-out dB':>12} {'SSIM':>7} "
          f"{'xyz rmse':>10} {'rel':>7}   {'':<10}")

    rad = (gt_grp - gt_grp.mean(1, keepdim=True)).norm(dim=-1).mean()
    best = None
    for s in [int(x) for x in a.splits.split(",") if x]:
        k = a.budget - s
        p_norm = rank_k_positions(gt_grp, s)
        xyz_k = (p_norm * sc + cen).detach()
        attr = fit_appearance(xyz_k, k)
        with torch.no_grad():
            p = float(np.mean([psnr(r, render_gaussians(*gaussians(xyz_k, attr), c))
                               for r, c in zip(test_ref, test)]))
            ss = float(np.mean([ssim(r, render_gaussians(*gaussians(xyz_k, attr), c))
                                for r, c in zip(test_ref, test)]))
            d = (p_norm.reshape(ng, g, 3) - gt_grp).norm(dim=-1)
            rmse = float(d.pow(2).mean().sqrt())
        tag = "  <- 현재" if s == 20 else ""
        print(f"{s:6d} {k:7d} {p:12.2f} {ss:7.3f} {rmse*sc:10.5f} "
              f"{rmse/float(rad):7.3f}{tag}")
        if best is None or p > best[2]:
            best = (s, k, p)
    print(f"\n최적 분배(렌더 기준): shape {best[0]} / appearance {best[1]}  ({best[2]:.2f} dB)")
    print("'rel' = 위치 오차 / 그룹 반경. 1.0 이면 그룹 내 구조가 사라진 것입니다.")


if __name__ == "__main__":
    main()
