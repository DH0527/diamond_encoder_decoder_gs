"""How many channels per point does *render-equivalent* appearance need?

Two different questions keep getting conflated, and they give answers an order of
magnitude apart:

  reconstruction   reproduce the GT attributes.  `oracle_attr_rank.py` says this
                   needs ~8 channels/point for 27 dB, and 1 channel/point only
                   reaches 16.1 dB -- worse than the geometry bottleneck itself.

  equivalence      produce *any* attributes that make the render match.
                   `oracle_attr_compensation.py` reached 26.69 dB on held-out
                   poses with rank-28 geometry, using 1.05M free parameters
                   (8x the whole latent), so it sized nothing.

The second is what the system actually needs, and it has never been sized. This
sizes it: the adapted attributes are constrained to a k-dimensional affine
subspace shared across groups -- a learned basis plus one code per group, which is
exactly what a decoder with k channels per group could express in the linear
limit. Sweeping k converts "channels per point" into dB, the same way the geometry
oracle did.

Fitted on `fit_views`, always reported on **held-out** poses, because attributes
fitted to one projection hide geometry error that any other pose exposes (measured:
36.97 dB fit vs 21.02 dB held-out for a 1-view fit).
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

A_DIM = 11          # log_scale 3 | quat 4 | logit opacity 1 | SH DC 3


def rank_k_positions(grp: torch.Tensor, k: int):
    ng, g, _ = grp.shape
    cen = grp.mean(1, keepdim=True)
    off = grp - cen
    rad = off.norm(dim=-1).mean(-1).clamp(min=1e-9)
    unit = (off / rad[:, None, None]).reshape(ng, g * 3).double()
    mu = unit.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(unit - mu, full_matrices=False)
    b = vh[: min(k, vh.shape[0])]
    rec = ((unit - mu) @ b.T @ b + mu).reshape(ng, g, 3).float()
    rec = rec - rec.mean(1, keepdim=True)
    return (rec * rad[:, None, None] + cen).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--geo_rank", type=int, default=28, help="the geometry the model actually has")
    ap.add_argument("--codes", type=str, default="8,16,32,64,128",
                    help="attribute code size per GROUP; divide by group_size for ch/point")
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
    A0 = t[:, 3:14].contiguous()                       # (ng*g, 11), the GT attributes

    cv = item["camera"].numpy()
    rng_f, rng_t = np.random.default_rng(0), np.random.default_rng(9999)
    fit_cams = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
    fit_cams += [Camera(camera_from_vector(perturb_camera_vector(cv, rng_f)), device=dev,
                        downscale=a.downscale) for _ in range(a.fit_views - 1)]
    test_cams = [Camera(camera_from_vector(perturb_camera_vector(cv, rng_t)), device=dev,
                        downscale=a.downscale) for _ in range(a.test_views)]

    def gaussians(xyz, attr):
        return (xyz, torch.exp(attr[:, 0:3].clamp(-15, 5)) * sc, attr[:, 3:7],
                torch.sigmoid(attr[:, 7:8]), sh_dc_to_rgb(attr[:, 8:11]))

    gt_xyz = gt_grp.reshape(-1, 3) * sc + cen
    with torch.no_grad():
        fit_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in fit_cams]
        test_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in test_cams]

    xyz_k = (rank_k_positions(gt_grp, a.geo_rank) * sc + cen).detach()
    sd = A0.std(0, keepdim=True).clamp(min=1e-6)

    def report(label, attr):
        with torch.no_grad():
            f = np.mean([psnr(r, render_gaussians(*gaussians(xyz_k, attr), c))
                         for r, c in zip(fit_ref, fit_cams)])
            p = np.mean([psnr(r, render_gaussians(*gaussians(xyz_k, attr), c))
                         for r, c in zip(test_ref, test_cams)])
            s = np.mean([ssim(r, render_gaussians(*gaussians(xyz_k, attr), c))
                         for r, c in zip(test_ref, test_cams)])
        print(f"  {label:34s} fit {f:6.2f} | held-out {p:6.2f} dB  SSIM {s:.3f}")
        return float(p)

    print(f"scene {item['name']}  groups {ng}  geometry rank {a.geo_rank} (= 현재 모델)")
    print(f"  fit {len(fit_cams)} views, held-out {len(test_cams)} views\n")
    report("GT attributes (적응 없음)", A0)

    for k in [int(x) for x in a.codes.split(",") if x]:
        # attr = A0_groupmean + basis(k x 704) applied to a per-group code(k)
        base = A0.reshape(ng, g, A_DIM).mean(1, keepdim=True).expand(ng, g, A_DIM)
        base = base.reshape(ng, g * A_DIM).clone()
        B = torch.zeros(k, g * A_DIM, device=dev)
        torch.nn.init.normal_(B, std=1.0 / (g * A_DIM) ** 0.5)
        C = torch.zeros(ng, k, device=dev)
        B.requires_grad_(True)
        C.requires_grad_(True)
        opt = torch.optim.Adam([B, C], lr=a.lr)
        sd_g = sd.repeat(1, g).reshape(1, g * A_DIM)
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            attr = (base + (C @ B) * sd_g).reshape(-1, A_DIM)
            loss = 0.0
            for r, c in zip(fit_ref, fit_cams):
                l, _, _ = photometric_loss(render_gaussians(*gaussians(xyz_k, attr), c), r, 0.2)
                loss = loss + l / len(fit_cams)
            loss.backward()
            opt.step()
        with torch.no_grad():
            attr = (base + (C @ B) * sd_g).reshape(-1, A_DIM)
        report(f"code {k:4d}/group = {k/g:.2f} ch/point", attr)

    print("\nz_compact 예산: 0.5 ch/point 를 기하와 appearance 가 나눠 씁니다.")
    print("기하 rank 28 = 0.44 ch/point 이므로 appearance 에 남는 것은 0.06 ch/point 뿐입니다.")


if __name__ == "__main__":
    main()
