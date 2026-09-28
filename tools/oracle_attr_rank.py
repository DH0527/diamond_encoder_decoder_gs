"""How many channels per point do the attributes actually need?

The geometry answer came from projecting group offsets onto their own leading k
principal directions and reading off ich / nn_unique / PSNR (`oracle_shape_rank`).
This asks the same question of the other half of a Gaussian, so that the channel
split between geometry and appearance is decided by a number rather than by
whatever slot happened to be free.

Per group: 64 points x 11 attribute channels (log_scale 3, quaternion 4, logit
opacity 1, SH DC 3) = 704 numbers. Project onto rank k, put the result back on the
**true** xyz, and render. Everything except the attributes is exact, so the dB
drop is caused by the attribute rank alone.

Reading the result:

    rank k  ->  k / 64 channels per point in this group's attribute code

so k = 64 is "1 channel per point", k = 128 is 2, and so on. Compare against
z_compact's total budget of 0.5 channels per point for geometry *and* appearance
together -- whatever the attributes get has to come out of the 28 shape channels.

Channels are standardised before the projection because they live on wildly
different scales (a quaternion component and a log scale are not comparable), and
un-standardised afterwards.
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
from can3tok.render import Camera, psnr, render_gaussians, sh_dc_to_rgb, ssim  # noqa: E402
from can3tok.train import make_datasets  # noqa: E402

ATTR = slice(3, 14)          # log_scale 3 | quat 4 | logit opacity 1 | SH DC 3
A_DIM = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--ranks", type=str, default="8,16,32,64,128,256,512,704")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--views", type=int, default=4,
                    help="held-out-style extra poses, so a per-view fit cannot flatter the result")
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
    xyz = t[:, 0:3] * sc + cen

    cams = [Camera(camera_from_vector(item["camera"].numpy()), device=dev, downscale=a.downscale)]
    if a.views > 1:
        from can3tok.render import perturb_camera_vector
        rng = np.random.default_rng(0)
        cams += [Camera(camera_from_vector(perturb_camera_vector(item["camera"].numpy(), rng)),
                        device=dev, downscale=a.downscale) for _ in range(a.views - 1)]

    def gaussians(attr):
        return (xyz, torch.exp(attr[:, 0:3].clamp(-15, 5)) * sc, attr[:, 3:7],
                torch.sigmoid(attr[:, 7:8]), sh_dc_to_rgb(attr[:, 8:11]))

    A = t[:, ATTR]                                    # (ng*g, 11)
    with torch.no_grad():
        refs = [render_gaussians(*gaussians(A), c) for c in cams]

    # Standardise per channel, then treat each group's 64x11 block as one vector.
    mu_c, sd_c = A.mean(0, keepdim=True), A.std(0, keepdim=True).clamp(min=1e-6)
    X = ((A - mu_c) / sd_c).reshape(ng, g * A_DIM).double()
    mu = X.mean(0, keepdim=True)
    _, sv, vh = torch.linalg.svd(X - mu, full_matrices=False)
    energy = (sv ** 2) / (sv ** 2).sum()

    print(f"scene {item['name']}  groups {ng}  attribute dim/group {g*A_DIM}  "
          f"views {len(cams)}")
    print(f"z_compact 전체 예산: 0.5 채널/점 (기하 + appearance 합쳐서)\n")
    print(f"{'rank k':>7} {'ch/point':>9} {'var kept':>9} {'PSNR':>7} {'SSIM':>6}   "
          f"{'scale':>7} {'rot':>7} {'opac':>7} {'color':>7}   (채널별 상대오차)")

    for k in [int(r) for r in a.ranks.split(",") if r]:
        k = min(k, vh.shape[0])
        b = vh[:k]
        rec = ((X - mu) @ b.T @ b + mu).reshape(ng * g, A_DIM).float() * sd_c + mu_c
        with torch.no_grad():
            imgs = [render_gaussians(*gaussians(rec), c) for c in cams]
            p = float(np.mean([psnr(r, i) for r, i in zip(refs, imgs)]))
            s = float(np.mean([ssim(r, i) for r, i in zip(refs, imgs)]))
        err = ((rec - A).abs().mean(0) / A.abs().mean(0).clamp(min=1e-6))
        e_sc, e_rt = float(err[0:3].mean()), float(err[3:7].mean())
        e_op, e_co = float(err[7:8].mean()), float(err[8:11].mean())
        print(f"{k:7d} {k/g:9.2f} {float(energy[:k].sum())*100:8.1f}% {p:7.2f} {s:6.3f}   "
              f"{e_sc:7.3f} {e_rt:7.3f} {e_op:7.3f} {e_co:7.3f}")


if __name__ == "__main__":
    main()
