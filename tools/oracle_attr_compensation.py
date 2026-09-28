"""Can attributes rescue a rank-limited point set?

`oracle_shape_rank.py` established that the geometry is budget-bound, not
loss-bound: 28 shape channels must describe 192 numbers (64 points x 3), the
linear ceiling at rank 28 is nn_unique 0.529 / ich 0.236 / 17.4 dB, and the model
already sits there. No reweighting moves that, because channels-per-point is
131072/262144 = 0.5 and is fixed by the two constraints.

So the question changes. If the GT point *set* cannot be reproduced at this
budget, must the render also be stuck at 17 dB? Not necessarily -- 3DGS is a
representation, not a measurement. A different set of Gaussians that renders the
same is equally correct, and the parameters that control how much of the image a
Gaussian covers are scale, rotation and opacity. A hole left by a missing point
can be filled by a neighbour that is larger and better oriented.

Those are exactly the parameters an image loss is *well* conditioned for: they
change the same pixels the Gaussian already occupies, so no transport is needed
(the ill-conditioning measured in audit_render_direction.py is specific to
positions).

This measures the ceiling of that idea, with no model in the loop: freeze the
rank-k positions, then optimise only the attributes against the reference render.
If PSNR recovers most of the way to the canonical 27.8 dB, then geometry accuracy
at 0.5 channels/point is not what caps render quality, and the attribute stage is
the main path rather than a follow-up. If it barely moves, the budget caps the
render too and only reducing K can help.
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
from can3tok.render import Camera, photometric_loss, psnr, render_gaussians, sh_dc_to_rgb, ssim  # noqa: E402
from can3tok.train import make_datasets  # noqa: E402


def rank_k_positions(grp: torch.Tensor, k: int):
    """Group offsets projected onto their own leading k principal directions."""
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
    ap.add_argument("--rank", type=int, default=28, help="the model's shape-channel count")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--free", type=str, default="scale,opacity",
                    help="which attributes may adapt: any of scale,rot,opacity,color")
    ap.add_argument("--fit_views", type=int, default=1,
                    help="views the attributes are fitted on")
    ap.add_argument("--test_views", type=int, default=3,
                    help="HELD-OUT views, never seen during fitting. Without these the number "
                         "is meaningless: attributes fitted to one projection can hide geometry "
                         "error that any other pose exposes")
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    item = val_ds[a.scene]
    dev, g = "cuda", int(targs.group_size)

    m = item["mask"].to(dev) > 0.5
    t = item["target"].to(dev).float()[m]
    ng = t.shape[0] // g
    t = t[: ng * g]
    sc = float(item["scale"])
    cen = item["center"].to(dev).reshape(1, 3)
    cam = Camera(camera_from_vector(item["camera"].numpy()), device=dev, downscale=a.downscale)

    gt_xyz = t[:, 0:3].reshape(ng, g, 3)
    # log scale, quaternion, logit opacity, SH DC -- the stored parameterisation,
    # which is also the one a decoder head would emit.
    p0 = {"scale": t[:, 3:6].clone(), "rot": t[:, 6:10].clone(),
          "opacity": t[:, 10:11].clone(), "color": t[:, 11:14].clone()}

    def gaussians(xyz, p):
        return (xyz * sc + cen, torch.exp(p["scale"].clamp(-15, 5)) * sc, p["rot"],
                torch.sigmoid(p["opacity"]), sh_dc_to_rgb(p["color"]))

    from can3tok.render import perturb_camera_vector

    cv = item["camera"].numpy()
    fit_cams = [cam]
    rng = np.random.default_rng(0)
    for _ in range(max(a.fit_views - 1, 0)):
        fit_cams.append(Camera(camera_from_vector(perturb_camera_vector(cv, rng)), device=dev,
                               downscale=a.downscale))
    # Held-out poses drawn from a *different* stream, never optimised against.
    rng_t = np.random.default_rng(9999)
    test_cams = [Camera(camera_from_vector(perturb_camera_vector(cv, rng_t)), device=dev,
                        downscale=a.downscale) for _ in range(a.test_views)]

    xyz_k = rank_k_positions(gt_xyz, a.rank).detach()
    gt_flat = gt_xyz.reshape(-1, 3)

    def refs(cams):
        with torch.no_grad():
            return [render_gaussians(*gaussians(gt_flat, p0), c) for c in cams]

    fit_ref, test_ref = refs(fit_cams), refs(test_cams)

    def report(label, p_):
        with torch.no_grad():
            f = np.mean([psnr(r, render_gaussians(*gaussians(xyz_k, p_), c))
                         for r, c in zip(fit_ref, fit_cams)])
            t = np.mean([psnr(r, render_gaussians(*gaussians(xyz_k, p_), c))
                         for r, c in zip(test_ref, test_cams)])
            s = np.mean([ssim(r, render_gaussians(*gaussians(xyz_k, p_), c))
                         for r, c in zip(test_ref, test_cams)])
        print(f"  {label:42s} fit {f:6.2f} dB | held-out {t:6.2f} dB  SSIM {s:.3f}")

    print(f"scene {item['name']}  n={xyz_k.shape[0]}  rank {a.rank}  free: {a.free}")
    print(f"  fitted on {len(fit_cams)} view(s), evaluated on {len(test_cams)} unseen view(s)\n")
    report(f"rank-{a.rank} positions, GT attributes", p0)

    free = [f for f in a.free.split(",") if f]
    p = {k: (v.clone().requires_grad_(k in free)) for k, v in p0.items()}
    opt = torch.optim.Adam([p[k] for k in free], lr=a.lr)
    for it in range(a.steps):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for r, c in zip(fit_ref, fit_cams):
            l, _, _ = photometric_loss(render_gaussians(*gaussians(xyz_k, p), c), r, 0.2)
            loss = loss + l / len(fit_cams)
        loss.backward()
        opt.step()

    report(f"rank-{a.rank} positions, ADAPTED attributes", p)
    for k in free:
        d = (p[k] - p0[k]).abs()
        print(f"    {k:8s} moved by  mean {float(d.mean()):.4f}  p99 {float(d.quantile(0.99)):.4f}")


if __name__ == "__main__":
    main()
