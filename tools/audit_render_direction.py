"""Does the render gradient actually point at the right answer for *positions*?

Magnitude was the previous question (``audit_render_scale.py``: the rasteriser
gradient is ~1000x a chamfer gradient per unit weight). Magnitude is not the whole
story. A term can be large and still be useless -- or harmful -- if it does not
point toward the correction.

The correction is known exactly here: the cloud is the target perturbed by known
noise, so ``target - pred`` is the ideal descent direction. So measure

    cos( -dL/dxyz , target - pred )

per point, for each loss, as a function of the perturbation size. A well-
conditioned geometry term stays near 1. A term that only sees the image should
degrade as soon as a point is displaced further than its own projected footprint,
because past that distance the gradient is the image gradient at the *wrong*
place and carries no information about where the point belongs.

Why this matters for the training schedule: vanilla 3DGS does optimise means by
image loss alone, but it relies on adaptive density control (clone / split /
prune) to fix the coverage errors gradient descent cannot reach. This design has a
fixed 262144-point budget and no densification, so it would be using the half of
3DGS's machinery that depends on the other half.
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

from can3tok.losses import intra_group_chamfer, multiscale_chamfer, point_group_residual_loss  # noqa: E402
from can3tok.losses import render_loss  # noqa: E402
from can3tok.render import Camera, render_gaussians, sh_dc_to_rgb  # noqa: E402
from can3tok.io_utils import camera_from_vector  # noqa: E402
from can3tok.train import make_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--sigmas", type=str, default="0.05,0.1,0.2,0.35,0.5,0.75")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--views", type=int, default=1,
                    help="isotropic noise puts ~1/3 of its energy along the view ray, which a "
                         "single projection cannot see at all -- so the single-view cosine is "
                         "capped near sqrt(2/3)=0.816. Raise this to separate that ceiling "
                         "from genuine ill-conditioning")
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    item = val_ds[a.scene]
    dev, g = "cuda", int(targs.group_size)

    tgt_full = item["target"].unsqueeze(0).to(dev).float()
    mask = item["mask"].unsqueeze(0).to(dev).float()
    txyz = tgt_full[..., 0:3]
    m = mask[0] > 0.5
    idx_live = torch.nonzero(m, as_tuple=False).reshape(-1)
    ng = int(m.sum()) // g
    idx = idx_live[: ng * g]
    tg = txyz[0][idx].reshape(ng, g, 3)
    rad = (tg - tg.mean(1, keepdim=True)).norm(dim=-1).mean(-1).clamp(min=1e-9)

    scales = [int(s) for s in str(targs.chamfer_scales).split(",") if s]
    sw = [float(s) for s in str(targs.chamfer_scale_weights).split(",") if s]

    # For reference: how big is a Gaussian's projected footprint, in the same
    # units as the perturbation? That is the distance past which an image-space
    # gradient stops knowing which way to push.
    cam = Camera(camera_from_vector(item["camera"].numpy()), device=dev, downscale=a.downscale)
    sc = float(item["scale"])
    w_xyz = txyz[0][idx_live] * sc + item["center"].to(dev).reshape(1, 3)
    depth = (w_xyz @ cam.world_view_transform[:3, :3]) + cam.world_view_transform[3, :3]
    depth = depth[:, 2].clamp(min=1e-3)
    gsz_world = torch.exp(tgt_full[0][idx_live, 3:6]).mean(-1) * sc
    px = gsz_world / depth * (cam.image_width / (2.0 * cam.tanfovx))
    print(f"scene {item['name']}  n={int(m.sum())}  render {cam.image_width}x{cam.image_height}")
    print(f"projected Gaussian radius: median {float(px.median()):.2f} px, "
          f"p90 {float(px.quantile(0.9)):.2f} px")
    med_rad_px = float((rad.median() * sc / depth.median()
                        * (cam.image_width / (2.0 * cam.tanfovx))))
    print(f"one group radius on screen: ~{med_rad_px:.1f} px "
          f"(so a 0.5-radius error is ~{0.5*med_rad_px:.1f} px)\n")

    print(f"{'sigma':>6} {'err px':>7}  " + "  ".join(f"{n:>16}" for n in
          (f"render(v{a.views})", "chamfer", "intra_chamfer", "xyz_residual")))

    for s in [float(x) for x in a.sigmas.split(",") if x]:
        gen = torch.Generator(device=dev).manual_seed(0)
        noise = torch.randn(tg.shape, generator=gen, device=dev) * (s * rad[:, None, None])
        pxyz = txyz.clone()
        pxyz[0, idx] = (tg + noise).reshape(-1, 3)
        pxyz = pxyz.detach().requires_grad_(True)
        want = (txyz - pxyz.detach())[0][idx]                    # ideal correction
        want_n = want / want.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        def cos_of(loss):
            if pxyz.grad is not None:
                pxyz.grad = None
            loss.backward()
            gr = -pxyz.grad[0][idx]
            n = gr.norm(dim=-1, keepdim=True)
            ok = (n > 1e-20).reshape(-1)
            if int(ok.sum()) == 0:
                return float("nan"), 0.0
            c = ((gr / n.clamp(min=1e-20)) * want_n).sum(-1)[ok]
            return float(c.mean()), float(ok.float().mean())

        row = []
        r = render_loss(pxyz, tgt_full, mask, item["camera"].unsqueeze(0),
                        item["center"].unsqueeze(0), item["scale"].reshape(1),
                        {"xyz": 0, "scale": 3, "rot": 6, "opacity": 10, "color": 11},
                        lam_dssim=float(getattr(targs, "render_lam_dssim", 0.2)),
                        downscale=a.downscale, views=a.views, min_coverage=0.0)
        row.append(cos_of(r["render"]))
        row.append(cos_of(multiscale_chamfer(pxyz, txyz, mask, scales, sw,
                                            balanced=bool(targs.balanced_chamfer))))
        row.append(cos_of(intra_group_chamfer(pxyz, txyz, mask, g,
                                             chunk_groups=int(targs.intra_chamfer_chunk))))
        row.append(cos_of(point_group_residual_loss(pxyz, txyz, mask, g)["xyz_residual"]))

        err_px = s * med_rad_px
        cells = "  ".join(f"{c:8.3f} @{f*100:4.0f}%" for c, f in row)
        print(f"{s:6.2f} {err_px:7.1f}  {cells}")

    print("\ncos = mean cosine between -dL/dxyz and (target - pred), over points that")
    print("received any gradient; @ = fraction of points that received one.")


if __name__ == "__main__":
    main()
