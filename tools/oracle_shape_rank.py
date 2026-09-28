"""How many distinct group shapes does a rank-k code buy, and what nn_unique?

Two hypotheses explain nn_unique 0.50, and they call for different fixes:

  H1 (supervision)  the decoder output carries no one-to-one term at all
                    (w_xyz / w_xyz_mse / w_xyz_hard / w_xyz_residual are 0), so
                    nothing holds slot identity after the unpack. Fix: turn the
                    1:1 term back on -- measured the best-conditioned signal
                    available (cos 0.866 with the correction, at 100% coverage).

  H2 (capacity)     the 28 shape channels carry effective rank ~11.8, the
                    shape->offset map emits ~7 distinct group shapes
                    (template_erank_pred 7.1 against 22.5 in the GT), and no loss
                    can make a rank-11 code express 22 shapes. Fix: change the
                    decomposition or the channel budget, not the weights.

They are separable without training. Take the template-aligned target offsets --
the exact thing the decoder is asked to produce -- project them onto their own
leading k principal directions, and measure what comes back. That is the best a
*perfect* linear decoder from a rank-k code could do, so it upper-bounds the
budget's contribution and lower-bounds the blame that belongs to the loss.

If a rank-12 projection already reaches nn_unique ~0.9, capacity is not the
binding constraint and H1 owns the gap. If it saturates near 0.5, H2 does and the
next run is aimed at the wrong thing.
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

from can3tok.train import make_datasets  # noqa: E402


def measure(pred_off: torch.Tensor, true_off: torch.Tensor, chunk: int = 512):
    """nn_unique, intra_chamfer and shape effective rank, as eval_utils computes them."""
    rad = true_off.norm(dim=-1).mean(-1).clamp(min=1e-9)
    uniq, ich = [], []
    for i in range(0, pred_off.shape[0], chunk):
        d = torch.cdist(pred_off[i : i + chunk], true_off[i : i + chunk])
        nn = d.argmin(dim=2)
        hit = torch.zeros_like(nn, dtype=torch.bool).scatter_(1, nn, True)
        uniq.append(hit.float().mean(1))
        ich.append(0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)) / rad[i : i + chunk])

    def erank(o):
        s = o.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-9)
        x = (o / s).reshape(o.shape[0], -1)
        x = x - x.mean(0, keepdim=True)
        ev = torch.linalg.eigvalsh((x.T @ x / max(x.shape[0] - 1, 1)).double()).clamp(min=0)
        p = ev / ev.sum().clamp(min=1e-30)
        p = p[p > 0]
        return float(torch.exp(-(p * p.log()).sum()))

    return float(torch.cat(uniq).mean()), float(torch.cat(ich).median()), erank(pred_off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--ranks", type=str, default="4,7,12,16,22,28,40,64,192")
    ap.add_argument("--render", action="store_true",
                    help="also render each rank-k reconstruction with the GT attributes, which "
                         "converts the rank axis into dB and says how much of the 17.2 dB is "
                         "bound by the per-group budget rather than by the loss")
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    item = val_ds[a.scene]
    g = int(targs.group_size)
    dev = "cuda"

    m = item["mask"].to(dev) > 0.5
    xyz = item["target"].to(dev).float()[m][:, 0:3]
    ng = xyz.shape[0] // g
    grp = xyz[: ng * g].reshape(ng, g, 3)
    off = grp - grp.mean(1, keepdim=True)
    # Normalise each group by its own radius: the decoder's `local` is built in that
    # frame (the analytic scale anchor carries the size), so the shape code only
    # ever has to express the *unit* shape.
    rad = off.norm(dim=-1).mean(-1).clamp(min=1e-9)
    unit = off / rad[:, None, None]

    x = unit.reshape(ng, g * 3).double()
    mu = x.mean(0, keepdim=True)
    xc = x - mu
    # Right singular vectors = the shape basis a linear decoder would learn.
    _, sv, vh = torch.linalg.svd(xc, full_matrices=False)
    energy = (sv ** 2) / (sv ** 2).sum()

    u_true, i_true, e_true = measure(unit, unit)
    print(f"scene {item['name']}  groups {ng}  group_size {g}  offset dim {g*3}")
    print(f"GT control: nn_unique {u_true:.3f}  ich {i_true:.4f}  shape_erank {e_true:.1f}")
    print(f"\nmodel today (step 2000, same metric): nn_unique 0.51  ich 0.251  "
          f"template_erank 7.1\n")
    cam = ref = gt_attr = None
    if a.render:
        from can3tok.render import Camera, render_gaussians, sh_dc_to_rgb, psnr, ssim
        from can3tok.io_utils import camera_from_vector

        cam = Camera(camera_from_vector(item["camera"].numpy()), device=dev, downscale=a.downscale)
        t = item["target"].to(dev).float()[m][: ng * g]
        sc = float(item["scale"])
        cen = item["center"].to(dev).reshape(1, 3)
        gt_attr = (torch.exp(t[:, 3:6]) * sc, t[:, 6:10],
                   torch.sigmoid(t[:, 10:11]), sh_dc_to_rgb(t[:, 11:14]))
        with torch.no_grad():
            ref = render_gaussians(grp.reshape(-1, 3) * sc + cen, *gt_attr, cam)

    head = f"{'rank k':>7} {'var kept':>9} {'shape_erank':>12} {'ich':>8} {'nn_unique':>10}"
    if a.render:
        head += f" {'PSNR':>7} {'SSIM':>6}"
    print(head)
    for k in [int(r) for r in a.ranks.split(",") if r]:
        k = min(k, vh.shape[0])
        basis = vh[:k]
        rec = (xc @ basis.T @ basis + mu).reshape(ng, g, 3).float()
        rec = rec - rec.mean(1, keepdim=True)
        u, i, e = measure(rec, unit)
        line = f"{k:7d} {float(energy[:k].sum())*100:8.1f}% {e:12.1f} {i:8.4f} {u:10.3f}"
        if a.render:
            # Put the offsets back on the true centroids at the true radii, so the
            # only thing degraded is the within-group arrangement.
            xyz_k = (rec * rad[:, None, None] + grp.mean(1, keepdim=True)).reshape(-1, 3)
            with torch.no_grad():
                img = render_gaussians(xyz_k * sc + cen, *gt_attr, cam)
            line += f" {psnr(ref, img):7.2f} {ssim(ref, img):6.3f}"
        print(line)


if __name__ == "__main__":
    main()
