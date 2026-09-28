"""Is w_render the right size next to the point-space terms?

A raw weight says nothing about balance when one term comes from a rasteriser and
the others from distances between points -- they have no reason to arrive at the
same scale, and the launcher's `--w_render 6` was a guess.

This measures ``|dL/dxyz|`` for each term on the *same* perturbed point cloud, at
the real 262k point count, with no model in the loop. Two reasons that is the
right comparison rather than the parameter-space audit in ``audit_losses.py``:

* it needs only the rasteriser and a chamfer, so it fits in memory beside a
  running job, where the parameter audit does not;
* the render gradient per point depends on how many Gaussians land on each pixel,
  so a reduced-point-count measurement would not transfer. 262k is the number
  that matters and 262k is what this uses.

The perturbation stands in for the decoder's error. It is isotropic noise at a
chosen multiple of the group radius, which is *not* what the decoder actually
produces (its error is more clustered -- that is the whole finding), so read the
ratios as an order of magnitude, not a calibration.
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
from can3tok.train import make_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True, help="a run's args.json, for the loader config")
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="perturbation as a multiple of the group radius; 0.5 matches the "
                         "measured rel_offset_p50 of 0.499")
    ap.add_argument("--views", type=int, default=1)
    ap.add_argument("--downscale", type=int, default=2)
    # The run whose args.json supplies the loader config may predate the terms
    # being weighed, so the weights are overridable independently of it.
    ap.add_argument("--w_render", type=float, default=None)
    ap.add_argument("--w_xyz_residual", type=float, default=None)
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    for k in ("w_render", "w_xyz_residual"):
        if getattr(a, k) is not None:
            setattr(targs, k, float(getattr(a, k)))
    _, val_ds = make_datasets(targs)
    item = val_ds[a.scene]
    dev = "cuda"
    g = int(targs.group_size)

    tgt_full = item["target"].unsqueeze(0).to(dev).float()
    mask = item["mask"].unsqueeze(0).to(dev).float()
    txyz = tgt_full[..., 0:3]

    # Perturb inside each group by sigma * that group's radius, so the error is
    # scale-appropriate everywhere rather than uniform in absolute units.
    m = mask[0] > 0.5
    ng = int(m.sum()) // g
    tg = txyz[0][m][: ng * g].reshape(ng, g, 3)
    rad = (tg - tg.mean(1, keepdim=True)).norm(dim=-1).mean(-1).clamp(min=1e-9)
    gen = torch.Generator(device=dev).manual_seed(0)
    noise = torch.randn(tg.shape, generator=gen, device=dev) * (a.sigma * rad[:, None, None])
    pxyz = txyz.clone()
    idx = torch.nonzero(m, as_tuple=False).reshape(-1)[: ng * g]
    pxyz[0, idx] = (tg + noise).reshape(-1, 3)
    pxyz = pxyz.detach().requires_grad_(True)

    weights = {"w_chamfer": float(targs.w_chamfer), "chamfer_scales": targs.chamfer_scales,
               "chamfer_scale_weights": targs.chamfer_scale_weights}
    scales = [int(s) for s in str(targs.chamfer_scales).split(",") if s]
    sw = [float(s) for s in str(targs.chamfer_scale_weights).split(",") if s]

    def gnorm(loss):
        if pxyz.grad is not None:
            pxyz.grad = None
        loss.backward()
        gr = pxyz.grad
        live = gr[0][m]
        return float(gr.norm()), float(live.norm(dim=-1).mean())

    rows = []

    w = float(targs.w_chamfer)
    if w:
        l = multiscale_chamfer(pxyz, txyz, mask, scales, sw, balanced=bool(targs.balanced_chamfer))
        rows.append(("w_chamfer", w, *gnorm(w * l)))

    w = float(targs.w_intra_chamfer)
    if w:
        l = intra_group_chamfer(pxyz, txyz, mask, g, chunk_groups=int(targs.intra_chamfer_chunk))
        rows.append(("w_intra_chamfer", w, *gnorm(w * l)))

    w = float(targs.w_xyz_residual)
    if w:
        d = point_group_residual_loss(pxyz, txyz, mask, g)
        rows.append(("w_xyz_residual", w, *gnorm(w * d["xyz_residual"])))

    w = float(getattr(targs, "w_render", 0.0))
    if w:
        r = render_loss(pxyz, tgt_full, mask, item["camera"].unsqueeze(0),
                        item["center"].unsqueeze(0), item["scale"].reshape(1),
                        {"xyz": 0, "scale": 3, "rot": 6, "opacity": 10, "color": 11},
                        lam_dssim=float(getattr(targs, "render_lam_dssim", 0.2)),
                        downscale=a.downscale, views=a.views,
                        min_coverage=float(getattr(targs, "render_min_coverage", 0.25)))
        rows.append((f"w_render (views={a.views})", w, *gnorm(w * r["render"])))

    print(f"scene {item['name']}  n={int(m.sum())}  groups={ng}  "
          f"perturbation = {a.sigma:.2f} x group radius\n")
    print(f"{'term':26s} {'weight':>8s} {'|w*dL/dxyz|':>13s} {'per-point':>11s} {'share':>7s}")
    tot = sum(r[2] for r in rows)
    for name, w, n, per in sorted(rows, key=lambda r: -r[2]):
        print(f"{name:26s} {w:8.2f} {n:13.4e} {per:11.3e} {n/max(tot,1e-12):6.1%}")


if __name__ == "__main__":
    main()
