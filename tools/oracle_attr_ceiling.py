"""How much is left on the table, and where.

`oracle_attr_target.py` says the best closed-form attribute target for the current
positions renders at 16.91 dB, and that 62.6% of predicted Gaussians are the
nearest point to no GT point at all. Two very different problems are hiding in
that gap, and they call for different fixes:

  ceiling   attributes fitted directly by gradient descent against the render,
            from the responsibility target as the starting point. This is the
            most any attribute decoder could do with *these* positions, so it
            bounds what raising the render weight can buy.

  spread    GT positions thinned to the fraction of predictions that actually
            own territory, with the responsibility aggregate as their attributes.
            Same effective point count as the prediction, but placed well. If
            this lands far above the ceiling, the loss is in the geometry's
            clumping and no attribute work will recover it.

The fit uses the training views and reports held-out ones, because an attribute
set that only works from the camera it was fitted on tells us nothing about what
the decoder should learn.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.io_utils import camera_from_vector  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.render import (Camera, perturb_camera_vector, psnr,  # noqa: E402
                            render_gaussians, sh_dc_to_rgb)
from can3tok.train import build_config, make_datasets  # noqa: E402


def responsibility(pred, gt_xyz, gt_att):
    """Aggregate onto each prediction the GT points it is the closest one to."""
    ng, g, _ = pred.shape                       # g predictions, gt_xyz may hold more
    d = torch.cdist(pred, gt_xyz)
    near = torch.gather(gt_att, 1, d.argmin(2).unsqueeze(-1).expand(-1, -1, 11))
    oh = torch.zeros(ng, g, gt_xyz.shape[1], device=pred.device)
    oh.scatter_(1, d.argmin(1).unsqueeze(1), 1.0)
    cnt = oh.sum(2, keepdim=True)
    w = oh / cnt.clamp(min=1.0)
    r = torch.bmm(w, gt_att)
    al = torch.sigmoid(gt_att[..., 7:8]).squeeze(-1)
    keep = 1.0 - torch.bmm(oh, torch.log1p(-al.clamp(max=0.999)).unsqueeze(-1)).exp()
    r[..., 7:8] = torch.logit(keep.clamp(1e-4, 1 - 1e-4))
    own_c = torch.bmm(w, gt_xyz)
    sp = (torch.bmm(w, gt_xyz.pow(2)) - own_c.pow(2)).clamp(min=0).sqrt().mean(-1, keepdim=True)
    r[..., 0:3] = torch.maximum(r[..., 0:3], torch.log(sp.clamp(min=1e-6)))
    orph = cnt.squeeze(-1) < 0.5
    r[orph] = near[orph]
    return r, orph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--fit_views", type=int, default=4)
    ap.add_argument("--test_views", type=int, default=3)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs = Namespace(**ck["args"])
    targs.out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    train_ds, val_ds = make_datasets(targs)
    cfg = build_config(targs, train_ds.target_dim, train_ds.sh_dim)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.cuda().eval()
    g, dev = int(cfg.group_size), "cuda"

    item = val_ds[a.scene]
    m = item["mask"].to(dev) > 0.5
    t = item["target"].to(dev).float()[m]
    ng = t.shape[0] // g
    t = t[: ng * g]
    sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
    with torch.no_grad():
        out = model(item["input"].unsqueeze(0).to(dev).float(),
                    item["mask"].unsqueeze(0).to(dev).float(),
                    run_decode=True, run_gen=False)
    pred = out["pred"][0][m][: ng * g, 0:3].float().reshape(ng, g, 3)
    gt_xyz, gt_att = t[:, 0:3].reshape(ng, g, 3), t[:, 3:14].reshape(ng, g, 11)

    cv = item["camera"].numpy()
    rng = np.random.default_rng(4242)
    fit = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)] + [
        Camera(camera_from_vector(perturb_camera_vector(cv, rng)), device=dev,
               downscale=a.downscale) for _ in range(a.fit_views - 1)]
    rng = np.random.default_rng(9999)
    held = [Camera(camera_from_vector(perturb_camera_vector(cv, rng)), device=dev,
                   downscale=a.downscale) for _ in range(a.test_views)]

    def gauss(xyz, att):
        x, at = xyz.reshape(-1, 3), att.reshape(-1, 11)
        return (x * sc + cen, torch.exp(at[:, 0:3].clamp(-15, 5)) * sc,
                torch.nn.functional.normalize(at[:, 3:7], dim=-1),
                torch.sigmoid(at[:, 7:8]), sh_dc_to_rgb(at[:, 8:11]))

    with torch.no_grad():
        ref_f = [render_gaussians(*gauss(gt_xyz, gt_att), c) for c in fit]
        ref_h = [render_gaussians(*gauss(gt_xyz, gt_att), c) for c in held]

    def ev(xyz, att):
        with torch.no_grad():
            return float(np.mean([psnr(r, render_gaussians(*gauss(xyz, att), c))
                                  for r, c in zip(ref_h, held)]))

    resp, orph = responsibility(pred, gt_xyz, gt_att)
    own = float((~orph).float().mean())
    print(f"scene {item['name']}  points {ng*g}  영역을 가진 예측점 {own:.1%}\n")
    print(f"{'설정':>34} {'held-out 3뷰':>13}")
    print(f"{'예측 위치 + responsibility 타깃':>34} {ev(pred, resp):13.2f}")

    # ---- ceiling: fit the attributes themselves against the render -------
    p = resp.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.02)
    for i in range(a.steps):
        c = fit[i % len(fit)]
        loss = (render_gaussians(*gauss(pred, p), c) - ref_f[i % len(fit)]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"{'예측 위치 + 렌더로 직접 최적화 (상한)':>34} {ev(pred, p.detach()):13.2f}")

    # ---- spread: same effective count, but well placed -------------------
    k = max(1, int(round(own * g)))
    idx = torch.argsort(torch.rand(ng, g, device=dev), dim=1)[:, :k]
    sx = torch.gather(gt_xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    sa, _ = responsibility(sx, gt_xyz, gt_att)
    print(f"{f'GT 위치 {k*ng/1000:.0f}k개만 ({own:.0%}) + 통합':>34} {ev(sx, sa):13.2f}")
    with torch.no_grad():
        print(f"{'GT 위치 전부 + GT attribute':>34} {ev(gt_xyz, gt_att):13.2f}")


if __name__ == "__main__":
    main()
