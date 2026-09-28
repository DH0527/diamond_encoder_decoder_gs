"""Which attribute target should the decoder be trained on?

The parameter loss currently pairs slot *i* of the prediction with slot *i* of
the target. Measured, that pairing is wrong almost always: only 7.0% of predicted
points have their slot partner as their nearest GT point, and the slot target
differs from the nearest one by 58-80% of the attributes' own standard deviation.
So for 93% of the points the loss is teaching the colour and opacity of some
other Gaussian.

Three candidate targets, rendered so they can be compared in the units that
matter:

  slot          what the loss uses today
  nearest       the attributes of the GT point the prediction actually sits on
  responsibility the aggregate of every GT point this prediction is the closest
                one to -- a Voronoi cell. This is the one that matches the goal
                "fewer Gaussians should still represent the original": if a
                prediction is the only point standing in for four GT Gaussians,
                it should take their combined colour and grow to cover their
                spread, not copy one of them.

Scale under `responsibility` is the union extent, not the mean: a Gaussian
replacing several has to cover the ground they covered, and their mean scale
would leave holes. Opacity is composited the way alpha actually composites,
1 - prod(1 - a), rather than averaged.

Reported against the same reference the training render loss uses -- a render of
the original Gaussians -- and on held-out poses as well, since a target that only
works from the capture camera is not a target.
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
                            render_gaussians, sh_dc_to_rgb, ssim)
from can3tok.train import build_config, make_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--test_views", type=int, default=3)
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
    gt_xyz = t[:, 0:3].reshape(ng, g, 3)
    gt_att = t[:, 3:14].reshape(ng, g, 11)

    cv = item["camera"].numpy()
    cams = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
    rng = np.random.default_rng(9999)
    held = [Camera(camera_from_vector(perturb_camera_vector(cv, rng)), device=dev,
                   downscale=a.downscale) for _ in range(a.test_views)]

    def gauss(xyz, att):
        return (xyz.reshape(-1, 3) * sc + cen,
                torch.exp(att.reshape(-1, 11)[:, 0:3].clamp(-15, 5)) * sc,
                att.reshape(-1, 11)[:, 3:7],
                torch.sigmoid(att.reshape(-1, 11)[:, 7:8]),
                sh_dc_to_rgb(att.reshape(-1, 11)[:, 8:11]))

    with torch.no_grad():
        ref = [render_gaussians(*gauss(gt_xyz, gt_att), c) for c in cams]
        ref_h = [render_gaussians(*gauss(gt_xyz, gt_att), c) for c in held]

    # ---- the three candidate targets -------------------------------------
    tgt = {"slot": gt_att.clone()}

    d = torch.cdist(pred, gt_xyz)                      # (ng, g_pred, g_gt)
    tgt["nearest"] = torch.gather(
        gt_att, 1, d.argmin(2).unsqueeze(-1).expand(-1, -1, 11))

    # responsibility: each GT point votes for its closest prediction
    owner = d.argmin(1)                                # (ng, g_gt) -> pred index
    oh = torch.zeros(ng, g, g, device=dev)
    oh.scatter_(1, owner.unsqueeze(1), 1.0)            # (ng, g_pred, g_gt)
    cnt = oh.sum(2, keepdim=True)                      # how many each prediction owns
    w = oh / cnt.clamp(min=1.0)
    resp = torch.bmm(w, gt_att)                        # mean of the owned attributes

    # opacity composites, it does not average
    al = torch.sigmoid(gt_att[..., 7:8]).squeeze(-1)
    keep = 1.0 - torch.bmm(oh, torch.log1p(-al.clamp(max=0.999)).unsqueeze(-1)).exp()
    resp[..., 7:8] = torch.logit(keep.clamp(1e-4, 1 - 1e-4))

    # scale must cover the ground the owned points covered, not their average
    own_c = torch.bmm(w, gt_xyz)
    spread = torch.bmm(w, gt_xyz.pow(2)) - own_c.pow(2)
    spread = spread.clamp(min=0).sqrt().mean(-1, keepdim=True)
    resp[..., 0:3] = torch.maximum(resp[..., 0:3], torch.log(spread.clamp(min=1e-6)))
    orphan = (cnt.squeeze(-1) < 0.5)                   # owns nothing -> keep nearest
    resp[orphan] = tgt["nearest"][orphan]
    tgt["responsibility"] = resp

    # Same thing with the opacity simply averaged instead of composited.
    # Compositing assumes the merged Gaussians were stacked along a view ray, so
    # that replacing them needs one that blocks as much light as all of them. But
    # "nearest in 3D" collects points lying side by side on a surface, and side by
    # side Gaussians tile the surface -- they cover different pixels and never
    # multiply their alphas. Coverage is already handled by the union-extent floor
    # on scale. If that reasoning is right this variant should render no worse.
    resp_mean = resp.clone()
    resp_mean[..., 7:8] = torch.bmm(w, gt_att)[..., 7:8]
    resp_mean[orphan] = tgt["nearest"][orphan]
    tgt["resp_mean_opacity"] = resp_mean

    print(f"scene {item['name']}  groups {ng}  points {ng*g}")
    print(f"예측점당 담당 GT 개수: 평균 {float(cnt.mean()):.2f}  "
          f"0개인 비율 {float(orphan.float().mean()):.1%}\n")
    print(f"{'attribute 타깃':>16} {'자기 카메라':>11} {'held-out 3뷰':>13} {'SSIM(held)':>11}")
    for name in ("slot", "nearest", "responsibility", "resp_mean_opacity"):
        with torch.no_grad():
            p0 = np.mean([psnr(r, render_gaussians(*gauss(pred, tgt[name]), c))
                          for r, c in zip(ref, cams)])
            ph = np.mean([psnr(r, render_gaussians(*gauss(pred, tgt[name]), c))
                          for r, c in zip(ref_h, held)])
            sh = np.mean([ssim(r, render_gaussians(*gauss(pred, tgt[name]), c))
                          for r, c in zip(ref_h, held)])
        print(f"{name:>16} {p0:11.2f} {ph:13.2f} {sh:11.3f}")

    with torch.no_grad():
        pg = np.mean([psnr(r, render_gaussians(*gauss(gt_xyz, gt_att), c))
                      for r, c in zip(ref_h, held)])
    print(f"\n참고: GT 위치 + GT attribute = {pg:.1f} dB (동일 이미지)")
    print("     'slot' 이 현재 파라미터 손실이 가르치는 목표입니다.")


if __name__ == "__main__":
    main()
