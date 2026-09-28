"""How many Gaussians does the render loss actually reach, and what sets that?

The attribute render covers a measured 34.9% of points at 4 views, and the
remaining 65% depend entirely on the parameter anchor -- which saturates 10 dB
below what the render reaches and, being an L1 to a per-point target, collapses
towards the conditional mean. That split is usually presented as a property of
rendering. It is not. It is a property of *which cameras* we render from.

Training orbits the frame's own camera by +-8 degrees (`perturb_camera_vector`,
`max_deg=8.0`). 3DGS optimises against a hundred-plus views spanning the scene,
and the feed-forward reconstruction models supervise on novel views far from the
input. Nothing forces the narrow orbit here: the reference is not a photograph,
it is `render(target Gaussians)`, which can be produced from any pose at all --
the function's own docstring says so.

So this sweeps the orbit width and counts, per point, whether the render gives it
any gradient. Also reported:

  covered      fraction of Gaussians receiving non-zero gradient
  frame        fraction of pixels the reference render is non-empty in -- a wide
               orbit that puts the scene out of frame buys coverage that is not
               real, and this is what would show it
  psnr(ref)    the reference render's own contrast against a grey image, as a
               second check that the view still contains something

If coverage rises steeply with angle, the 65% is an artefact of the camera
sampling and the right fix is more diverse views, not a better anchor.
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
from can3tok.render import (Camera, perturb_camera_vector,  # noqa: E402
                            render_gaussians, sh_dc_to_rgb)
from can3tok.train import build_config, make_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", type=int, default=160)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--angles", type=str, default="8,20,45,90,180")
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = Namespace(**ck["args"])
    args.out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    tr, val = make_datasets(args)
    cfg = build_config(args, tr.target_dim, tr.sh_dim)
    model = build_model(cfg)
    cur = model.state_dict()
    model.load_state_dict({k: v for k, v in ck["model"].items()
                           if k in cur and cur[k].shape == v.shape}, strict=False)
    model.cuda().eval()
    g, dev = int(cfg.group_size), "cuda"

    item = val[a.scene]
    m = item["mask"].to(dev) > 0.5
    t = item["target"].to(dev).float()[m]
    n = (t.shape[0] // g) * g
    t = t[:n]
    sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
    with torch.no_grad():
        out = model(item["input"].unsqueeze(0).to(dev).float(),
                    item["mask"].unsqueeze(0).to(dev).float(),
                    run_decode=True, run_gen=False)
    xyz = out["pred"][0][m][:n, 0:3].detach().float()
    A_gt = t[:, 3:14]
    # The gradient has to be taken at the PREDICTION, not at the ground truth.
    # Rendering A_gt against a reference that is also A_gt gives a residual of
    # exactly zero, and |0| has zero subgradient in torch, so every point reports
    # "no gradient" regardless of the camera -- which is what the first version of
    # this tool measured.
    A_pred = out.get("attr_pred")
    if A_pred is None:
        raise SystemExit("checkpoint has no attribute decoder")
    A_pred = A_pred[0][m][:n, 3:14].detach().float()

    def gaussians(x, at):
        return (x * sc + cen, torch.exp(at[:, 0:3].clamp(-15, 5)) * sc,
                torch.nn.functional.normalize(at[:, 3:7], dim=-1),
                torch.sigmoid(at[:, 7:8]), sh_dc_to_rgb(at[:, 8:11]))

    cv = item["camera"].numpy()
    print(f"scene {item['name']}  {n} points  {a.views} views per setting")
    print(f"{'orbit':>7} {'covered':>9} {'frame':>8} {'ref contrast':>13}")

    for deg in [float(x) for x in a.angles.split(",") if x]:
        rng = np.random.default_rng(1234)
        cams = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
        cams += [Camera(camera_from_vector(perturb_camera_vector(cv, rng, max_deg=deg)),
                        device=dev, downscale=a.downscale)
                 for _ in range(a.views - 1)]
        hit = torch.zeros(n, device=dev, dtype=torch.bool)
        occ, con = [], []
        for c in cams:
            with torch.no_grad():
                ref = render_gaussians(*gaussians(xyz, A_gt), c)
            occ.append(float((ref.abs().amax(0) > 1e-3).float().mean()))
            con.append(float(ref.std()))
            A = A_pred.clone().requires_grad_(True)
            img = render_gaussians(*gaussians(xyz, A), c)
            (img - ref.detach()).abs().mean().backward()
            hit |= A.grad.abs().sum(-1) > 0
        print(f"{deg:6.0f}° {float(hit.float().mean()):9.1%} "
              f"{float(np.mean(occ)):8.1%} {float(np.mean(con)):13.4f}")

    print("\ncovered = 렌더 gradient를 받는 점의 비율. frame 이 유지되면서")
    print("covered 가 오르면, 65% 미커버는 렌더의 성질이 아니라 카메라 선택의 결과입니다.")


if __name__ == "__main__":
    main()
