"""Are the Gaussians the render never sees different from the ones it does?

3DGS optimises only what a view can see, and it needs no parameter anchor for the
rest: the Gaussians persist across 30k iterations and 100+ cameras, so coverage
accumulates until every one of them has been supervised many times.

This model is not that. It is feed-forward over 2700 scenes, so the points do not
persist -- what persists is the decoder's weights. A point the render cannot see
in this scene is still produced by the same function that was fitted on points it
could see, in this and every other scene. Generalisation, not accumulation, is
what covers it.

Which means the anchor is only necessary if the uncovered points are
*systematically unlike* the covered ones. If both sets are drawn from the same
distribution, a decoder fitted on the visible 35% already predicts the other 65%
correctly and the anchor is redundant -- worse than redundant, since it is the
term that collapses within-group variation towards the mean.

So: split the points by whether the render reached them and compare. Reported per
channel as the gap between the two group means in units of the pooled standard
deviation -- a effect size, so 0.1 is "the same population" and 1.0 is "a
different one". Position and local density are included too, since a bias there
is what would make appearance differ downstream.
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

CH = (("log_scale", slice(3, 6)), ("rot_w", slice(6, 7)),
      ("opacity", slice(10, 11)), ("color", slice(11, 14)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scenes", type=str, default="160,40,220")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--jitter", type=float, default=8.0)
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

    acc = {}
    cov_frac = []
    for si in [int(x) for x in a.scenes.split(",") if x]:
        item = val[si]
        m = item["mask"].to(dev) > 0.5
        t = item["target"].to(dev).float()[m]
        n = (t.shape[0] // g) * g
        t = t[:n]
        sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
        with torch.no_grad():
            out = model(item["input"].unsqueeze(0).to(dev).float(),
                        item["mask"].unsqueeze(0).to(dev).float(),
                        run_decode=True, run_gen=False)
        xyz = out["pred"][0][m][:n, 0:3].float()
        A = t[:, 3:14]

        cv = item["camera"].numpy()
        rng = np.random.default_rng(1234 + si)
        cams = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
        cams += [Camera(camera_from_vector(perturb_camera_vector(cv, rng, max_deg=a.jitter)),
                        device=dev, downscale=a.downscale) for _ in range(a.views - 1)]

        seen = torch.zeros(n, dtype=torch.bool, device=dev)
        for c in cams:
            with torch.no_grad():
                _, radii = render_gaussians(
                    xyz * sc + cen, torch.exp(A[:, 0:3].clamp(-15, 5)) * sc,
                    torch.nn.functional.normalize(A[:, 3:7], dim=-1),
                    torch.sigmoid(A[:, 7:8]), sh_dc_to_rgb(A[:, 8:11]), c,
                    return_radii=True)
            seen |= radii > 0
        cov_frac.append(float(seen.float().mean()))

        # depth along the frame camera's optical axis, and how crowded each point is
        R = torch.from_numpy(cv[4:13].reshape(3, 3).astype(np.float32)).to(dev)
        T = torch.from_numpy(cv[13:16].astype(np.float32)).to(dev)
        depth = ((xyz * sc + cen) @ R.T + T)[:, 2]
        feats = {"depth": depth[:, None], "opacity": A[:, 10:11]}
        for name, sl in CH:
            feats[name] = A[:, sl]
        for name, v in feats.items():
            # float64: the float32 variance of a 200k-element tensor with values
            # spanning several units came out non-finite for the colour channel.
            v = v.double().mean(-1)
            c1, c0 = v[seen], v[~seen]
            if c1.numel() < 8 or c0.numel() < 8:
                continue
            pooled = torch.sqrt(0.5 * (c1.var() + c0.var())).clamp(min=1e-12)
            d = float((c1.mean() - c0.mean()) / pooled)
            if np.isfinite(d):
                acc.setdefault(name, []).append(d)

    print(f"{len(cov_frac)} scenes, {a.views} views, orbit {a.jitter:.0f}°  "
          f"covered {np.mean(cov_frac):.1%}\n")
    print(f"{'quantity':>10} {'effect size (covered - uncovered)':>34}")
    for k, v in acc.items():
        d = float(np.mean(v))
        note = ("같은 분포" if abs(d) < 0.2 else
                "약한 차이" if abs(d) < 0.5 else "체계적 차이")
        print(f"{k:>10} {d:+22.3f}   {note}")
    print("\n|d| < 0.2 이면 커버된 점으로 학습한 디코더가 나머지도 맞게 예측합니다")
    print("-> 앵커는 불필요. |d| > 0.5 이면 분포 이동이 있어 보정이 필요합니다.")


if __name__ == "__main__":
    main()
