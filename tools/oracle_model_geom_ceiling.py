"""What could this model's geometry reach if its attributes were optimal?

The claim "0.5 channels per point caps the render near 17 dB" came from
`oracle_shape_rank`, which reconstructs GT positions in a rank-k subspace and
renders them **with GT attributes**. That measures reproduction of the GT point
set, and it is the wrong ceiling for this system: `oracle_attr_compensation`
already showed the same rank-28 positions reaching 26.69 dB on held-out poses
once the attributes were allowed to adapt, and `oracle_attr_code_size` reached
21.43 dB with only 8 appearance channels per group.

Both of those used PCA positions, not the model's. This closes that gap: freeze
the **trained model's own predicted positions**, then optimise the attributes
per scene against a multi-view render. The result separates two very different
diagnoses that the aggregate PSNR cannot:

  * if it lands near 15 dB, the geometry really is the wall and the budget
    argument stands;
  * if it lands near 21-26 dB, the geometry is fine, an equivalent Gaussian set
    exists on top of it, and the gap is that the *encoder-produced* appearance
    code is not finding it -- a generalisation problem, not a capacity one.

Held-out poses are always reported separately from the fitted ones. Measured
earlier on this data, fitting attributes to a single view gives 36.97 dB there
and 21.02 dB elsewhere, so a fit-only number means nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.io_utils import camera_from_vector, load_npz_state  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.render import (Camera, perturb_camera_vector, photometric_loss,  # noqa: E402
                            psnr, render_gaussians, sh_dc_to_rgb, ssim)
from can3tok.train import build_config, make_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--indices", default="160,240,183")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fit_views", type=int, default=4)
    ap.add_argument("--test_views", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    targs = Namespace(**ck["args"])
    targs.out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    train_ds, val_ds = make_datasets(targs)
    cfg = build_config(targs, train_ds.target_dim, train_ds.sh_dim)
    model = build_model(cfg, allow_padded_tokens=getattr(targs, "allow_padded_tokens", False))
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()
    model.cfg.decoder_refine_alpha = 1.0
    dev = a.device
    print(f"ckpt {a.ckpt}  step {ck.get('step')}")
    print(f"  fit {a.fit_views} views, evaluate on {a.test_views} UNSEEN views\n")

    agg = {}
    for i in [int(x) for x in a.indices.split(",") if x]:
        item = val_ds[i]
        gs = load_npz_state(val_ds.files[i])
        cv = item["camera"].numpy()
        T = lambda k, d: torch.from_numpy(np.asarray(gs[k], np.float32)).to(dev).reshape(-1, d)
        gt = (T("xyz", 3), T("scaling", 3), T("rot", 4), T("opacity", 1),
              sh_dc_to_rgb(T("color", 3)))

        rng_f, rng_t = np.random.default_rng(0), np.random.default_rng(9999)
        fit_cams = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
        fit_cams += [Camera(camera_from_vector(perturb_camera_vector(cv, rng_f)), device=dev,
                            downscale=a.downscale) for _ in range(a.fit_views - 1)]
        test_cams = [Camera(camera_from_vector(perturb_camera_vector(cv, rng_t)), device=dev,
                            downscale=a.downscale) for _ in range(a.test_views)]
        with torch.no_grad():
            fit_ref = [render_gaussians(*gt, c) for c in fit_cams]
            test_ref = [render_gaussians(*gt, c) for c in test_cams]

        m = item["mask"].to(dev) > 0.5
        tgt = item["target"].to(dev).float()[m]
        sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
        with torch.no_grad():
            out = model(item["input"].unsqueeze(0).to(dev).float(),
                        item["mask"].unsqueeze(0).to(dev).float(),
                        run_decode=True, run_gen=False, gen_noise_std=0.0)
        pr = out.get("attr_pred", out["pred"])[0][m].float()
        xyz = (pr[:, 0:3] * sc + cen).detach()          # the model's geometry, frozen

        def gaussians(p):
            return (xyz, torch.exp(p[:, 0:3].clamp(-15, 5)) * sc, p[:, 3:7],
                    torch.sigmoid(p[:, 7:8]), sh_dc_to_rgb(p[:, 8:11]))

        def report(tag, p):
            with torch.no_grad():
                f = np.mean([psnr(r, render_gaussians(*gaussians(p), c))
                             for r, c in zip(fit_ref, fit_cams)])
                t = np.mean([psnr(r, render_gaussians(*gaussians(p), c))
                             for r, c in zip(test_ref, test_cams)])
                s = np.mean([ssim(r, render_gaussians(*gaussians(p), c))
                             for r, c in zip(test_ref, test_cams)])
            agg.setdefault(tag, []).append((f, t, s))
            return f, t, s

        report("모델 자기 attribute", pr[:, 3:14].clone())
        report("GT attribute", tgt[:, 3:14].clone())

        p = pr[:, 3:14].clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=a.lr)
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for r, c in zip(fit_ref, fit_cams):
                l, _, _ = photometric_loss(render_gaussians(*gaussians(p), c), r, 0.2)
                loss = loss + l / len(fit_cams)
            loss.backward()
            opt.step()
        report("장면별 최적 attribute", p.detach())
        print(f"  [{i}] {item['name']} 완료")

    print(f"\n{'설정':>22} {'fit':>8} {'held-out':>10} {'SSIM':>7}")
    for tag, rows in agg.items():
        f, t, s = [float(np.mean([r[k] for r in rows])) for k in range(3)]
        print(f"{tag:>22} {f:8.2f} {t:10.2f} {s:7.3f}")
    own = float(np.mean([r[1] for r in agg["모델 자기 attribute"]]))
    best = float(np.mean([r[1] for r in agg["장면별 최적 attribute"]]))
    print(f"\n이 기하 위에서 attribute 만으로 얻을 수 있는 여유: {best - own:+.2f} dB")
    print("크면 기하가 아니라 appearance 경로가 병목입니다.")


if __name__ == "__main__":
    main()
