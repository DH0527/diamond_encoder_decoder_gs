"""How many appearance channels per group does the render actually need?

Same question as `oracle_attr_code_size.py`, with two corrections:

  * **grouping.** That tool compacts the mask and cuts the result into runs of
    `group_size`, which is only the real grouping for Morton chunks. Under fixed
    anchors the runs straddle anchor boundaries -- measured on these two scenes
    the compacted groups have a median radius 25-39x the true one -- so both the
    rank-k geometry it substitutes and the per-group attribute code it fits are
    defined over groups the model does not have.
  * **the point set is held fixed** between the reference and every variant, so a
    dB difference is the attribute code and nothing else.

Attributes are constrained to `mean + code_k @ basis`, a k-dimensional affine
subspace shared across groups: exactly what k channels per group can express in
the linear limit. Fitted on `fit_views`, always reported on held-out poses.
"""
from __future__ import annotations
import argparse, json, os, sys
from argparse import Namespace
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from can3tok.io_utils import camera_from_vector            # noqa: E402
from can3tok.render import (Camera, perturb_camera_vector, photometric_loss,  # noqa: E402
                            psnr, render_gaussians, sh_dc_to_rgb, ssim)
from can3tok.train import make_datasets                    # noqa: E402

A_DIM = 11          # log_scale 3 | quat 4 | logit opacity 1 | SH DC 3


def full_groups(target, mask, g):
    """Slot-layout grouping: (n_full, g, dim) over groups with all g slots live."""
    n = target.shape[0]
    ng = n // g
    t = target[: ng * g].reshape(ng, g, -1)
    cnt = (mask[: ng * g].reshape(ng, g) > 0.5).sum(1)
    sel = cnt == g
    return t[sel], int(sel.sum())


def rank_k_positions(grp, k):
    """Best rank-k linear reconstruction of the centred, radius-normalised offsets."""
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
    ap.add_argument("--scene", type=int, default=40)
    ap.add_argument("--geo_rank", type=int, default=19, help="shape channels the layout keeps")
    ap.add_argument("--codes", default="1,2,4,8,16")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fit_views", type=int, default=4)
    ap.add_argument("--test_views", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    item = val_ds[a.scene]
    g, dev = int(targs.group_size), "cuda"

    grp, n_full = full_groups(item["target"].to(dev).float(),
                              item["mask"].to(dev).float(), g)
    ng = grp.shape[0]
    sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
    gt_grp = grp[..., 0:3]
    A0 = grp[..., 3:14].reshape(-1, A_DIM).contiguous()

    cv = item["camera"].numpy()
    rf, rt = np.random.default_rng(0), np.random.default_rng(9999)
    fit_cams = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
    fit_cams += [Camera(camera_from_vector(perturb_camera_vector(cv, rf)), device=dev,
                        downscale=a.downscale) for _ in range(a.fit_views - 1)]
    test_cams = [Camera(camera_from_vector(perturb_camera_vector(cv, rt)), device=dev,
                        downscale=a.downscale) for _ in range(a.test_views)]

    def gaussians(xyz, attr):
        return (xyz, torch.exp(attr[:, 0:3].clamp(-15, 5)) * sc, attr[:, 3:7],
                torch.sigmoid(attr[:, 7:8]), sh_dc_to_rgb(attr[:, 8:11]))

    gt_xyz = gt_grp.reshape(-1, 3) * sc + cen
    with torch.no_grad():
        fit_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in fit_cams]
        test_ref = [render_gaussians(*gaussians(gt_xyz, A0), c) for c in test_cams]

    xyz_k = (rank_k_positions(gt_grp, a.geo_rank) * sc + cen).detach()
    sd = A0.std(0, keepdim=True).clamp(min=1e-6)

    def report(label, attr, xyz):
        with torch.no_grad():
            f = np.mean([psnr(r, render_gaussians(*gaussians(xyz, attr), c))
                         for r, c in zip(fit_ref, fit_cams)])
            p = np.mean([psnr(r, render_gaussians(*gaussians(xyz, attr), c))
                         for r, c in zip(test_ref, test_cams)])
            s = np.mean([ssim(r, render_gaussians(*gaussians(xyz, attr), c))
                         for r, c in zip(test_ref, test_cams)])
        print(f"  {label:38s} fit {f:6.2f} | held-out {p:6.2f} dB  SSIM {s:.3f}")
        return float(p)

    print(f"scene {item['name']}  slot-layout full groups {ng} (of {len(item['mask'])//g})"
          f"  points {ng*g:,}")
    print(f"  geometry: rank {a.geo_rank}   fit {len(fit_cams)} views, held-out {len(test_cams)}\n")
    report("GT geometry + GT attributes (상한)", A0, gt_xyz)
    report(f"rank-{a.geo_rank} geometry + GT attributes", A0, xyz_k)

    for k in [int(x) for x in a.codes.split(",") if x]:
        base = A0.reshape(ng, g, A_DIM).mean(1, keepdim=True).expand(ng, g, A_DIM)
        base = base.reshape(ng, g * A_DIM).clone()
        B = torch.zeros(k, g * A_DIM, device=dev)
        torch.nn.init.normal_(B, std=1.0 / (g * A_DIM) ** 0.5)
        C = torch.zeros(ng, k, device=dev)
        B.requires_grad_(True); C.requires_grad_(True)
        opt = torch.optim.Adam([B, C], lr=a.lr)
        sd_g = sd.repeat(1, g).reshape(1, g * A_DIM)
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            attr = (base + (C @ B) * sd_g).reshape(-1, A_DIM)
            loss = 0.0
            for r, c in zip(fit_ref, fit_cams):
                l, _, _ = photometric_loss(render_gaussians(*gaussians(xyz_k, attr), c), r, 0.2)
                loss = loss + l / len(fit_cams)
            loss.backward(); opt.step()
        with torch.no_grad():
            attr = (base + (C @ B) * sd_g).reshape(-1, A_DIM)
        report(f"rank-{a.geo_rank} geo + appearance {k:2d} ch/group", attr, xyz_k)


if __name__ == "__main__":
    main()
