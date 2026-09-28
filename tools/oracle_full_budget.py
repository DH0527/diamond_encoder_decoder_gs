"""One latent budget in, one held-out dB out.

Every earlier oracle sized one axis with the others held fixed: shape channels at
4096 cells, appearance channels at 4096 cells, or the cell-count trade with
geometry only. A latent budget is all three at once, and multiplying the separate
penalties assumes a separability nothing has shown.

This measures the whole configuration end to end:

    cells C          -> k-means anchors on the scene, points partitioned to the
                        nearest one, q = live/C points per cell
    shape channels   -> positions replaced by their rank-k linear reconstruction
                        inside each cell
    appearance ch    -> attributes replaced by mean + code_k @ shared basis,
                        fitted through the rasteriser on fit views

and reports PSNR/SSIM on HELD-OUT poses against the same point set rendered with
GT positions and GT attributes, so a dB difference is the budget and nothing else.
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

A_DIM = 11


def kmeans(x, k, iters, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g, device=dev)[:k]].clone()
    for _ in range(iters):
        lab = torch.empty(x.shape[0], dtype=torch.long, device=dev)
        for i in range(0, x.shape[0], 200_000):
            lab[i:i + 200_000] = torch.cdist(x[i:i + 200_000], c).argmin(1)
        cnt = torch.bincount(lab, minlength=k)
        s = torch.zeros_like(c).index_add_(0, lab, x)
        c = torch.where((cnt > 0).unsqueeze(-1), s / cnt.clamp(min=1).unsqueeze(-1).float(), c)
    return c


def partition(feat, xyz, anchors, q, dev):
    """Nearest-anchor partition, then q points per anchor that has them."""
    C = anchors.shape[0]
    lab = torch.empty(xyz.shape[0], dtype=torch.long, device=dev)
    for i in range(0, xyz.shape[0], 200_000):
        lab[i:i + 200_000] = torch.cdist(xyz[i:i + 200_000], anchors).argmin(1)
    order = torch.argsort(lab)
    lab_s, feat_s = lab[order], feat[order]
    cnt = torch.bincount(lab_s, minlength=C)
    start = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnt.cumsum(0)[:-1]])
    full = (cnt >= q).nonzero().squeeze(1)
    if full.numel() == 0:
        return None
    idx = start[full].unsqueeze(1) + torch.arange(q, device=dev).unsqueeze(0)
    return feat_s[idx]                       # (n_full, q, F)


def rank_k_positions(grp, k):
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
    ap.add_argument("--configs", default="4096:19:8,4096:3:8,1024:3:10,2048:3:9",
                    help="cells:shape_channels:appearance_channels")
    ap.add_argument("--kmeans_iters", type=int, default=12)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fit_views", type=int, default=4)
    ap.add_argument("--test_views", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda"

    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val_ds = make_datasets(targs)
    it = val_ds[a.scene]
    m = it["mask"].to(dev) > 0.5
    feat = it["target"].to(dev).float()[m][:, 0:14]      # xyz3 + attrs11
    sc, cen = float(it["scale"]), it["center"].to(dev).reshape(1, 3)
    N = feat.shape[0]

    cv = it["camera"].numpy()
    rf, rt = np.random.default_rng(0), np.random.default_rng(9999)
    fit_cams = [Camera(camera_from_vector(cv), device=dev, downscale=a.downscale)]
    fit_cams += [Camera(camera_from_vector(perturb_camera_vector(cv, rf)), device=dev,
                        downscale=a.downscale) for _ in range(a.fit_views - 1)]
    test_cams = [Camera(camera_from_vector(perturb_camera_vector(cv, rt)), device=dev,
                        downscale=a.downscale) for _ in range(a.test_views)]

    def gauss(xyz, at):
        return (xyz, torch.exp(at[:, 0:3].clamp(-15, 5)) * sc, at[:, 3:7],
                torch.sigmoid(at[:, 7:8]), sh_dc_to_rgb(at[:, 8:11]))

    print(f"scene {it['name']}  live points {N:,}")
    print(f"\n{'config':>22} {'latent':>9} {'cells':>7} {'pts/cell':>9} "
          f"{'kept pts':>9} {'held-out dB':>12} {'SSIM':>7}")

    for spec in [s for s in a.configs.split(",") if s]:
        C, ks, ka = (int(v) for v in spec.split(":"))
        q = max(int(N // C), 8)
        anch = kmeans(feat[:, 0:3], C, a.kmeans_iters, 0, dev)
        grp = partition(feat, feat[:, 0:3], anch, q, dev)
        if grp is None:
            print(f"{spec:>22}  (cells too many for this snapshot)"); continue
        ng = grp.shape[0]
        gt_grp = grp[..., 0:3]
        A0 = grp[..., 3:14].reshape(-1, A_DIM).contiguous()
        gt_xyz = gt_grp.reshape(-1, 3) * sc + cen
        with torch.no_grad():
            fit_ref = [render_gaussians(*gauss(gt_xyz, A0), c) for c in fit_cams]
            test_ref = [render_gaussians(*gauss(gt_xyz, A0), c) for c in test_cams]
        xyz_k = (rank_k_positions(gt_grp, ks) * sc + cen).detach()

        sd = A0.std(0, keepdim=True).clamp(min=1e-6)
        base = A0.reshape(ng, q, A_DIM).mean(1, keepdim=True).expand(ng, q, A_DIM)
        base = base.reshape(ng, q * A_DIM).clone()
        B = torch.zeros(ka, q * A_DIM, device=dev)
        torch.nn.init.normal_(B, std=1.0 / (q * A_DIM) ** 0.5)
        Cc = torch.zeros(ng, ka, device=dev)
        B.requires_grad_(True); Cc.requires_grad_(True)
        opt = torch.optim.Adam([B, Cc], lr=a.lr)
        sd_g = sd.repeat(1, q).reshape(1, q * A_DIM)
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            at = (base + (Cc @ B) * sd_g).reshape(-1, A_DIM)
            loss = 0.0
            for r, c in zip(fit_ref, fit_cams):
                l, _, _ = photometric_loss(render_gaussians(*gauss(xyz_k, at), c), r, 0.2)
                loss = loss + l / len(fit_cams)
            loss.backward(); opt.step()
        with torch.no_grad():
            at = (base + (Cc @ B) * sd_g).reshape(-1, A_DIM)
            p = np.mean([psnr(r, render_gaussians(*gauss(xyz_k, at), c))
                         for r, c in zip(test_ref, test_cams)])
            s = np.mean([ssim(r, render_gaussians(*gauss(xyz_k, at), c))
                         for r, c in zip(test_ref, test_cams)])
        # centroid 2 + occupancy 1 assumed on top of shape+appearance
        total = (2 + 1 + ks + ka) * C
        print(f"{f'{C}c s{ks} a{ka}':>22} {total:>9,} {C:>7,} {q:>9} {ng*q:>9,} "
              f"{p:>12.2f} {s:>7.3f}")


if __name__ == "__main__":
    main()
