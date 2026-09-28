"""Is the appearance code an optimisation problem or an information problem?

Measured on the trained model: freezing its geometry and optimising the
attributes per scene reaches 26.92 dB on held-out views, against 14.33 dB for
what the model actually produces. 12.6 dB sits on top of geometry the model
already has, so the gap is not the budget and not the loss balance (the render
term is 85.5% of the attribute gradient, and the reconstruction weights have
already decayed to their floor).

That leaves two possibilities, and they call for opposite fixes:

  optimisation   the code the decoder needs IS a function of what the encoder
                 sees, and training simply has not found it -> more render
                 steps, a different schedule, a better conditioned path.

  information    the code depends on something the encoder never receives ->
                 more channels into the pack (`attr_pack_dim`), which is a
                 structural change.

The experiment separates them in three phases, all with the model frozen:

  1. optimise the appearance code C per scene, **decoder frozen**, against a
     multi-view render. C* is then by construction the code this exact decoder
     would need. Its gain over the encoder's own code is the prize on offer.

  2. ask whether C* is predictable from what the pack actually carries: a group's
     own 64x11 GT attributes, summarised. Fit on groups from train scenes, score
     on groups from held-out scenes. This is the information question, asked
     without involving the encoder's weights at all -- if a direct regressor
     cannot find C* from the pack's own contents, no amount of training will.

  3. render with the *predicted* code. Landing near C* means learnable;
     landing near the encoder's current code means the information is not there.

Everything is reported on held-out views, because attributes fitted to one
projection score 36.97 dB there and 21.02 dB elsewhere on this data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.config import channel_budget  # noqa: E402
from can3tok.io_utils import camera_from_vector, load_npz_state  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.render import (Camera, perturb_camera_vector, photometric_loss,  # noqa: E402
                            psnr, render_gaussians, sh_dc_to_rgb, ssim)
from can3tok.train import build_config, make_datasets  # noqa: E402


def scene_cameras(cv, n_fit, n_test, dev, downscale):
    rf, rt = np.random.default_rng(0), np.random.default_rng(9999)
    fit = [Camera(camera_from_vector(cv), device=dev, downscale=downscale)]
    fit += [Camera(camera_from_vector(perturb_camera_vector(cv, rf)), device=dev,
                   downscale=downscale) for _ in range(n_fit - 1)]
    test = [Camera(camera_from_vector(perturb_camera_vector(cv, rt)), device=dev,
                   downscale=downscale) for _ in range(n_test)]
    return fit, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_scenes", default="40,63,74,120,138")
    ap.add_argument("--test_scenes", default="160,200,240")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fit_views", type=int, default=4)
    ap.add_argument("--test_views", type=int, default=3)
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--reg_steps", type=int, default=4000)
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
    for p in model.parameters():
        p.requires_grad_(False)
    model.cfg.decoder_refine_alpha = 1.0
    dev, g = a.device, int(cfg.group_size)
    c_app = int(channel_budget(cfg).get("appearance", 0))
    print(f"ckpt {a.ckpt}  step {ck.get('step')}  appearance channels {c_app}")
    print(f"  train {a.train_scenes} | held-out {a.test_scenes}\n")

    # ---------------- phase 1: per-scene optimal code, decoder frozen -------
    store = {}
    for tag, spec in (("train", a.train_scenes), ("test", a.test_scenes)):
        for i in [int(s) for s in spec.split(",") if s]:
            item = val_ds[i]
            gs = load_npz_state(val_ds.files[i])
            T = lambda k, d: torch.from_numpy(np.asarray(gs[k], np.float32)).to(dev).reshape(-1, d)
            gt = (T("xyz", 3), T("scaling", 3), T("rot", 4), T("opacity", 1),
                  sh_dc_to_rgb(T("color", 3)))
            fit_cams, test_cams = scene_cameras(item["camera"].numpy(), a.fit_views,
                                                a.test_views, dev, a.downscale)
            with torch.no_grad():
                fit_ref = [render_gaussians(*gt, c) for c in fit_cams]
                test_ref = [render_gaussians(*gt, c) for c in test_cams]
                x = item["input"].unsqueeze(0).to(dev).float()
                mk = item["mask"].unsqueeze(0).to(dev).float()
                z_raw, anchors, _ = model.encoder(x, mk)
                z_compact, _ = model.compressor(z_raw, anchors)
                z_hat, ctx = model.decompressor(z_compact, shortcut_alpha=0.0)
                pred, _ = model.decoder(z_hat, ctx["cell_vec"], ctx["scale"],
                                        centroid=ctx["centroid"], count=ctx["count"],
                                        group_vec=ctx.get("group_vec"),
                                        appear=ctx.get("appearance"))
            xyz_in = pred[..., 0:3].detach()
            scale_in = ctx["scale"].detach()
            c_enc = ctx["appearance"].detach().clone()       # what the encoder produced
            m = item["mask"].to(dev) > 0.5
            sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)

            def render_with(code, cams):
                out = model.attr_decoder(xyz_in, code, scale_in)["pred"][0][m].float()
                return [render_gaussians(out[:, 0:3] * sc + cen,
                                         torch.exp(out[:, 3:6].clamp(-15, 5)) * sc,
                                         out[:, 6:10], torch.sigmoid(out[:, 10:11]),
                                         sh_dc_to_rgb(out[:, 11:14]), c) for c in cams]

            def score(code):
                with torch.no_grad():
                    return (float(np.mean([psnr(r, im) for r, im in
                                           zip(test_ref, render_with(code, test_cams))])),
                            float(np.mean([ssim(r, im) for r, im in
                                           zip(test_ref, render_with(code, test_cams))])))

            code = c_enc.clone().requires_grad_(True)
            opt = torch.optim.Adam([code], lr=a.lr)
            for _ in range(a.steps):
                opt.zero_grad(set_to_none=True)
                loss = 0.0
                for r, im in zip(fit_ref, render_with(code, fit_cams)):
                    l, _, _ = photometric_loss(im, r, 0.2)
                    loss = loss + l / len(fit_cams)
                loss.backward()
                opt.step()
            p_enc, s_enc = score(c_enc)
            p_opt, s_opt = score(code.detach())

            # group features the pack actually carries: the group's own attributes
            tgt = item["target"].to(dev).float()[m]
            ng = tgt.shape[0] // g
            at = tgt[: ng * g, 3:14].reshape(ng, g, 11)
            feat = torch.cat([at.mean(1), at.std(1),
                              tgt[: ng * g, 0:3].reshape(ng, g, 3).mean(1),
                              ctx["scale"][0, :ng].detach().log()], dim=-1)
            store.setdefault(tag, []).append(dict(
                idx=i, name=item["name"], feat=feat.cpu(),
                c_enc=c_enc[0, :ng].cpu(), c_opt=code.detach()[0, :ng].cpu(),
                p_enc=p_enc, p_opt=p_opt, s_enc=s_enc, s_opt=s_opt,
                item=i, ng=ng))
            print(f"  [{tag}] {item['name']:20s} 인코더 {p_enc:6.2f} dB -> 최적 {p_opt:6.2f} dB "
                  f"({p_opt - p_enc:+.2f})")

    print("\n--- 1단계: 이 디코더가 필요로 하는 코드의 가치 ---")
    for tag in ("train", "test"):
        e = float(np.mean([r["p_enc"] for r in store[tag]]))
        o = float(np.mean([r["p_opt"] for r in store[tag]]))
        print(f"  {tag:6s} 인코더 코드 {e:6.2f} dB | 장면별 최적 코드 {o:6.2f} dB  ({o - e:+.2f})")

    # ---------------- phase 2: is C* predictable from the pack's contents? ---
    Xtr = torch.cat([r["feat"] for r in store["train"]]).to(dev)
    Ytr = torch.cat([r["c_opt"] for r in store["train"]]).to(dev)
    Xte = torch.cat([r["feat"] for r in store["test"]]).to(dev)
    Yte = torch.cat([r["c_opt"] for r in store["test"]]).to(dev)
    Ete = torch.cat([r["c_enc"] for r in store["test"]]).to(dev)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
    ym, ys = Ytr.mean(0, keepdim=True), Ytr.std(0, keepdim=True).clamp(min=1e-6)

    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.GELU(),
                        nn.Linear(512, 512), nn.GELU(),
                        nn.Linear(512, Ytr.shape[1])).to(dev)
    o = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(0)
    for _ in range(a.reg_steps):
        idx = torch.randint(0, Xtr_n.shape[0], (4096,), generator=gen, device=dev)
        o.zero_grad(set_to_none=True)
        ((net(Xtr_n[idx]) - (Ytr[idx] - ym) / ys) ** 2).mean().backward()
        o.step()
    net.eval()
    with torch.no_grad():
        Yp = net(Xte_n) * ys + ym

    def r2(p, t):
        return float((1 - ((p - t) ** 2).sum(0) /
                      ((t - t.mean(0, keepdim=True)) ** 2).sum(0).clamp(min=1e-12)).mean())

    print("\n--- 2단계: 최적 코드가 pack 내용물로부터 예측되는가 (held-out 그룹) ---")
    print(f"  R^2(예측 -> 최적 코드)   {r2(Yp, Yte):+.3f}")
    print(f"  R^2(인코더 -> 최적 코드) {r2(Ete, Yte):+.3f}   <- 현재 인코더가 얼마나 맞추고 있나")

    # ---------------- phase 3: render with the predicted code ---------------
    print("\n--- 3단계: 예측된 코드로 렌더 (held-out 장면 / held-out 뷰) ---")
    print(f"{'scene':>18} {'인코더':>8} {'예측':>8} {'최적':>8}")
    off, rows = 0, []
    for r in store["test"]:
        i, ng = r["idx"], r["ng"]
        item = val_ds[i]
        gs = load_npz_state(val_ds.files[i])
        Tf = lambda k, d: torch.from_numpy(np.asarray(gs[k], np.float32)).to(dev).reshape(-1, d)
        gt = (Tf("xyz", 3), Tf("scaling", 3), Tf("rot", 4), Tf("opacity", 1),
              sh_dc_to_rgb(Tf("color", 3)))
        _, test_cams = scene_cameras(item["camera"].numpy(), a.fit_views, a.test_views,
                                     dev, a.downscale)
        with torch.no_grad():
            test_ref = [render_gaussians(*gt, c) for c in test_cams]
            x = item["input"].unsqueeze(0).to(dev).float()
            mk = item["mask"].unsqueeze(0).to(dev).float()
            z_raw, anchors, _ = model.encoder(x, mk)
            z_compact, _ = model.compressor(z_raw, anchors)
            z_hat, ctx = model.decompressor(z_compact, shortcut_alpha=0.0)
            pred, _ = model.decoder(z_hat, ctx["cell_vec"], ctx["scale"],
                                    centroid=ctx["centroid"], count=ctx["count"],
                                    group_vec=ctx.get("group_vec"),
                                    appear=ctx.get("appearance"))
            code = ctx["appearance"].clone()
            code[0, :ng] = Yp[off: off + ng]
            m = item["mask"].to(dev) > 0.5
            sc, cen = float(item["scale"]), item["center"].to(dev).reshape(1, 3)
            out = model.attr_decoder(pred[..., 0:3], code, ctx["scale"])["pred"][0][m].float()
            imgs = [render_gaussians(out[:, 0:3] * sc + cen,
                                     torch.exp(out[:, 3:6].clamp(-15, 5)) * sc,
                                     out[:, 6:10], torch.sigmoid(out[:, 10:11]),
                                     sh_dc_to_rgb(out[:, 11:14]), c) for c in test_cams]
            p_pred = float(np.mean([psnr(rr, im) for rr, im in zip(test_ref, imgs)]))
        off += ng
        rows.append((r["p_enc"], p_pred, r["p_opt"]))
        print(f"{os.path.splitext(r['name'])[0]:>18} {r['p_enc']:8.2f} {p_pred:8.2f} {r['p_opt']:8.2f}")
    e, p, o_ = [float(np.mean([x[k] for x in rows])) for k in range(3)]
    print(f"{'평균':>18} {e:8.2f} {p:8.2f} {o_:8.2f}")
    frac = (p - e) / max(o_ - e, 1e-9)
    print(f"\n예측 코드가 회복한 격차: {100 * frac:.1f}%")
    print("  높으면(>50%) 정보는 pack 에 있고 학습/최적화 문제입니다.")
    print("  낮으면(<20%) pack 이 그 정보를 담고 있지 않습니다 -> attr_pack_dim 을 늘려야 합니다.")


if __name__ == "__main__":
    main()
