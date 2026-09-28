"""Render a held-out scene from a checkpoint and compare it against the original.

Every number this project has produced so far is point-space chamfer. This is
the first measurement of what the reconstruction actually *looks* like, which is
what decides the open questions in `PLAN.md`: the Gaussian budget K, whether
attributes can be decoded instead of stored, and whether attribute supervision
should be parameter loss or render loss.

The npz carries no ground-truth photograph, so the reference is
``render(original Gaussians)``. Three renders are produced per scene so a
failure is attributable to one stage rather than to "the model":

    A  original      every Gaussian in the npz
    B  canonical     only the <=max_points the sampler selected, GT attributes
    C  reconstruction predicted xyz, attributes copied from the nearest GT point

B vs A is the cost of selection/truncation. C vs B is the cost of the
autoencoder's geometry alone -- attributes are given away for free, so this is a
*lower* bound on the eventual error, and the point is to see whether geometry at
today's accuracy is already the limiting factor. C vs A is the end-to-end number.

A fourth render, ``snap``, replaces each predicted point with its nearest GT
point. It renders a strict subset of the original Gaussians, so it is the
ceiling reachable by improving nothing but where the points sit -- if C is close
to snap, the geometry error is already spread evenly and the remaining loss is
elsewhere.

Run with the env that has the rasteriser:
    /home/super/anaconda3/envs/can3tok/bin/python tools/render_compare.py --ckpt ...
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

from can3tok.io_utils import load_npz_state, load_replay_dict  # noqa: E402
from can3tok.model import build_model  # noqa: E402
from can3tok.render import Camera, render_gaussians, sh_dc_to_rgb, psnr, ssim  # noqa: E402
from can3tok.schedule import apply_eval_schedule  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--indices", type=str, default="0,1,2,3")
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--branches", type=str, default="codec,gen")
    p.add_argument("--downscale", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_png", action="store_true")
    p.add_argument("--test_views", type=int, default=0,
                   help="extra HELD-OUT poses, orbited around the frame's own camera. The "
                        "frame camera alone cannot tell equivalence from a per-view fit: the "
                        "attribute oracle scored 36.97 dB on the pose it was fitted to and "
                        "21.02 dB on unseen ones. Any claim about learned attributes needs this")
    return p.parse_args()


def load_run(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = Namespace(**ck["args"])
    # stats.json lives beside the checkpoint; make_datasets resolves it via out_dir.
    args.out_dir = os.path.dirname(os.path.abspath(ckpt_path))
    train_ds, val_ds = make_datasets(args)
    cfg = build_config(args, train_ds.target_dim, train_ds.sh_dim)
    model = build_model(cfg, allow_padded_tokens=getattr(args, "allow_padded_tokens", False))
    # Non-strict, and it reports. This tool reads checkpoints written by older
    # configs than the code it runs under, so a tensor added since -- e.g. a
    # function-preserving gate initialised to 1.0 -- must not stop the analysis.
    # Silence would be worse than the crash it replaces, so anything missing or
    # unexpected is printed and the caller can judge whether it matters.
    _in = model.load_state_dict(ck["model"], strict=False)
    if _in.missing_keys:
        print(f"  [load_run] not in ckpt, left at init ({len(_in.missing_keys)}): "
              + ", ".join(sorted(_in.missing_keys)[:6])
              + (" ..." if len(_in.missing_keys) > 6 else ""))
    if _in.unexpected_keys:
        print(f"  [load_run] in ckpt but unused ({len(_in.unexpected_keys)}): "
              + ", ".join(sorted(_in.unexpected_keys)[:6])
              + (" ..." if len(_in.unexpected_keys) > 6 else ""))
    model.to(device).eval()
    step = int(ck.get("step", -1))
    apply_eval_schedule(model, max(step, 0), args)
    return model, cfg, args, val_ds, step


def attrs_from_nn(query_xyz: torch.Tensor, gt_xyz: torch.Tensor, gt: dict, chunk: int = 4096):
    """Copy scale/rot/opacity/colour from each query point's nearest GT point.

    This is deliberate oracle attribute transfer: the latent carries xyz only
    today (``patch_dim = g * 4``), so rendering the prediction at all requires
    attributes from somewhere, and taking the nearest GT point's is the choice
    that leaves the geometry error as the only variable.
    """
    idx = torch.empty(query_xyz.shape[0], dtype=torch.long, device=query_xyz.device)
    for s in range(0, query_xyz.shape[0], chunk):
        e = min(s + chunk, query_xyz.shape[0])
        idx[s:e] = torch.cdist(query_xyz[s:e], gt_xyz).argmin(dim=1)
    return {k: v[idx] for k, v in gt.items()}, idx


def render_set(xyz, a, cam):
    return render_gaussians(xyz, a["scaling"], a["rot"], a["opacity"], a["color"], cam)


def main():
    a = parse_args()
    dev = a.device
    model, cfg, targs, val_ds, step = load_run(a.ckpt, dev)
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)),
                                        "render", f"step{max(step,0):08d}")
    os.makedirs(out_dir, exist_ok=True)
    branches = [b for b in a.branches.split(",") if b]
    indices = [int(i) for i in a.indices.split(",") if i != ""]
    indices = [i for i in indices if i < len(val_ds)]
    print(f"ckpt {a.ckpt}  step {step}  val {len(val_ds)} files  rendering {indices}")

    rows = []
    for i in indices:
        item = val_ds[i]
        path = val_ds.files[i]
        raw = load_replay_dict(path)
        cam = Camera(raw["camera"], device=dev, downscale=a.downscale)
        held = []
        if a.test_views > 0:
            from can3tok.render import perturb_camera_vector
            from can3tok.io_utils import camera_from_vector
            cv = item["camera"].numpy()
            rng = np.random.default_rng(9999 + i)
            held = [Camera(camera_from_vector(perturb_camera_vector(cv, rng)),
                           device=dev, downscale=a.downscale) for _ in range(a.test_views)]
        gs = load_npz_state(path)

        T = lambda k, d: torch.from_numpy(np.asarray(gs[k], np.float32)).to(dev).reshape(-1, d)
        gt_xyz = T("xyz", 3)
        gt_attr = {
            "scaling": T("scaling", 3),
            "rot": T("rot", 4),
            "opacity": T("opacity", 1),
            "color": sh_dc_to_rgb(T("color", 3)),
        }

        center = item["center"].to(dev).reshape(1, 3)
        scale = float(item["scale"])
        mask = item["mask"].to(dev) > 0.5
        target = item["target"].to(dev).float()
        sel_xyz = target[mask][:, 0:3] * scale + center

        with torch.no_grad():
            x = item["input"].unsqueeze(0).to(dev).float()
            m = item["mask"].unsqueeze(0).to(dev).float()
            # R8+ can encode more points than the decoder emits.  Match
            # train.run_eval here; otherwise the render silently encodes the
            # smaller decoder layout and is not the checkpoint's real output.
            ex = item.get("enc_input")
            em = item.get("enc_mask")
            run_gen = bool(cfg.use_gen_branch) and "gen" in branches
            out = model(
                x, m, run_decode=True, run_gen=run_gen, gen_noise_std=0.0,
                enc_x=None if ex is None else ex.unsqueeze(0).to(dev).float(),
                enc_mask=None if em is None else em.unsqueeze(0).to(dev).float(),
            )

        # A -- the original scene, and the reference every other render is scored against
        img_a = render_set(gt_xyz, gt_attr, cam)

        # B -- only what the sampler kept. Attributes come from the same NN
        # transfer as the reconstruction so B and C differ in geometry only.
        # Exact identity, not a nearest-neighbour lookup: the dataset records which
        # npz row each slot came from. An NN map is only 0.991 injective here
        # because 0.16-0.17% of these Gaussians are exactly coincident and more
        # sit inside the float32 round-trip error, so it would quietly attribute
        # part of the canonical loss to a lookup artefact.
        sel_src = item["source_index"].to(dev)[mask]
        sel_attr = {k: v[sel_src] for k, v in gt_attr.items()}
        img_b = render_set(sel_xyz, sel_attr, cam)

        rec = {"orig": img_a, "canon": img_b}
        row = {"idx": i, "name": item["name"], "n_gt": int(gt_xyz.shape[0]),
               "n_sel": int(sel_xyz.shape[0]),
               "canon_psnr": psnr(img_a, img_b), "canon_ssim": ssim(img_a, img_b),
               "canon_src_unique": float(sel_src.unique().numel()) / max(sel_src.numel(), 1),
               # Validates the index map. val_ds runs with augment off and
               # crop_prob 0, so the selected xyz must equal the indexed GT xyz to
               # within the float32 normalise round trip (~3e-5). A larger value
               # means the mapping is stale, not that the model is wrong.
               "canon_xyz_err": float((sel_xyz - gt_xyz[sel_src]).abs().max())}

        for br in branches:
            key = "pred" if br == "codec" else "gen_pred"
            if key not in out:
                continue
            pred = out[key][0].float()
            # `_own` must read the module that learns the attributes. With the
            # separate attribute decoder, out["pred"]'s attribute channels have no
            # loss on them at all, so measuring the gate here would have scored an
            # untrained head. attr_pred also carries the bounded position nudge,
            # which is part of what ships.
            a_key = "attr_pred" if br == "codec" else "gen_attr_pred"
            attr = out[a_key][0].float() if a_key in out else pred
            pred_xyz = pred[mask][:, 0:3] * scale + center
            pr_attr, nn_idx = attrs_from_nn(pred_xyz, gt_xyz, gt_attr)
            img_c = render_set(pred_xyz, pr_attr, cam)
            img_s = render_set(gt_xyz[nn_idx], pr_attr, cam)  # snapped ceiling
            rec[br] = img_c
            rec[br + "_snap"] = img_s

            # The model's OWN attributes, once it has any. Every render number
            # before this substituted GT scale/rotation/opacity/colour so that the
            # image difference isolated geometry; that made sense while nothing but
            # xyz was trained. With attribute heads present, the honest end-to-end
            # number is this one, and the gap to `img_c` is exactly what the
            # learned attributes cost.
            if attr.shape[-1] > 11:
                pa = attr[mask]
                own = {
                    "scaling": torch.exp(pa[:, 3:6].clamp(-15, 5)) * scale,
                    "rot": pa[:, 6:10],
                    "opacity": torch.sigmoid(pa[:, 10:11]),
                    "color": sh_dc_to_rgb(pa[:, 11:14]),
                }
                own_xyz = pa[:, 0:3] * scale + center       # includes the nudge
                rec[br + "_own"] = render_set(own_xyz, own, cam)
                row[f"{br}_psnr_own"] = psnr(img_a, rec[br + "_own"])
                row[f"{br}_ssim_own"] = ssim(img_a, rec[br + "_own"])
                if held:
                    hp, hs, hg = [], [], []
                    for c_h in held:
                        ref_h = render_set(gt_xyz, gt_attr, c_h)
                        hp.append(psnr(ref_h, render_set(own_xyz, own, c_h)))
                        hs.append(ssim(ref_h, render_set(own_xyz, own, c_h)))
                        hg.append(psnr(ref_h, render_set(pred_xyz, pr_attr, c_h)))
                    row[f"{br}_psnr_own_held"] = float(np.mean(hp))
                    row[f"{br}_ssim_own_held"] = float(np.mean(hs))
                    row[f"{br}_psnr_held"] = float(np.mean(hg))
            # How much of the scene the prediction actually reaches. If every
            # predicted point had a distinct nearest GT point this would be 1.0;
            # a low value means predictions pile onto the same surface patches
            # and leave the rest of the scene uncovered, which no amount of
            # positional accuracy fixes. This separates "slightly off" from
            # "missing", and the two need different repairs.
            uniq = float(nn_idx.unique().numel()) / max(nn_idx.numel(), 1)
            row.update({
                f"{br}_psnr": psnr(img_a, img_c), f"{br}_ssim": ssim(img_a, img_c),
                f"{br}_psnr_vs_canon": psnr(img_b, img_c),
                f"{br}_snap_psnr": psnr(img_a, img_s),
                f"{br}_nn_unique": uniq,
            })

        rows.append(row)
        keys = ", ".join(f"{k} {v:.2f}" for k, v in row.items() if "psnr" in k)
        print(f"[{i:3d}] {row['name']:24s} n {row['n_gt']:7d}->{row['n_sel']:7d}  {keys}")

        if not a.no_png:
            import torchvision.utils as vu
            order = ["orig", "canon"] + [k for k in rec if k not in ("orig", "canon")]
            grid = torch.stack([rec[k].clamp(0, 1) for k in order])
            stem = os.path.join(out_dir, os.path.splitext(item["name"])[0])
            vu.save_image(grid, stem + ".png", nrow=2)
            # Each PNG gets its own numbers and its own tile legend. Without this a
            # PNG left over from one invocation sits next to a summary json written
            # by another, which is exactly how a throwaway smoke image (val index
            # 1 = 3DGS optimisation step 150, a scene whose *original* renders as
            # fog) ended up looking like a result.
            with open(stem + ".json", "w") as f:
                json.dump({"tiles_row_major_nrow2": order, "ckpt": a.ckpt,
                           "step": step, **row}, f, indent=2)

    if rows:
        agg = {k: float(np.mean([r[k] for r in rows if k in r]))
               for k in rows[0] if isinstance(rows[0][k], float)}
        print("\nmean over scenes:")
        for k, v in sorted(agg.items()):
            print(f"  {k:24s} {v:8.3f}")
        # Name the summary after the scenes it covers, so two invocations with
        # different --indices cannot overwrite each other's numbers.
        tag = "-".join(str(i) for i in indices)
        name = f"render_metrics_idx{tag}.json" if len(tag) <= 60 else "render_metrics.json"
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump({"step": step, "ckpt": a.ckpt, "indices": indices,
                       "branches": branches, "downscale": a.downscale,
                       "scenes": rows, "mean": agg}, f, indent=2)
        print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
