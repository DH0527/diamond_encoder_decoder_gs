"""Held-out photographic metrics broken out PER SCENE.

train.py logs one mean over eval_val_indices, which in a multi-scene run mixes
two scenes whose GT ceilings differ -- so the single number cannot say whether a
scene is doing badly or merely has a lower ceiling. This reports each scene
separately, with its own psnr_gt_photo, plus the SSIM ratio against the GT
Gaussians (needed because once the PSNR gap goes negative, PSNR alone stops
distinguishing "better" from "smoother").
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.render_compare import load_run          # noqa: E402
from can3tok.eval_utils import render_eval_metrics  # noqa: E402
import can3tok.train as T                           # noqa: E402
from argparse import Namespace                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--args_json", default="")
    ap.add_argument("--per_scene", type=int, default=6, help="val samples per scene")
    a = ap.parse_args()

    model, cfg, targs, val_ds, step = load_run(a.ckpt, "cuda")
    aj = a.args_json or os.path.join(os.path.dirname(a.ckpt), "args.json")
    args = Namespace(**json.load(open(aj)))
    views, _ = T.load_eval_views(args)
    print(f"ckpt step {step}   held-out 사진 {len(views)} 장\n")

    scenes = getattr(val_ds, "file_scene", None)
    if scenes is None:
        scenes = np.zeros(len(val_ds), np.int64)
    roots = getattr(val_ds, "roots", ["(single)"])

    hdr = f"{'씬':34s}{'n':>4s}{'PSNR':>8s}{'GT':>8s}{'gap':>8s}{'SSIM':>8s}{'GT SSIM':>9s}{'SSIM비':>8s}"
    print(hdr); print("-" * len(hdr))
    for si, root in enumerate(roots):
        idx = [i for i in range(len(val_ds)) if int(scenes[i]) == si][: a.per_scene]
        if not idx:
            continue
        acc = {}
        for i in idx:
            it = val_ds[i]
            with torch.no_grad():
                ex, em = it.get("enc_input"), it.get("enc_mask")
                out = model(it["input"].unsqueeze(0).cuda().float(),
                            it["mask"].unsqueeze(0).cuda().float(),
                            run_decode=True, run_gen=False,
                            enc_x=None if ex is None else ex.unsqueeze(0).cuda().float(),
                            enc_mask=None if em is None else em.unsqueeze(0).cuda().float())
            ap_ = out.get("attr_pred", out["pred"]).float()[0]
            sk = it.get("scene_key") or f"scene{int(it.get('scene', si))}"
            pm = (out["presence"].float()[0]
                  if int(getattr(args, "eval_pred_mask", 0)) else None)
            rendered = render_eval_metrics(
                    ap_, it["target"].cuda().float(), it["mask"].cuda().float(),
                    it["center"], float(it["scale"]), views, val_ds.layout,
                    downscale=int(getattr(args, "render_downscale", 2)),
                    scene_id=sk, pred_mask=pm)
            rendered.pop("_eval_view_ids", None)
            for k, v in rendered.items():
                acc.setdefault(k, []).append(v)
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        p, g = m.get("psnr_photo", 0), m.get("psnr_gt_photo", 0)
        sp, sg = m.get("ssim_photo", 0), m.get("ssim_gt_photo", 0)
        name = root.split("/output/")[-1].replace("/replay_seg", "")
        print(f"{name:34s}{len(idx):4d}{p:8.2f}{g:8.2f}{g - p:+8.2f}{sp:8.3f}{sg:9.3f}"
              f"{sp / max(sg, 1e-9):8.3f}")


if __name__ == "__main__":
    main()
