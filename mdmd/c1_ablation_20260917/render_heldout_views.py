"""Decode one snapshot per scene and render held-out eval cameras (true other views)."""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "detail_bottleneck_20260917"))
from common import *
from can3tok.train import load_eval_views
from can3tok.eval_utils import filter_eval_views, unpack_eval_view

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--indices", default="166,382")
ap.add_argument("--n_views", type=int, default=4)
ap.add_argument("--out_dir", required=True)
A = ap.parse_args()
os.makedirs(A.out_dir, exist_ok=True)

D = load(A.ckpt, "val", 166, 1)
val = D["val"]
val.holdout_own_photo = False
model, cfg = D["model"], D["cfg"]
views, held = load_eval_views(D["args"])
print("held pool idx", sorted(held), "n_views", len(views), flush=True)

def save_rgb(t, path):
    arr = (t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
    Image.fromarray(arr.round().astype(np.uint8)).save(path)

summary = []
for index in [int(x) for x in A.indices.split(",")]:
    it = val[index]
    name = str(it.get("name", f"idx{index}")).replace(".npz", "")
    scene_key = it.get("scene_key") or f"scene{int(it['scene'])}"
    x = it["input"][None].cuda().float()
    mk = it["mask"][None].cuda().float()
    tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
    gtm = it["mask"].cuda() > 0.5
    cen = it["center"].cuda().float().view(1, 3)
    sc = float(it["scale"])
    kw = {}
    if "enc_input" in it:
        kw = dict(enc_x=it["enc_input"][None].cuda().float(),
                  enc_mask=it["enc_mask"][None].cuda().float())
    if "group_anchor" in it:
        kw["group_anchor"] = it["group_anchor"][None].cuda().float()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(x, mk, run_decode=True, run_gen=False, **kw)
    pr = o.get("attr_pred", o["pred"]).float()[0]
    gt = tg[0]
    sv = filter_eval_views(views, scene_key)
    print(f"index {index} {name} scene={scene_key} heldout={len(sv)}", flush=True)
    # spread picks: first, 1/3, 2/3, last
    if len(sv) > A.n_views:
        pick = [sv[int(round(i * (len(sv) - 1) / (A.n_views - 1)))] for i in range(A.n_views)]
    else:
        pick = sv
    for j, view in enumerate(pick):
        cam_vec, photo, view_scene, image_id, pool_i = unpack_eval_view(view)
        cam = Camera(camera_from_vector(np.asarray(cam_vec, np.float32)),
                     device="cuda", downscale=1)
        def rend(t):
            return render_gaussians(*gauss(t, gtm, cen, sc), cam)
        im = rend(pr).clamp(0, 1)
        gtg = rend(gt).clamp(0, 1)
        ref = photo.cuda().float()
        if ref.shape[-2:] != im.shape[-2:]:
            ref = F.interpolate(ref[None], size=im.shape[-2:], mode="bilinear",
                                align_corners=False)[0]
        tag = os.path.splitext(os.path.basename(str(image_id)))[0]
        stem = f"idx{index:03d}_{name}_v{j}_{tag}"
        save_rgb(ref, os.path.join(A.out_dir, f"{stem}_photo.png"))
        save_rgb(gtg, os.path.join(A.out_dir, f"{stem}_gtg.png"))
        save_rgb(im, os.path.join(A.out_dir, f"{stem}_pred.png"))
        rec = {
            "index": index, "name": name, "scene": scene_key,
            "view": tag, "pool": int(pool_i),
            "psnr_vs_gtg": float(psnr(im, gtg)),
            "psnr_vs_photo": float(psnr(im, ref)),
            "psnr_gtg_vs_photo": float(psnr(gtg, ref)),
        }
        summary.append(rec)
        print(json.dumps(rec), flush=True)

json.dump(summary, open(os.path.join(A.out_dir, "summary.json"), "w"), indent=2)
