"""Render full-frame photo / GTG / pred for several val indices. Loads the ckpt once."""
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "detail_bottleneck_20260917"))
from common import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--indices", default="25,76,127,166,229,280,331,382")
ap.add_argument("--out_dir", required=True)
A = ap.parse_args()
os.makedirs(A.out_dir, exist_ok=True)
indices = [int(x) for x in A.indices.split(",") if x]

D = load(A.ckpt, "val", indices[0], 1)
val = D["val"]
val.holdout_own_photo = False
model, cfg = D["model"], D["cfg"]
summary = []

def save_rgb(t, path):
    arr = (t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
    Image.fromarray(arr.round().astype(np.uint8)).save(path)

for index in indices:
    it = val[index]
    name = str(it.get("name", f"idx{index}")).replace(".npz", "")
    scene = int(it.get("scene_id", -1)) if "scene_id" in it else -1
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
    cam = Camera(camera_from_vector(it["camera"].numpy().astype(np.float32)),
                 device="cuda", downscale=1)

    def rend(t):
        return render_gaussians(*gauss(t, gtm, cen, sc), cam)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(x, mk, run_decode=True, run_gen=False, **kw)
    pr = o.get("attr_pred", o["pred"]).float()[0]
    gt = tg[0]
    photo = it["photo"].cuda().float()
    if photo.shape[-2:] != (cam.image_height, cam.image_width):
        photo = F.interpolate(photo[None], size=(cam.image_height, cam.image_width),
                              mode="bilinear", align_corners=False)[0]
    im = rend(pr).clamp(0, 1)
    gtg = rend(gt).clamp(0, 1)
    stem = f"idx{index:03d}_{name}"
    save_rgb(photo, os.path.join(A.out_dir, f"{stem}_photo.png"))
    save_rgb(gtg, os.path.join(A.out_dir, f"{stem}_gtg.png"))
    save_rgb(im, os.path.join(A.out_dir, f"{stem}_pred.png"))
    pad = 8
    _, h, w = im.shape
    canvas = torch.zeros(3, h, w * 3 + pad * 2, device=im.device)
    canvas[:, :, 0:w] = photo
    canvas[:, :, w + pad:2 * w + pad] = gtg
    canvas[:, :, 2 * w + 2 * pad:] = im
    save_rgb(canvas, os.path.join(A.out_dir, f"{stem}_row.png"))
    rec = {
        "index": index,
        "name": name,
        "scene": scene,
        "cam": it["camera"][:3].tolist(),
        "hw": [int(cam.image_height), int(cam.image_width)],
        "psnr_vs_gtg": float(psnr(im, gtg)),
        "psnr_vs_photo": float(psnr(im, photo)),
        "psnr_gtg_vs_photo": float(psnr(gtg, photo)),
    }
    summary.append(rec)
    print(json.dumps(rec), flush=True)

json.dump(summary, open(os.path.join(A.out_dir, "summary.json"), "w"), indent=2)
