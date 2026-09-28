"""Save full-frame photo / GT-gaussian / pred for a C1-family checkpoint.

Same camera protocol as measure_c1_gate.py: holdout off, index 166 (step_028630).
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "detail_bottleneck_20260917"))
from common import *

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--out", required=True)
A = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)

D = load(A.ckpt, "val", A.index, 1)
val = D["val"]
val.holdout_own_photo = False
it = val[A.index]
D["it"] = it
print(f"[setup] {it.get('name')} holdout={int(val.holdout_own_photo)} "
      f"cam={it['camera'][:3].tolist()}", flush=True)
model, cfg = D["model"], D["cfg"]
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

def save_rgb(t, path):
    arr = (t.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
    Image.fromarray(arr.round().astype(np.uint8)).save(path)
    print("wrote", path, arr.shape)

save_rgb(photo, f"{A.out}_full_photo.png")
save_rgb(gtg, f"{A.out}_full_gtg.png")
save_rgb(im, f"{A.out}_full_pred.png")

# 2x2: photo | gtg / pred | blank-label strip is unnecessary; stack 1x3
pad = 8
c, h, w = im.shape
canvas = torch.zeros(3, h, w * 3 + pad * 2, device=im.device)
canvas[:, :, 0:w] = photo
canvas[:, :, w + pad:2 * w + pad] = gtg
canvas[:, :, 2 * w + 2 * pad:] = im
save_rgb(canvas, f"{A.out}_full_row.png")
print(f"psnr pred vs gtg {float(psnr(im, gtg)):.3f}  pred vs photo {float(psnr(im, photo)):.3f}")
