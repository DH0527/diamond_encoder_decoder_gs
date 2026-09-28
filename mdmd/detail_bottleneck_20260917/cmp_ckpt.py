"""체크포인트 두 개를 같은 시점에서 렌더해 crop 으로 나란히 본다.

PSNR 은 이 문제에 눈이 멀었다 (난간을 제대로 그리는 오라클 렌더가 base 보다 1.5dB 낮다).
그래서 숫자와 함께 그림을 봐야 한다. 숫자는 참고로만 찍는다.
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)      # 이름=경로
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--out", required=True)
A = ap.parse_args()

CROPS = {"A_hood_rail": (440, 880, 120, 230), "B_front_rail": (250, 540, 170, 500),
         "D_letters": (695, 855, 222, 295)}

panes, stats = [], {}
ref = gtg = None
for spec in A.ckpts:
    name, path = spec.split("=", 1)
    D = load(path, "val", A.index, 1)
    model, cfg, it = D["model"], D["cfg"], D["it"]
    x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
    tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
    gtm = it["mask"].cuda() > 0.5
    cen = it["center"].cuda().float().view(1,3); sc = float(it["scale"])
    kw = {}
    if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
    if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()
    cam = Camera(camera_from_vector(it["camera"].numpy().astype(np.float32)), device="cuda", downscale=1)
    def rend(t): return render_gaussians(*gauss(t, gtm, cen, sc), cam)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(x, mk, run_decode=True, run_gen=False, **kw)
    with torch.no_grad():
        pr = o.get("attr_pred", o["pred"]).float()[0]
        im = rend(pr).clamp(0, 1)
        if ref is None:
            p = it["photo"].cuda().float()
            ref = p if p.shape[-2:] == (cam.image_height, cam.image_width) else \
                F.interpolate(p[None], size=(cam.image_height, cam.image_width),
                              mode="bilinear", align_corners=False)[0]
            gtg = rend(tg[0]).clamp(0, 1)
        # 이방성도 같이. 난간은 여기서 결정된다.
        ls = pr[gtm][:, 3:6].clamp(-15, 5)
        an = (ls.max(-1).values - ls.min(-1).values).exp()
        stats[name] = {"psnr_vs_gtg": float(psnr(im, gtg)), "psnr_vs_photo": float(psnr(im, ref)),
                       "aniso_p50": float(torch.quantile(an.float(), 0.5)),
                       "aniso_lt4_pct": 100*float((an < 4).float().mean())}
        for cn, (x0,x1,y0,y1) in CROPS.items():
            stats[name][cn] = float(psnr(im[:, y0:y1, x0:x1], gtg[:, y0:y1, x0:x1]))
    panes.append((name, im.cpu()))
    print(f"[{name:10s}] " + json.dumps({k: round(v,3) for k,v in stats[name].items()}), flush=True)
    del model, o
    torch.cuda.empty_cache()

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [("photo", ref.cpu()), ("gt gaussians", gtg.cpu())] + panes
for cn, (x0,x1,y0,y1) in CROPS.items():
    fig, ax = plt.subplots(len(rows), 1, figsize=(9.5, 2.3*len(rows)))
    for a, (nm, im) in zip(ax, rows):
        a.imshow(im[:, y0:y1, x0:x1].clamp(0,1).permute(1,2,0).numpy())
        t = nm if nm in ("photo", "gt gaussians") else \
            f"{nm}   aniso p50 {stats[nm]['aniso_p50']:.2f}   <4x {stats[nm]['aniso_lt4_pct']:.1f}%"
        a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{A.out}_{cn}.png", dpi=120)
    print("wrote", f"{A.out}_{cn}.png")
json.dump(stats, open(A.out + ".json", "w"), indent=1)
