"""이방성만 먼저 키우는 계획이 성립하는지 오라클로 판정한다.

H2 의 전제는 "먼저 길게, 그 다음에 방향" 이었다. 그 전제가 참이려면 방향이 틀린 채로
길어진 중간 상태가 최소한 나빠지지는 않아야 한다. 여기서는 예측에서 스케일만 GT 로
갈아끼우고(회전은 예측 그대로) 렌더한다. 그게 base 보다 나쁘면 순차 계획은 무너진다.

  s_only    스케일만 GT, 회전은 예측(42도 틀림)  <- H2 가 향하는 중간 상태
  r_only    회전만 GT, 스케일은 예측(구에 가까움)
  s_and_r   둘 다 GT
  all_attr  모든 속성 GT (상한)
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/T16k_20260913_093332/ckpt_step00070000.pt")
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--out", required=True)
A = ap.parse_args()

CROPS = {"A_hood_rail": (440, 880, 120, 230), "D_letters": (695, 855, 222, 295)}

D = load(A.ckpt, "val", A.index, 1)
model, cfg, it = D["model"], D["cfg"], D["it"]
x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim][0]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1, 3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()
cam = Camera(camera_from_vector(it["camera"].numpy().astype(np.float32)), device="cuda", downscale=1)
def rend(t): return render_gaussians(*gauss(t, gtm, cen, sc), cam).clamp(0, 1)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
base = o.get("attr_pred", o["pred"]).float()[0]

# 살아있는 슬롯끼리만 최근접. 죽은 슬롯이 섞이면 대응이 오염된다.
with torch.no_grad():
    bl, tl = base[gtm], tg[gtm]
    idx = torch.empty(bl.shape[0], dtype=torch.long, device=bl.device)
    for s in range(0, bl.shape[0], 4096):
        e = min(s + 4096, bl.shape[0])
        idx[s:e] = torch.cdist(bl[s:e, :3], tl[:, :3]).argmin(dim=1)
    matched = tl[idx]                                   # 예측 점마다 대응하는 GT 속성

SL, RQ = slice(3, 6), slice(6, 10)
def variant(use_scale, use_rot, use_all=False):
    v = base.clone(); live = v[gtm]
    if use_all: live[:, 3:] = matched[:, 3:]
    else:
        if use_scale: live[:, SL] = matched[:, SL]
        if use_rot:   live[:, RQ] = matched[:, RQ]
    v[gtm] = live
    return v

VAR = [("base", base), ("s_only", variant(1, 0)), ("r_only", variant(0, 1)),
       ("s_and_r", variant(1, 1)), ("all_attr", variant(0, 0, True))]

p = it["photo"].cuda().float()
ref = p if p.shape[-2:] == (cam.image_height, cam.image_width) else \
    F.interpolate(p[None], size=(cam.image_height, cam.image_width), mode="bilinear", align_corners=False)[0]
gtg = rend(tg)

panes, stats = [], {}
with torch.no_grad():
    for nm, t in VAR:
        im = rend(t)
        ls = t[gtm][:, SL].clamp(-15, 5)
        an = (ls.max(-1).values - ls.min(-1).values).exp()
        st = {"psnr_vs_gtg": float(psnr(im, gtg)), "aniso_p50": float(torch.quantile(an.float(), 0.5))}
        for cn, (x0, x1, y0, y1) in CROPS.items():
            st[cn] = float(psnr(im[:, y0:y1, x0:x1], gtg[:, y0:y1, x0:x1]))
        stats[nm] = st; panes.append((nm, im.cpu()))
        print(f"[{nm:9s}] " + json.dumps({k: round(v, 3) for k, v in st.items()}), flush=True)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [("photo", ref.cpu()), ("gt gaussians", gtg.cpu())] + panes
for cn, (x0, x1, y0, y1) in CROPS.items():
    fig, ax = plt.subplots(len(rows), 1, figsize=(9.5, 2.1 * len(rows)))
    for a, (nm, im) in zip(ax, rows):
        a.imshow(im[:, y0:y1, x0:x1].detach().permute(1, 2, 0).numpy())
        t = nm if nm not in stats else f"{nm}   aniso p50 {stats[nm]['aniso_p50']:.2f}   crop PSNR {stats[nm][cn]:.2f}"
        a.set_title(t, fontsize=8); a.axis("off")
    fig.tight_layout(); fig.savefig(f"{A.out}_{cn}.png", dpi=120)
    print("wrote", f"{A.out}_{cn}.png")
json.dump(stats, open(A.out + ".json", "w"), indent=1)
