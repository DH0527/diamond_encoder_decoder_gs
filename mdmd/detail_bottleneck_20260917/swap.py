"""난간이 없는 이유가 위치인가 속성인가. 최적화 없이 교차 렌더로 가른다.

viewgen 이 보여준 것: 한 시점만 보고 attr_decoder 를 최적화하면 그 시점은 +9dB,
안 본 시점은 -1dB. 즉 사다리의 이득은 3D 가 아니었다. 그러면 남은 질문은 단순하다.

  mix_geo  = 예측 위치 + GT 속성   난간이 나오면 위치는 이미 맞고 속성이 벽
  mix_attr = GT 위치   + 예측 속성   난간이 나오면 속성은 이미 맞고 위치가 벽

렌더 목적함수는 attr_detach_geometry=1 때문에 위치를 건드릴 수 없으므로, 위치가
벽이라면 렌더 손실을 아무리 키워도 난간은 나오지 않는다.
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--views", type=int, default=6)
ap.add_argument("--out", default="")
A = ap.parse_args()

CROPS = {"A_hood_rail": (440, 880, 120, 230), "B_front_rail": (250, 540, 170, 500),
         "D_letters": (695, 855, 222, 295)}

D = load(A.ckpt, "val", A.index, 1)
model, cfg, val = D["model"], D["cfg"], D["val"]
val.extra_real_views = int(A.views); val.view_downscale = 2; val.keep_extra_fullres = True
it = val[A.index]

x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1,3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()

def mkcam(v): return Camera(camera_from_vector(np.asarray(v, np.float32)), device="cuda", downscale=1)
def fit(im, c):
    if im.shape[-2:] == (c.image_height, c.image_width): return im
    return F.interpolate(im[None], size=(c.image_height, c.image_width), mode="bilinear",
                         align_corners=False)[0]

cam = mkcam(it["camera"].numpy()); ref = fit(it["photo"].cuda().float(), cam)
ecams = [mkcam(v) for v in it["extra_cams"].numpy()]
eimgs = [fit(im.cuda().float(), c) for im, c in zip(it["extra_imgs"], ecams)]
def rend(t, c): return render_gaussians(*gauss(t, gtm, cen, sc), c)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
pred = o.get("attr_pred", o["pred"]).float()[0]
gt = tg[0]
print(f"[setup] {it['name']}  pts {int(gtm.sum())}  held views {len(ecams)}  dim {gt.shape[-1]}")

def nn_take(q, src):
    """q 의 각 점에 대해 src 에서 3D 최근접 점의 속성을 가져온다.

    index 대응을 가정하면 안 된다. 속성 손실이 index-wise 가 아니라 집합 매칭
    (attr_match_mode=responsibility) 이라서 예측 i번과 GT i번은 같은 대상이 아니고,
    실제로 index 로 섞으면 base 20.42 가 13 으로 무너진다.
    """
    ql, sl = q[gtm], src[gtm]                  # 죽은 슬롯(182,323개)은 후보에서 뺀다
    idx = torch.empty(ql.shape[0], dtype=torch.long, device=q.device)
    for s in range(0, ql.shape[0], 4096):
        e = min(s + 4096, ql.shape[0])
        idx[s:e] = torch.cdist(ql[s:e, :3], sl[:, :3]).argmin(dim=1)
    out = q.clone()
    out[gtm] = torch.cat([ql[:, :3], sl[idx, 3:]], dim=-1)
    return out

VAR = {
    "gtg":      gt,
    "base":     pred,
    # 예측 위치 + (가장 가까운 GT 의) GT 속성. 난간이 나오면 위치는 이미 맞다.
    "pxyz_gatt": nn_take(pred, gt),
    # GT 위치 + (가장 가까운 예측의) 예측 속성. 난간이 나오면 속성은 이미 맞다.
    "gxyz_patt": nn_take(gt, pred),
    # 기계 점검: GT 위치에 GT 속성을 최근접으로 옮기면 gtg 와 같아야 한다.
    "nn_sanity": nn_take(gt, gt),
}

res, imgs = {}, {}
with torch.no_grad():
    g_main = rend(gt, cam).clamp(0,1)
    g_held = [rend(gt, c).clamp(0,1) for c in ecams]
    for k, t in VAR.items():
        m = rend(t, cam).clamp(0,1); imgs[k] = m
        r = {"main_photo": float(psnr(m, ref)), "main_gtg": float(psnr(m, g_main))}
        for cn, (x0,x1,y0,y1) in CROPS.items():
            r[cn] = float(psnr(m[:, y0:y1, x0:x1], g_main[:, y0:y1, x0:x1]))
        hp, hg = [], []
        for c, gh, ph in zip(ecams, g_held, eimgs):
            q = rend(t, c).clamp(0,1)
            hp.append(float(psnr(q, ph))); hg.append(float(psnr(q, gh)))
        r["held_photo"] = float(np.mean(hp)); r["held_gtg"] = float(np.mean(hg))
        res[k] = r
        print(f"[{k:9s}] " + json.dumps({a: round(b,2) for a,b in r.items()}), flush=True)

print("\n===== 위치 대 속성 (main view, GT 가우시안 렌더 기준) =====")
for k in ["base","pxyz_gatt","gxyz_patt","nn_sanity","gtg"]:
    v = res[k]
    print(f"{k:9s} A_hood {v['A_hood_rail']:6.2f}  B_front {v['B_front_rail']:6.2f}  "
          f"D_letters {v['D_letters']:6.2f}  | held_photo {v['held_photo']:6.2f}")

if A.out:
    json.dump(res, open(A.out + ".json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["gtg", "base", "pxyz_gatt", "gxyz_patt"]
    for cn, (x0,x1,y0,y1) in CROPS.items():
        fig, ax = plt.subplots(len(order)+1, 1, figsize=(9, 2.2*(len(order)+1)))
        ax[0].imshow(ref[:, y0:y1, x0:x1].permute(1,2,0).cpu().numpy())
        ax[0].set_title("photo", fontsize=8); ax[0].axis("off")
        for i, k in enumerate(order):
            ax[i+1].imshow(imgs[k][:, y0:y1, x0:x1].permute(1,2,0).cpu().numpy())
            ax[i+1].set_title(k, fontsize=8); ax[i+1].axis("off")
        fig.tight_layout(); fig.savefig(f"{A.out}_{cn}.png", dpi=115)
        print("wrote", f"{A.out}_{cn}.png")
