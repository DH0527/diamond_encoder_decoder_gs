"""여러 시점으로 속성을 감독하면 난간이 3D 로 서는가.

앞의 두 측정이 여기로 이끈다.
  viewgen : 한 시점만 보고 attr_decoder 를 최적화하면 그 시점 +11dB, 안 본 시점 -1.5dB.
            즉 한 장 오라클은 속성에 대해 아무것도 증명하지 못한다 (속성은 한 시점에서
            과소결정이고, train.py 주석의 측정도 같은 말을 한다).
  swap    : 예측 위치에 GT 속성을 최근접으로 옮기면 난간이 선으로 나온다. 위치는 이미
            난간을 덮고 있고 속성이 벽이다.

그래서 감독은 여러 시점에서, 채점은 감독에 쓰지 않은 시점에서 한다. main view 를
피팅에서 제외해 두면 이미 있는 crop 상자(A_hood / D_letters)가 그대로 미관측 지표가 된다.
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from can3tok.render import photometric_loss

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--iters", type=int, default=600)
ap.add_argument("--lr_d", type=float, default=1e-4)
ap.add_argument("--views", type=int, default=14)     # 풀에서 받아올 시점 수
ap.add_argument("--fit", type=int, default=8)        # 그중 감독에 쓸 수
ap.add_argument("--per_step", type=int, default=2)   # 스텝마다 뽑는 수 (학습 루프와 같은 방식)
ap.add_argument("--scope", default="blocks", choices=["blocks", "attrd"])
ap.add_argument("--edge", type=float, default=1.0)
# 대조군이 실제 학습 손실과 같아야 한다. 학습은 평범한 L1 이 아니라 edge_gain=4 로
# 가중한 L1 에 DSSIM 0.2 를 섞어 쓴다. 그 상태 위에서 Sobel 항이 무엇을 더하는지가
# 물어야 할 질문이다. (--real_loss 0 이면 이전처럼 평범한 L1 + Sobel)
ap.add_argument("--real_loss", type=int, default=0)
ap.add_argument("--edge_gain", type=float, default=4.0)
ap.add_argument("--lam_dssim", type=float, default=0.2)
ap.add_argument("--view_seed", type=int, default=0,
                help="Fixes extra-view sampling so edge-weight runs share cameras.")
ap.add_argument("--out", default="")
A = ap.parse_args()

CROPS = {"A_hood_rail": (440, 880, 120, 230), "B_front_rail": (250, 540, 170, 500),
         "D_letters": (695, 855, 222, 295)}

D = load(A.ckpt, "val", A.index, 1)
model, cfg, val = D["model"], D["cfg"], D["val"]
val.extra_real_views = int(A.views); val.view_downscale = 2; val.keep_extra_fullres = True
# 학습용 홀드아웃은 끈다. data.py 가 시드 없는 default_rng() 로 대체 시점을 뽑기 때문에
# (1063행) 켜두면 프로세스마다 MAIN 시점이 달라져 런 사이 비교가 무효가 된다. 여기서
# 목표는 GT 가우시안 렌더라서 사진 홀드아웃은 재는 것과 무관하다.
val.holdout_own_photo = False
val.extra_view_seed = int(A.view_seed)
it = val[A.index]

x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1,3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()

def mkcam(v): return Camera(camera_from_vector(np.asarray(v, np.float32)), device="cuda", downscale=1)
cam = mkcam(it["camera"].numpy())
def fit_im(im, c):
    if im.shape[-2:] == (c.image_height, c.image_width): return im
    return F.interpolate(im[None], size=(c.image_height, c.image_width), mode="bilinear",
                         align_corners=False)[0]
ref = fit_im(it["photo"].cuda().float(), cam)
allc = [mkcam(v) for v in it["extra_cams"].numpy()]
alli = [fit_im(im.cuda().float(), c) for im, c in zip(it["extra_imgs"], allc)]
nf = min(int(A.fit), max(len(allc) - 1, 0))
fitc, fiti = allc[:nf], alli[:nf]
hldc, hldi = allc[nf:], alli[nf:]
if nf == 0: raise SystemExit("감독에 쓸 시점이 없다.")
print(f"[setup] {it['name']}  pts {int(gtm.sum())}  fit views {len(fitc)}  "
      f"held views {1 + len(hldc)} (main + {len(hldc)})  "
      f"holdout={int(val.holdout_own_photo)} view_seed={A.view_seed} "
      f"real_loss={A.real_loss} w_sobel={A.edge} "
      f"cam0={it['extra_cams'][0,:3].tolist() if it['extra_cams'].numel() else 'none'}")

def rend(t, c): return render_gaussians(*gauss(t, gtm, cen, sc), c)
def sobel(t):
    k = t.new_tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]).view(1,1,3,3).repeat(3,1,1,1)
    gx = F.conv2d(t[None], k, padding=1, groups=3)
    gy = F.conv2d(t[None], k.transpose(2,3), padding=1, groups=3)
    return torch.cat([gx,gy],1)[0]

with torch.no_grad():
    g_main = rend(tg[0], cam).clamp(0,1)
    g_fit  = [rend(tg[0], c).clamp(0,1) for c in fitc]
    g_hld  = [rend(tg[0], c).clamp(0,1) for c in hldc]
E_fit = [sobel(g).detach() for g in g_fit]

def score(pred):
    with torch.no_grad():
        m = rend(pred, cam).clamp(0,1)
        o = {"MAIN_gtg": float(psnr(m, g_main)), "MAIN_photo": float(psnr(m, ref))}
        for cn,(x0,x1,y0,y1) in CROPS.items():
            o[cn] = float(psnr(m[:, y0:y1, x0:x1], g_main[:, y0:y1, x0:x1]))
        f = [float(psnr(rend(pred, c).clamp(0,1), g)) for c, g in zip(fitc, g_fit)]
        h = [float(psnr(rend(pred, c).clamp(0,1), g)) for c, g in zip(hldc, g_hld)]
        o["fit_gtg"] = float(np.mean(f)) if f else 0.0
        o["held_gtg"] = float(np.mean(h)) if h else 0.0
    return o, m

res = {}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
z0 = o["z_compact"].detach().float().clone()
with torch.no_grad():
    base = o.get("attr_pred", o["pred"]).float()[0]
res["base"], img_base = score(base)
print("[base ] " + json.dumps({k: round(v,2) for k,v in res["base"].items()}))

model.train()
for m in model.modules():
    if isinstance(m, torch.nn.Dropout): m.eval()
model.cfg.checkpoint_decode = True
for mod in model.modules():
    if hasattr(mod, "patch_chunk"): mod.patch_chunk = 32
for p in model.parameters(): p.requires_grad_(False)
tgt_mod = model.attr_decoder.blocks if A.scope == "blocks" else model.attr_decoder
bp = list(tgt_mod.parameters())
for p in bp: p.requires_grad_(True)
print(f"[{A.scope}] {sum(p.numel() for p in bp)/1e6:.3f}M params, "
      f"스텝마다 {A.per_step}/{len(fitc)} 시점")

zf = z0.clone()
def pred_now():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = model.decode_compact(zf)
    return d.get("attr_pred", d["pred"]).float()[0]

opt = torch.optim.Adam(bp, lr=A.lr_d)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.iters, eta_min=0.0)
rng = np.random.default_rng(0); t0 = time.time()
for i in range(A.iters):
    opt.zero_grad(set_to_none=True)
    pr = pred_now()
    js = rng.choice(len(fitc), size=min(A.per_step, len(fitc)), replace=False)
    loss = 0.0
    for j in js:
        im = rend(pr, fitc[j])
        if A.real_loss:
            loss = loss + photometric_loss(im, g_fit[j], lam_dssim=A.lam_dssim,
                                           edge_gain=A.edge_gain, w_sobel=A.edge)[0]
        else:
            loss = loss + (im - g_fit[j]).abs().mean() + A.edge * (sobel(im) - E_fit[j]).abs().mean()
    loss = loss / len(js)
    loss.backward(); opt.step(); sch.step()
    if (i+1) % max(A.iters//6,1) == 0:
        s, _ = score(pred_now())
        print(f"[{A.scope}] {i+1:4d}/{A.iters} L1 {float(loss):.5f}  "
              + json.dumps({k: round(v,2) for k,v in s.items()}), flush=True)
res[A.scope], img_fin = score(pred_now())
print(f"[{A.scope}] done in {time.time()-t0:.0f}s")

print("\n===== MAIN 과 held 는 감독에 쓰지 않은 시점 =====")
for k in ["base", A.scope]:
    v = res[k]
    print(f"{k:8s} fit {v['fit_gtg']:6.2f} | held {v['held_gtg']:6.2f}  MAIN {v['MAIN_gtg']:6.2f}  "
          f"A_hood {v['A_hood_rail']:6.2f}  B_front {v['B_front_rail']:6.2f}  D_letters {v['D_letters']:6.2f}")

if A.out:
    json.dump(res, open(A.out + ".json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cn,(x0,x1,y0,y1) in CROPS.items():
        bps = res["base"].get(cn, res["base"]["MAIN_gtg"])
        fps = res[A.scope].get(cn, res[A.scope]["MAIN_gtg"])
        panes = [("photo", ref), ("gtg", g_main),
                 (f"base {bps:.2f}dB", img_base),
                 (f"{A.scope} w_sobel={A.edge} {fps:.2f}dB", img_fin)]
        fig, ax = plt.subplots(len(panes), 1, figsize=(9, 2.2*len(panes)))
        for a,(nm,im) in zip(ax, panes):
            a.imshow(im[:, y0:y1, x0:x1].clamp(0,1).permute(1,2,0).cpu().numpy())
            a.set_title(nm, fontsize=8); a.axis("off")
        fig.tight_layout(); fig.savefig(f"{A.out}_{cn}.png", dpi=115)
        print("wrote", f"{A.out}_{cn}.png")
