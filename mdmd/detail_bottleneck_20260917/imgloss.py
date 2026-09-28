"""어떤 이미지 손실이 난간을 그린 답을 더 좋게 채점하는가.

렌더 손실로 가려면 먼저 이게 성립해야 한다. s_and_r 은 난간이 실제로 선으로 나오는
렌더인데 평범한 PSNR 로는 base 보다 낮다 (18.45 vs 20.42). 손실이 정답을 오답으로
채점하면 그 손실로 학습해도 난간은 안 나온다. 최적점에 난간이 있는 손실을 골라야 한다.

  판정: loss(s_and_r) < loss(base) 여야 그 손실은 난간 쪽으로 민다.
        해상도도 같이 본다 -- 1~2픽셀 구조는 다운스케일에서 손실에 아예 안 보인다.
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/T16k_20260913_093332/ckpt_step00070000.pt")
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--views", type=int, default=4)
A = ap.parse_args()

RAIL = (440, 880, 120, 230)

D = load(A.ckpt, "val", A.index, 1)
model, cfg, val = D["model"], D["cfg"], D["val"]
val.extra_real_views = int(A.views); val.view_downscale = 2; val.keep_extra_fullres = True
it = val[A.index]
x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim][0]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1, 3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
base = o.get("attr_pred", o["pred"]).float()[0]

with torch.no_grad():
    P, T = base[gtm], tg[gtm]
    idx = torch.empty(P.shape[0], dtype=torch.long, device=P.device)
    for s in range(0, P.shape[0], 4096):
        e = min(s + 4096, P.shape[0])
        idx[s:e] = torch.cdist(P[s:e, :3], T[:, :3]).argmin(dim=1)
    M = T[idx]
    ls = M[:, 3:6].clamp(-15., 5.)
    gt_an = (ls.max(-1).values - ls.min(-1).values).exp()
    def sub(sel):
        """GT 스케일+회전을 sel 인 점에만 넣는다. 나머지는 base 그대로."""
        v = base.clone(); live = v[gtm].clone()
        live[sel, 3:6] = M[sel, 3:6]; live[sel, 6:10] = M[sel, 6:10]
        v[gtm] = live
        return v
    allsel = torch.ones_like(gt_an, dtype=torch.bool)
    # 전체를 갈아끼우면 벽/면까지 최근접 전이의 오차를 받는다. 난간·전선·모서리 같은
    # 바늘만 고치면 얻는 것과 잃는 것이 분리된다.
    VARS = [("s_and_r_all", sub(allsel)), ("needles_30", sub(gt_an > 30)),
            ("needles_100", sub(gt_an > 100)), ("needles_300", sub(gt_an > 300))]
    for nm, _ in VARS[1:]:
        t = float(nm.split("_")[1])
        print(f"[{nm}] 바뀐 점 {int((gt_an > t).sum())} / {gt_an.shape[0]}")

def mkcam(v): return Camera(camera_from_vector(np.asarray(v, np.float32)), device="cuda", downscale=1)
cams = [mkcam(it["camera"].numpy())] + [mkcam(v) for v in it["extra_cams"].numpy()]

# ---- 후보 손실 ---------------------------------------------------------------
def _blur(a, ks=5, s=1.5):
    r = ks // 2
    g = torch.exp(-torch.arange(-r, r + 1, device=a.device, dtype=a.dtype) ** 2 / (2 * s * s)); g = g / g.sum()
    a = F.conv2d(a, g.view(1, 1, 1, -1).expand(a.shape[1], 1, 1, ks), padding=(0, r), groups=a.shape[1])
    return F.conv2d(a, g.view(1, 1, -1, 1).expand(a.shape[1], 1, ks, 1), padding=(r, 0), groups=a.shape[1])

def _grad(a):
    gx = a[..., :, 1:] - a[..., :, :-1]
    gy = a[..., 1:, :] - a[..., :-1, :]
    return gx, gy

def l_l1(p, t):     return (p - t).abs().mean()
def l_l2(p, t):     return ((p - t) ** 2).mean()
def l_grad(p, t):
    pgx, pgy = _grad(p); tgx, tgy = _grad(t)
    return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()
def l_hipass(p, t): return ((p - _blur(p)) - (t - _blur(t))).abs().mean()
def l_edgew(p, t):
    """GT 엣지 세기로 가중한 L1. 평평한 면이 예산을 먹는 것을 막는다."""
    tgx, tgy = _grad(t)
    e = F.pad(tgx.abs().mean(1, keepdim=True), (0, 1)) + F.pad(tgy.abs().mean(1, keepdim=True), (0, 0, 0, 1))
    w = 1.0 + 20.0 * e / e.mean().clamp(min=1e-8)
    return ((p - t).abs().mean(1, keepdim=True) * w).sum() / w.sum() / 1.0
def l_gradmag(p, t):
    """엣지 세기 자체를 맞춘다. 흐린 렌더는 엣지가 약해서 반드시 벌점을 받는다."""
    pgx, pgy = _grad(p); tgx, tgy = _grad(t)
    pm = (pgx[..., :-1, :].pow(2) + pgy[..., :, :-1].pow(2) + 1e-12).sqrt()
    tm = (tgx[..., :-1, :].pow(2) + tgy[..., :, :-1].pow(2) + 1e-12).sqrt()
    return (pm - tm).abs().mean()
def l_ssim(p, t):
    mp, mt = _blur(p), _blur(t)
    vp, vt = _blur(p * p) - mp * mp, _blur(t * t) - mt * mt
    cv = _blur(p * t) - mp * mt
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mp * mt + c1) * (2 * cv + c2)) / ((mp * mp + mt * mt + c1) * (vp + vt + c2))
    return 1.0 - s.mean()

LOSSES = [("L1", l_l1), ("L2", l_l2), ("grad_L1", l_grad), ("hipass_L1", l_hipass),
          ("edge_weighted_L1", l_edgew), ("grad_magnitude", l_gradmag), ("1-SSIM", l_ssim)]

def rend(t, c): return render_gaussians(*gauss(t, gtm, cen, sc), c).clamp(0, 1)

def collect(ds, crop_only):
    """해상도 ds 에서 base 대비 각 변형의 손실. 여러 시점 평균."""
    acc = {nm: {v: 0.0 for v, _ in [("base", 0)] + VARS} for nm, _ in LOSSES}
    with torch.no_grad():
        for c in cams:
            if crop_only and c is not cams[0]: continue
            g = rend(tg, c)
            ims = [("base", rend(base, c))] + [(v, rend(t, c)) for v, t in VARS]
            if crop_only:
                x0, x1, y0, y1 = RAIL
                g = g[:, y0:y1, x0:x1]
                ims = [(v, a[:, y0:y1, x0:x1]) for v, a in ims]
            g = g[None]
            if ds > 1: g = F.interpolate(g, scale_factor=1.0 / ds, mode="area")
            for v, a in ims:
                a = a[None]
                if ds > 1: a = F.interpolate(a, scale_factor=1.0 / ds, mode="area")
                for nm, fn in LOSSES: acc[nm][v] += float(fn(a, g))
    return acc

names = ["base"] + [v for v, _ in VARS]
for crop_only in (False, True):
    tag = "난간 crop 만" if crop_only else f"전체 화면, 시점 {len(cams)}개 평균"
    for ds in (1, 2):
        print(f"\n===== {tag}   ds={ds}   (base 대비 % 변화, 음수면 개선) =====")
        r = collect(ds, crop_only)
        print(f"{'손실':>18s} " + "".join(f"{v:>16s}" for v in names[1:]))
        for nm, _ in LOSSES:
            b = r[nm]["base"]
            print(f"{nm:>18s} " + "".join(f"{100*(r[nm][v]-b)/b:+15.1f}%" for v in names[1:]))
print("\n음수가 나오는 칸이 있어야 그 손실로 렌더 학습을 할 때 그 구조가 살아난다.")
