"""어떤 손실 형태가 스케일과 회전을 동시에 가르치는가.

aniso_first.py 가 확정한 것: 난간은 스케일과 회전이 같이 맞아야 나온다. 스케일만
맞추면 42도 틀린 방향으로 길어져 가시가 되고, 이건 base 보다 나쁘다 (17.95 vs 21.89).
따라서 필요한 것은 단계가 아니라 둘을 같이 미는 손실이다.

여기서는 후보 손실을 실제 attr_decoder 에 피팅한다. 용량이 제한된 상태에서 그 손실을
끝까지 따라가면 어디에 도달하는지가 질문이다. 한 샘플에 과적합시키는 것은 의도한
것이다 -- 가장 관대한 조건에서도 난간이 안 나오면 학습에서는 더 안 나온다.

  direct  log-scale L1 + quat L1        이전에 시도해서 실패한 것. GT 축 순서가 무작위라
                                        같은 타원체를 축 이름만 다르게 쓴 경우도 벌점을 받는다
  cur     현재 covariance3d_loss        크기 오차가 방향을 덮는다 (GT 회전 줘도 0.0% 감소)
  trace   trace 정규화 공분산 + 크기항   크기를 나눠내 모양과 방향만 남긴다
  whiten  Ct^-1/2 Cp Ct^-1/2 vs I       세 축의 상대 오차를 같은 무게로
  h2      이방성 + cur                  지금 돌던 것

Σ = R S² R^T 는 축 순서에 불변이므로 공분산 계열은 direct 의 모호성이 없다.
"""
import os, sys, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from can3tok.losses import _quat_to_R

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/T16k_20260913_093332/ckpt_step00070000.pt")
ap.add_argument("--form", required=True, choices=["direct", "cur", "trace", "whiten", "h2"])
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--iters", type=int, default=600)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--views", type=int, default=4)
# 한계가 어디인지 가르는 축. z 를 열면 인코더가 z 에 정보를 못 담은 탓인지 분리되고,
# scope=all 로 열면 현재 구조 전체의 천장이 나온다.
ap.add_argument("--free_z", type=int, default=0)
ap.add_argument("--scope", default="attrd", choices=["attrd", "dec", "none"])
ap.add_argument("--lr_z", type=float, default=3e-3)
ap.add_argument("--out", default="")
A = ap.parse_args()

CROPS = {"A_hood_rail": (440, 880, 120, 230), "D_letters": (695, 855, 222, 295)}

D = load(A.ckpt, "val", A.index, 1)
model, cfg, val = D["model"], D["cfg"], D["val"]
val.extra_real_views = int(A.views); val.view_downscale = 2; val.keep_extra_fullres = True
it = val[A.index]
x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
cen = it["center"].cuda().float().view(1, 3); sc = float(it["scale"])
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()

def mkcam(v): return Camera(camera_from_vector(np.asarray(v, np.float32)), device="cuda", downscale=1)
def fit_im(im, c):
    if im.shape[-2:] == (c.image_height, c.image_width): return im
    return F.interpolate(im[None], size=(c.image_height, c.image_width), mode="bilinear", align_corners=False)[0]
cam = mkcam(it["camera"].numpy()); ref = fit_im(it["photo"].cuda().float(), cam)
ecams = [mkcam(v) for v in it["extra_cams"].numpy()]
def rend(t, c): return render_gaussians(*gauss(t, gtm, cen, sc), c)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
z0 = o["z_compact"].detach().float().clone()
with torch.no_grad():
    base = o.get("attr_pred", o["pred"]).float()[0]
gt = tg[0]

def nn_take(q, src):
    idx = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
    for s in range(0, q.shape[0], 4096):
        e = min(s + 4096, q.shape[0])
        idx[s:e] = torch.cdist(q[s:e, :3], src[:, :3]).argmin(dim=1)
    return src[idx]
with torch.no_grad():
    TA = gt[:, 3:].clone()
    TA[gtm] = nn_take(base[gtm], gt[gtm])[:, 3:]
    TA = TA.contiguous()
    TL = TA[gtm]                                        # 목표: [log_scale 3, quat 4, op 1, sh ...]

# ---- 목표 공분산과 화이트닝 행렬은 한 번만 -----------------------------------
def cov(ls, q):
    s = torch.exp(ls.clamp(-15.0, 5.0)); R = _quat_to_R(F.normalize(q, dim=-1))
    return torch.einsum("...ij,...j,...kj->...ik", R, s * s, R)
with torch.no_grad():
    Ct = cov(TL[:, 0:3], TL[:, 3:7])
    nrm = Ct.flatten(-2).norm(dim=-1)
    n_cur = nrm.clamp(min=0.01 * nrm.median().clamp(min=1e-12))[..., None, None]
    tr_t = Ct.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-30)
    ev, EV = torch.linalg.eigh(Ct.double())
    ev = ev.clamp(min=ev.max(dim=-1, keepdim=True).values * 1e-6)     # 상대 하한. 절대 하한은 inf 를 만든다
    W = ((EV * ev.rsqrt().unsqueeze(-2)) @ EV.transpose(-1, -2)).float()
    I3 = torch.eye(3, device=Ct.device)

def obj(p):
    """후보 손실. opacity 와 sh 의 L1 은 모든 후보에 공통으로 둔다 (질문 대상이 아니다)."""
    pl = p[gtm]
    ps, pq, po, pc = pl[:, 3:6], pl[:, 6:10], pl[:, 10:11], pl[:, 11:]
    ts, tq, to, tc = TL[:, 0:3], TL[:, 3:7], TL[:, 7:8], TL[:, 8:]
    L = (po - to).abs().mean() + (pc - tc).abs().mean()
    if A.form == "direct":
        qn, qt = F.normalize(pq, dim=-1), F.normalize(tq, dim=-1)
        sgn = torch.sign((qn * qt).sum(-1, keepdim=True)); sgn = torch.where(sgn == 0, 1.0, sgn)
        return L + (ps - ts).abs().mean() + (qn - sgn * qt).abs().mean()
    Cp = cov(ps, pq)
    if A.form in ("cur", "h2"):
        L = L + ((Cp / n_cur - Ct / n_cur) ** 2).flatten(-2).sum(-1).mean()
        if A.form == "h2":
            # 정렬 후 평균 제거: 크기와 방향을 빼고 순수 모양만
            a = ps.clamp(-15.0, 5.0).sort(dim=-1, descending=True).values
            b = ts.clamp(-15.0, 5.0).sort(dim=-1, descending=True).values
            L = L + (ps - ts).abs().mean() + \
                ((a - a.mean(-1, keepdim=True)) - (b - b.mean(-1, keepdim=True))).abs().mean()
        return L
    if A.form == "trace":
        tp = Cp.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-30)
        shape = ((Cp / tp[..., None, None] - Ct / tr_t[..., None, None]) ** 2).flatten(-2).sum(-1).mean()
        size = (tp.log() - tr_t.log()).abs().mean()          # 크기는 따로, 로그로
        return L + shape + 0.1 * size
    M = W @ Cp @ W                                                   # whiten
    return L + ((M - I3) ** 2).flatten(-2).sum(-1).mean()

def diag(p):
    """이방성과 방향 오차. 이 두 개가 aniso_first 의 s_and_r 로 가야 한다."""
    with torch.no_grad():
        pl = p[gtm]
        ls = pl[:, 3:6].clamp(-15.0, 5.0)
        an = (ls.max(-1).values - ls.min(-1).values).exp()
        qn = F.normalize(pl[:, 6:10], dim=-1); qt = F.normalize(TL[:, 3:7], dim=-1)
        ang = 2.0 * (qn * qt).sum(-1).abs().clamp(max=1.0).acos() * 180.0 / np.pi
        gan = (TL[:, 0:3].clamp(-15.0, 5.0).max(-1).values - TL[:, 0:3].clamp(-15.0, 5.0).min(-1).values).exp()
        big = gan > 8
        return {"aniso_p50": float(torch.quantile(an.float(), 0.5)),
                "rot_deg_p50": float(torch.quantile(ang.float(), 0.5)),
                "rot_deg_p50_gtaniso8": float(torch.quantile(ang[big].float(), 0.5))}

with torch.no_grad():
    g_main = rend(gt, cam).clamp(0, 1)
    g_hld = [rend(gt, c).clamp(0, 1) for c in ecams]

def score(p):
    with torch.no_grad():
        m = rend(p, cam).clamp(0, 1)
        s = {"MAIN_gtg": float(psnr(m, g_main))}
        for cn, (x0, x1, y0, y1) in CROPS.items():
            s[cn] = float(psnr(m[:, y0:y1, x0:x1], g_main[:, y0:y1, x0:x1]))
        h = [float(psnr(rend(p, c).clamp(0, 1), g)) for c, g in zip(ecams, g_hld)]
        s["held_gtg"] = float(np.mean(h)) if h else 0.0
        s.update(diag(p))
    return s, m

res = {}
res["base"], img_base = score(base)
print("[base    ] " + json.dumps({k: round(v, 3) for k, v in res["base"].items()}), flush=True)
with torch.no_grad():
    sr = base.clone(); live = sr[gtm].clone()
    live[:, 3:6] = TL[:, 0:3]; live[:, 6:10] = TL[:, 3:7]; sr[gtm] = live   # aniso_first 의 s_and_r
res["s_and_r"], img_sr = score(sr)
print("[s_and_r ] " + json.dumps({k: round(v, 3) for k, v in res["s_and_r"].items()}), flush=True)

model.train()
for m in model.modules():
    if isinstance(m, torch.nn.Dropout): m.eval()
model.cfg.checkpoint_decode = True
for mod in model.modules():
    if hasattr(mod, "patch_chunk"): mod.patch_chunk = 32
for p in model.parameters(): p.requires_grad_(False)
if A.scope == "attrd":  bp = list(model.attr_decoder.parameters())
elif A.scope == "dec":  bp = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
else:                   bp = []
for p in bp: p.requires_grad_(True)
zf = z0.clone().requires_grad_(bool(A.free_z))
groups = ([{"params": bp, "lr": A.lr}] if bp else []) + \
         ([{"params": [zf], "lr": A.lr_z}] if A.free_z else [])
print(f"[scope {A.scope} free_z {A.free_z}] {sum(p.numel() for p in bp)/1e6:.3f}M params"
      f"{' + z ' + str(tuple(zf.shape)) if A.free_z else ''}")

def pred_now():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = model.decode_compact(zf)
    return d.get("attr_pred", d["pred"]).float()[0]

opt = torch.optim.Adam(groups)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, A.iters, eta_min=0.0)
t0 = time.time()
for i in range(A.iters):
    opt.zero_grad(set_to_none=True)
    obj(pred_now()).backward(); opt.step(); sch.step()
    if (i + 1) % max(A.iters // 6, 1) == 0:
        s, _ = score(pred_now())
        print(f"[{A.form:7s}] {i+1:4d}/{A.iters}  " + json.dumps({k: round(v, 3) for k, v in s.items()}), flush=True)
res[A.form], img_fin = score(pred_now())
print(f"[{A.form:7s}] {time.time()-t0:.0f}s")

print(f"\n===== {A.form} : 손실은 이미지를 보지 않는다 =====")
for k in ["base", A.form, "s_and_r"]:
    v = res[k]
    print(f"{k:8s} aniso {v['aniso_p50']:6.2f}  rot {v['rot_deg_p50_gtaniso8']:5.1f}deg | "
          f"MAIN {v['MAIN_gtg']:6.2f}  held {v['held_gtg']:6.2f}  "
          f"A_hood {v['A_hood_rail']:6.2f}  D_letters {v['D_letters']:6.2f}")

if A.out:
    json.dump(res, open(A.out + ".json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for cn, (x0, x1, y0, y1) in CROPS.items():
        panes = [("photo", ref), ("gtg", g_main), ("base", img_base),
                 (A.form, img_fin), ("s_and_r (oracle)", img_sr)]
        fig, ax = plt.subplots(len(panes), 1, figsize=(9, 2.2 * len(panes)))
        for a, (nm, im) in zip(ax, panes):
            a.imshow(im[:, y0:y1, x0:x1].clamp(0, 1).detach().permute(1, 2, 0).cpu().numpy())
            t = nm if nm not in res else f"{nm}   aniso {res[nm]['aniso_p50']:.2f}  rot {res[nm]['rot_deg_p50_gtaniso8']:.0f}deg  PSNR {res[nm][cn]:.2f}"
            a.set_title(t, fontsize=8); a.axis("off")
        fig.tight_layout(); fig.savefig(f"{A.out}_{cn}.png", dpi=115)
        print("wrote", f"{A.out}_{cn}.png")
