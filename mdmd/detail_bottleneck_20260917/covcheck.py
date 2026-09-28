"""공분산 손실이 난간을 볼 수 있는가. 방법 문제인지 전달 문제인지 가른다.

covariance3d_loss 는 d = ||Cp/n - Ct/n||^2, n = clamp(||Ct||_F, min=0.01*median).
얇은 가우시안은 ||Ct||_F ~ sigma_max^2 로 작아서 하한에 걸릴 수 있고, 걸리면 그 점의
오차는 scale-free 가 아니라 절대값이 되어 기여가 0 으로 눌린다.

그래서 세 가지를 재고, 셋 다 같은 방향을 가리키는지 본다.
  1. 하한에 걸리는 점의 비율 (전체 / 이방성 8배 초과)
  2. 버킷별 손실 기여 몫: 총합의 몇 %가 얇은 것에서 오는가
  3. 민감도: 예측의 회전만 GT 로 바꿨을 때 / 스케일만 바꿨을 때 손실이 얼마나 내려가는가
     회전을 정답으로 줘도 손실이 안 내려가면 그 손실로는 회전을 가르칠 수 없다.

대응은 최근접 GT 로 잡는다. 실제 손실은 sinkhorn 매칭을 쓰므로 이 수치는 근사지만,
하한과 정규화가 만드는 구조적 성질은 매칭과 무관하다.
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from can3tok.losses import _quat_to_R

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--out", default="")
A = ap.parse_args()

D = load(A.ckpt, "val", A.index, 1)
model, cfg, it = D["model"], D["cfg"], D["it"]
x  = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
gtm = it["mask"].cuda() > 0.5
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
pl = o.get("attr_pred", o["pred"]).float()[0][gtm]
gl = tg[0][gtm]

def nn_idx(q, src):
    i = torch.empty(q.shape[0], dtype=torch.long, device=q.device)
    for s in range(0, q.shape[0], 4096):
        e = min(s+4096, q.shape[0]); i[s:e] = torch.cdist(q[s:e,:3], src[:,:3]).argmin(1)
    return i
gl = gl[nn_idx(pl, gl)]                        # per-point 대응

def cov(ls, q):
    s = torch.exp(ls.clamp(-15., 5.)); R = _quat_to_R(F.normalize(q, dim=-1))
    return torch.einsum("...ij,...j,...kj->...ik", R, s*s, R)

Ct = cov(gl[:, 3:6], gl[:, 6:10])
nrm = Ct.flatten(-2).norm(dim=-1)
floor = 0.01 * nrm.median().clamp(min=1e-12)
n = nrm.clamp(min=floor)[..., None, None]
def d_of(ls, q):
    return ((cov(ls, q)/n - Ct/n)**2).flatten(-2).sum(-1)

ls_p, q_p = pl[:, 3:6], pl[:, 6:10]
ls_t, q_t = gl[:, 3:6], gl[:, 6:10]
V = {"pred": d_of(ls_p, q_p),
     "GT rot 만":   d_of(ls_p, q_t),
     "GT scale 만": d_of(ls_t, q_p),
     "GT 둘 다":    d_of(ls_t, q_t)}

lsq = ls_t.clamp(-15,5)
aniso = (lsq.max(-1).values - lsq.min(-1).values).exp()
under = (nrm < floor)
print(f"[setup] 점 {pl.shape[0]}  floor {float(floor):.3e}  ||Ct|| p10 {float(torch.quantile(nrm,0.1)):.3e} "
      f"p50 {float(nrm.median()):.3e} p90 {float(torch.quantile(nrm,0.9)):.3e}")
print(f"[하한  ] 하한에 걸린 점 {int(under.sum())}/{under.numel()} = {100*float(under.float().mean()):.1f}%"
      f"   이방성 8배 초과 중에서는 {100*float(under[aniso>8].float().mean()):.1f}%")

tot = float(V["pred"].sum())
print(f"\n[기여  ] 총 손실합 {tot:.3e} 의 버킷별 몫")
res = {"floor": float(floor), "under_frac": float(under.float().mean()), "buckets": {}}
for lo, hi in [(1,2),(2,4),(4,8),(8,1e9)]:
    m = (aniso >= lo) & (aniso < hi)
    if int(m.sum()) < 100: continue
    tag = f"{lo}-{hi}" if hi < 1e8 else f">{lo}"
    share = 100*float(V["pred"][m].sum())/tot
    row = {"n": int(m.sum()), "share_pct": share}
    print(f"  aniso {tag:>4s}  n {int(m.sum()):6d}  손실 몫 {share:5.1f}%  "
          f"점당 평균 d {float(V['pred'][m].mean()):.3e}")
    for k, v in V.items():
        row[k] = float(v[m].mean())
    res["buckets"][tag] = row

print(f"\n[민감도] 버킷별 점당 평균 d. 회전을 정답으로 줘도 안 내려가면 그 손실은 회전을 못 가르친다")
hdr = f"  {'aniso':>5s}  " + "  ".join(f"{k:>11s}" for k in V)
print(hdr)
for tag, row in res["buckets"].items():
    print(f"  {tag:>5s}  " + "  ".join(f"{row[k]:11.3e}" for k in V))
print("\n[전체  ] " + "  ".join(f"{k} {float(v.mean()):.3e}" for k, v in V.items()))

# ---- 대안 형태가 회전에 반응하는가 ------------------------------------------
# 현재 형태의 문제는 크기 오차가 방향 오차를 가린다는 것이다. 그러니 크기를 나눠낸
# 형태들을 같은 민감도 테스트에 걸어, 회전만 정답으로 줬을 때 실제로 내려가는지 본다.
def d_cur(ls, q):  return ((cov(ls,q)/n - Ct/n)**2).flatten(-2).sum(-1)
def d_trace(ls, q):
    """trace 로 나눠 크기를 제거. 남는 것은 모양 + 방향."""
    C = cov(ls, q)
    tp = C.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-30)[..., None, None]
    tt = Ct.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(min=1e-30)[..., None, None]
    return ((C/tp - Ct/tt)**2).flatten(-2).sum(-1)
ev, EV = torch.linalg.eigh(Ct.double())
W = (EV * ev.clamp(min=1e-30).rsqrt().unsqueeze(-2)) @ EV.transpose(-1,-2)   # Ct^{-1/2}
I3 = torch.eye(3, dtype=torch.float64, device=Ct.device)
def d_whiten(ls, q):
    """Ct^{-1/2} Cp Ct^{-1/2} 를 I 와 비교. 세 축의 상대 오차를 같은 무게로 본다."""
    M = W @ cov(ls, q).double() @ W
    return ((M - I3)**2).flatten(-2).sum(-1).float()

def sens(fn, name):
    a, b, c = fn(ls_p, q_p).mean(), fn(ls_p, q_t).mean(), fn(ls_t, q_p).mean()
    drop_r = 100.0*(1.0 - float(b)/float(a)) if float(a) > 0 else 0.0
    drop_s = 100.0*(1.0 - float(c)/float(a)) if float(a) > 0 else 0.0
    print(f"  {name:22s} pred {float(a):10.3e}  GT rot {float(b):10.3e} ({drop_r:5.1f}% 감소)"
          f"  GT scale {float(c):10.3e} ({drop_s:5.1f}% 감소)")
    return {"pred": float(a), "gt_rot": float(b), "gt_scale": float(c),
            "rot_drop_pct": drop_r, "scale_drop_pct": drop_s}

print("\n[대안  ] 회전만 정답으로 줬을 때 손실이 내려가는 형태여야 회전을 가르칠 수 있다")
res["forms"] = {"current_frob_norm": sens(d_cur, "현재 (||Ct|| 정규화)"),
                "trace_normalised":  sens(d_trace, "trace 정규화"),
                "whitened":          sens(d_whiten, "whitened (Ct^-1/2)")}
m8 = aniso > 8
print("  -- 이방성 8배 초과만 --")
for nm, fn in [("현재", d_cur), ("trace 정규화", d_trace), ("whitened", d_whiten)]:
    a, b = float(fn(ls_p,q_p)[m8].mean()), float(fn(ls_p,q_t)[m8].mean())
    print(f"  {nm:22s} pred {a:10.3e}  GT rot {b:10.3e} ({100*(1-b/a) if a>0 else 0:5.1f}% 감소)")

# ---- 왜 어떤 형태로도 회전이 안 배워지는가 -----------------------------------
# 등방성 공분산은 회전에 불변이다. R S S^T R^T 에서 S 가 구면이면 R 이 사라진다.
# 즉 예측이 구에 가까우면 회전에 대한 기울기는 형태와 무관하게 정확히 0 이다.
lp = ls_p.clamp(-15,5)
a_p = (lp.max(-1).values - lp.min(-1).values).exp()
a_t = aniso
q = lambda t, x: float(torch.quantile(t.float(), x))
print("\n[이방성] 회전이 관측되려면 예측 자체가 이방적이어야 한다")
print(f"  예측 aniso  p10 {q(a_p,.1):6.2f}  p50 {q(a_p,.5):6.2f}  p90 {q(a_p,.9):7.2f}")
print(f"  GT   aniso  p10 {q(a_t,.1):6.2f}  p50 {q(a_t,.5):6.2f}  p90 {q(a_t,.9):7.2f}")
for th in (1.5, 2.0, 4.0):
    print(f"  예측 aniso < {th}: {100*float((a_p<th).float().mean()):5.1f}%    "
          f"GT aniso < {th}: {100*float((a_t<th).float().mean()):5.1f}%")
m8 = aniso > 8
print(f"  GT 가 8배 초과인 점들에서의 예측 aniso  p50 {q(a_p[m8],.5):6.2f}  p90 {q(a_p[m8],.9):6.2f}")
res["aniso"] = {"pred_p50": q(a_p,.5), "gt_p50": q(a_t,.5),
                "pred_p50_where_gt_gt8": q(a_p[m8],.5),
                "pred_frac_below_2": float((a_p<2).float().mean())}
if A.out: json.dump(res, open(A.out + ".json", "w"), indent=1)
