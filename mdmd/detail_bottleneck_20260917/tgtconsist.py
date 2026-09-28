"""목표 속성장이 애초에 위치의 함수인가.

cap_A/B/C 가 보인 것: z 를 완전히 풀어도 이방성이 7.98 에서 멈춘다 (목표 15.51).
잠재도 파라미터 수도 아니라면, 남는 가능성은 목표가 학습 가능한 함수가 아니라는 것이다.

attr_decoder 의 점당 입력은 위치뿐이다 (그룹 코드는 256점이 공유). 그러므로 공간적으로
붙어 있는 두 점은 거의 같은 속성을 받아야 한다. 그런데 3DGS 해는 유일하지 않아서 인접한
가우시안이 서로 다른 방향을 가질 수 있다. 그러면 어떤 디코더도 국소 평균밖에 낼 수 없고,
방향이 어긋난 타원체들의 평균은 더 둥글다. 그게 7.98 의 정체라면 숫자가 맞아야 한다.

  ind_aniso   개별 GT 이방성                      15.51 이어야 한다
  avg_aniso   이웃 GT 공분산을 평균한 것의 이방성   <- 디코더가 낼 수 있는 상한
  axis_spread 이웃 GT 장축들이 서로 벌어진 각도
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from can3tok.losses import _quat_to_R

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/T16k_20260913_093332/ckpt_step00070000.pt")
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--k", type=int, default=8)
A = ap.parse_args()

D = load(A.ckpt, "val", A.index, 1)
model, cfg, it = D["model"], D["cfg"], D["it"]
x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim][0]
gtm = it["mask"].cuda() > 0.5
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
    M = T[idx]                                    # 예측 점마다의 목표 속성

    def cov(ls, q):
        s = torch.exp(ls.clamp(-15., 5.)); R = _quat_to_R(F.normalize(q, dim=-1))
        return torch.einsum("...ij,...j,...kj->...ik", R, s * s, R)
    def aniso_of(C):
        ev = torch.linalg.eigvalsh(C.double()).clamp(min=1e-30)
        return (ev[..., -1] / ev[..., 0]).sqrt().float()
    Cm = cov(M[:, 3:6], M[:, 6:10])
    ind = aniso_of(Cm)

    N = P.shape[0]
    avg = torch.empty(N, device=P.device)
    spread = torch.empty(N, device=P.device)
    dist = torch.empty(N, device=P.device)
    for s in range(0, N, 2048):
        e = min(s + 2048, N)
        d = torch.cdist(P[s:e, :3], P[:, :3])
        tk = d.topk(A.k, largest=False)
        nb = tk.indices                                        # (b, k), 자기 포함
        dist[s:e] = tk.values[:, 1:].mean(dim=1)
        avg[s:e] = aniso_of(Cm[nb].mean(dim=1))                # 이웃 공분산의 평균
        R = _quat_to_R(F.normalize(M[nb][..., 6:10], dim=-1))
        j = M[nb][..., 3:6].clamp(-15., 5.).argmax(dim=-1)
        ax = torch.gather(R, 3, j.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 1)).squeeze(-1)
        ax = F.normalize(ax, dim=-1)                           # (b, k, 3) 장축들
        c = (ax[:, :1] * ax).sum(-1).abs().clamp(max=1.0)  # 자기 자신과의 각도
        spread[s:e] = (c[:, 1:].acos() * 180.0 / np.pi).mean(dim=1)

def q(v, m=None):
    v = v if m is None else v[m]
    return f"p50 {float(torch.quantile(v.float(),0.5)):6.2f}"

big = ind > 8
print(f"\n[k={A.k} 이웃, {N}점]  이웃 간 평균 거리 {q(dist)}")
print(f"\n  개별 GT 이방성                : {q(ind)}")
print(f"  이웃 GT 공분산 평균의 이방성  : {q(avg)}   <- 위치만 보는 디코더의 상한")
print(f"  이웃 장축들이 벌어진 각도     : {q(spread)}")
print(f"\n[GT 이방성 >8 인 점만  n={int(big.sum())}]")
print(f"  개별                          : {q(ind, big)}")
print(f"  이웃 평균                     : {q(avg, big)}")
print(f"  장축 벌어짐                   : {q(spread, big)}")
for k in (2, 4, 8, 16):
    if k > A.k: continue
print(f"\ncap_A/B/C 가 도달한 값은 7.98 이었다. 위 '이웃 평균' 과 비교하라.")
