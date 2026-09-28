"""방향 정보가 이미 예측 위치 안에 들어 있는가.

lossform.py 가 보인 것: 손실을 어떻게 써도 attr_decoder 는 목표 방향장을 못 낸다
(이방성 최대 6.04 vs 목표 15.51, 방향 35도). 손실이 아니라 표현의 문제다.

난간의 방향은 원래 이웃 구조에 들어 있다. 난간 위 점의 k-최근접 이웃은 선을 이루므로
그 국소 공분산의 주축이 곧 난간 방향이다. 이게 사실이면 head 에 국소 PCA 특징을
넣어주는 것만으로 방향이 공짜로 나온다. 셀당 8채널 코드로 점마다 다른 방향을 짜내는
것과는 완전히 다른 문제가 된다.

  국소 PCA 주축 vs GT 장축   <- 이 각도가 작으면 정보는 위치에 있다
   현재 예측 장축 vs GT 장축   <- 비교 기준
"""
import os, sys, json, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from can3tok.losses import _quat_to_R

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/T16k_20260913_093332/ckpt_step00070000.pt")
ap.add_argument("--index", type=int, default=166)
ap.add_argument("--k", type=int, default=16)
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
    # 예측 위치에 GT 를 최근접으로 붙인다 (aniso_first 와 같은 대응)
    idx = torch.empty(P.shape[0], dtype=torch.long, device=P.device)
    for s in range(0, P.shape[0], 4096):
        e = min(s + 4096, P.shape[0])
        idx[s:e] = torch.cdist(P[s:e, :3], T[:, :3]).argmin(dim=1)
    M = T[idx]

    def major(ls, q):
        """가장 큰 축의 방향과 이방성."""
        R = _quat_to_R(F.normalize(q, dim=-1))
        j = ls.clamp(-15., 5.).argmax(dim=-1)
        v = torch.gather(R, 2, j.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        an = (ls.clamp(-15., 5.).max(-1).values - ls.clamp(-15., 5.).min(-1).values).exp()
        return F.normalize(v, dim=-1), an
    gt_ax, gt_an = major(M[:, 3:6], M[:, 6:10])
    pr_ax, _ = major(P[:, 3:6], P[:, 6:10])

    # k-NN 국소 PCA. 이웃은 예측 위치만 쓴다 (학습 때 쓸 수 있는 정보여야 한다)
    N = P.shape[0]
    pca_ax = torch.empty(N, 3, device=P.device)
    lin = torch.empty(N, device=P.device)          # 선형성 = 1 - l2/l1
    for s in range(0, N, 2048):
        e = min(s + 2048, N)
        d = torch.cdist(P[s:e, :3], P[:, :3])
        nb = P[d.topk(A.k + 1, largest=False).indices, :3]      # (b, k+1, 3), 자기 포함
        nb = nb - nb.mean(dim=1, keepdim=True)
        C = nb.transpose(1, 2) @ nb / A.k
        ev, EV = torch.linalg.eigh(C.double())
        pca_ax[s:e] = F.normalize(EV[..., -1].float(), dim=-1)
        ev = ev.float().clamp(min=0)
        lin[s:e] = 1.0 - (ev[:, 1] / ev[:, 2].clamp(min=1e-20)).clamp(max=1.0)

    def ang(a, b):   # 축은 부호가 없다
        return ((a * b).sum(-1).abs().clamp(max=1.0).acos() * 180.0 / np.pi)
    a_pca, a_pred = ang(pca_ax, gt_ax), ang(pr_ax, gt_ax)

def q(v, m=None):
    v = v if m is None else v[m]
    return f"p50 {float(torch.quantile(v.float(),0.5)):5.1f}  p90 {float(torch.quantile(v.float(),0.9)):5.1f}"

print(f"\n[전체 {P.shape[0]}점]  GT 장축과의 각도 (축이라 0~90도)")
print(f"  국소 PCA 주축 : {q(a_pca)}")
print(f"  현재 예측 장축: {q(a_pred)}")
for lo, hi, tag in [(8, 1e9, "GT 이방성 >8"), (30, 1e9, "GT 이방성 >30"), (100, 1e9, "GT 이방성 >100")]:
    m = (gt_an > lo) & (gt_an <= hi)
    if int(m.sum()) < 50: continue
    print(f"\n[{tag}  n={int(m.sum())}]")
    print(f"  국소 PCA 주축 : {q(a_pca, m)}")
    print(f"  현재 예측 장축: {q(a_pred, m)}")
for t in (0.5, 0.7, 0.9):
    m = (lin > t) & (gt_an > 8)
    if int(m.sum()) < 50: continue
    print(f"\n[이웃이 선형 >{t}  &  GT 이방성 >8   n={int(m.sum())}]")
    print(f"  국소 PCA 주축 : {q(a_pca, m)}")
    print(f"  현재 예측 장축: {q(a_pred, m)}")
