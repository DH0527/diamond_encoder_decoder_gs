"""바늘을 그리려면 위치가 얼마나 정확해야 하는가.

imgloss.py 가 보인 것: GT 스케일+회전을 바늘에만 넣어도 거의 모든 이미지 손실이
나빠진다. 바늘은 길고 얇아서, 위치가 짧은 축의 굵기만큼만 틀려도 획이 난간을 벗어나
엉뚱한 곳에 그려진다. 둥근 블롭은 같은 위치 오차에 둔감하다.

그래서 재야 할 것은 위치 오차를 바늘 자신의 굵기로 나눈 값이다.

  err / sigma_min  <= 1 이어야 획이 제자리에 놓인다
  err / sigma_max     획 길이에 비해 얼마나 벗어났는가
"""
import os, sys, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/T16k_20260913_093332/ckpt_step00070000.pt")
ap.add_argument("--index", type=int, default=166)
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
    # 두 방향 다 본다. GT 바늘 하나하나가 예측 점 중 가장 가까운 것과 얼마나 떨어졌는가가
    # "그 바늘을 그릴 수 있는가" 의 질문이다.
    d_gt = torch.empty(T.shape[0], device=T.device)
    for s in range(0, T.shape[0], 4096):
        e = min(s + 4096, T.shape[0])
        d_gt[s:e] = torch.cdist(T[s:e, :3], P[:, :3]).min(dim=1).values
    ls = T[:, 3:6].clamp(-15., 5.)
    smin, smax = ls.min(-1).values.exp(), ls.max(-1).values.exp()
    an = smax / smin

def q(v):
    return (f"p50 {float(torch.quantile(v.float(),0.5)):8.3f}  "
            f"p90 {float(torch.quantile(v.float(),0.9)):8.3f}")

print(f"\n[GT 가우시안 {T.shape[0]}개]  각 GT 가 가장 가까운 예측 점까지의 거리 (정규화 좌표)")
for lo, tag in [(0, "전체"), (8, "이방성 >8"), (30, ">30"), (100, ">100"), (300, ">300")]:
    m = an > lo
    if int(m.sum()) < 50: continue
    e, mn, mx = d_gt[m], smin[m], smax[m]
    print(f"\n  {tag:12s} n={int(m.sum()):6d}")
    print(f"    위치 오차            {q(e)}")
    print(f"    짧은 축 sigma_min    {q(mn)}")
    print(f"    긴 축   sigma_max    {q(mx)}")
    print(f"    오차 / sigma_min     {q(e/mn.clamp(min=1e-12))}   <- 1 이하여야 제자리")
    print(f"    오차 / sigma_max     {q(e/mx.clamp(min=1e-12))}")
    print(f"    오차 < sigma_min 인 비율  {100*float((e < mn).float().mean()):5.1f}%")
