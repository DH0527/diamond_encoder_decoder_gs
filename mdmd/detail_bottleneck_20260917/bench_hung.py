"""Hungarian 매칭의 시간이 어디로 가는지. GPU 로 옮길 수 있는지 판단하기 위한 것.

현재 구현은 (1024, 256, 256) 비용 행렬 268MB 를 CPU 로 넘긴 뒤 그룹 1024개를 파이썬
for 문으로 돌며 scipy 를 호출한다. 후보는 세 갈래다.
  A. 전송을 줄인다: 살아 있는 부분행렬만 넘긴다 (268MB -> 약 25MB). 결과 동일.
  B. 루프를 병렬화한다: scipy 는 C++ 이라 GIL 을 놓을 수 있다. 결과 동일.
  C. GPU 로 옮긴다: auction 알고리즘. 결과가 최적해와 다를 수 있으므로 일치율을 재야 한다.
"""
import os, sys, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--index", type=int, default=166)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()

D = load(A.ckpt, "val", A.index, 1)
model, cfg, it = D["model"], D["cfg"], D["it"]
x = it["input"][None].cuda().float(); mk = it["mask"][None].cuda().float()
tg = it["target"][None].cuda().float()[..., :cfg.target_dim]
kw = {}
if "enc_input" in it: kw = dict(enc_x=it["enc_input"][None].cuda().float(), enc_mask=it["enc_mask"][None].cuda().float())
if "group_anchor" in it: kw["group_anchor"] = it["group_anchor"][None].cuda().float()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    o = model(x, mk, run_decode=True, run_gen=False, **kw)
pred = o.get("attr_pred", o["pred"]).float()
gsz = int(cfg.group_size)
b, n, c = pred.shape
ng = n // gsz; bg = b * ng
p = pred[:, :ng*gsz].reshape(bg, gsz, c)
t = tg[:, :ng*gsz].reshape(bg, gsz, c)
m = mk[:, :ng*gsz].reshape(bg, gsz) > 0.5
print(f"[setup] groups {bg}  slots {gsz}  live {int(m.sum())}  "
      f"그룹당 live 평균 {float(m.sum())/bg:.1f}  코어 {os.cpu_count()}")

def T(fn, reps=A.reps, sync=True):
    if sync: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): r = fn()
    if sync: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, r

# ---- 지금 방식 --------------------------------------------------------------
def cost_full():
    with torch.no_grad():
        cost = torch.cdist(p[..., 0:3], t[..., 0:3])
        inv = ~m
        cost = cost.masked_fill(inv.unsqueeze(2), 1.0e6)
        cost = cost.masked_fill(inv.unsqueeze(1), 1.0e6)
        return cost
dt_cost, cost = T(cost_full)
dt_xfer, cost_np = T(lambda: cost.cpu().numpy())
m_np = m.cpu().numpy()
print(f"[현재] cdist+mask {dt_cost*1e3:7.1f} ms   전송 {dt_xfer*1e3:7.1f} ms "
      f"({cost.numel()*4/1e6:.0f}MB)")

def loop_now():
    out = []
    for i in range(bg):
        li = np.flatnonzero(m_np[i])
        if li.size < 8: continue
        r, col = linear_sum_assignment(cost_np[i][np.ix_(li, li)])
        out.append((i, li[r], li[col]))
    return out
dt_loop, ref = T(loop_now)
print(f"[현재] scipy 루프  {dt_loop*1e3:7.1f} ms   합계 {(dt_cost+dt_xfer+dt_loop)*1e3:7.1f} ms")

# ---- A. 살아 있는 부분행렬만 전송 -------------------------------------------
def small_xfer():
    with torch.no_grad():
        subs = []
        for i in range(bg):
            li = torch.nonzero(m[i], as_tuple=True)[0]
            if li.numel() < 8: subs.append(None); continue
            d = torch.cdist(p[i, li, 0:3][None], t[i, li, 0:3][None])[0]
            subs.append(d.cpu().numpy())
        return subs
dt_small, subs = T(small_xfer, reps=1)
tot_mb = sum(s.nbytes for s in subs if s is not None)/1e6
print(f"[A] live 부분행렬만: 준비+전송 {dt_small*1e3:7.1f} ms  ({tot_mb:.0f}MB) "
      f"-- 그룹마다 커널을 띄우므로 오히려 느릴 수 있다")

# ---- B. scipy 루프을 스레드로 --------------------------------------------------
def loop_threads(w):
    def one(i):
        li = np.flatnonzero(m_np[i])
        if li.size < 8: return None
        r, col = linear_sum_assignment(cost_np[i][np.ix_(li, li)])
        return (i, li[r], li[col])
    with ThreadPoolExecutor(max_workers=w) as ex:
        return [z for z in ex.map(one, range(bg)) if z is not None]
for w in (4, A.workers, 16):
    dt, res = T(lambda w=w: loop_threads(w))
    same = all(np.array_equal(a[2], b[2]) for a, b in zip(ref, res))
    print(f"[B] 스레드 {w:2d}개: {dt*1e3:7.1f} ms  ({dt_loop/dt:4.2f}x)  결과 동일: {same}")

# ---- C. GPU auction ----------------------------------------------------------
def auction(cost, live, iters=200, eps=None):
    """배치 auction. 최소비용 배정. cost: (G, S, S), live: (G, S) bool.

    죽은 슬롯은 이미 1e6 으로 막혀 있으니 그대로 두고 배정한 뒤 live 만 걸러 쓴다.
    eps-complementary slackness 로 종료. 최적해와 다를 수 있어 일치율을 재야 한다.
    """
    G, S, _ = cost.shape
    C = -cost                                  # auction 은 이익 최대화
    if eps is None: eps = (C.abs().amax() / (S * 10.0)).clamp(min=1e-6)
    price = torch.zeros(G, S, device=cost.device)
    owner = torch.full((G, S), -1, dtype=torch.long, device=cost.device)   # 대상 -> 입찰자
    assign = torch.full((G, S), -1, dtype=torch.long, device=cost.device)  # 입찰자 -> 대상
    for _ in range(iters):
        un = (assign < 0) & live
        if not bool(un.any()): break
        v = C - price.unsqueeze(1)                     # (G, S 입찰자, S 대상)
        v = v.masked_fill(~live.unsqueeze(1), -1e9)
        top2 = v.topk(2, dim=-1)
        best, second = top2.values[..., 0], top2.values[..., 1]
        j = top2.indices[..., 0]
        bid = price.gather(1, j) + (best - second) + eps
        # 대상마다 최고 입찰만 채택. scatter_reduce 로 한 번에.
        bid_m = torch.where(un, bid, torch.full_like(bid, -1e9))
        winner_bid = torch.full((G, S), -1e9, device=cost.device)
        winner_bid.scatter_reduce_(1, j, bid_m, reduce="amax", include_self=True)
        take = un & (bid_m >= winner_bid.gather(1, j))
        if not bool(take.any()): break
        gi, bi = torch.nonzero(take, as_tuple=True)
        tj = j[gi, bi]
        prev = owner[gi, tj]
        has_prev = prev >= 0
        assign[gi[has_prev], prev[has_prev]] = -1
        owner[gi, tj] = bi
        assign[gi, bi] = tj
        price[gi, tj] = bid[gi, bi]
    return assign

# ---- D. live 를 앞으로 모아 잘라 보내기 + 스레드 --------------------------------
# 전송량은 죽은 슬롯이 지배한다. 그룹마다 live 를 앞으로 정렬해 (G, kmax, kmax) 만
# 보내면 된다. GPU 커널 두 번과 전송 한 번이라 A 의 커널 폭주가 없다.
kcnt = m.sum(1)
kmax = int(kcnt.max())
print(f"\n[D] 그룹당 live: 평균 {float(kcnt.float().mean()):.1f}  최대 {kmax}  "
      f"자르면 {bg*kmax*kmax*4/1e6:.0f}MB (현재 {cost.numel()*4/1e6:.0f}MB)")
order = torch.argsort((~m).to(torch.int8), dim=1, stable=True)     # live 먼저
ord_k = order[:, :kmax]
pinned = torch.empty((bg, kmax, kmax), dtype=torch.float32, pin_memory=True)

def pack_xfer():
    with torch.no_grad():
        c2 = torch.cdist(p[..., 0:3], t[..., 0:3])
        c2 = c2.masked_fill((~m).unsqueeze(2), 1.0e6).masked_fill((~m).unsqueeze(1), 1.0e6)
        idx = ord_k
        c2 = c2.gather(1, idx.unsqueeze(-1).expand(-1, -1, gsz))
        c2 = c2.gather(2, idx.unsqueeze(1).expand(-1, kmax, -1))
        pinned.copy_(c2, non_blocking=True)
        torch.cuda.synchronize()
        return pinned.numpy()
dt_pack, cp_np = T(pack_xfer)
k_np = kcnt.cpu().numpy(); ord_np = ord_k.cpu().numpy()

def loop_packed(w=16):
    def one(i):
        k = int(k_np[i])
        if k < 8: return None
        r, col = linear_sum_assignment(cp_np[i][:k, :k])
        o = ord_np[i]
        return (i, o[r], o[col])
    with ThreadPoolExecutor(max_workers=w) as ex:
        return [z for z in ex.map(one, range(bg)) if z is not None]
dt_lp, res_p = T(loop_packed)
ref_cost = {i: float(cost_np[i][r, cl].sum()) for i, r, cl in ref}
new_cost = {i: float(cost_np[i][r, cl].sum()) for i, r, cl in res_p}
same_cost = all(abs(new_cost[i] - ref_cost[i]) < 1e-4 for i in ref_cost)
print(f"[D] 준비+전송 {dt_pack*1e3:6.1f} ms + 스레드16 루프 {dt_lp*1e3:6.1f} ms "
      f"= {(dt_pack+dt_lp)*1e3:6.1f} ms   현재 {674.2:.1f} ms 대비 {674.2/((dt_pack+dt_lp)*1e3):.2f}x")
print(f"[D] 배정 총비용 동일: {same_cost}   그룹 수 {len(res_p)} vs {len(ref)}")

live = m
dt_auc, asg = T(lambda: auction(cost, live))
cost_ref = sum(float(cost_np[i][r, cl].sum()) for i, r, cl in ref)
tot = 0.0; nmatched = 0
asg_np = asg.cpu().numpy()
for i, r, cl in ref:
    a = asg_np[i][r]
    ok = a >= 0
    tot += float(cost_np[i][r[ok], a[ok]].sum()); nmatched += int(ok.sum())
print(f"[C] GPU auction: {dt_auc*1e3:7.1f} ms  ({dt_loop/dt_auc:4.2f}x vs scipy 루프)")
print(f"    총비용 auction {tot:.4f} vs 최적 {cost_ref:.4f}  "
      f"({100*(tot/cost_ref-1):+.2f}%)  배정된 점 {nmatched}/{sum(len(r) for _,r,_ in ref)}")
