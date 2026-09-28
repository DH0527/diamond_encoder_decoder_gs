"""잡음 바닥 보정판: 셀 내부 패턴의 '실재하는' 변동 중 3차원이 담는 몫."""
import os,sys,argparse
import numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--index",type=int,default=166); ap.add_argument("--bins",type=int,default=6)
ap.add_argument("--sigma",type=float,default=0.9); ap.add_argument("--minpts",type=int,default=64)
ap.add_argument("--sweep",type=str,default="")
A=ap.parse_args()
D=load(A.ckpt,"val",A.index,1); it=D["it"]; cfg=D["cfg"]; cam=D["cam"]
G=int(cfg.group_size); NC=int(cfg.max_points)//G; B=A.bins
t=it["target"].cuda().float()[...,:cfg.target_dim]; m=(it["mask"].cuda()>0.5)
cen=it["center"].cuda().float().view(1,3); sc=float(it["scale"])
xyz=t[:,0:3]*sc+cen; cell=torch.arange(t.shape[0],device=t.device)//G
SWEEP=[int(v) for v in A.sweep.split(",")] if A.sweep else [B]
# 셀 물리 반경 (정규화 단위 -> 원 좌표계)
_rads=[]
for c in range(NC):
    s_=m&(cell==c)
    if int(s_.sum())<A.minpts: continue
    pp=xyz[s_]; _rads.append(float((pp-pp.mean(0,keepdim=True)).norm(dim=1).median()))
print(f"셀 반경(중앙값) {np.median(_rads):.4f} (정규화 단위)")
ctr=None
def soft_desc(q):
    # q: (n,3) 셀-로컬 정규화 좌표 -> 부드러운 3D 히스토그램 (B^3)
    w=[torch.exp(-0.5*((q[:,k:k+1]-ctr[None,:])/A.sigma)**2) for k in range(3)]
    h=torch.einsum('na,nb,nc->abc',w[0],w[1],w[2]).reshape(-1)
    return h/(h.sum()+1e-8)
def localize(p):
    q=p-p.mean(0,keepdim=True)
    U,S_,Vh=torch.linalg.svd(q,full_matrices=False)
    return (q@Vh.T)/(S_/np.sqrt(max(p.shape[0]-1,1))).clamp(min=1e-8)
g=torch.Generator(device='cuda').manual_seed(0)
print("\n bins  셀크기대비빈폭  신호비율   3d      4d      6d      8d     12d  (잡음보정 설명분산)")
for B in SWEEP:
  ctr=torch.linspace(-2.5,2.5,B,device=xyz.device)
  rows=[];half=[]
  for c in range(NC):
    s=m&(cell==c); n=int(s.sum())
    if n<A.minpts: continue
    q=localize(xyz[s])
    rows.append(soft_desc(q).cpu().numpy())
    perm=torch.randperm(n,generator=g,device=q.device)
    half.append((soft_desc(q[perm[:n//2]]).cpu().numpy(), soft_desc(q[perm[n//2:]]).cpu().numpy()))
  X=np.array(rows); H1=np.array([h[0] for h in half]); H2=np.array([h[1] for h in half])
  Xc=X-X.mean(0,keepdims=True); tot=(Xc**2).sum()
  noise_full=((H1-H2)**2).sum()/4.0
  rel=max(1e-9,1-noise_full/tot)
  sv=np.linalg.svd(Xc,full_matrices=False,compute_uv=False); cum=np.cumsum(sv**2/(sv**2).sum())
  f=lambda k: 100*min(cum[k-1]/rel,1.0)
  print(f"  {B:3d}    {5.0/B:.2f} std      {100*rel:5.1f}%  {f(3):5.1f}%  {f(4):5.1f}%  {f(6):5.1f}%  {f(8):5.1f}%  {f(12):5.1f}%")
