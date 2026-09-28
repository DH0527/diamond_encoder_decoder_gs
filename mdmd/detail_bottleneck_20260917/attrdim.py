"""셀 내부 '속성 구조'의 고유차원: appearance 8 채널로 충분한가."""
import os,sys,argparse
import numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from can3tok.losses import ATTR_STD
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--index",type=int,default=166); ap.add_argument("--minpts",type=int,default=100)
A=ap.parse_args()
D=load(A.ckpt,"val",A.index,1); it=D["it"]; cfg=D["cfg"]; model=D["model"]
G=int(cfg.group_size); NC=int(cfg.max_points)//G
t=it["target"].cuda().float()[...,:cfg.target_dim]; m=(it["mask"].cuda()>0.5)
cen=it["center"].cuda().float().view(1,3); sc=float(it["scale"])
xyz=t[:,0:3]*sc+cen; cell=torch.arange(t.shape[0],device=t.device)//G
sd=torch.tensor(ATTR_STD[:11],device='cuda').clamp(min=1e-3)
ATT=torch.cat([t[:,3:6],t[:,6:10]*torch.sign(t[:,6:7]+1e-12),t[:,10:11],t[:,11:14]],1)/sd
def celldesc(idx):
    p=xyz[idx]; a=ATT[idx]; n=p.shape[0]
    q=p-p.mean(0,keepdim=True)
    U,S_,Vh=torch.linalg.svd(q,full_matrices=False)
    q=(q@Vh.T)/(S_/np.sqrt(max(n-1,1))).clamp(min=1e-8)
    ac=a-a.mean(0,keepdim=True)
    X=torch.cat([q,torch.ones(n,1,device=q.device)],1)
    sol=torch.linalg.lstsq(X,ac).solution[:3]          # 11x3 위치의존 기울기
    return torch.cat([ac.std(0), sol.reshape(-1)]).cpu().numpy()   # 11 + 33 = 44
rows=[];h1=[];h2=[]
g=torch.Generator(device='cuda').manual_seed(0)
for c in range(NC):
    s=m&(cell==c); n=int(s.sum())
    if n<A.minpts: continue
    idx=torch.nonzero(s,as_tuple=True)[0]
    rows.append(celldesc(idx))
    pm=idx[torch.randperm(n,generator=g,device=idx.device)]
    h1.append(celldesc(pm[:n//2])); h2.append(celldesc(pm[n//2:]))
X=np.array(rows); H1=np.array(h1); H2=np.array(h2)
Xc=X-X.mean(0,keepdims=True); tot=(Xc**2).sum()
noise=((H1-H2)**2).sum()/4.0; rel=max(1e-9,1-noise/tot)
print(f"셀 {X.shape[0]}개, 기술자 44차원 (셀내 std 11 + 위치의존 기울기 33)")
print(f"실재 신호 비율 (split-half 보정) {100*rel:.1f}%")
sv=np.linalg.svd(Xc,full_matrices=False,compute_uv=False); cum=np.cumsum(sv**2/(sv**2).sum())
print("누적 설명분산(잡음보정): " + "  ".join(f"{k}d {100*min(cum[k-1]/rel,1):.0f}%" for k in [1,2,3,4,6,8,11,16,24]))
