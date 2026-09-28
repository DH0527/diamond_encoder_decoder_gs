"""잡음 보정판: 셀 내부 패턴 중 '씬 고정' 성분 vs '스냅샷마다 변하는' 성분."""
import os,sys,argparse
import numpy as np, torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--bins",type=int,default=8); ap.add_argument("--sigma",type=float,default=0.9)
ap.add_argument("--nsnap",type=int,default=10); ap.add_argument("--minpts",type=int,default=100)
ap.add_argument("--scene",default="train_colmap")
A=ap.parse_args()
D=load(A.ckpt,"val",166,1); cfg=D["cfg"]; tr=D["tr"]
G=int(cfg.group_size); NC=int(cfg.max_points)//G; B=A.bins
ctr=torch.linspace(-2.5,2.5,B,device="cuda")
def soft_desc(q):
    w=[torch.exp(-0.5*((q[:,k:k+1]-ctr[None,:])/A.sigma)**2) for k in range(3)]
    h=torch.einsum('na,nb,nc->abc',w[0],w[1],w[2]).reshape(-1)
    return h/(h.sum()+1e-8)
def localize(p):
    q=p-p.mean(0,keepdim=True)
    U,S_,Vh=torch.linalg.svd(q,full_matrices=False)
    return (q@Vh.T)/(S_/np.sqrt(max(p.shape[0]-1,1))).clamp(min=1e-8)
cand=[i for i,f in enumerate(tr.files) if "/"+A.scene in f]
step=max(1,len(cand)//A.nsnap); pick=cand[::step][:A.nsnap]
print(f"{A.scene} 스냅샷 {len(pick)}개")
g=torch.Generator(device='cuda').manual_seed(0)
Ds=[]; Hs=[]
for i in pick:
    it=tr[i]
    t=it["target"].cuda().float()[...,:cfg.target_dim]; m=(it["mask"].cuda()>0.5)
    xyz=t[:,0:3]*float(it["scale"])+it["center"].cuda().float().view(1,3)
    cell=torch.arange(t.shape[0],device=t.device)//G
    d=np.full((NC,B**3),np.nan,np.float32); h1=np.full_like(d,np.nan); h2=np.full_like(d,np.nan)
    for c in range(NC):
        s=m&(cell==c); n=int(s.sum())
        if n<A.minpts: continue
        q=localize(xyz[s]); d[c]=soft_desc(q).cpu().numpy()
        pm=torch.randperm(n,generator=g,device=q.device)
        h1[c]=soft_desc(q[pm[:n//2]]).cpu().numpy(); h2[c]=soft_desc(q[pm[n//2:]]).cpu().numpy()
    Ds.append(d); Hs.append((h1,h2)); print("  ",it["name"],flush=True)
X=np.stack(Ds); ok=~np.isnan(X).any(axis=(0,2)); X=X[:,ok,:]
H1=np.stack([h[0] for h in Hs])[:,ok,:]; H2=np.stack([h[1] for h in Hs])[:,ok,:]
T,NCok,dd=X.shape
print(f"모든 스냅샷에서 유효한 셀 {NCok}개")
noise = ((H1-H2)**2).sum()/4.0/T          # 스냅샷당 잡음 총량
mu_cell=X.mean(0); glob=X.mean((0,1))
tot=((X-glob)**2).sum()
between=T*((mu_cell-glob)**2).sum()
within=((X-mu_cell[None])**2).sum()
within_true=max(0.0, within-noise*T*0)    # within 은 T개 관측의 편차합
# within 안에 포함된 잡음: 각 관측의 잡음 * (1-1/T)
noise_in_within = ((H1-H2)**2).sum()/4.0*(1-1.0/T)
within_sig=max(0.0,within-noise_in_within)
between_sig=max(0.0,between-((H1-H2)**2).sum()/4.0/T)
tot_sig=within_sig+between_sig
print(f"\n원자료:      셀간(고정) {100*between/tot:5.1f}%   스냅샷간 {100*within/tot:5.1f}%")
print(f"잡음 보정후: 셀간(고정) {100*between_sig/tot_sig:5.1f}%   스냅샷간 {100*within_sig/tot_sig:5.1f}%")
print(f"  (스냅샷간 변동의 {100*noise_in_within/within:.1f}% 는 표본 잡음이었음)")
R=(X-mu_cell[None]).reshape(-1,dd); Rc=R-R.mean(0,keepdims=True)
sv=np.linalg.svd(Rc,full_matrices=False,compute_uv=False); cum=np.cumsum(sv**2/(sv**2).sum())
relR=max(1e-9, 1-noise_in_within/ (Rc**2).sum())
print("  씬고정 성분을 준 뒤 남는 동적 성분 (잡음보정): " + "  ".join(f"{k}d {100*min(cum[k-1]/relR,1):.0f}%" for k in [1,2,3,4,6,8]))
