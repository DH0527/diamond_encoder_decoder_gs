"""CPU checks of actual loss formulas; synthetic, not an empirical root-cause claim."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from can3tok.losses import covariance3d_loss, intra_group_attr_set_loss, _quat_to_R
torch.set_num_threads(2)
layout={'xyz':0,'scale':3,'rot':6,'opacity':10,'color':11}
target=torch.zeros(1,2,14);target[...,3:6]=-6;target[...,6]=1
mask=torch.ones(1,2)
out={'covariance_scale_ratios':[], 'splat_clamped_hinge':[], 'quaternion_normalization':[],
     'notes':'Synthetic checks. These do not establish which term triggered a historical batch.'}
for k in (1,2,5,10,30):
    a=torch.tensor(math.log(k),requires_grad=True)
    p=target+torch.cat([torch.zeros(3),a.repeat(3),torch.zeros(8)])[None,None]
    loss=covariance3d_loss(p,target,mask,layout,{})
    grad=torch.autograd.grad(loss,a)[0]
    out['covariance_scale_ratios'].append({'k':k,'loss':float(loss),'d_loss_d_uniform_logscale':float(grad)})
for r in (1.,2.,4.,6.,10.,100.):
    logratio=torch.tensor(math.log(r),requires_grad=True)
    loss=(logratio.exp()-1).relu().clamp(max=4).square()
    grad=torch.autograd.grad(loss,logratio)[0]
    out['splat_clamped_hinge'].append({'radius_over_ref':r,'loss':float(loss),'d_loss_d_logratio':float(grad)})
for norm in (1.,1e-2,1e-4,1e-6):
    q=torch.tensor([norm,0.,0.,0.],requires_grad=True)
    qhat=torch.nn.functional.normalize(q,dim=-1)
    g=torch.autograd.grad(qhat[1],q)[0]
    out['quaternion_normalization'].append({'raw_norm':norm,'jacobian_component':float(g[1])})
# Equivalent covariance via x/y scale permutation + 90-degree z rotation.
p=target.clone();t=target.clone();t[...,3:6]=torch.tensor([-4.,-6.,-7.])
p[...,3:6]=torch.tensor([-6.,-4.,-7.]);p[...,6:10]=torch.tensor([2**-.5,0.,0.,2**-.5])
out['equivalent_covariance_axis_permutation']={
    'covariance_loss':float(covariance3d_loss(p,t,mask,layout,{})),
    'attribute_set_loss':float(intra_group_attr_set_loss(p,t,mask,2,layout))}
(Path(__file__).resolve().parent/'loss_math.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
