"""Frozen output diagnostic: locate pixels affected by fully detached Gaussians.
Opacity removal is a diagnostic counterfactual, NOT a training/pruning change.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np, torch
from PIL import Image
import torch.nn.functional as F
from can3tok.render import Camera,render_gaussians,sh_dc_to_rgb,psnr,photometric_loss
from can3tok.io_utils import camera_from_vector
OUT=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
results={}
for scene in ['train','truck']:
    d=np.load(OUT/f'frozen_B1g_42000_{scene}.npz')
    p=torch.tensor(d['pred'],device='cuda'); pm=torch.tensor(d['pred_mask'],device='cuda')
    center=torch.tensor(d['center'],device='cuda');scale=float(d['scale'])
    ref=torch.tensor(d['photo'],device='cuda')
    cam=Camera(camera_from_vector(d['camera']),device='cuda',downscale=2)
    def gauss(x):
        z=x[pm]
        return(z[:,:3]*scale+center,z[:,3:6].clamp(-15,5).exp()*scale,
               F.normalize(z[:,6:10],dim=-1),z[:,10:11].sigmoid(),sh_dc_to_rgb(z[:,11:14]))
    with torch.no_grad():
        gp=gauss(p);W=cam.world_view_transform
        depth=(gp[0]@W[:3,:3])[:,2]+W[3,2]
        r=.5*cam.image_width/cam.tanfovx*gp[1].amax(-1)/depth.clamp(min=.001)
        near=depth<(.1*depth.median().clamp(min=.01)).clamp(min=.01)
        fat=r>4*r.median().clamp(min=.001); gate=near|fat
        im,radii=render_gaussians(*gp,cam,return_radii=True)
        removed=render_gaussians(*gp[:3],gp[3]*(~gate[:,None]),gp[4],cam)
        height=im.shape[1];sky=slice(0,max(1,round(height*.3)))
        affected=(im-removed).abs().amax(0)>.05
        rec={'full_psnr':psnr(im,ref),'guarded_removed_psnr':psnr(removed,ref),
             'top30pct_psnr':psnr(im[:,sky],ref[:,sky]),
             'top30pct_guarded_removed_psnr':psnr(removed[:,sky],ref[:,sky]),
             'visible_gaussians_gated_fraction':float(gate[radii>0].float().mean()),
             'pixels_changed_over_0.05_fraction':float(affected.float().mean()),'gradients':{}}
        images=torch.cat([ref,im,removed],dim=2).cpu().clamp(0,1).permute(1,2,0).numpy()
        Image.fromarray((images*255).astype('uint8')).save(OUT/f'guard_removal_{scene}.png')
    for mode in ['all_detached','geometry_detached']:
        leaf=p.clone().requires_grad_(True);g=gauss(leaf)
        g=tuple(torch.where(gate[:,None],t.detach(),t) if mode=='all_detached' or i<3 else t
                for i,t in enumerate(g))
        im=render_gaussians(*g,cam)
        loss=60*photometric_loss(im,ref)[0]
        gr=torch.autograd.grad(loss,leaf)[0][pm][gate]
        rec['gradients'][mode]={'guarded_opacity_norm':float(gr[:,10:11].norm()),
                                'guarded_color_norm':float(gr[:,11:14].norm())}
    results[scene]=rec
(OUT/'guard_probe.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
