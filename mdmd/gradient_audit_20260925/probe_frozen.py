"""Frozen checkpoint inference and output-leaf gradients. Never updates model weights.

Use an isolated GPU, e.g. CUDA_VISIBLE_DEVICES=3. This is NOT a replay of the
historical bad minibatch and leaf norms are NOT parameter gradient norms.
"""
from pathlib import Path
import os, sys, json, argparse, gc
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import numpy as np
import torch
import torch.nn.functional as F
from can3tok.train import build_parser, build_config, make_datasets, load_eval_views
from can3tok.model import build_model
from can3tok.schedule import apply_eval_schedule, effective_weights
from can3tok.losses import covariance3d_loss, sinkhorn_parameter_target, _detach_near_gaussians
from can3tok.render import Camera, render_gaussians, sh_dc_to_rgb, photometric_loss, psnr
from can3tok.io_utils import camera_from_vector
from PIL import Image

OUT=Path(__file__).resolve().parent
torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.15)
torch.manual_seed(0); np.random.seed(0)
argsj=json.loads((ROOT/'runs/B1g_20260925_053253/args.json').read_text())
args=build_parser().parse_args(['--root',argsj['root'],'--out_dir',str(OUT)])
vars(args).update(argsj)
args.patch_chunk=16; args.pool_chunk=8
tr, val=make_datasets(args)
views,held=load_eval_views(args)
for ds in (tr,val):
    ds.view_exclude=set(held); ds.extra_real_views=0
    # Diagnostic scores are explicitly own-camera, not held-out scores.
    # Disable only this local dataset's camera substitution for reproducibility.
    ds.holdout_own_photo=False
indices=[]
for scene, desired in [(0,'step_028630'),(1,'step_028750')]:
    candidates=[i for i,p in enumerate(val.files) if int(val.file_scene[i])==scene]
    exact=[i for i in candidates if desired in val.files[i]]
    indices.append(exact[0] if exact else candidates[-1])
items=[val[i] for i in indices]
cfg=build_config(args,tr.target_dim,tr.sh_dim)
model=build_model(cfg,allow_padded_tokens=args.allow_padded_tokens).cuda().eval()
layout={'xyz':0,'scale':3,'rot':6,'opacity':10,'color':11}

def qstats(t):
    t=t.detach().float().reshape(-1)
    return dict(zip(['min','p01','p50','p99','max'],torch.quantile(t,torch.tensor([0.,.01,.5,.99,1.],device=t.device)).cpu().tolist()))

def gauss(p,m,cen,sc):
    z=p[m]
    return (z[:,:3]*sc+cen, z[:,3:6].clamp(-15,5).exp()*sc,
            F.normalize(z[:,6:10],dim=-1),z[:,10:11].sigmoid(),sh_dc_to_rgb(z[:,11:14]))

def gradstats(g):
    return {k:float(g[...,s:e].norm()) for k,s,e in [('all',0,14),('xyz',0,3),('logscale',3,6),('quaternion',6,10),('opacity',10,11),('dc',11,14)]}

def saveim(t,path):
    Image.fromarray((t.detach().cpu().clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')).save(path)

CKPTS={
    'B1_30000':'runs/B1_bgcap_20260922_024641/ckpt_step00030000.pt',
    'B1_34000':'runs/B1_bgcap_20260922_024641/ckpt_step00034000.pt',
    'B1_36000':'runs/B1_bgcap_20260922_024641/ckpt_step00036000.pt',
    'B1g_42000':'runs/B1g_20260924_105249/ckpt_step00042000.pt'}
results={}
for name,cp in CKPTS.items():
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    apply_eval_schedule(model,int(ck['step']),args)
    eff=effective_weights(int(ck['step']),args)
    del ck; gc.collect()
    results[name]={}
    for ix,it in zip(indices,items):
        scene=Path(it['scene_key']).name; print(name,scene,it['name'],flush=True)
        x=it['input'][None].cuda().float(); mk=it['mask'][None].cuda().float()
        tg=it['target'][None].cuda().float()[...,:cfg.target_dim]
        m=mk[0]>.5; cen=it['center'].cuda().float()[None]; sc=float(it['scale'])
        kw={'enc_x':it['enc_input'][None].cuda().float(),'enc_mask':it['enc_mask'][None].cuda().float(),
            'group_anchor':it['group_anchor'][None].cuda().float()}
        rawq=[]
        hook=model.attr_decoder.head_rot.register_forward_hook(lambda mod,inp,out:rawq.append(out.detach().float().norm(dim=-1)))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(x,mk,run_decode=True,run_gen=False,**kw)
        hook.remove()
        p=out['attr_pred'].detach().float()[0]
        pred_presence=out['presence'].detach().float()
        pm=pred_presence[0]>0
        if int(pm.sum())<64: pm=m
        scg=out['decoded_scale'].detach().float()
        rawqn=torch.cat([q.reshape(-1) for q in rawq])
        # Camera is the NPZ's own camera, fixed across checkpoints; holdout is
        # respected for training, but this diagnostic explicitly reads its photo.
        cam=Camera(camera_from_vector(it['camera'].numpy().astype(np.float32)),device='cuda',downscale=2)
        ref=it['photo'].cuda().float()
        if ref.numel()==0:
            from PIL import Image as PILImage
            photomap=json.loads(Path(args.photo_map).read_text())
            path=photomap.get(str(val.files[ix]),photomap.get(it['name']))
            if isinstance(path,dict): path=path.get('image',path.get('path'))
            ref=torch.from_numpy(np.array(PILImage.open(path).convert('RGB')).copy()).permute(2,0,1).cuda().float()/255
        if ref.shape[-2:]!=(cam.image_height,cam.image_width):
            ref=F.interpolate(ref[None],size=(cam.image_height,cam.image_width),mode='bilinear',align_corners=False)[0]
        record={'index':ix,'file':str(val.files[ix]),'camera_scope':'own camera, not necessarily held out',
                'render_mask':'prediction presence > 0; target uses dataset mask',
                'predicted_live':int(pm.sum()),'mask_disagreement':int((pm!=m).sum()),
                'raw_rotation_head_norm':qstats(rawqn), 'num_points_original':int(it['num_points_original']),
                'num_points_valid':int(it['num_points_valid']),'num_points_used':int(m.sum()),
                'aniso_pred':qstats((p[m,3:6].amax(-1)-p[m,3:6].amin(-1)).exp()),
                'aniso_target':qstats((tg[0,m,3:6].amax(-1)-tg[0,m,3:6].amin(-1)).exp()),
                'group_scale':qstats(scg)}
        with torch.no_grad():
            ex=kw['enc_x'][0,:,:3].reshape(-1,2048,3); em=kw['enc_mask'][0].reshape(-1,2048,1)
            tx=tg[0,:,:3].reshape(-1,256,3); tm=mk[0].reshape(-1,256,1)
            ec=(ex*em).sum(1)/em.sum(1).clamp(min=1)
            tc=(tx*tm).sum(1)/tm.sum(1).clamp(min=1)
            ts=(((tx-tc[:,None]).square()*tm).sum(1)/tm.sum(1).clamp(min=1)).sqrt().amax(-1)
            validg=tm.sum((1,2))>1
            record['encoder_target_center_gap_over_target_extent']=qstats((ec-tc).norm(dim=-1)[validg]/ts[validg].clamp(min=1e-5))
        with torch.no_grad():
            gp=gauss(p,pm,cen,sc); gt=gauss(tg[0],m,cen,sc)
            image,radii=render_gaussians(*gp,cam,return_radii=True)
            target_image=render_gaussians(*gt,cam)
            record.update(photo_psnr=psnr(image,ref),target_photo_psnr=psnr(target_image,ref),teacher_psnr=psnr(image,target_image))
            W=cam.world_view_transform
            z=(gp[0]@W[:3,:3])[:,2]+W[3,2]
            zgt=(gt[0]@W[:3,:3])[:,2]+W[3,2]
            f=.5*cam.image_width/cam.tanfovx
            radius=f*gp[1].amax(-1)/z.clamp(min=.001)
            z_min=(.1*z.median().clamp(min=.01)).clamp(min=.01)
            near=z<z_min; fat=radius>4*radius.median().clamp(min=.001); gate=near|fat
            vis=radii>0
            record['guard']={'all_fraction':float(gate.float().mean()),'near_fraction':float(near.float().mean()),
                'fat_fraction':float(fat.float().mean()),'visible_count':int(vis.sum()),
                'visible_detach_fraction':float(gate[vis].float().mean()),'depth':qstats(z),
                'actual_visible_radius':qstats(radii[vis]),'proxy_radius':qstats(radius)}
            z_min_gt=(.1*zgt.median().clamp(min=.01)).clamp(min=.01)
            rr=f*gt[1].amax(-1)/zgt.clamp(min=.001)
            rref=torch.quantile(rr[zgt>z_min_gt],.8).clamp(min=.001)
            record['splat_hinge_saturated_visible_fraction']=float(((radius/rref>=5)&(z>z_min_gt))[vis].float().mean())
            record['render_leaf']={}
        saveim(torch.cat([ref,image,target_image],dim=2),OUT/f'{name}_{scene}_photo_pred_target.png')
        for frac in [0.,.1]:
            leaf=p.detach().clone().requires_grad_(True)
            g=gauss(leaf,pm,cen,sc)
            g=_detach_near_gaussians(g,cam,frac) if frac>0 else g
            im=render_gaussians(*g,cam)
            loss=60*photometric_loss(im,ref,lam_dssim=float(args.render_lam_dssim))[0]
            gr=torch.autograd.grad(loss,leaf)[0]
            norm2=gr.square().sum(-1); top=norm2.topk(max(1,int(len(norm2)*.001))).values.sum()/norm2.sum().clamp(min=1e-30)
            record['render_leaf'][str(frac)]={'weighted_loss':float(loss), 'norms':gradstats(gr),
                                             'top_0.1pct_gradient_energy_fraction':float(top)}
            del loss,gr,im,g,leaf
        with torch.no_grad():
            matched=sinkhorn_parameter_target(p[None,:,:3],tg,mk,args.group_size,scale=scg,
                        epsilon=args.sinkhorn_epsilon,iterations=args.sinkhorn_iterations,chunk_groups=16,
                        pred_attr=p[None],attr_weight=args.sinkhorn_attr_weight,layout=layout)
        for key,tt in [('raw_target',tg),('sinkhorn_target',matched)]:
            leaf=p.detach().clone().requires_grad_(True)
            loss=eff['w_cov3d']*covariance3d_loss(leaf[None],tt,mk,layout,{})
            gr=torch.autograd.grad(loss,leaf)[0]
            record[key+'_cov3d']={'weight':eff['w_cov3d'],'weighted_loss':float(loss),'norms':gradstats(gr)}
            del leaf,loss,gr
        results[name][scene]=record
        if name=='B1g_42000':
            np.savez_compressed(OUT/f'frozen_{name}_{scene}.npz',pred=p.cpu().numpy(),
                target=tg[0].cpu().numpy(),mask=m.cpu().numpy(),pred_mask=pm.cpu().numpy(),
                center=cen.cpu().numpy(),scale=np.asarray(sc),camera=it['camera'].numpy(),
                photo=ref.cpu().numpy(),group_scale=scg.cpu().numpy())
        (OUT/'frozen_probe.json').write_text(json.dumps(results,indent=2))
        print(json.dumps(record),flush=True)
        del out,kw,x,mk,tg,p,rawq,rawqn,matched,gp,gt,image,target_image,ref
        gc.collect();torch.cuda.empty_cache()
print('Finished; max torch GPU allocation GB',torch.cuda.max_memory_allocated()/2**30,flush=True)
