"""Read-only, full-resolution Gaussian and handrail ablations for T16k.

No optimizer, checkpoint writes, or changes to training code. Coordinate and
attribute swaps use per-cell position-only Hungarian matching, not raw slots.
"""
from pathlib import Path
from argparse import ArgumentParser, Namespace
import importlib.util
import json
import sys
import gc

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
from can3tok.train import make_datasets, build_config
from can3tok.model import build_model
from can3tok.schedule import schedule_flags
from can3tok.io_utils import load_npz_state,camera_from_vector
from can3tok.render import Camera,render_gaussians,sh_dc_to_rgb,psnr,ssim
from can3tok.losses import _quat_to_R

OUT=Path(__file__).parent
ROIS={3:{'upper_rail':[395,118,620,345],'front_rail':[275,185,350,450],
         'lettering':[615,205,855,322]},93:{'near_rail':[470,350,725,544]}}
LINES=[[(411,331),(412,207),(450,133),(462,135),(610,181)],
       [(470,145),(470,318)],[(523,155),(524,318)],[(574,170),(575,323)]]


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    print(name,flush=True)


def qnt(a):
    a=np.asarray(a).ravel()
    return dict(zip(('p10','p50','p90','p99'),np.quantile(a,[.1,.5,.9,.99]).tolist())) if a.size else {}


def png(name,t):
    arr=(t.detach().float().clamp(0,1).cpu().permute(1,2,0).numpy()*255).round().astype(np.uint8)
    Image.fromarray(arr).save(OUT/name)


def metrics(im,photo,target,rois):
    row={'photo_psnr':psnr(im,photo),'target_psnr':psnr(im,target),'photo_ssim':ssim(im,photo)}
    row['rois']={}
    for name,(x0,y0,x1,y1) in rois.items():
        a,b,c=[x[:,y0:y1,x0:x1].clamp(0,1) for x in (im,photo,target)]
        gx=lambda x:torch.diff(x,dim=-1)
        gy=lambda x:torch.diff(x,dim=-2)
        row['rois'][name]={'photo_psnr':psnr(a,b),'target_psnr':psnr(a,c),
                          'photo_l1':float((a-b).abs().mean()),
                          'gradient_l1':float(((gx(a)-gx(b)).abs().mean()+(gy(a)-gy(b)).abs().mean())/2)}
    return row


def matching(pred,target,mask):
    gi=np.flatnonzero(mask); matched=np.full(len(mask),-1,np.int64)
    for base in range(0,len(mask),256):
        idx=base+np.flatnonzero(mask[base:base+256])
        if not len(idx):continue
        r,c=linear_sum_assignment(cdist(pred[idx,:3],target[idx,:3]))
        matched[idx[r]]=idx[c]
    return gi,matched


def projection(gs,cam):
    xyz,scale,rot,op,col=gs
    h=torch.cat([xyz,torch.ones_like(xyz[:,:1])],-1)@cam.full_proj_transform
    xy=(h[:,:2]/h[:,3:4]+1)*h.new_tensor([cam.image_width,cam.image_height])/2-.5
    view=xyz@cam.world_view_transform[:3,:3]+cam.world_view_transform[3,:3]
    z=view[:,2].clamp(min=1e-4)
    fx=cam.image_width/(2*cam.tanfovx);fy=cam.image_height/(2*cam.tanfovy)
    j=torch.zeros(len(xyz),2,3,device=xyz.device)
    j[:,0,0]=fx/z;j[:,1,1]=fy/z;j[:,0,2]=-fx*view[:,0]/z.square();j[:,1,2]=-fy*view[:,1]/z.square()
    r=_quat_to_R(rot);cov=(r*scale.square()[:,None,:])@r.transpose(1,2)
    a=cam.world_view_transform[:3,:3].T
    cv=a@cov@a.T
    s2=j@cv@j.transpose(1,2)+torch.eye(2,device=xyz.device)*.3
    ev=torch.linalg.eigvalsh(s2).clamp(min=0).sqrt()
    return xy,view[:,2],ev


def attention_probe(model,ex,em,enc_src,focus_src,groups):
    enc=model.encoder;pool=enc.pooler;blk=pool.attn[0]
    d=pool.q.shape[-1];heads=blk.attn.num_heads
    rows=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for g in groups:
            xyz=ex[0,g*2048:(g+1)*2048,:3];m=em[0,g*2048:(g+1)*2048]
            cen=(xyz*m[:,None]).sum(0)/m.sum().clamp(min=1)
            loc=(xyz-cen)*m[:,None]
            at=ex[0,g*2048:(g+1)*2048,3:14]*m[:,None]
            feats=torch.cat([loc,(at-enc.attr_mean_p.reshape(1,-1))/enc.attr_std_p.reshape(1,-1)],-1)
            h=pool.embed(feats[None]);qq=blk.norm_q(pool.q);kk=blk.norm_kv(h)
            w=blk.attn.in_proj_weight;b=blk.attn.in_proj_bias
            qq=F.linear(qq,w[:d],b[:d]).reshape(1,16,heads,d//heads).transpose(1,2)
            kk=F.linear(kk,w[d:2*d],b[d:2*d]).reshape(1,2048,heads,d//heads).transpose(1,2)
            logits=(qq.float()@kk.float().transpose(-1,-2))/(d//heads)**.5
            logits=logits.masked_fill(m[None,None,None,:]<=.5,float('-inf'))
            weights=logits.softmax(-1)[0].mean(0) # Q,P, average heads
            focus=torch.as_tensor(np.isin(enc_src[g*2048:(g+1)*2048],focus_src),device=ex.device)
            mass=weights[:,focus].sum(-1)
            valid=weights[:,m>.5]
            effective=torch.exp(-(valid*valid.clamp(min=1e-30).log()).sum(-1))
            rows.append(dict(group=int(g),input_count=int(m.sum()),focus_count=int(focus.sum()),
                             focus_fraction=float(focus.sum()/m.sum().clamp(min=1)),
                             focus_attention_mean=float(mass.mean()),focus_attention_max_query=float(mass.max()),
                             attention_effective_points_mean=float(effective.mean()),
                             xyz_rms=float(loc[m>.5].square().mean().sqrt()),
                             standardized_attr_rms=float(feats[m>.5,3:].square().mean().sqrt())))
    return rows


def main():
    ap=ArgumentParser();ap.add_argument('--device',default='cuda:3');a=ap.parse_args()
    torch.set_num_threads(4);torch.cuda.set_device(a.device);dev=a.device
    ckpath=ROOT/'runs/T16k_20260913_093332/ckpt_step00070000.pt'
    ck=torch.load(ckpath,map_location='cpu',mmap=True,weights_only=False)
    args=Namespace(**ck['args']);_,ds=make_datasets(args);ds.extra_real_views=0
    # Capture source identity of the larger encoder packing.
    sels=[];assigns=[];old_select=ds._select;old_assign=ds._anchor_assign
    def sel(*x,**kw):
        r=old_select(*x,**kw);sels.append(r.copy());return r
    def ass(*x,**kw):
        r=old_assign(*x,**kw);assigns.append(r.copy());return r
    ds._select=sel;ds._anchor_assign=ass
    it=ds[166];ds._select=old_select;ds._anchor_assign=old_assign
    gs=load_npz_state(ds.files[166]);sc=float(it['scale']);cen=it['center'].to(dev)
    raw_norm=(gs['xyz']-it['center'].numpy())/sc
    kept=np.flatnonzero(np.all(np.abs(raw_norm)<=1,axis=-1))
    enc_src=np.full(2097152,-1,np.int64);live=assigns[1]>=0
    enc_src[live]=kept[sels[1][assigns[1][live]]]
    cfg=build_config(args,ds.target_dim,ds.sh_dim)
    model=build_model(cfg,allow_padded_tokens=args.allow_padded_tokens)
    status=model.load_state_dict(ck['model'],strict=True)
    flags=schedule_flags(int(ck['step']),args)
    for k in ('decoder_refine_alpha','folding_res_gain','attr_detach_geometry'):setattr(cfg,k,flags[k])
    model.set_encoder_residual(flags['encoder_residual']);model.to(dev).eval()
    del ck
    x=it['input'][None].to(dev);m=it['mask'][None].to(dev)
    ex=it['enc_input'][None].to(dev);em=it['enc_mask'][None].to(dev)
    unpack_chunks=[]
    hook=model.decoder.unpack.register_forward_hook(lambda mod,inp,out:unpack_chunks.append(out.detach()))
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(x,m,run_gen=False,enc_x=ex,enc_mask=em)
    hook.remove();print('model forward complete',flush=True)
    target=it['target'][:,:14].to(dev).float();pred=out['attr_pred'][0].float()
    mask=m[0]>.5;pm=out['presence'][0]>0
    gi,match=matching(pred.cpu().numpy(),target.cpu().numpy(),mask.cpu().numpy())
    idx=torch.as_tensor(gi,device=dev);mi=torch.as_tensor(match[gi],device=dev)
    pp=pred[idx];tt=target[mi]
    def unpack(t):
        return(t[:,:3]*sc+cen,t[:,3:6].clamp(-15,5).exp()*sc,t[:,6:10],t[:,10:11].sigmoid(),sh_dc_to_rgb(t[:,11:14]))
    def from_raw(sel=None):
        values=[torch.as_tensor(gs[k],device=dev).float().reshape(-1,d) for k,d in [('xyz',3),('scaling',3),('rot',4),('opacity',1),('color',3)]]
        values[-1]=sh_dc_to_rgb(values[-1]);return tuple(v if sel is None else v[sel] for v in values)
    variants={'original_dc':from_raw(),'filtered_dc':from_raw(torch.as_tensor(kept,device=dev)),
              'target_dc':unpack(target[idx]),'model':unpack(pred[pm]),'model_gt_mask':unpack(pp)}
    for name,sl in [('gt_covariance',slice(3,10)),('gt_opacity',slice(10,11)),('gt_color',slice(11,14)),('gt_all_attributes',slice(3,14)),('gt_xyz',slice(0,3))]:
        v=pp.clone();v[:,sl]=tt[:,sl];variants[name]=unpack(v)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        _,ctx=model.decompressor(out['z_compact'])
        geo=out['pred'][...,:3].clone()
        baseline=model.attr_decoder(geo,ctx['attr_code'],ctx['scale'])['pred'][0].float()
        shifted=geo.clone();dead=(m<.5)
        offsets=ctx['scale'].repeat_interleave(256,dim=1)
        shifted[...,0]+=dead*offsets[...,0]*.5
        alt=model.attr_decoder(shifted,ctx['attr_code'],ctx['scale'])['pred'][0].float()
    variants['dead_slot_shift']=unpack(alt[idx])
    # Copy raw data for later CPU analysis, with exact source indices.
    np.savez_compressed(OUT/'gaussian_bundle.npz',target=target.cpu().numpy(),pred=pred.cpu().numpy(),
                        coarse=out['pred'][0,:,:3].float().cpu().numpy(),mask=mask.cpu().numpy(),presence=pm.cpu().numpy(),
                        source_index=it['source_index'].numpy(),match=match,center=it['center'].numpy(),scale=sc,
                        cell_scale=ctx['scale'][0].float().cpu().numpy(),enc_source=enc_src,
                        z_compact=out['z_compact'][0].float().cpu().numpy())
    save('setup.json',dict(checkpoint=str(ckpath),snapshot=ds.files[166],strict_load=str(status),
                          rois=ROIS,lines=LINES,mask_swap='GT mask; per-cell xyz Hungarian assignment',
                          attr_redecode_max=float((baseline-out['attr_pred'][0]).abs().max()),
                          inactive_shift_live_attribute_mae=float((alt[idx,3:]-baseline[idx,3:]).abs().mean())))
    spec=importlib.util.spec_from_file_location('reference_sh','/data/daeho/aabb/gaussian-splatting/utils/sh_utils.py')
    shmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(shmod)
    shcoeff=torch.cat([torch.as_tensor(gs['color'],device=dev).float().reshape(-1,1,3),
                       torch.as_tensor(gs['sh'],device=dev).float().reshape(-1,15,3)],1).transpose(1,2)
    vp=json.load(open(args.view_pool));pool=next(v for k,v in vp.items() if 'train_colmap' in k)
    allscores={}
    for vi in (3,93):
        view=pool[vi];cam=Camera(camera_from_vector(view['cam']),device=dev,downscale=1)
        photo=torch.as_tensor(np.array(Image.open(view['image']).convert('RGB')),device=dev).permute(2,0,1).float()/255
        png(f'view{vi}_photo.png',photo)
        direction=F.normalize(variants['original_dc'][0]-cam.camera_center,dim=-1)
        colors=(shmod.eval_sh(3,shcoeff,direction)+.5).clamp(min=0)
        raw=variants['original_dc'];variants['original_sh3']=(*raw[:4],colors)
        with torch.no_grad():
            reference=render_gaussians(*variants['target_dc'],cam)
            rows={}
            for name,values in variants.items():
                im=render_gaussians(*values,cam)
                png(f'view{vi}_{name}.png',im)
                rows[name]=metrics(im,photo,reference,ROIS[vi])
            # Image actually available from a downscaled extra-view target.
            pil=Image.open(view['image']).convert('RGB').resize((488,272),Image.Resampling.BILINEAR)
            low=torch.as_tensor(np.array(pil),device=dev).permute(2,0,1).float()[None]/255
            up=F.interpolate(low,size=(544,977),mode='bilinear',align_corners=False)[0]
            png(f'view{vi}_half_target.png',up)
            rows['half_target']=metrics(up,photo,reference,ROIS[vi])
        allscores[str(vi)]=rows;save('render_metrics.json',allscores)
        if vi!=3:continue
        # d sum(rgb on a fixed handrail band) / d per-Gaussian rgb gives its
        # alpha-compositing contribution to that band, including occlusion.
        band=Image.new('L',(977,544));draw=ImageDraw.Draw(band)
        for line in LINES:draw.line(line,fill=255,width=7)
        band.save(OUT/'view3_rail_band.png')
        bw=torch.as_tensor(np.array(band),device=dev).float()/255
        base=variants['target_dc'];co=base[-1].detach().clone().requires_grad_(True)
        im=render_gaussians(*base[:4],co,cam)
        grad=torch.autograd.grad((im*bw).sum(),co)[0]
        contribution=grad.mean(-1).clamp(min=0).detach()
        order=torch.argsort(contribution,descending=True)
        csum=contribution[order].cumsum(0)
        nn=int(torch.searchsorted(csum,.9*csum[-1]))+1
        focus_idx=order[:nn];focus_slots=idx[focus_idx]
        focus_src=it['source_index'].numpy()[focus_slots.cpu().numpy()]
        invmatch=np.full(len(mask),-1,np.int64);invmatch[match[gi]]=np.arange(len(gi))
        pi=torch.as_tensor(invmatch[focus_slots.cpu().numpy()],device=dev)
        xy_t,z_t,sig_t=projection(base,cam);xy_p,z_p,sig_p=projection(variants['model_gt_mask'],cam)
        disp=(xy_p[pi]-xy_t[focus_idx]).norm(dim=-1)
        allpred=pred[pm].cpu().numpy();dd,nearest=cKDTree(allpred[:,:3]).query(target[focus_slots,:3].cpu().numpy(),workers=4)
        xy_all,_,_=projection(variants['model'],cam)
        nn_disp=(xy_all[torch.as_tensor(nearest,device=dev)]-xy_t[focus_idx]).norm(dim=-1)
        counts=m[0].reshape(1024,256).sum(-1)
        groups=np.unique(focus_slots.cpu().numpy()//256)
        # Limit attention inspection to 16 cells with the greatest teacher band mass.
        cg=torch.zeros(1024,device=dev).scatter_add_(0,idx//256,contribution)
        topgroups=torch.argsort(cg,descending=True)[:16].cpu().numpy()
        attn=attention_probe(model,ex,em,enc_src,focus_src,topgroups)
        save('rail_geometry.json',dict(focus_gaussians=nn,focus_cells=len(groups),band_pixels=int((bw>0).sum()),
            target_count=counts[groups].cpu().tolist(),
            matched_projected_error_px=qnt(disp.cpu()),nearest3d_projected_error_px=qnt(nn_disp.cpu()),
            nearest3d_distance_world=qnt(dd*sc),
            target_sigma_min_px=qnt(sig_t[focus_idx,0].cpu()),pred_sigma_min_px=qnt(sig_p[pi,0].cpu()),
            target_sigma_max_px=qnt(sig_t[focus_idx,1].cpu()),pred_sigma_max_px=qnt(sig_p[pi,1].cpu()),
            target_alpha=qnt(base[3][focus_idx].cpu()),pred_alpha=qnt(variants['model_gt_mask'][3][pi].cpu()),
            attention=attn))
        np.savez_compressed(OUT/'rail_contributions.npz',contribution=contribution.cpu().numpy(),
                            target_live_slots=gi,focus_slots=focus_slots.cpu().numpy(),focus_src=focus_src)
        del im,co,grad
    # Compare each geometry stage on the real valid-slot centroids.
    uc=torch.cat(unpack_chunks,dim=1)[0,:,:768].float().reshape(-1,3)+ctx['centroid'][0].repeat_interleave(256,dim=0)
    centers=ctx['centroid'][0].repeat_interleave(256,dim=0)
    stages={'fold':out['direct_xyz'][0].reshape(-1,3).float()+centers,
            'fold_plus_deep':out['learned_xyz'][0].reshape(-1,3).float()+centers,
            'unpack':uc,'refine':out['pred'][0,:,:3].float(),'nudge':pred[:,:3]}
    tg=target[:,:3].reshape(1024,256,3);wm=m[0].reshape(1024,256,1);den=wm.sum(1).clamp(min=1)
    targetcenter=(tg*wm).sum(1)/den
    stage_rows={}
    for name,xyz in stages.items():
        pc=(xyz.reshape(1024,256,3)*wm).sum(1)/den
        error=(pc-targetcenter).norm(dim=-1)
        stage_rows[name]={'centroid_error_world':qnt((error[den[:,0]>0]*sc).cpu()),
                          'centroid_error_over_extent':qnt((error/(ctx['scale'][0,:,0].float().clamp(min=1e-6)))[den[:,0]>0].cpu())}
    save('decoder_stages.json',stage_rows)
    print('all probes complete',flush=True)


if __name__=='__main__':main()
