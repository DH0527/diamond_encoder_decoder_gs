"""CPU follow-up statistics and scientific comparison figures; no training."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from can3tok.encoder import GroupPointPooler,ATTR_MEAN,ATTR_STD
from can3tok.io_utils import camera_from_vector,load_npz_state,gaussian_target_channels
from can3tok.render import Camera,sh_dc_to_rgb
from probe_rail_detail import projection,qnt,ROIS,metrics

OUT=Path(__file__).parent
torch.set_num_threads(4)

def save(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def read(name):return np.array(Image.open(OUT/name).convert('RGB'))

def pool_comparison_figure():
    names=['baseline','no_local_xyz','mean_local_attributes']
    paths=[OUT/f'view3_pool_{name}.png' for name in names]
    if not all(path.exists() for path in paths):
        return
    titles=['Original pool input','Local xyz removed; anchors preserved','Local attributes replaced by cell mean']
    x0,y0,x1,y1=ROIS[3]['upper_rail']
    fig,axes=plt.subplots(1,3,figsize=(12,4.7),layout='constrained')
    for ax,path,title in zip(axes,paths,titles):
        ax.imshow(read(path.name)[y0:y1,x0:x1],interpolation='nearest')
        ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('Frozen T16k 70k: full codec counterfactual, identical GT mask',fontsize=12)
    fig.savefig(OUT/'crop_pool_counterfactual.png',dpi=160);plt.close(fig)

def main():
    b=np.load(OUT/'gaussian_bundle.npz');r=np.load(OUT/'rail_contributions.npz')
    setup=json.load(open(OUT/'setup.json'));ck=torch.load(setup['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    m=b['mask'];gi=np.flatnonzero(m);match=b['match'];source=b['source_index']
    pred=torch.from_numpy(b['pred'][gi]);target=torch.from_numpy(b['target'][gi]);scale=float(b['scale']);center=torch.from_numpy(b['center'])
    def gs(t):return(t[:,:3]*scale+center,t[:,3:6].clamp(-15,5).exp()*scale,t[:,6:10],t[:,10:11].sigmoid(),sh_dc_to_rgb(t[:,11:14]))
    vp=json.load(open(ROOT/'assets/view_pool_speedy_both.json'))
    pool=next(v for k,v in vp.items() if 'train_colmap' in k)
    cam=Camera(camera_from_vector(pool[3]['cam']),device='cpu')
    xy,z,sig=projection(gs(target),cam);px,pz,psig=projection(gs(pred),cam)
    band=np.array(Image.open(OUT/'view3_rail_band.png'))>0
    uv=xy.round().long().numpy();in_frame=(uv[:,0]>=0)&(uv[:,0]<977)&(uv[:,1]>=0)&(uv[:,1]<544)
    hit=np.zeros(len(gi),bool);hit[in_frame]=band[uv[in_frame,1],uv[in_frame,0]]
    selected=hit&(z.numpy()>.2)&(sig[:,1].numpy()<20)&(target[:,10].sigmoid().numpy()>.1)&(r['contribution']>0)
    ii=np.flatnonzero(selected)
    inv=np.full(len(m),-1);inv[match[gi]]=np.arange(len(gi));pi=inv[gi[ii]]
    pairgood=(pz[pi].numpy()>.2)&(px[pi,0].numpy()>0)&(px[pi,0].numpy()<977)&(px[pi,1].numpy()>0)&(px[pi,1].numpy()<544)
    gt_ii=ii[pairgood];pr_ii=pi[pairgood]
    active=b['pred'][b['presence']];_,near=cKDTree(active[:,:3]).query(b['target'][gi[ii],:3],workers=4)
    ap,az,_=projection(gs(torch.from_numpy(active)),cam)
    ngood=az[near].numpy()>.2
    nnerror=(ap[near]-xy[ii]).norm(dim=-1).numpy()
    ptn=(px[pr_ii]-xy[gt_ii]).norm(dim=-1).numpy()
    cell=gi[ii]//256
    bias=ck['model']['attr_decoder.head_scale.bias'].float().numpy()
    a=ck['model']['attr_decoder.scale_base_a'].float().numpy()
    lo=np.log(b['cell_scale'][cell].clip(1e-6))*a+bias-3
    gtlog=b['target'][gi[ii],3:6]
    target_small=gtlog.min(1)<lo.min(1)
    info=dict(selection='projected GT centers inside annotated 7-pixel rail band; depth > .2; sigma_max < 20 pixels; alpha > .1; nonzero contribution',
              count=len(ii),cells=len(np.unique(cell)),paired_front_and_inframe_count=int(pairgood.sum()),
              matched_displacement_px=qnt(ptn),nearest3d_displacement_px=qnt(nnerror[ngood]),
              nearest3d_displacement_gt_2px_fraction=float((nnerror[ngood]>2).mean()),
              target_sigma_min_px=qnt(sig[gt_ii,0]),pred_sigma_min_px=qnt(psig[pr_ii,0]),
              sigma_min_pair_ratio=qnt(psig[pr_ii,0]/sig[gt_ii,0]),
              target_sigma_max_px=qnt(sig[gt_ii,1]),pred_sigma_max_px=qnt(psig[pr_ii,1]),
              projected_anisotropy_target=qnt(sig[gt_ii,1]/sig[gt_ii,0]),
              projected_anisotropy_pred=qnt(psig[pr_ii,1]/psig[pr_ii,0]),
              count_per_cell=qnt(m.reshape(1024,256).sum(1)[np.unique(cell)]),
              scale_band_target_min_below_any_allowed_axis_fraction=float(target_small.mean()),
              scale_band_bias=bias.tolist(),scale_band_slope=a.tolist())
    save('focused_geometry.json',info)
    # Encoder sensitivity with anchors held fixed: destroy only within-cell xyz,
    # then destroy only within-cell attributes; measure the first pool output.
    raw=load_npz_state(setup['snapshot']);tar,_=gaussian_target_channels(raw,b['center'],scale)
    unfloored=(raw['scaling'][source[gi[ii]]].min(1)/scale)>1e-8
    info['scale_band_below_floor_excluding_clipped_target_fraction']=float(target_small[unfloored].mean())
    info['unfloored_target_count']=int(unfloored.sum())
    save('focused_geometry.json',info)
    initial=json.load(open(OUT/'rail_geometry.json'))
    groups=[x['group'] for x in initial['attention']]
    pooler=GroupPointPooler(14,1024,d=128,n_q=16,heads=4,blocks=1)
    state={k.removeprefix('encoder.pooler.'):v for k,v in ck['model'].items() if k.startswith('encoder.pooler.')}
    pooler.load_state_dict(state,strict=True);pooler.eval()
    feats=[];masks=[]
    for g in groups:
        src=b['enc_source'][g*2048:(g+1)*2048];live=src>=0
        ft=np.zeros((2048,14),np.float32);ft[live]=tar[src[live],:14]
        ft[live,:3]-=ft[live,:3].mean(0)
        ft[:,3:]=(ft[:,3:]-np.array(ATTR_MEAN[:11]))/np.array(ATTR_STD[:11])
        feats.append(ft);masks.append(live)
    ft=torch.from_numpy(np.stack(feats));mm=torch.from_numpy(np.stack(masks)).float()
    geo=ft.clone();geo[:,:,:3]=0
    attr=ft.clone();mean=(ft[:,:,3:]*mm[:,:,None]).sum(1)/mm.sum(1)[:,None]
    attr[:,:,3:]=torch.where(mm[:,:,None]>.5,mean[:,None,:],attr[:,:,3:])
    with torch.no_grad():
        base=pooler(ft[None],mm[None])[0]
        no_geo=pooler(geo[None],mm[None])[0]
        no_attr=pooler(attr[None],mm[None])[0]
    den=base.norm(dim=-1).clamp(min=1e-12)
    dg=(no_geo-base).norm(dim=-1)/den;da=(no_attr-base).norm(dim=-1)/den
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16):
        bf=pooler(ft[None],mm[None])[0]
        bg=pooler(geo[None],mm[None])[0]
    bdiff=(bg.float()-bf.float()).norm(dim=-1)/bf.float().norm(dim=-1).clamp(min=1e-12)
    save('pool_sensitivity.json',dict(precision='CPU FP32; first pool output only, all params frozen; anchors held fixed',
         groups=groups,geometry_collapsed_relative_change=qnt(dg),attributes_collapsed_relative_change=qnt(da),
         cpu_bf16_geometry_collapsed_relative_change=qnt(bdiff),cpu_bf16_unchanged_output_fraction=float((bf==bg).float().mean()),
         per_group=[dict(group=g,geometry=float(x),attributes=float(y)) for g,x,y in zip(groups,dg,da)]))
    # Exactly mimic PIL bilinear extra-view downsample, then runtime upsample.
    rows=json.load(open(OUT/'render_metrics.json'))
    for vi in (3,93):
        photo=Image.open(pool[vi]['image']).convert('RGB')
        low=torch.from_numpy(np.array(photo.resize((488,272),Image.Resampling.BILINEAR))).permute(2,0,1).float()/255
        up=F.interpolate(low[None],size=(544,977),mode='bilinear',align_corners=False)[0]
        original=torch.from_numpy(np.array(photo)).permute(2,0,1).float()/255
        ref=torch.from_numpy(read(f'view{vi}_target_dc.png')).permute(2,0,1).float()/255
        Image.fromarray((up.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)).save(OUT/f'view{vi}_half_target.png')
        rows[str(vi)]['half_target']=metrics(up,original,ref,ROIS[vi])
        rows[str(vi)]['half_target']['note']='PIL bilinear downsample, torch bilinear upsample; target metric uses saved 8-bit target PNG'
    save('render_metrics.json',rows)
    # Full-image overview, with identical ROI rectangles on all panels.
    names=['photo','original_sh3','target_sh3','target_dc','model','half_target']
    titles=['Photograph','Original NPZ: SH degree 3','Selected target: SH degree 3','Selected target: DC only','T16k 70k reconstruction','Actual extra-view target at full size']
    fig,axes=plt.subplots(2,3,figsize=(18,7.8),layout='constrained')
    for ax,name,title in zip(axes.flat,names,titles):
        ax.imshow(read(f'view3_{name}.png'));ax.set_title(title,fontsize=12);ax.axis('off')
        for rect in ROIS[3].values():
            x0,y0,x1,y1=rect;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#f24cff',lw=.8))
    fig.savefig(OUT/'overview.png',dpi=150);plt.close(fig)
    for vi,roi in [(3,'upper_rail'),(93,'near_rail')]:
        x0,y0,x1,y1=ROIS[vi][roi]
        fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
        for ax,name,title in zip(axes.flat,names,titles):
            ax.imshow(read(f'view{vi}_{name}.png')[y0:y1,x0:x1],interpolation='nearest')
            ax.set_title(title,fontsize=10);ax.axis('off')
        fig.suptitle(f'Fixed ROI: view {vi}, {roi}; nearest-neighbor display zoom',fontsize=12)
        fig.savefig(OUT/f'crop_view{vi}_{roi}.png',dpi=160);plt.close(fig)
    swaps=['target_dc','model_gt_mask','gt_covariance','gt_xyz','gt_all_attributes','dead_slot_shift']
    labels=['Target Gaussian set','Model, common GT mask','Only covariance replaced','Only xyz replaced','All attributes replaced','Only inactive xyz shifted upstream']
    x0,y0,x1,y1=ROIS[3]['upper_rail']
    fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,name,title in zip(axes.flat,swaps,labels):
        ax.imshow(read(f'view3_{name}.png')[y0:y1,x0:x1],interpolation='nearest');ax.axis('off');ax.set_title(title,fontsize=10)
    fig.suptitle('Coupled xyz/attribute ablations: Hungarian matching within each cell')
    fig.savefig(OUT/'crop_ablation.png',dpi=160);plt.close(fig)
    pool_comparison_figure()
    print(json.dumps(info,indent=2));print('figures and focused statistics complete')


if __name__=='__main__':main()
