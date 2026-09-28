"""Read-only diagnostics for the 2026-09-15 audit; does not train or edit the model.

Run from the repository root using the can3tok Python environment.
--model additionally loads a checkpoint and renders on the explicitly selected GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from argparse import Namespace
from can3tok import losses as L
from can3tok.io_utils import load_npz_state, load_replay_dict
from can3tok.train import make_datasets, build_config, load_eval_views
from can3tok.schedule import schedule_flags
from can3tok.template import fibonacci_ball

OUT = Path(__file__).parent


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    print(name, json.dumps(obj, ensure_ascii=False), flush=True)


def quant(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    return dict(zip(('p0', 'p50', 'p90', 'p99', 'p100'),
                    np.quantile(x, [0, .5, .9, .99, 1]).tolist())) if x.size else {}


def checkpoint_metadata():
    rows=[]
    for run,step in [('S16k_20260910_012456',62000),
                     ('T16k_20260913_093332',70000),
                     ('E1_20260915_041813',2000)]:
        p=ROOT/'runs'/run/f'ckpt_step{step:08d}.pt'
        ck=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
        w=ck['model']['decoder.unpack.weight']; b=ck['model']['decoder.unpack.bias']
        identity=torch.eye(w.shape[0],w.shape[1]); difference=w-identity
        rows.append(dict(run=run,step=int(ck['step']),
                         score_metric=ck.get('args',{}).get('score_metric'),
                         best_score=ck.get('best_score'),
                         best_file_exists=(p.parent/'ckpt_best.pt').exists(),
                         unpack_shape=list(w.shape),
                         unpack_relative_frobenius=float(difference.norm()/identity.norm()),
                         unpack_max_abs_difference=float(difference.abs().max()),
                         unpack_bias_norm=float(b.norm()),
                         unpack_xyz_aux_block_norm=float(w[:768,768:].norm()),
                         latent_scale=float(ck['model']['latent_scale'])))
        del ck,w,b,identity,difference
    save('checkpoint_metadata.json',rows)


def cpu_probes(args):
    torch.manual_seed(1515)
    layout = dict(xyz=0, scale=3, rot=6, opacity=10, color=11, sh=14)
    x = torch.zeros(100, 3)
    x[:, 0] = torch.linspace(-.99, .99, 100)
    si = L._sample_spatial_indices(x, torch.ones(100), 60)
    t = torch.randn(1, 256, 14)
    t[..., 6:10] = torch.nn.functional.normalize(t[..., 6:10], dim=-1)
    aset = float(L.intra_group_attr_set_loss(t, t, torch.ones(1, 256), 256, layout))
    # Ordered target is perfect on valid slots; moving a dead slot changes the
    # current global projection loss, while the rendered live set is unchanged.
    gt = torch.zeros(1, 16, 3)
    gt[0, :8] = torch.randn(8, 3) * .1
    m = torch.cat([torch.ones(1, 8), torch.zeros(1, 8)], 1)
    p1 = gt.clone(); p2 = p1.clone(); p2[0, 8:] = .8
    hist = [float(L.projected_histogram_loss(p, gt, m, samples=16)) for p in (p1, p2)]
    from can3tok.render import _edge_weight, _EDGE_W_CACHE
    _EDGE_W_CACHE.clear()
    image = torch.zeros(3, 8, 8)
    w1 = _edge_weight(image, 4).clone()
    image[:, :, 4:] = 1
    stale = _edge_weight(image, 4).clone()
    _EDGE_W_CACHE.clear()
    fresh = _edge_weight(image, 4).clone()
    ball = fibonacci_ball(256)
    res = dict(
        sampling_100_to_60=dict(count=len(si), selected_min=int(si.min()),
                               selected_max=int(si.max()), unsampled_tail=100-1-int(si.max())),
        attr_set_gt_self_256=aset,
        proj_hist_only_dead_slots_changed=hist,
        edge_cache_same_pointer_changed_image=dict(stale_equals_old=bool(torch.equal(stale,w1)),
                                                   max_error=float((stale-fresh).abs().max())),
        template_prefix_centroid_norm={str(n):float(np.linalg.norm(ball[:n].mean(0)))
                                       for n in [32,64,80,128,192,256]},
        schedule={str(s):{k:v for k,v in schedule_flags(s,args).items()
                         if k in ('attr_detach_geometry','render_downscale','folding_res_gain')}
                  for s in [0,12000,40000,70000]})
    save('cpu_probes.json',res)


def data_probes(args):
    train, val = make_datasets(args)
    val.extra_real_views = 0
    views, held = load_eval_views(args)
    vp = json.load(open(args.view_pool))
    pm = json.load(open(args.photo_map))
    held_paths = {e['image'] for pool in vp.values() for j,e in enumerate(pool) if j in held}
    leaks = []
    for p,si in zip(train.files,train.file_scene):
        e = pm.get(p,pm.get(str(Path(p).resolve()),pm.get(Path(p).name)))
        ip = e.get('image') if isinstance(e,dict) else e
        if ip in held_paths: leaks.append(dict(npz=p,scene=int(si),image=ip))
    save('view_audit.json',dict(held_indices=sorted(held),pool_sizes={k:len(v) for k,v in vp.items()},
                              eval_views=len(views), scene_tag_in_view_tuple=False,
                              own_photo_leak_count=len(leaks),own_photo_leaks=leaks))
    rows = []
    for vi in [25,166,229,382]:
        path = val.files[vi]
        raw = load_replay_dict(path); gs = load_npz_state(path)
        selected, assignments = [], []
        orig_select, orig_assign = val._select, val._anchor_assign
        def capture_select(*a,**kw):
            r=orig_select(*a,**kw); selected.append(r.copy()); return r
        def capture_assign(*a,**kw):
            r=orig_assign(*a,**kw); assignments.append(r.copy()); return r
        val._select, val._anchor_assign = capture_select, capture_assign
        it=val[vi]
        val._select, val._anchor_assign = orig_select, orig_assign
        masks=[it['mask'].numpy().reshape(1024,256),it['enc_mask'].numpy().reshape(1024,2048)]
        pts=[it['target'].numpy()[:,:3].reshape(1024,256,3),
             it['enc_input'].numpy()[:,:3].reshape(1024,2048,3)]
        counts=[m.sum(1) for m in masks]
        means=[(p*m[...,None]).sum(1)/np.maximum(c[:,None],1) for p,m,c in zip(pts,masks,counts)]
        stds=[np.sqrt((((p-mu[:,None])**2)*m[...,None]).sum(1)/np.maximum(c[:,None],1)).max(1)
              for p,m,c,mu in zip(pts,masks,counts,means)]
        owners=[]
        for sel,pos,cap in zip(selected,assignments,[256,2048]):
            owner=np.full(int(it['num_points_valid']),-1,np.int32)
            live=pos>=0; owner[sel[pos[live]]]=np.flatnonzero(live)//cap; owners.append(owner)
        td=owners[0]>=0; common=td & (owners[1]>=0)
        valid=(counts[0]>1)&(counts[1]>1)
        row=dict(val_index=vi,path=path,scene=int(it['scene']),
                 original=int(it['num_points_original']),valid=int(it['num_points_valid']),
                 target_live=int(counts[0].sum()),encoder_live=int(counts[1].sum()),
                 raw_dtypes={k:str(np.asarray(v).dtype) for k,v in raw['gaussians'].items()},
                 raw_sh_shape=list(np.asarray(raw['gaussians']['features_rest']).shape),
                 loader_target_dim=val.target_dim,loader_input_dim=val.input_dim,
                 raw_pruning_scores='pruning_scores' in raw,loaded_pruning_scores='pruning_scores' in gs,
                 target_missing_from_encoder=int((td & (owners[1]<0)).sum()),
                 target_owner_changed_fraction=float((owners[0][common]!=owners[1][common]).mean()),
                 partial_nonempty_group_fraction=float(((counts[0]>0)&(counts[0]<256)).sum()/max((counts[0]>0).sum(),1)),
                 target_count_quantiles=quant(counts[0]),encoder_count_quantiles=quant(counts[1]),
                 count_prior_mae=float(np.abs(np.clip(counts[1],0,256)-counts[0]).mean()),
                 center_mismatch_norm=quant(np.linalg.norm(means[1][valid]-means[0][valid],axis=1)),
                 center_mismatch_over_encoder_extent=quant(np.linalg.norm(means[1][valid]-means[0][valid],axis=1)/np.maximum(stds[1][valid],1e-6)),
                 encoder_extent=quant(stds[1][valid]),
                 extent_ratio_encoder_over_target=quant(stds[1][valid]/np.maximum(stds[0][valid],1e-6)),
                 raw_linear_scale_zero_fraction=float((np.asarray(raw['gaussians']['scaling'])==0).mean()))
        rows.append(row); save('data_probes.json',rows)
        del it,raw,gs,pts,masks; gc.collect()
    return val


def spectral(x, valid=None):
    x=x.detach().float().reshape(1024,-1)
    if valid is not None:
        x=x[valid]
    x=x-x.mean(0,keepdim=True)
    with torch.autocast('cuda',enabled=False):
        eig=torch.linalg.eigvalsh(x@x.T).clamp(min=0)
    p=eig/eig.sum().clamp(min=1e-20)
    return dict(rows=x.shape[0],width=x.shape[1],
                covariance_entropy_rank=float(torch.exp(-(p*(p+1e-30).log()).sum())),
                top_variance_fraction=float(p[-1]),rms=float(x.square().mean().sqrt()))


def model_probes(args,val,checkpoint,device,ranks_only=False):
    torch.cuda.set_device(device)
    from can3tok.model import build_model
    from can3tok.render import Camera,render_gaussians,sh_dc_to_rgb,psnr
    from can3tok.io_utils import camera_from_vector
    from scipy.spatial import cKDTree
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=build_config(args,val.target_dim,val.sh_dim)
    model=build_model(cfg,allow_padded_tokens=args.allow_padded_tokens)
    status=model.load_state_dict(ck['model'],strict=True)
    step=int(ck['step']); saved_best=float(ck.get('best_score',float('nan')))
    flags=schedule_flags(step,args)
    for key in ['decoder_refine_alpha','folding_res_gain','attr_detach_geometry']:
        setattr(cfg,key,flags[key])
    model.set_encoder_residual(flags['encoder_residual'])
    cfg.attr_teacher_prob=0.; model.to(device).eval()
    meta=dict(checkpoint=str(checkpoint),step=step,saved_best_score=saved_best,
              score_metric=args.score_metric,strict_load=str(status),latent_scale=float(model.latent_scale),
              cfg_target_dim=cfg.target_dim,cfg_sh_dim=cfg.sh_dim)
    del ck
    views,held=load_eval_views(args)
    vp=json.load(open(args.view_pool)); pool_names=list(vp)
    # load_eval_views concatenates equal-size per-scene subsets in dict order.
    nview=len(held)
    results=[]
    for vi in [166,382]:
        it=val[vi]; si=int(it['scene'])
        x=it['input'][None].to(device); m=it['mask'][None].to(device)
        ex=it['enc_input'][None].to(device); em=it['enc_mask'][None].to(device)
        stages={}; handles=[]
        for name,module in [('pooler',model.encoder.pooler),('pack',model.encoder.pack),
                            ('token_merge',model.compressor.token_merge),('mid',model.compressor.mid)]:
            # Pooler is called in chunks; retain all chunks and concatenate later.
            stages[name]=[]
            handles.append(module.register_forward_hook(lambda mod,inp,out,n=name:stages[n].append(out.detach())))
        stages['pre_merge']=[]
        handles.append(model.compressor.token_merge.register_forward_pre_hook(
            lambda mod,inp:stages['pre_merge'].append(inp[0].detach())))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(x,m,run_gen=False,enc_x=ex,enc_mask=em)
        for h in handles:h.remove()
        ranks={name:spectral(torch.cat(seq,dim=1)) for name,seq in stages.items()}
        nonempty=em.reshape(1024,-1).sum(-1)>0
        ranks_nonempty={name:spectral(torch.cat(seq,dim=1),nonempty) for name,seq in stages.items()}
        del stages
        z=out['z_compact']
        # Same autocast as forward, including attribute decode, using raw compact.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            decoded=model.decode_compact(z)
        pub_diff=dict(xyz_max=float((decoded['pred'][...,:3]-out['pred'][...,:3]).abs().max()),
                      attributes_mae_vs_training=float((decoded['pred'][...,3:]-out['attr_pred'][...,3:]).abs().mean()),
                      attr_pred_in_public='attr_pred' in decoded)
        save('model_progress.json',dict(meta=meta,val_index=vi,ranks=ranks,ranks_nonempty=ranks_nonempty,public_decode_difference=pub_diff))
        if ranks_only:
            results.append(dict(val_index=vi,path=val.files[vi],ranks=ranks,ranks_nonempty=ranks_nonempty))
            save('rank_probes.json',dict(meta=meta,results=results))
            del decoded,x,ex,em,out,it,z,m
            gc.collect();torch.cuda.empty_cache()
            continue
        del decoded,x,ex,em
        target=it['target'].to(device).float()[:,:14]
        pred=out['attr_pred'][0].float()
        gt_mask=m[0]>0.5; pm=out['presence'][0]>0
        sc=float(it['scale']); center=it['center'].to(device)
        def unpack(t,mask):
            t=t[mask]
            return(t[:,:3]*sc+center,t[:,3:6].clamp(-15,5).exp()*sc,t[:,6:10],
                   t[:,10:11].sigmoid(),sh_dc_to_rgb(t[:,11:14]))
        sets={'target':unpack(target,gt_mask),'oracle_mask':unpack(pred,gt_mask),'pred_mask':unpack(pred,pm)}
        per_view=[]
        with torch.no_grad():
            for j,(cv,photo) in enumerate(views):
                camera=Camera(camera_from_vector(cv),device=device,downscale=2)
                ref=torch.nn.functional.interpolate(photo[None].to(device),size=(camera.image_height,camera.image_width),
                                                    mode='bilinear',align_corners=False)[0]
                scores={}
                for name,gs in sets.items():
                    image=render_gaussians(*gs,camera)
                    scores[name]=psnr(image,ref)
                per_view.append(dict(view_scene=j//nview,held_index=sorted(held)[j%nview],scores=scores))
        summary={}
        for which,rows in [('own', [r for r in per_view if r['view_scene']==si]),
                           ('foreign',[r for r in per_view if r['view_scene']!=si]),('mixed',per_view)]:
            summary[which]={k:float(np.mean([r['scores'][k] for r in rows])) for k in sets}
        pg=pred[pm].cpu().numpy(); tg=target[gt_mask].cpu().numpy()
        tree=cKDTree(tg[:,:3]); dd,nn=tree.query(pg[:,:3],workers=4)
        aniso_p=np.exp(pg[:,3:6].max(1)-pg[:,3:6].min(1))
        aniso_t=np.exp(tg[:,3:6].max(1)-tg[:,3:6].min(1))
        gt_nn=tree.query(tg[:,:3],k=2,workers=4)[0][:,1]
        threshold=np.median(gt_nn[gt_nn>0])
        # Perfect self target on the actual 256-slot groups (not sampled subsets).
        with torch.no_grad():
            aset=float(L.intra_group_attr_set_loss(target[None],target[None],m,256,val.layout,chunk_groups=48))
        row=dict(val_index=vi,scene=si,path=val.files[vi],ranks=ranks,ranks_nonempty=ranks_nonempty,
                 public_decode_difference=pub_diff,render_psnr=summary,per_view=per_view,
                 target_presence_count=int(gt_mask.sum()),pred_presence_count=int(pm.sum()),
                 presence_false_positive=int((pm & ~gt_mask).sum()),presence_false_negative=int((~pm & gt_mask).sum()),
                 actual_attr_set_gt_self=aset,
                 anisotropy_pred=quant(aniso_p),anisotropy_target=quant(aniso_t),
                 nn_matched_anisotropy_pred_for_top20_gt=quant(aniso_p[aniso_t[nn]>=np.quantile(aniso_t,.8)]),
                 pred_to_gt_normalized=quant(dd),pred_to_gt_world=quant(dd*sc),
                 gt_nonzero_nn_median_world=float(threshold*sc),
                 pred_gt_nn_gt5x_fraction=float((dd>5*threshold).mean()),
                 pred_gt_nn_gt50x_fraction=float((dd>50*threshold).mean()))
        results.append(row); save('model_probes.json',dict(meta=meta,results=results))
        del out,pred,target,m,it,sets,z;gc.collect();torch.cuda.empty_cache()


def main():
    global OUT
    p=argparse.ArgumentParser()
    p.add_argument('--args',default='runs/T16k_20260913_093332/args.json')
    p.add_argument('--model',action='store_true')
    p.add_argument('--model-only',action='store_true')
    p.add_argument('--ranks-only',action='store_true')
    p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--out-dir',default='')
    p.add_argument('--device',default='cuda:3')
    p.add_argument('--checkpoint',default='runs/T16k_20260913_093332/ckpt_step00070000.pt')
    a=p.parse_args(); args=Namespace(**json.load(open(a.args)))
    if a.out_dir:
        OUT=Path(a.out_dir);OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if a.metadata_only:
        checkpoint_metadata();return
    if a.model_only:
        _,val=make_datasets(args);val.extra_real_views=0
    else:
        cpu_probes(args); val=data_probes(args)
    if a.model or a.model_only:model_probes(args,val,a.checkpoint,a.device,a.ranks_only)


if __name__=='__main__':main()
