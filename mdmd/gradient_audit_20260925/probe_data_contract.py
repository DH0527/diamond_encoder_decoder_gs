"""CPU-only audit of the currently configured encoder/target cell contract."""
from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np,torch
from argparse import Namespace
from can3tok.train import make_datasets
OUT=Path(__file__).resolve().parent
torch.set_num_threads(2)
a=Namespace(**json.loads((ROOT/'runs/B1g_20260925_053253/args.json').read_text()))
a.extra_real_views=0;a.holdout_own_photo=0
tr,ds=make_datasets(a)
frozen=json.loads((OUT/'frozen_probe.json').read_text())['B1g_42000']
results={}
def quant(x):
    return dict(zip(['min','p01','p50','p99','max'],np.quantile(x,[0,.01,.5,.99,1]).tolist()))
for scene,v in frozen.items():
    it=ds[v['index']]
    em=it['enc_mask'].numpy()>.5;tm=it['mask'].numpy()>.5
    ec=em.reshape(1024,2048).sum(1);tc=tm.reshape(1024,256).sum(1)
    occ=2*ec/256.-1
    record={'file':ds.files[v['index']], 'encoder_count':quant(ec),'target_count':quant(tc),
            'encoder_count_exceeds_256_fraction':float((ec>256).mean()),
            'target_partial_cell_fraction':float(((tc>0)&(tc<256)).mean()),
            'encoder_full_but_target_partial_fraction':float(((ec>=256)&(tc<256)&(tc>0)).mean()),
            'count_prior_abs_error_mean':float(np.abs(np.minimum(ec,256)-tc).mean()),
            'raw_occupancy_anchor':quant(occ)}
    ex=it['enc_input'].numpy()[em,:14];tx=it['input'].numpy()[tm,:14]
    # Exact 14-channel row identity; duplicates are explicitly excluded when
    # estimating owner mismatch. This is a diagnostic, not a source-ID contract.
    width=min(ex.shape[1],tx.shape[1]);dt=np.dtype((np.void,width*4))
    ek=np.ascontiguousarray(ex[:,:width],dtype=np.float32).view(dt).reshape(-1)
    tk=np.ascontiguousarray(tx[:,:width],dtype=np.float32).view(dt).reshape(-1)
    uq,first,cnt=np.unique(ek,return_index=True,return_counts=True)
    pos=np.searchsorted(uq,tk);pos=np.minimum(pos,len(uq)-1)
    exists=uq[pos]==tk;unique=exists&(cnt[pos]==1)
    eg=np.flatnonzero(em)//2048;tg=np.flatnonzero(tm)//256
    egmatched=eg[first[pos]]
    record.update(exact_row_channels=width, exact_target_rows_found_fraction=float(exists.mean()),
                  exact_target_rows_uniquely_found_fraction=float(unique.mean()),
                  different_cell_among_unique_matches_fraction=float((egmatched[unique]!=tg[unique]).mean()))
    results[scene]=record
    print(scene,json.dumps(record),flush=True)
(OUT/'data_contract.json').write_text(json.dumps(results,indent=2))
