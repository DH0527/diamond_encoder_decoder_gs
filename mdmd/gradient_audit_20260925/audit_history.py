"""Read-only training audit. Snapshot sources/logs; parse logged window statistics."""
from pathlib import Path
import json, re, hashlib, shutil, datetime, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUNS = ['C1_l1_vanilla_20260921_005556', 'B1_bgcap_20260922_024641',
        'B1g_20260924_105249', 'B1g_20260925_053253']
snap = OUT / 'snapshot'
manifest = {'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': {}}
files = list((ROOT / 'can3tok').glob('*.py'))
files += [ROOT / 'scripts/launch_b1g_resume_30k.sh']
for run in RUNS:
    files += [ROOT / f'runs/{run}.log', ROOT / f'runs/{run}/args.json', ROOT / f'runs/{run}/train_metrics.jsonl']
for src in files:
    if not src.exists():
        continue
    rel = src.relative_to(ROOT)
    dst = snap / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    data = dst.read_bytes()
    manifest['files'][str(rel)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))

def extract(pattern, line):
    m = re.search(pattern, line)
    return float(m[1]) if m else None

all_rows, summary = [], {}
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
for run in RUNS:
    rows, evals, skips, loss_skips, anchors = [], [], [], [], []
    lines = (snap / f'runs/{run}.log').read_text(errors='replace').splitlines()
    for line in lines:
        if re.search(r'^\[\w+\] step \d+/\d+ loss ', line):
            row = {'run': run, 'step': int(extract(r'step (\d+)', line))}
            for key, pat in {'loss':r' loss ([\d.e+\-]+)', 'gn':r' gn ([\d.e+\-]+)',
                             'cov3d':r' c3d ([\d.e+\-]+)', 'radius':r'U\[rad ([\d.e+\-]+)',
                             'rattr':r' rattr ([\d.e+\-]+)', 'aniso':r'shp\[p50 ([\d.e+\-]+)',
                             'aniso_gt':r'shp\[p50 [\d.e+\-]+/([\d.e+\-]+)',
                             'ich':r' ich ([\d.e+\-]+)', 'centroid':r' cen ([\d.e+\-]+)',
                             'ramp_sobel':r' sob ([\d.e+\-]+)', 'lr':r' lr ([\d.e+\-]+)'}.items():
                row[key] = extract(pat, line)
            rows.append(row)
        elif 'eval step' in line:
            evals.append({'step':int(extract(r'eval step (\d+)', line)),
                          'photo':extract(r'PSNR photo=([\d.]+)', line),
                          'teacher':extract(r'teacher=([\d.]+)', line),
                          'train':extract(r'\| s0 ([\d.]+)', line),
                          'truck':extract(r'\| s1 ([\d.]+)', line), 'line':line})
        elif '[GSKIP]' in line:
            skips.append({'step':int(extract(r'step (\d+)', line)),
                          'norm':extract(r'gnorm ([\d.e+\-]+)', line),
                          'cumulative':int(extract(r'\((\d+) grad-skips', line)), 'line':line})
        elif '[SKIP]' in line:
            loss_skips.append(line)
        elif '[GSPAN]' in line:
            anchors.append(line)
    windows = {}
    for lo, hi in [(0,12000),(12000,20000),(20000,30000),(30000,32000),
                   (32000,34000),(34000,35000),(35000,37000),(37000,40000),
                   (40000,41000),(41000,42000),(42000,44000)]:
        rr = [r for r in rows if lo < r['step'] <= hi]
        if not rr: continue
        windows[f'{lo}-{hi}'] = {'logged_windows':len(rr)}
        for k in ('gn','loss','aniso','rattr','ich'):
            vals = [r[k] for r in rr if r[k] is not None]
            if vals:
                windows[f'{lo}-{hi}'][k] = {'median':float(np.median(vals)), 'max':max(vals)}
    summary[run] = {'first':rows[0] if rows else None, 'last':rows[-1] if rows else None,
                    'windows':windows, 'evals':evals, 'grad_skips':skips,
                    'loss_skips':loss_skips, 'anchors':anchors,
                    'max_logged_norm':max(rows,key=lambda r:r['gn'] or 0) if rows else None}
    all_rows += rows
    label = run.replace('_202609',' / Sep ')
    xx = [r['step'] for r in rows]
    axs[0].plot(xx,[r['gn'] for r in rows],label=label,lw=1)
    axs[1].plot(xx,[r['loss'] for r in rows],lw=1)
    axs[2].plot([r['step'] for r in evals],[r['photo'] for r in evals],marker='.',lw=1)
    axs[3].plot(xx,[r['aniso'] for r in rows],lw=1)
    if skips:
        axs[0].scatter([r['step'] for r in skips],[r['norm'] for r in skips],marker='x',s=12)
axs[0].set_yscale('log')
axs[0].set_ylabel('Pre-clip norm (logged)')
axs[0].legend(fontsize=8)
axs[1].set_ylabel('Loss (logged)')
axs[2].set_ylabel('Photo PSNR / dB')
axs[3].set_ylabel('Pred. anisotropy p50')
axs[3].set_xlabel('Attempted step (skips included)')
for ax in axs:
    for step in (12000,34100,40000,43000): ax.axvline(step,ls='--',c='gray',alpha=.3)
    ax.grid(alpha=.2)
fig.suptitle('History audit: logged averages are not per-step gradients; x = sampled skip events')
fig.tight_layout()
fig.savefig(OUT / 'training_history.png',dpi=160)
(OUT / 'history.json').write_text(json.dumps(summary,indent=2))
with (OUT / 'logged_windows.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
args={run:json.loads((snap/f'runs/{run}/args.json').read_text()) for run in RUNS}
diff={k:{r:a.get(k,'<absent>') for r,a in args.items()} for k in sorted(set().union(*(a.keys() for a in args.values())))
      if len({json.dumps(a.get(k,'<absent>'),sort_keys=True) for a in args.values()})>1}
(OUT / 'args_diff.json').write_text(json.dumps(diff,indent=2))
for run,s in summary.items():
    print(run, 'last',s['last']['step'], 'max logged gn',s['max_logged_norm']['gn'],
          'skip lower bound',s['grad_skips'][-1]['cumulative'] if s['grad_skips'] else 0)
    print('anchors',s['anchors'])
    print('windows',json.dumps(s['windows']))
