"""Is the shape-channel budget the wall? Measured on the grouping the model uses.

`oracle_shape_rank.py` and `oracle_nonlinear_shape.py` both build their group
vectors by compacting the mask and cutting the result into runs of `group_size`.
That is correct for Morton chunks and wrong for fixed anchors, where each group
is filled to its own count: the compacted runs straddle anchor boundaries.
Measured on this run's two scenes the compacted groups have a median radius
25-39x the true one, and every `ich` is divided by that radius -- so the numbers
those tools report for an anchor run are not comparable to `group_error_breakdown`,
which is what the training log prints.

This reproduces the same question with the slot-layout grouping and the same
prefix rule the eval uses, so PCA-k / MLP-k / the model can be read on one axis.
"""
from __future__ import annotations
import argparse, os, sys
from argparse import Namespace
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def group_offsets(target, mask, gs, q):
    """Centred, radius-normalised offsets of the leading q slots of full groups."""
    t = target[..., 0:3].reshape(-1, gs, 3).float()
    cnt = (mask.reshape(-1, gs) > 0.5).sum(-1)
    sel = cnt >= q
    g = t[sel][:, :q]
    cen = g.mean(1, keepdim=True)
    off = g - cen
    rad = off.norm(dim=-1).mean(-1).clamp(min=1e-9)
    return (off / rad[:, None, None]).reshape(-1, q * 3)


@torch.no_grad()
def measure(pred, true, q, chunk=512):
    """Identical formula to eval_utils.group_error_breakdown."""
    p = pred.reshape(-1, q, 3); t = true.reshape(-1, q, 3)
    p = p - p.mean(1, keepdim=True)
    rad = t.norm(dim=-1).mean(-1).clamp(min=1e-9)
    uq, ic = [], []
    for i in range(0, p.shape[0], chunk):
        d = torch.cdist(p[i:i+chunk], t[i:i+chunk])
        nn_ = d.argmin(dim=2)
        hit = torch.zeros_like(nn_, dtype=torch.bool).scatter_(1, nn_, True)
        uq.append(hit.float().mean(1))
        ic.append(0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)) / rad[i:i+chunk])
    return float(torch.cat(uq).mean()), float(torch.cat(ic).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--train_scenes", default="40,63,120,204,300")
    ap.add_argument("--test_scenes", default="160,224,340")
    ap.add_argument("--q", type=int, default=64, help="slots per group required (full group)")
    ap.add_argument("--dims", default="19")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    import json
    from can3tok.train import make_datasets
    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val = make_datasets(targs)
    gs = int(targs.group_size); q = int(a.q); dev = "cuda"

    def collect(idxs):
        outs = []
        for i in idxs:
            it = val[int(i)]
            outs.append(group_offsets(it["target"].to(dev).float(),
                                      it["mask"].to(dev).float(), gs, q))
        return torch.cat(outs, 0)

    tr = collect([x for x in a.train_scenes.split(",") if x])
    te = collect([x for x in a.test_scenes.split(",") if x])
    D = q * 3
    print(f"slot-layout grouping | q={q} slots  offset dim {D}  "
          f"train groups {tr.shape[0]}  HELD-OUT groups {te.shape[0]}")
    u_gt, i_gt = measure(te, te, q)
    print(f"GT control (te vs te): nn_unique {u_gt:.3f}  ich {i_gt:.4f}\n")

    print(f"{'method':>26} {'nn_unique':>10} {'ich':>9}   set")
    for dim in [int(x) for x in a.dims.split(",") if x]:
        mu = tr.mean(0, keepdim=True)
        U, S, V = torch.pca_lowrank(tr - mu, q=min(dim + 8, D - 1))
        B = V[:, :dim]
        rec = lambda X: ((X - mu) @ B) @ B.T + mu
        u_l, i_l = measure(rec(te), te, q)
        print(f"{'PCA-' + str(dim):>26} {u_l:10.3f} {i_l:9.4f}   HELD-OUT")

        torch.manual_seed(0)
        enc = torch.nn.Sequential(torch.nn.Linear(D, a.hidden), torch.nn.GELU(),
                                  torch.nn.Linear(a.hidden, a.hidden), torch.nn.GELU(),
                                  torch.nn.Linear(a.hidden, dim)).to(dev)
        dec = torch.nn.Sequential(torch.nn.Linear(dim, a.hidden), torch.nn.GELU(),
                                  torch.nn.Linear(a.hidden, a.hidden), torch.nn.GELU(),
                                  torch.nn.Linear(a.hidden, D)).to(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
        for it in range(a.steps):
            idx = torch.randint(0, tr.shape[0], (min(a.batch, tr.shape[0]),), device=dev)
            x = tr[idx]
            loss = (dec(enc(x)) - x).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            u_n, i_n = measure(dec(enc(te)), te, q)
        print(f"{'MLP AE-' + str(dim):>26} {u_n:10.3f} {i_n:9.4f}   HELD-OUT")

    print("\n모델 실측(같은 지표, 같은 그룹핑): nn_unique 0.446-0.485  ich 0.344-0.433")


if __name__ == "__main__":
    main()
