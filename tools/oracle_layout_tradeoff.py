"""At a FIXED total latent budget, is it better to have many small cells or few large ones?

16,384 numbers for 262,144 Gaussians is 0.0625 per Gaussian no matter how it is
arranged. What differs is the structure:

    4 ch x 4096 cells   ->  4 numbers describe 64 points
   16 ch x 1024 cells   -> 16 numbers describe 256 points
   32 ch x  512 cells   -> 32 numbers describe 512 points

Larger cells amortise a code over more points, which helps if the region is
coherent and hurts if it is diverse. This measures it directly: build C k-means
anchors on the scene, take the q = N/C nearest points to each, and reconstruct
the centred radius-normalised offsets through a k-dimensional bottleneck --
the same PCA / MLP pair and the same ich / nn_unique formula the training log
prints, so every row is on one axis.
"""
from __future__ import annotations
import argparse, json, os, sys
from argparse import Namespace
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.oracle_shape_budget_slotwise import measure  # noqa: E402


def kmeans(x, k, iters, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    c = x[torch.randperm(x.shape[0], generator=g, device=dev)[:k]].clone()
    for _ in range(iters):
        lab = torch.empty(x.shape[0], dtype=torch.long, device=dev)
        for i in range(0, x.shape[0], 200_000):
            lab[i:i + 200_000] = torch.cdist(x[i:i + 200_000], c).argmin(1)
        cnt = torch.bincount(lab, minlength=k)
        s = torch.zeros_like(c).index_add_(0, lab, x)
        nz = cnt > 0
        c = torch.where(nz.unsqueeze(-1), s / cnt.clamp(min=1).unsqueeze(-1).float(), c)
    return c


def groups_for(xyz, anchors, q, dev):
    """TRUE partition: every point goes to its nearest anchor, then each anchor
    keeps up to q of its own points. Taking topk(q) per anchor independently --
    which an earlier version did -- is not a partition: a point lands in several
    groups and others land in none, and it scored 4ch/64pts at nn_unique 0.245
    against 0.489 for the same budget under the real slot layout.

    Only anchors that filled all q slots are kept, so every group enters at the
    same width and the comparison is not a comparison of fill rates.
    """
    C = anchors.shape[0]
    lab = torch.empty(xyz.shape[0], dtype=torch.long, device=dev)
    for i in range(0, xyz.shape[0], 200_000):
        lab[i:i + 200_000] = torch.cdist(xyz[i:i + 200_000], anchors).argmin(1)
    order = torch.argsort(lab)
    lab_s, xyz_s = lab[order], xyz[order]
    cnt = torch.bincount(lab_s, minlength=C)
    start = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), cnt.cumsum(0)[:-1]])
    full = (cnt >= q).nonzero().squeeze(1)
    if full.numel() == 0:
        return None
    idx = start[full].unsqueeze(1) + torch.arange(q, device=dev).unsqueeze(0)
    g = xyz_s[idx]                       # (n_full, q, 3)
    cen = g.mean(1, keepdim=True)
    off = g - cen
    rad = off.norm(dim=-1).mean(-1).clamp(min=1e-9)
    return (off / rad[:, None, None]), rad


def abs_chamfer(pred, true, rad, q, chunk=256):
    """Chamfer in SCENE units. ich divides by the group radius, so it flatters
    large groups; this multiplies the radius back and is the only cross-layout
    comparable number here."""
    p = pred.reshape(-1, q, 3); t = true.reshape(-1, q, 3)
    p = p - p.mean(1, keepdim=True)
    out = []
    for i in range(0, p.shape[0], chunk):
        d = torch.cdist(p[i:i+chunk], t[i:i+chunk])
        out.append(0.5 * (d.min(2).values.mean(1) + d.min(1).values.mean(1)))
    return float((torch.cat(out) * rad).median())


def fit_eval(tr, te, k, q, hidden, steps, batch, lr, dev, te_rad=None):
    D = q * 3
    trf, tef = tr.reshape(-1, D), te.reshape(-1, D)
    mu = trf.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(trf - mu, q=min(k + 8, D - 1, trf.shape[0] - 1))
    B = V[:, :k]
    rec_l = ((tef - mu) @ B) @ B.T + mu
    u_l, i_l = measure(rec_l, tef, q)
    a_l = abs_chamfer(rec_l, tef, te_rad, q)
    torch.manual_seed(0)
    enc = torch.nn.Sequential(torch.nn.Linear(D, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, k)).to(dev)
    dec = torch.nn.Sequential(torch.nn.Linear(k, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, D)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, trf.shape[0], (min(batch, trf.shape[0]),), device=dev)
        x = trf[idx]
        loss = (dec(enc(x)) - x).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rec_n = dec(enc(tef))
        u_n, i_n = measure(rec_n, tef, q)
        a_n = abs_chamfer(rec_n, tef, te_rad, q)
    return (u_l, i_l, a_l), (u_n, i_n, a_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--train_scenes", default="40,120,204,300")
    ap.add_argument("--test_scenes", default="160,224")
    ap.add_argument("--layouts", default="4:64,16:256,32:512",
                    help="comma list of channels:points_per_cell")
    ap.add_argument("--kmeans_iters", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    dev = "cuda"

    from can3tok.train import make_datasets
    targs = Namespace(**json.load(open(a.args_json)))
    targs.out_dir = os.path.dirname(os.path.abspath(a.args_json))
    _, val = make_datasets(targs)

    def load(idxs):
        out = []
        for i in idxs:
            it = val[int(i)]
            m = it["mask"].to(dev) > 0.5
            out.append(it["target"].to(dev).float()[m][:, 0:3])
        return out

    tr_pts = load([x for x in a.train_scenes.split(",") if x])
    te_pts = load([x for x in a.test_scenes.split(",") if x])
    print(f"train snapshots {len(tr_pts)}  held-out {len(te_pts)}  "
          f"points {[int(p.shape[0]) for p in te_pts]}")
    print(f"\n{'layout':>22} {'cells':>7} {'pts/cell':>9} "
          f"{'MLP uniq':>9} {'MLP ich':>9} {'MLP abs_chamfer':>16}   <- abs 가 유일한 교차비교 지표")

    for spec in [s for s in a.layouts.split(",") if s]:
        k, q = (int(v) for v in spec.split(":"))
        C = 262144 // q
        def build(pts_list):
            gs, rs = [], []
            for p in pts_list:
                if p.shape[0] < q:
                    continue
                anch = kmeans(p, C, a.kmeans_iters, 0, dev)
                r = groups_for(p, anch, q, dev)
                if r is not None:
                    gs.append(r[0]); rs.append(r[1])
            return torch.cat(gs, 0), torch.cat(rs, 0)
        tr, _ = build(tr_pts); te, te_rad = build(te_pts)
        (u_l, i_l, a_l), (u_n, i_n, a_n) = fit_eval(tr, te, k, q, a.hidden, a.steps,
                                                    a.batch, a.lr, dev, te_rad=te_rad)
        print(f"{f'{k}ch x {C}cells':>22} {C:>7,} {q:>9} "
              f"{u_n:>9.3f} {i_n:>9.4f} {a_n:>16.5f}")


if __name__ == "__main__":
    main()
