"""FiLM, or something else? How the group code should enter the attribute decoder.

The code is one 16-dim vector per group; the tokens are 64 slots. FiLM is the
standard way to inject a global vector into a set of tokens, which is why it was
chosen, but it has a specific limitation worth pricing: `gam` and `bet` broadcast
across all 64 slots, so the code cannot say "slot 5 is red, slot 6 is blue". It
can only rescale and shift a slot basis that is otherwise fixed across every scene
in the dataset. Whatever distinguishes one slot from another has to come from
`slot_emb` or from the point's own position.

That matters here more than usual, because 52-78% of the attribute variance is
*within* a group -- precisely the part a broadcast modulation cannot address
directly.

So: same code budget, same width, same depth, same optimiser, only the way the
code reaches the tokens changes.

    none      code ignored entirely            (floor: slot + position only)
    film      tok * (1 + gam) + bet            (what the model does)
    concat    code appended to every token     (the MLP forms its own interaction)
    xattn     each slot cross-attends to the   (slots can read different parts
              code split into several tokens    of the code)
    film+cat  both

Fitted as an autodecoder -- the per-group codes are free variables learned with
the network -- so this measures what the mechanism can *express*, with the encoder
taken out of the question. Supervised on the attributes directly rather than
through the rasteriser: it is the conditioning path being compared, and a render
fit would add its own optimisation noise to a question that does not need it.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.layers import FourierFeatures, SelfAttentionBlock  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402

A_DIM = 11


class Cond(nn.Module):
    def __init__(self, kind, k, d, g, pe_dim, layers, heads, n_code_tok=4):
        super().__init__()
        self.kind, self.G, self.d = kind, g, d
        self.slot = nn.Embedding(g, d)
        nn.init.normal_(self.slot.weight, std=0.02)
        self.film = nn.Linear(k, 2 * d) if kind in ("film", "film+cat") else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.n_ct = n_code_tok
        if kind == "xattn":
            self.to_tok = nn.Linear(k, n_code_tok * d)
            self.xa = nn.MultiheadAttention(d, heads, batch_first=True)
            self.xn = nn.LayerNorm(d)
        extra = k if kind in ("concat", "film+cat") else 0
        self.inp = nn.Sequential(nn.Linear(d + pe_dim + extra, d), nn.SiLU(),
                                 nn.Linear(d, d), nn.SiLU())
        self.blocks = nn.ModuleList([SelfAttentionBlock(d, heads=heads, dropout=0.0)
                                     for _ in range(layers)])
        self.head = nn.Linear(d, A_DIM)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, code, pe):
        b = pe.shape[0]
        tok = self.slot.weight.view(1, self.G, -1).expand(b, -1, -1)
        if self.film is not None:
            gam, bet = self.film(code[:, None]).chunk(2, -1)
            tok = tok * (1.0 + gam) + bet
        feats = [tok, pe]
        if self.kind in ("concat", "film+cat"):
            feats.append(code[:, None].expand(b, self.G, code.shape[-1]))
        h = self.inp(torch.cat(feats, -1))
        if self.kind == "xattn":
            ct = self.to_tok(code).reshape(b, self.n_ct, self.d)
            h = h + self.xa(self.xn(h), ct, ct, need_weights=False)[0]
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--scenes", type=int, default=8,
                    help="scenes sharing ONE network. With a single scene the network "
                         "simply memorises position -> attribute and the code is "
                         "unnecessary, which is not the situation the model is in")
    ap.add_argument("--groups", type=int, default=512)
    ap.add_argument("--code", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--kinds", type=str, default="none,film,concat,xattn,film+cat")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = Namespace(**ck["args"])
    args.out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    tr, _ = make_datasets(args)
    cfg = build_config(args, tr.target_dim, tr.sh_dim)
    g, dev = int(cfg.group_size), "cuda"

    # SLOT-LAYOUT grouping. Compacting the mask and cutting the result into runs
    # of `group_size` -- what this did before -- is only the real grouping for
    # Morton chunks; under fixed anchors the runs straddle anchor boundaries and
    # the "within-group" variance this tool is built to measure is then the
    # variance of groups the model does not have.
    chunks = []
    for si in range(int(a.scenes)):
        it = tr[(a.scene + si * 37) % len(tr)]
        tt = it["target"].cuda().float()
        mk = it["mask"].cuda().float()
        n_all = tt.shape[0] // g
        tg = tt[: n_all * g].reshape(n_all, g, -1)
        cnt = (mk[: n_all * g].reshape(n_all, g) > 0.5).sum(1)
        tg = tg[cnt == g]
        if tg.shape[0] == 0:
            continue
        chunks.append(tg[: int(a.groups)])
    t = torch.cat(chunks, 0)
    ng = t.shape[0]
    xyz, A = t[..., 0:3], t[..., 3:14]
    sd = A.reshape(-1, A_DIM).std(0).clamp(min=1e-6)
    mu = A.reshape(-1, A_DIM).mean(0)
    Y = (A - mu) / sd

    pe_g = FourierFeatures(3, cfg.num_freqs_xyz, include_input=True).to(dev)
    with torch.no_grad():
        loc = xyz - xyz.mean(1, keepdim=True)
        loc = loc / loc.norm(dim=-1, keepdim=True).mean(1, keepdim=True).clamp(min=1e-6)
        PE = torch.cat([pe_g(xyz), pe_g(loc)], -1)

    # how much of the variance is inside a group -- the part a broadcast cannot see
    wg = float((Y - Y.mean(1, keepdim=True)).var()) / float(Y.var())
    print(f"{a.scenes} scenes, one shared network  {ng} groups x {g}  code {a.code}  "
          f"within-group variance share {wg:.1%}\n")
    print(f"{'conditioning':>10} {'params':>9} {'nrmse':>8} {'within-group nrmse':>20}")

    for kind in [s for s in a.kinds.split(",") if s]:
        torch.manual_seed(0)
        net = Cond(kind, a.code, a.dim, g, PE.shape[-1], a.layers, int(cfg.heads)).to(dev)
        C = torch.zeros(ng, a.code, device=dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": net.parameters(), "lr": 3e-4},
                                {"params": [C], "lr": 1e-2}])
        for _ in range(a.steps):
            opt.zero_grad(set_to_none=True)
            loss = (net(C, PE) - Y).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            P = net(C, PE)
            e = float((P - Y).pow(2).mean().sqrt())
            dP, dY = P - P.mean(1, keepdim=True), Y - Y.mean(1, keepdim=True)
            ew = float((dP - dY).pow(2).mean().sqrt() / dY.pow(2).mean().sqrt())
        npar = sum(p.numel() for p in net.parameters())
        tag = "  <- 현재" if kind == "film" else ""
        print(f"{kind:>10} {npar/1e6:8.2f}M {e:8.4f} {ew:20.4f}{tag}")

    print("\nnrmse 1.0 = 평균 예측과 동일. 'within-group' 은 그룹 평균을 뺀 뒤의 오차로,")
    print("브로드캐스트 변조가 원리적으로 다루기 어려운 성분입니다.")


if __name__ == "__main__":
    main()
