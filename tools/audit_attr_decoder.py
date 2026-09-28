"""Does every input the attribute decoder is supposed to use actually reach it?

Three bugs this session were of one kind: the code looked right, the loss was
logged, and no signal was flowing. The optimiser never received the decoder's
parameters; the render's detach silently zeroed the position nudge; the eval read
a tensor nothing trained. None of them were visible by reading the module.

So this measures rather than inspects. For each input the decoder is supposed to
condition on, perturb it and see whether the output moves, and check the gradient
comes back:

  latent      the per-group appearance code out of z_compact
  position    the point's own xyz -- the only per-point input, and therefore the
              only thing that can distinguish slots within a group
  attention   whether the self-attention blocks change their input at all
  slot basis  the learned per-slot embedding

The position check is split into between-group and within-group, because they are
not the same question. Positions arrive scene-normalised, and a group's radius is
~0.003 of the scene, so a Fourier encoding tuned to the scene may be nearly
constant across the 64 points of one group -- which is exactly where 52-78% of the
attribute variance lives. If the within-group column is tiny, the decoder is blind
to the variation it most needs to reproduce.
"""

from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from can3tok.model import build_model  # noqa: E402
from can3tok.train import build_config, make_datasets  # noqa: E402


def split_var(x, gsz):
    """(N, D) -> (between-group std, within-group std) treating rows as grouped."""
    n = (x.shape[0] // gsz) * gsz
    v = x[:n].reshape(-1, gsz, x.shape[-1]).float()
    return float(v.mean(1).std()), float((v - v.mean(1, keepdim=True)).std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--groups", type=int, default=256,
                    help="groups to audit; the training job holds the GPUs, and the "
                         "quantities here are per-group so a subset is representative")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = Namespace(**ck["args"])
    args.out_dir = os.path.dirname(os.path.abspath(a.ckpt))
    if int(getattr(args, "attr_decoder_layers", 0)) <= 0:
        args.attr_decoder_layers = 4
    tr, _ = make_datasets(args)
    cfg = build_config(args, tr.target_dim, tr.sh_dim)
    model = build_model(cfg)
    # Drop tensors whose shape no longer matches, so the audit still runs across
    # an architecture change instead of refusing to load.
    cur = model.state_dict()
    sd_ck = {k: v for k, v in ck["model"].items()
             if k in cur and cur[k].shape == v.shape}
    skipped = [k for k in ck["model"] if k not in sd_ck]
    if skipped:
        print(f"note: {len(skipped)} tensors skipped (shape changed): {skipped[:3]}")
    missing, _ = model.load_state_dict(sd_ck, strict=False)
    model.cuda().train()
    dec, g = model.attr_decoder, int(cfg.group_size)
    # "is it in the state_dict" is not "has it been trained" -- the tensors are
    # saved from step 0 whether or not the optimiser ever touched them. Compare
    # against the initialisation instead.
    step = int(ck.get("step", -1))
    at_init = bool(abs(float(dec.head_scale.bias.mean()) + 7.58) < 1e-6
                   and float(dec.head_nudge.weight.abs().max()) == 0.0)
    print(f"ckpt step {step}  attr_start {getattr(args,'attr_start','?')}  -> decoder "
          f"{'AT INIT (nothing to measure functionally)' if at_init else 'trained'}")
    print(f"num_freqs_xyz={cfg.num_freqs_xyz}  appearance channels={dec.c_app}  "
          f"attn layers={len(dec.blocks)}\n")

    item = tr[a.scene]
    x = item["input"].unsqueeze(0).cuda().float()
    m = item["mask"].unsqueeze(0).cuda().float()
    with torch.no_grad():
        z_raw, anc, _ = model.encode(x, m)
        zc, _ = model.compress(z_raw, anc)
        zh, ctx = model.decompressor(zc, shortcut_alpha=float(cfg.shortcut_alpha_eval))
        pred, _ = model.decoder(zh, ctx["cell_vec"], ctx["scale"], centroid=ctx["centroid"],
                                count=ctx["count"], group_vec=ctx.get("group_vec"),
                                appear=ctx.get("appearance"))
    ng_a = min(int(a.groups), int(ctx["appearance"].shape[1]))
    xyz = pred[..., : ng_a * g, 0:3].detach().contiguous()
    app = ctx["appearance"][:, :ng_a].detach().contiguous()
    sc = ctx["scale"][:, :ng_a].detach().contiguous()
    m = m[:, : ng_a * g].contiguous()
    v = m[0] > 0.5

    # ---- 1. does the position encoding resolve WITHIN a group? -----------
    with torch.no_grad():
        pe = dec.xyz_pe(xyz[0][v][: (int(v.sum()) // g) * g])
    b_pe, w_pe = split_var(pe, g)
    b_xyz, w_xyz = split_var(xyz[0][v], g)
    print(f"{'signal':<26} {'between-group':>14} {'within-group':>13} {'ratio':>8}")
    print(f"{'raw xyz':<26} {b_xyz:14.5f} {w_xyz:13.5f} {w_xyz/max(b_xyz,1e-9):8.4f}")
    print(f"{'xyz Fourier encoding':<26} {b_pe:14.5f} {w_pe:13.5f} "
          f"{w_pe/max(b_pe,1e-9):8.4f}")
    # What the same encoding would resolve if it were given group-local
    # coordinates instead of scene-normalised ones. Attribute variance is 52-78%
    # within-group, so an input whose within-group component is a fraction of its
    # between-group one is the wrong way round for the job.
    with torch.no_grad():
        q = xyz[0][v][: (int(v.sum()) // g) * g].reshape(-1, g, 3)
        loc = q - q.mean(1, keepdim=True)
        loc = loc / loc.norm(dim=-1).mean(1, keepdim=True).clamp(min=1e-9)[..., None]
        pel = dec.xyz_pe(loc.reshape(-1, 3))
    b_l, w_l = split_var(pel, g)
    print(f"{'  same, group-local':<26} {b_l:14.5f} {w_l:13.5f} "
          f"{w_l/max(b_l,1e-9):8.4f}")
    if getattr(dec, "loc_pe", None) is not None:
        with torch.no_grad():
            comb = torch.cat([dec.xyz_pe(q.reshape(-1, 3)), dec.loc_pe(loc.reshape(-1, 3))], -1)
        b_c, w_c = split_var(comb, g)
        print(f"{'  CONCAT (what it now gets)':<26} {b_c:14.5f} {w_c:13.5f} "
              f"{w_c/max(b_c,1e-9):8.4f}")

    # ---- 2. output sensitivity to each input -----------------------------
    with torch.no_grad():
        base = dec(xyz, app, sc, use_checkpoint=False)["pred"]
        # latent: shuffle the appearance code across groups. If the output does
        # not move, the decoder is not reading the latent at all.
        perm = torch.randperm(app.shape[1], device=app.device)
        d_app = (dec(xyz, app[:, perm], sc, use_checkpoint=False)["pred"] - base)
        # position: displace by a tenth of a group radius
        jit = torch.randn_like(xyz) * sc.mean() * 0.1
        d_xyz_out = (dec(xyz + jit, app, sc, use_checkpoint=False)["pred"] - base)
    # Per channel, not pooled: the 11 attribute channels sit at very different
    # levels (log_scale -7.58, quaternion w 1.0, opacity -2.13, colour 0), so a
    # pooled std is dominated by the offsets between channels and every real
    # response divides down to 0.0000.
    def rel(d):
        b = base[0][v][:, 3:].std(0).clamp(min=1e-9)
        return float((d[0][v][:, 3:].std(0) / b).mean())
    print(f"\n{'perturbation':<26} {'output change / output std':>28}")
    print(f"{'shuffle appearance code':<26} {rel(d_app):28.4f}")
    print(f"{'jitter xyz by 0.1 radius':<26} {rel(d_xyz_out):28.4f}")

    # ---- 3. does attention do anything? ----------------------------------
    with torch.no_grad():
        nb = min(64, ng_a)
        p0 = xyz[:, : nb * g].reshape(-1, g, 3)
        a0 = app[:, :nb].reshape(-1, 1, dec.c_app)
        tok = dec.slot_emb.weight.view(1, g, -1).expand(p0.shape[0], -1, -1)
        if dec.film is not None:
            gam, bet = dec.film(a0).chunk(2, -1)
            tok = tok * (1.0 + gam) + bet
        f0 = [tok, dec.xyz_pe(p0)]
        if getattr(dec, "loc_pe", None) is not None:
            l0 = p0 - p0.mean(1, keepdim=True)
            l0 = l0 / l0.norm(dim=-1, keepdim=True).mean(1, keepdim=True).clamp(min=1e-6)
            f0.append(dec.loc_pe(l0))
        h0 = dec.in_proj(torch.cat(f0, dim=-1))
        h = h0
        for blk in dec.blocks:
            h = blk(h)
        print(f"{'attention delta / input':<26} {float((h-h0).std()/h0.std().clamp(min=1e-9)):28.4f}")
        # measure whatever path this configuration actually uses
        if dec.cond == "xattn":
            ct = dec.to_code_tok(a0).reshape(p0.shape[0], dec.n_code_tok, -1)
            dh = dec.xattn(dec.xnorm(h0), ct, ct, need_weights=False)[0]
            val = float(dh.std() / h0.std().clamp(min=1e-9))
        else:
            val = float((tok - dec.slot_emb.weight).std() / dec.slot_emb.weight.std())
        print(f"{'code conditioning (' + dec.cond + ')':<26} {val:28.4f}")

    # ---- 4. gradient reaches every part ----------------------------------
    model.zero_grad(set_to_none=True)
    out = dec(xyz, app, sc, use_checkpoint=False)["pred"]
    # all 14 channels, not just the attributes: head_nudge lives on the xyz
    # output, and excluding it made the probe report "NO GRADIENT" for a head that
    # was wired correctly.
    out.pow(2).mean().backward()
    print(f"\n{'submodule':<26} {'grad norm':>14}")
    for name in ("slot_emb", "film", "to_code_tok", "xattn", "in_proj", "blocks", "head_color",
                 "head_opacity", "head_scale", "head_nudge"):
        mod = getattr(dec, name, None)
        if mod is None:
            continue
        gn = sum(float(p.grad.pow(2).sum()) for p in mod.parameters()
                 if p.grad is not None) ** 0.5
        flag = "   <-- NO GRADIENT" if gn < 1e-12 else ""
        print(f"{name:<26} {gn:14.4e}{flag}")


if __name__ == "__main__":
    main()
