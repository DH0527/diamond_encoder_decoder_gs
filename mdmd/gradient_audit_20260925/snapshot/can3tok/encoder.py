"""Patch encoder: exact identity pack + optional zero-initialised learned residual.

The identity pack is a non-parametric shortcut in the sense of DC-AE's Residual
Autoencoding (ICLR'25): ``z_raw`` literally *is* the Morton patch
(``group_size`` xyz triplets followed by ``group_size`` mask flags), so nothing
is lost before the bottleneck and the rest of the network only has to learn a
residual on top of a perfect signal.

The learned residual branch stays exactly zero until ``set_residual_enabled(True)``
is called. That matters: while no geometry loss is active, a trainable encoder
could trivially collapse ``z_raw`` into something easy to compress and the
``z_raw`` reconstruction loss would happily follow it down.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from .config import Can3TokConfig, patch_layout
from .layers import (MLP, CrossAttentionBlock, ResBlock2d, SelfAttentionBlock,
                     residual_head)

# Measured over 2.17M points across 11 scenes, in the order the target packs them:
# log_scale 3 | quaternion 4 | logit opacity 1 | SH DC 3.
ATTR_MEAN = (-8.2933, -8.5553, -8.5972, 0.9118, -0.0164, -0.0319, -0.0113,
             0.7002, 0.1821, 0.1609, -0.0383)
ATTR_STD = (1.5558, 1.4651, 1.6287, 0.1302, 0.2092, 0.2536, 0.2053,
            3.7138, 1.3148, 1.2916, 1.2338)


class GroupAttributeEncoder(nn.Module):
    """Compress a group's per-point attributes into one number per slot.

    The pack has always been xyz and a mask, so scale, rotation, opacity and
    colour never entered ``z_raw`` and therefore never reached ``z_compact``.
    Measured consequence: predicting those attributes from geometry alone gives a
    held-out R^2 of -0.99 (opacity), -0.25 (rotation), -0.23 (colour) -- worse
    than emitting the dataset mean. Two scenes differing only in colour receive
    the *identical* latent.

    This is the missing path. It takes the ``g x A_in`` attribute block and emits
    ``g`` numbers, which occupy the pack's aux slot where the mask used to sit.

    Why the mask can be given up: prefix packing makes groups all-or-nothing --
    measured, exactly **one** group per scene is partially filled -- so the mask
    carries 0.40 bits per slot out of fp16's 16. Emptiness is already recoverable
    from the log-extent anchor, and the presence head separately receives the
    count prior.

    Why the compression happens here and not per point: a per-point MLP squeezing
    11 numbers into 1 has to decide what to keep before seeing anything else in
    the group. One self-attention layer over the 64 slots lets it drop what a
    neighbour already carries, which is exactly the redundancy worth removing.

    Why a per-slot embedding is not optional: without it the module is a per-slot
    MLP, a self-attention layer and a per-slot linear, all of which are
    permutation-equivariant, so *every slot computes the same function*. Slot 3
    then cannot specialise to carry mean opacity while slot 7 carries mean red --
    slots can only differ through their own attribute values, which is a very poor
    code. Measured in isolation, training this module to push a group's mean
    attributes through its own 64-number output gives held-out R^2 0.158 without
    the embedding and 0.984 with it, uniformly across all 11 channels. Train R^2
    is 1.000 either way, i.e. the 64 numbers were always wide enough; the failure
    was that the only reachable code memorised groups instead of encoding them.
    The attribute decoder already carries a slot embedding for the same reason.
    """

    def __init__(self, group_size: int, in_dim: int, hidden: int, heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.G = int(group_size)
        self.in_dim = int(in_dim)
        # Per-CHANNEL standardisation against fixed dataset statistics, not
        # LayerNorm. LayerNorm normalises each point across its own 11 channels,
        # and those channels are not commensurable -- measured over 2.17M points:
        #
        #     log_scale  mean -8.29 -8.56 -8.60   std 1.56 1.47 1.63
        #     quaternion mean  0.91 -0.02 -0.03   std 0.13 0.21 0.25 0.21
        #     opacity    mean  0.70                std 3.71
        #     colour DC  mean  0.18  0.16 -0.04   std 1.31 1.29 1.23
        #
        # A per-point mean over that is dominated by log_scale's -8.5, and
        # subtracting it takes the point's *own* colour and opacity offset with
        # it -- which is exactly the signal this module exists to carry. Measured
        # end to end, held-out R^2 of recovering a group's mean attributes fell
        # 0.998 -> 0.747 across this one operation.
        #
        # Buffers rather than constants so a different dataset can be restandardised
        # without touching the code, and rather than running statistics so train
        # and eval cannot diverge.
        self.register_buffer("ch_mean", torch.tensor(ATTR_MEAN[:self.in_dim]))
        self.register_buffer("ch_std", torch.tensor(ATTR_STD[:self.in_dim]).clamp(min=1e-3))
        self.embed = MLP([self.in_dim, hidden, hidden], last_act=True, dropout=dropout)
        # Breaks permutation equivariance so a slot can own a share of the code.
        # Small init: it is an offset on an already-informative embedding, not a
        # signal of its own.
        self.slot_emb = nn.Parameter(torch.randn(1, self.G, hidden) * 0.02)
        self.mix = SelfAttentionBlock(hidden, heads=heads, dropout=dropout)
        self.out = nn.Linear(hidden, 1)
        # Small but NOT zero. A zero last layer makes d(loss)/d(hidden) exactly
        # zero, so `embed` and `mix` get no gradient on the first step and the
        # layer itself starts from a point where its own gradient is the only
        # thing moving. Measured after 16000 steps of a run initialised this way,
        # `out.weight` was still |w| 3.9e-4 while `embed` reached 1.5e-1 and `mix`
        # 6e-2 -- two orders of magnitude behind, i.e. a trained front end feeding
        # a nearly closed valve. The same mistake was found and fixed twice before
        # in this code base (fold_head / shape_xyz, and the attribute decoder's
        # cross-attention out_proj); this is the third site.
        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)

    def forward(self, attr_g: torch.Tensor, mask_g: torch.Tensor) -> torch.Tensor:
        """(B, groups, G, A_in) + (B, groups, G) -> (B, groups, G)."""
        b, ng, g, _ = attr_g.shape
        x = (attr_g.reshape(b * ng, g, self.in_dim) - self.ch_mean) / self.ch_std
        h = self.embed(x) + self.slot_emb[:, :g]
        h = self.mix(h, key_padding_mask=(mask_g.reshape(b * ng, g) <= 0.5))
        return self.out(h).reshape(b, ng, g) * mask_g


class GroupPointPooler(nn.Module):
    """Reduce a group's variable point set to the pack's fixed patch width.

    The identity pack this replaces is only definable when the encoder reads
    exactly as many points as the decoder emits: it literally copies 64 xyz
    triplets into z_raw. Once the encoder is allowed to read more points than the
    decoder will produce -- which is the whole point, since the loader otherwise
    throws away, before the bottleneck, information measured at 2.50 dB on average
    and 5.32 dB at worst -- there is no identity to copy and the reduction has to
    be learned.

    Learned queries cross-attend to the group's points, which is the same
    construction 3DShape2VecSet, Can3Tok and COD-VAE use to turn a point set into
    a fixed latent set. Crucially the network, not a hand-written sampling rule,
    now decides what survives; measured on over-budget snapshots the old rule's
    loss did not even track its keep ratio (92% kept lost 5.17 dB, 56% kept lost
    2.92 dB), so *which* points are dropped dominates *how many*.

    PROGRESSIVE COMPRESSION (``pool_blocks`` > 1).

    One cross-attention is a single collapse: 2048 points -> 16 queries in one
    step, and the queries never see each other, so nothing in the reduction can
    notice that two queries have latched onto the same structure. COD-VAE
    (arXiv:2503.08737) reaches 64 latents where VecSet needed 1024 by staging the
    same reduction instead -- per block: cross-attend points into the summary,
    let the summary self-attend, then push the summary back into the point
    features so the next block reads globally-informed points.

    ``pool_blocks=1`` reproduces the previous single-collapse path exactly (no
    self-attention, no feedback, same parameter shapes), so the old behaviour is
    the default and the staged path is opt-in.
    """

    def __init__(self, in_dim: int, patch_dim: int, d: int = 128, n_q: int = 8,
                 heads: int = 4, dropout: float = 0.0, blocks: int = 1,
                 feedback: bool = True):
        super().__init__()
        self.n_q = int(n_q)
        self.blocks = max(int(blocks), 1)
        self.embed = MLP([int(in_dim), d, d], last_act=True, dropout=dropout)
        self.q = nn.Parameter(torch.randn(1, self.n_q, d) * 0.02)
        self.attn = nn.ModuleList(
            [CrossAttentionBlock(d, heads=heads, dropout=dropout) for _ in range(self.blocks)])
        # Queries coordinate between reductions. Absent in the single-collapse
        # path, which is why two queries could redundantly summarise the same
        # points with nothing to separate them.
        self.qsa = nn.ModuleList(
            [SelfAttentionBlock(d, heads=heads, dropout=dropout)
             for _ in range(self.blocks - 1)]) if self.blocks > 1 else nn.ModuleList()
        # Points read the running summary before the next reduction. Cheap: the
        # key/value side is n_q=16, not the 2048-point side.
        self.back = nn.ModuleList(
            [CrossAttentionBlock(d, heads=heads, dropout=dropout)
             for _ in range(self.blocks - 1)]) if (self.blocks > 1 and feedback) else nn.ModuleList()
        self.out = nn.Linear(self.n_q * d, int(patch_dim))
        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, ng, P, in_dim) + (B, ng, P) -> (B, ng, patch_dim)."""
        b, ng, p, f = feats.shape
        h = self.embed(feats.reshape(b * ng, p, f))
        pad = (mask.reshape(b * ng, p) <= 0.5)
        z = self.q.to(h.dtype).expand(b * ng, -1, -1)
        for i in range(self.blocks):
            z = self.attn[i](z, h, key_padding_mask=pad)
            if i < len(self.qsa):
                z = self.qsa[i](z)
            if i < len(self.back):
                h = self.back[i](h, z)
        return self.out(z.reshape(b * ng, -1)).reshape(b, ng, -1)


class PatchPackEncoder(nn.Module):
    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay = patch_layout(cfg)
        self.group_size = lay["group_size"]
        self.tokens_per_group = lay["tokens_per_group"]
        self.num_groups = lay["num_groups"]
        self.patch_dim = lay["patch_dim"]
        self.token_dim = lay["token_dim"]
        self.raw_cells = lay["raw_cells"]
        self.raw_cells_used = lay["raw_cells_used"]

        self.pack = nn.Linear(self.patch_dim, self.token_dim)
        self._init_identity()
        # Frozen identity is the right choice only while ``w_z_raw`` supervises
        # z_raw against the pack: then the pack IS the target and learning it
        # would let both sides drift together. Once the objective is the render,
        # freezing it leaves the encoder's geometry path with zero trainable
        # parameters -- measured, ||dp|| on the encoder group is exactly 0 for
        # every logged step of every run so far, i.e. the "encoder" was a Morton
        # sort and a Hungarian assignment. Identity remains the initialisation
        # either way, so nothing about the starting point changes.
        if not bool(getattr(cfg, "pack_trainable", False)):
            for p in self.pack.parameters():
                p.requires_grad_(False)

        self.residual_enabled = False
        if cfg.encoder_residual:
            hid = int(cfg.encoder_residual_hidden)
            # patch + [centroid(3), extent(3), count(1)]
            self.residual = residual_head([self.patch_dim + 7, hid, hid, self.token_dim])
            nn.init.zeros_(self.residual.net[-1].weight)
            nn.init.zeros_(self.residual.net[-1].bias)
            # Small but NOT zero, and this is not the same call as a zero-init
            # output layer. The branch is `tanh(gate) * residual(feat)`, a product
            # of two independently zero-initialised factors, which is an exact
            # zero-gradient saddle: d/d(residual weights) = tanh(0) * (...) = 0,
            # and d/d(gate) = sech^2(0) * <upstream, residual(feat)> = 0 because
            # residual(feat) is itself 0. So NEITHER factor can move, at any
            # learning rate, forever. Measured: 4000 steps after
            # encoder_residual_start enabled this branch, residual_gate and the
            # last layer's weight and bias were all still exactly 0.0 while
            # attr_encoder moved normally (1.0e-3 per 4000 steps).
            #
            # A nonzero gate breaks the product while preserving the property the
            # zero init was for: the branch output is still exactly zero at init,
            # because residual's last layer is still zero. Only the gradient path
            # changes.
            self.residual_gate = nn.Parameter(torch.full((1,), 1e-2))
        else:
            self.residual = None
            self.residual_gate = None

        self.map_blocks = nn.Sequential(
            *[ResBlock2d(int(cfg.latent_channels)) for _ in range(int(cfg.map_blocks))]
        )

        # Learned pooling, used only when the encoder reads more points than the
        # decoder emits. group_in == group_size keeps the exact identity pack.
        mip = int(getattr(cfg, "max_input_points", 0)) or int(cfg.max_points)
        self.group_in = max(1, mip // self.num_groups)
        self.pooler = (
            GroupPointPooler(3 + int(getattr(cfg, "attr_pack_dim", 0)), self.patch_dim,
                             d=int(getattr(cfg, "pool_dim", 128)),
                             n_q=int(getattr(cfg, "pool_queries", 16)),
                             heads=4, dropout=cfg.dropout,
                             blocks=int(getattr(cfg, "pool_blocks", 1)),
                             feedback=bool(getattr(cfg, "pool_feedback", True)))
            if self.group_in > self.group_size else None
        )
        # Attention memory in the pooler is O(pool_chunk * group_in), so a fixed
        # chunk makes the cost scale with the per-cell cap. Measured on this
        # dataset, cap 144 drops 14-23% of the points of the densest snapshots
        # (max cell count 768) and cap 512 drops 0.0-0.2%, so the cap has to be
        # raisable -- and at a fixed chunk of 512 that would be a 3.6x memory
        # jump for no reason. Keeping the PRODUCT constant instead makes the cap a
        # free parameter. Explicit --pool_chunk still overrides.
        _pc = int(getattr(cfg, "pool_chunk", -1))
        self.pool_chunk = _pc if _pc > 0 else max(16, 73728 // max(self.group_in, 1))
        _a = int(getattr(cfg, "attr_pack_dim", 0))
        self.register_buffer("attr_mean_p", torch.tensor(ATTR_MEAN[:_a]).view(1, 1, 1, -1)
                             if _a > 0 else torch.zeros(1))
        self.register_buffer("attr_std_p", torch.tensor(ATTR_STD[:_a]).clamp(min=1e-3).view(1, 1, 1, -1)
                             if _a > 0 else torch.ones(1))

        a_in = int(getattr(cfg, "attr_pack_dim", 0))
        self.attr_encoder = (
            GroupAttributeEncoder(self.group_size, a_in, int(cfg.attr_pack_hidden),
                                  heads=4, dropout=cfg.dropout)
            if a_in > 0 else None
        )
        # `_build_patch_pooled` builds the patch from the points directly and never
        # calls attr_encoder -- the pooler already reads the attribute channels, so
        # the appearance path is intact, but this module is then 59,073 parameters
        # sitting in the graph receiving no gradient. It stays built so a checkpoint
        # written with the identity pack still loads, and is frozen so the
        # silent-freeze guard does not have to keep reporting it.
        if self.pooler is not None and self.attr_encoder is not None:
            for p in self.attr_encoder.parameters():
                p.requires_grad_(False)

    def _init_identity(self) -> None:
        nn.init.zeros_(self.pack.weight)
        nn.init.zeros_(self.pack.bias)
        n = min(self.patch_dim, self.token_dim)
        with torch.no_grad():
            self.pack.weight[:n, :n] = torch.eye(n)

    def set_residual_enabled(self, flag: bool) -> None:
        """Only flips whether the branch is *used*.

        ``requires_grad`` is deliberately left alone: toggling it mid-training
        would invalidate DDP's gradient buckets. Because the last layer and the
        gate are zero-initialised the branch contributes exactly nothing until
        its learning rate becomes non-zero, so it is safe to enable from step 0
        and keep the autograd graph static.
        """
        self.residual_enabled = bool(flag) and self.residual is not None

    # ------------------------------------------------------------------
    def build_patch(self, x: torch.Tensor, mask: torch.Tensor,
                    group_anchor: torch.Tensor | None = None):
        if self.pooler is not None:
            return self._build_patch_pooled(x, mask, group_anchor=group_anchor)
        b, s, _ = x.shape
        g = self.group_size
        total = self.num_groups * g
        if s < total:
            x = F.pad(x, (0, 0, 0, total - s))
            mask = F.pad(mask, (0, total - s))
        elif s > total:
            x = x[:, :total]
            mask = mask[:, :total]
        m = mask.to(x.dtype)
        xyz = x[..., 0:3] * m[..., None]
        xyz_g = xyz.view(b, self.num_groups, g, 3)
        m_g = m.view(b, self.num_groups, g)
        if bool(self.cfg.residual_pack):
            # Store within-group offsets so z_raw L1 cannot be solved by a
            # centroid broadcast. Absolute pose lives only in z_compact anchors.
            cnt = m_g.sum(dim=-1, keepdim=True).clamp(min=1.0)
            sample_cen = (xyz_g * m_g.unsqueeze(-1)).sum(dim=2, keepdim=True) / cnt.unsqueeze(-1)
            cen = (group_anchor[:, :self.num_groups].unsqueeze(2)
                   if group_anchor is not None else sample_cen)
            packed = (xyz_g - cen) * m_g.unsqueeze(-1)
        else:
            packed = xyz_g
        # aux block: the mask historically, an appearance code once attributes are
        # in play. Same width either way, so z_raw and the whole downstream stack
        # keep their shapes.
        if self.attr_encoder is not None:
            a_in = int(self.cfg.attr_pack_dim)
            attr_g = x[..., 3 : 3 + a_in].view(b, self.num_groups, g, a_in) * m_g[..., None]
            aux = self.attr_encoder(attr_g, m_g)
        else:
            aux = m_g
        patch = torch.cat(
            [packed.reshape(b, self.num_groups, g * 3), aux], dim=-1
        )
        return patch, xyz_g, m_g

    def _build_patch_pooled(self, x: torch.Tensor, mask: torch.Tensor,
                            group_anchor: torch.Tensor | None = None):
        """Variable-capacity path: pool group_in points per group into the patch."""
        b, s, _ = x.shape
        p = self.group_in
        total = self.num_groups * p
        if s < total:
            x = F.pad(x, (0, 0, 0, total - s)); mask = F.pad(mask, (0, total - s))
        elif s > total:
            x = x[:, :total]; mask = mask[:, :total]
        m = mask.to(x.dtype)
        xyz = x[..., 0:3] * m[..., None]
        xyz_g = xyz.view(b, self.num_groups, p, 3)
        m_g = m.view(b, self.num_groups, p)
        cnt = m_g.sum(-1, keepdim=True).clamp(min=1.0)
        sample_cen = (xyz_g * m_g.unsqueeze(-1)).sum(2, keepdim=True) / cnt.unsqueeze(-1)
        cen = (group_anchor[:, :self.num_groups].unsqueeze(2)
               if group_anchor is not None else sample_cen)
        loc = (xyz_g - cen) * m_g.unsqueeze(-1)
        if bool(getattr(self.cfg, "normalize_pooler_xyz", False)):
            # Same scalar extent the compact channel stores: max-axis std, then
            # the encode_scale/decode_scale floor. Do this on metric xyz, before
            # the pooler, never on the learned 1024-d patch.
            from .compressor import decode_scale, encode_scale
            var = ((xyz_g - cen) ** 2 * m_g.unsqueeze(-1)).sum(2) / cnt.clamp(min=1.0)
            extent = var.clamp(min=0).sqrt()
            scale = decode_scale(encode_scale(extent)).clamp(min=1e-5)
            loc = loc / scale.view(b, self.num_groups, 1, 1)
            loc = loc * m_g.unsqueeze(-1)
        a_in = int(getattr(self.cfg, "attr_pack_dim", 0))
        feats = [loc]
        if a_in > 0:
            attr = x[..., 3:3 + a_in].view(b, self.num_groups, p, a_in) * m_g[..., None]
            feats.append((attr - self.attr_mean_p) / self.attr_std_p)
        feats = torch.cat(feats, dim=-1)
        # Chunk over groups: the full (B*4096, 144, d) attention does not fit.
        # Every chunk's activations otherwise stay in the graph at once, so the
        # chunking bounds compute but not memory. A staged pooler holds several
        # times more on the wide point side, so checkpoint the chunks whenever
        # more than one reduction runs.
        use_ckpt = bool(self.training) and int(getattr(self.cfg, "pool_blocks", 1)) > 1
        outs = []
        for st in range(0, self.num_groups, self.pool_chunk):
            en = min(st + self.pool_chunk, self.num_groups)
            fc, mc = feats[:, st:en], m_g[:, st:en]
            outs.append(checkpoint(self.pooler, fc, mc, use_reentrant=False)
                        if use_ckpt else self.pooler(fc, mc))
        patch = torch.cat(outs, dim=1)
        # Anchors stay analytic and exact, as before.
        return patch, xyz_g, m_g

    @staticmethod
    def patch_anchors(xyz_g: torch.Tensor, mask_g: torch.Tensor,
                      anchor_centroid: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        """Analytic per-group statistics used as non-parametric latent anchors."""
        cnt = mask_g.sum(dim=-1, keepdim=True)
        denom = cnt.clamp(min=1.0)
        sample_centroid = (xyz_g * mask_g[..., None]).sum(dim=2) / denom
        var = ((xyz_g - sample_centroid[:, :, None, :]) ** 2 * mask_g[..., None]).sum(dim=2) / denom
        centroid = (anchor_centroid[:, :xyz_g.shape[1]]
                    if anchor_centroid is not None else sample_centroid)
        extent = var.clamp(min=0).sqrt()
        return {
            "centroid": centroid,
            "extent": extent,
            "count": cnt.squeeze(-1),
            "valid": (cnt.squeeze(-1) > 0).to(xyz_g.dtype),
        }

    def tokens_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        c = int(self.cfg.latent_channels)
        cells = tokens.reshape(b, self.raw_cells_used, c)
        if self.raw_cells > self.raw_cells_used:
            pad = cells.new_zeros(b, self.raw_cells - self.raw_cells_used, c)
            cells = torch.cat([cells, pad], dim=1)
        h, w = int(self.cfg.latent_hw[0]), int(self.cfg.latent_hw[1])
        return cells.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                group_anchor: torch.Tensor | None = None):
        patch, xyz_g, mask_g = self.build_patch(x, mask, group_anchor=group_anchor)
        anchors = self.patch_anchors(xyz_g, mask_g, anchor_centroid=group_anchor)
        z_tok = self.pack(patch)
        if self.residual_enabled and self.residual is not None:
            feat = torch.cat(
                [patch, anchors["centroid"], anchors["extent"], anchors["count"].unsqueeze(-1) / self.group_size],
                dim=-1,
            )
            z_tok = z_tok + torch.tanh(self.residual_gate) * self.residual(feat)
        z_map = self.tokens_to_map(z_tok)
        if len(self.map_blocks) > 0:
            z_map = self.map_blocks(z_map)
        return z_map, anchors, patch
