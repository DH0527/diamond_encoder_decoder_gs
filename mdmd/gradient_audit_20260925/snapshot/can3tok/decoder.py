"""Codec decoder: identity unpack + neighbourhood-aware per-point refinement.

The coarse prediction comes from inverting the identity pack, so at
initialisation the decoder already reproduces whatever ``z_raw_hat`` holds. The
refinement stage differs from the previous implementation in two important ways:
(1) cross-attention memory also sees the 3x3 window of compact cells (seam
removal); (2) each refine layer runs point self-attention *before* cross
attention so the 32 slots in a group can coordinate local geometry, instead of
attending to memory independently.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import Can3TokConfig, channel_budget, patch_layout
from .layers import (
    MLP,
    CrossAttentionBlock,
    FourierFeatures,
    SelfAttentionBlock,
    neighbor_gather,
    residual_head,
)
from .morton import grid_zorder_permutation


class CodecDecoder(nn.Module):
    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay = patch_layout(cfg)
        self.G = lay["group_size"]
        self.T = lay["tokens_per_group"]
        self.C = int(cfg.latent_channels)
        self.M = lay["merge"]
        self.num_groups = lay["num_groups"]
        self.cells = lay["compact_cells"]
        self.patch_dim = lay["patch_dim"]
        self.token_dim = lay["token_dim"]
        self.patch_chunk = max(int(cfg.patch_chunk), 1)
        d = int(cfg.model_dim)

        h, w = int(cfg.compact_latent_hw[0]), int(cfg.compact_latent_hw[1])
        perm = grid_zorder_permutation(h, w) if cfg.latent_zorder else torch.arange(h * w)
        self.register_buffer("cell_to_grid", perm.long(), persistent=False)
        self.grid_h, self.grid_w = h, w

        self.unpack = nn.Linear(self.token_dim, self.patch_dim)
        self._init_identity()

        self.mem_proj = nn.Linear(self.C, d)
        self.offset_pe = FourierFeatures(1, cfg.num_freqs_slot, include_input=True)
        self.slot_query = MLP([self.offset_pe.out_dim, d, d], last_act=True)
        self.slot_embedding = nn.Embedding(self.G, d) if cfg.use_slot_embedding else None
        self.xyz_pe = FourierFeatures(3, cfg.num_freqs_xyz, include_input=True)
        # Coarse xyz -> query (gen already had this; codec previously used slot index only).
        self.q_xyz = MLP([self.xyz_pe.out_dim, d, d], last_act=True, dropout=cfg.dropout)
        n_layers = max(1, int(cfg.decoder_layers))
        self.self_blocks = nn.ModuleList(
            [SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout) for _ in range(n_layers)]
        )
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout) for _ in range(n_layers)]
        )
        self.xyz_residual = residual_head([d + self.xyz_pe.out_dim, d, d, 3])
        self.presence_head = nn.Linear(d, 1)

        self.attr_mlp = None
        self.head_scale = self.head_rot = self.head_opacity = self.head_color = self.head_sh = None
        self.c_app = int(channel_budget(cfg).get("appearance", 0))
        if cfg.target_dim > 3:
            self.attr_mlp = MLP([d + self.xyz_pe.out_dim + self.c_app, d, d],
                                last_act=True, dropout=cfg.dropout)
            self.head_scale = nn.Linear(d, 3)
            self.head_rot = nn.Linear(d, 4)
            self.head_opacity = nn.Linear(d, 1)
            if cfg.target_dim > 11:
                self.head_color = nn.Linear(d, 3)
            if cfg.sh_dim > 0:
                self.head_sh = nn.Linear(d, int(cfg.sh_dim))
        self._init_heads()

    def _init_identity(self) -> None:
        nn.init.zeros_(self.unpack.weight)
        nn.init.zeros_(self.unpack.bias)
        n = min(self.patch_dim, self.token_dim)
        with torch.no_grad():
            self.unpack.weight[:n, :n] = torch.eye(n)

    def _init_heads(self) -> None:
        if self.slot_embedding is not None:
            nn.init.normal_(self.slot_embedding.weight, std=0.02)
        nn.init.zeros_(self.xyz_residual.net[-1].weight)
        nn.init.zeros_(self.xyz_residual.net[-1].bias)
        nn.init.zeros_(self.presence_head.weight)
        nn.init.zeros_(self.presence_head.bias)
        if self.head_scale is not None:
            nn.init.normal_(self.head_scale.weight, std=1e-3)
            nn.init.constant_(self.head_scale.bias, -5.5)
        if self.head_rot is not None:
            nn.init.normal_(self.head_rot.weight, std=1e-3)
            nn.init.zeros_(self.head_rot.bias)
            with torch.no_grad():
                self.head_rot.bias[0] = 1.0
        for head, bias in ((self.head_opacity, -2.0), (self.head_color, 0.0), (self.head_sh, 0.0)):
            if head is not None:
                nn.init.normal_(head.weight, std=1e-3)
                nn.init.constant_(head.bias, bias)

    # ------------------------------------------------------------------
    def set_refine_trainable(self, flag: bool) -> None:
        mods = [
            self.mem_proj,
            self.slot_query,
            self.q_xyz,
            self.self_blocks,
            self.blocks,
            self.xyz_residual,
            self.presence_head,
        ]
        if self.slot_embedding is not None:
            mods.append(self.slot_embedding)
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(bool(flag))

    def _tokens(self, z_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z_map.shape
        cells = z_map.reshape(b, c, h * w).transpose(1, 2)
        return cells[:, : self.num_groups * self.T].reshape(b, self.num_groups, self.T, c)

    def _cell_neighbors(self, cell_vec: torch.Tensor) -> torch.Tensor:
        """(B, cells, d) in cell order -> (B, cells, 9, d) spatial neighbours."""
        b, n, d = cell_vec.shape
        grid = cell_vec.new_zeros(b, self.grid_h * self.grid_w, d)
        grid[:, self.cell_to_grid] = cell_vec
        nb = neighbor_gather(grid, self.grid_h, self.grid_w, radius=1)
        return nb[:, self.cell_to_grid]

    def _group_neighbors(self, group_vec: torch.Tensor) -> torch.Tensor:
        """(B, cells, M, d) -> (B, cells, 9*M, d) spatial neighbour group tokens."""
        b, cells, m, d = group_vec.shape
        flat = group_vec.reshape(b, cells, m * d)
        grid = flat.new_zeros(b, self.grid_h * self.grid_w, m * d)
        grid[:, self.cell_to_grid] = flat
        nb = neighbor_gather(grid, self.grid_h, self.grid_w, radius=1)[:, self.cell_to_grid]
        return nb.reshape(b, cells, 9 * m, d)

    def _bounded_log_scale(self, raw):
        """Keep the emitted log-scale inside a sane band.

        Unbounded, `exp(clamp(-15, 5))` lets one Gaussian grow 148x. The render
        loss actively pushes that way -- enlarging a Gaussian is the cheapest way
        to cover a hole left by a missing point -- and the attribute oracle hit a
        CUDA illegal memory access at rank 128 doing exactly this, because a few
        enormous splats blow up the rasteriser's tile allocation.

        Same treatment every additive path in the geometry decoder already gets:
        a tanh cap around a sensible centre rather than a hard clamp, so the
        gradient never dies at the boundary. The centre is the head's own bias
        (initialised to -5.5, the data's typical log scale) and the band is
        +-`scale_log_cap` in log space, i.e. e^3 ~ 20x either way by default.
        """
        cap = float(getattr(self.cfg, "attr_scale_log_cap", 3.0))
        if cap <= 0:
            return raw
        return self.head_scale.bias.view(1, 1, -1) + torch.tanh(
            (raw - self.head_scale.bias.view(1, 1, -1)) / cap
        ) * cap

    def _attributes(self, h, xyz, attr_xyz, g0: int, g1: int, appear=None):
        b, chunk, gsz, _ = xyz.shape
        if self.attr_mlp is None:
            rest = xyz.new_zeros(b, chunk, gsz, int(self.cfg.target_dim) - 3)
            return torch.cat([xyz, rest], dim=-1)
        # Scheduled sampling on the conditioning position. p=1 is teacher forcing
        # (heads see the true xyz and learn A(x) cleanly), p=0 is the inference
        # condition (heads see only what the decoder produced). Mixing per point
        # rather than per batch means every step contains both regimes, so the
        # heads never get a clean run of one and overfit to it.
        p = float(getattr(self.cfg, "attr_teacher_prob", 1.0)) if self.training else 0.0
        if attr_xyz is not None and p > 0.0:
            xyz_cond = attr_xyz[:, g0 * gsz : g1 * gsz].view(b, chunk, gsz, 3)
            if p < 1.0:
                keep = torch.rand(b, chunk, gsz, 1, device=xyz.device) < p
                xyz_cond = torch.where(keep, xyz_cond, xyz)
        else:
            xyz_cond = xyz
        feat = [h.reshape(b * chunk, gsz, -1), self.xyz_pe(xyz_cond.reshape(b * chunk, gsz, 3))]
        if self.c_app > 0 and appear is not None:
            # One code per group, broadcast to its slots: appearance is a property
            # of the region, and the per-point variation it has to explain comes
            # from the position that is already in `feat`.
            feat.append(appear[:, g0:g1].reshape(b * chunk, 1, self.c_app).expand(-1, gsz, -1))
        elif self.c_app > 0:
            feat.append(h.new_zeros(b * chunk, gsz, self.c_app))
        attr = self.attr_mlp(torch.cat(feat, dim=-1))
        parts = [
            xyz,
            self._bounded_log_scale(self.head_scale(attr)).view(b, chunk, gsz, 3),
            F.normalize(self.head_rot(attr), dim=-1).view(b, chunk, gsz, 4),
            self.head_opacity(attr).view(b, chunk, gsz, 1),
        ]
        if self.head_color is not None:
            parts.append(self.head_color(attr).view(b, chunk, gsz, 3))
        if self.head_sh is not None:
            parts.append(self.head_sh(attr).view(b, chunk, gsz, int(self.cfg.sh_dim)))
        return torch.cat(parts, dim=-1)

    def _decode_range(self, tokens, nb, scale, centroid, count, g0: int, g1: int, attr_xyz=None,
                      appear=None):
        b = tokens.shape[0]
        chunk = g1 - g0
        gsz = self.G
        d = int(self.cfg.model_dim)
        tok = tokens[:, g0:g1]
        patch = self.unpack(tok.reshape(b, chunk, self.token_dim))
        coarse = patch[..., : gsz * 3].reshape(b * chunk, gsz, 3)
        # Residual-pack: tokens carry offsets; restore absolute pose from anchors.
        if bool(self.cfg.residual_pack) and centroid is not None:
            cen = centroid[:, g0:g1].reshape(b * chunk, 1, 3)
            coarse = coarse + cen
        mask_value = patch[..., gsz * 3 : gsz * 4].reshape(b, chunk, gsz)

        mem_tok = self.mem_proj(tok)
        memory = mem_tok.reshape(b * chunk, self.T, d)
        if nb is not None:
            cell_idx = torch.arange(g0, g1, device=tokens.device) // self.M
            ctx = nb[:, cell_idx].reshape(b * chunk, -1, d)
            memory = torch.cat([memory, ctx], dim=1)

        offset = torch.arange(gsz, device=tokens.device, dtype=tokens.dtype)
        offset = (offset / max(float(gsz - 1), 1.0)).view(1, gsz, 1).expand(b * chunk, gsz, 1)
        q = self.slot_query(self.offset_pe(offset))
        if self.slot_embedding is not None:
            q = q + self.slot_embedding.weight.view(1, gsz, d)
        q = q + mem_tok.mean(dim=2).reshape(b * chunk, 1, d)
        q = q + self.q_xyz(self.xyz_pe(coarse))
        # Mask empty slots so self-attn does not mix pad into real points.
        slot_pad = mask_value.reshape(b * chunk, gsz) <= 0.5
        h = q
        for sa, ca in zip(self.self_blocks, self.blocks):
            h = sa(h, key_padding_mask=slot_pad)
            h = ca(h, memory)

        residual = torch.tanh(self.xyz_residual(torch.cat([h, self.xyz_pe(coarse)], dim=-1)))
        # refinement budget is a fraction of the group's own extent
        s = scale[:, g0:g1].reshape(b * chunk, 1, 1)
        refine = float(getattr(self.cfg, "decoder_refine_alpha", 1.0))
        xyz = (coarse + residual * s * float(self.cfg.residual_scale) * refine).view(b, chunk, gsz, 3)
        # Match gen: soft count prior + unpack mask + learned residual. Without the
        # count prior, codec presence collapses to "all present" on padded frames.
        presence = (mask_value - 0.5) * 8.0 + self.presence_head(h).view(b, chunk, gsz)
        if count is not None:
            slot = torch.arange(gsz, device=tokens.device, dtype=tokens.dtype).view(1, 1, gsz)
            occ = (count[:, g0:g1].unsqueeze(-1) - slot).clamp(0.0, 1.0)
            presence = presence + (occ - 0.5) * 4.0
        pred = self._attributes(h.view(b, chunk, gsz, d), xyz, attr_xyz, g0, g1, appear)
        return pred, presence

    def forward(
        self,
        z_map: torch.Tensor,
        cell_vec: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        centroid: Optional[torch.Tensor] = None,
        count: Optional[torch.Tensor] = None,
        group_vec: Optional[torch.Tensor] = None,
        attr_xyz=None,
        appear=None,
        use_checkpoint: Optional[bool] = None,
    ):
        tokens = self._tokens(z_map)
        if scale is None:
            scale = tokens.new_full((tokens.shape[0], self.num_groups, 1), 0.02)
        nb = None
        if self.cfg.decoder_neighbor_context:
            if bool(getattr(self.cfg, "decoder_neighbor_group_tokens", True)) and group_vec is not None:
                nb = self._group_neighbors(group_vec)
            elif cell_vec is not None:
                nb = self._cell_neighbors(cell_vec)
        # `use_checkpoint=False` is required by any caller that runs this module
        # through `functional_call` with detached parameters -- the teacher-cycle
        # branch. Under checkpointing with use_reentrant=False, which tensors get
        # saved depends on what requires grad, so a detached-parameter pass saves a
        # different set than the trainable pass, and the recompute during backward
        # fails with "Recomputed values ... have different metadata". Observed as a
        # hard crash on the first step the decoder turns on.
        use_ckpt = bool(self.training) and bool(self.cfg.checkpoint_decode)
        if use_checkpoint is not None:
            use_ckpt = bool(use_checkpoint)
        preds, pres = [], []
        for g0 in range(0, self.num_groups, self.patch_chunk):
            g1 = min(g0 + self.patch_chunk, self.num_groups)
            if use_ckpt:
                p, o = checkpoint(
                    self._decode_range, tokens, nb, scale, centroid, count, g0, g1, attr_xyz,
                    appear,
                    use_reentrant=False,
                )
            else:
                p, o = self._decode_range(tokens, nb, scale, centroid, count, g0, g1, attr_xyz, appear)
            preds.append(p.reshape(p.shape[0], -1, int(self.cfg.target_dim)))
            pres.append(o.reshape(o.shape[0], -1))
        pred = torch.cat(preds, dim=1)[:, : int(self.cfg.max_points)]
        presence = torch.cat(pres, dim=1)[:, : int(self.cfg.max_points)]
        return pred, presence
