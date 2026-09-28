"""Generative decoder: z_compact -> points, without touching z_raw.

This is the path a world model (DIAMOND) actually uses, so it gets a real
hierarchy instead of expanding one 32-d vector into 64 points in a single step:

    cell code -> windowed cell context -> ``merge`` group tokens
              -> ``group_size`` point queries per group
              -> coarse xyz anchored on the centroid channels
              -> point self-attn then cross-attn refine (groups, cell, 3x3)

Anchoring the coarse position on the centroid channels is the reason this
converges much faster than a free-form expansion: the generative branch starts
from approximately the right place and only has to learn the local arrangement.
Set-style cross-attention decoding follows VecSet / COD-VAE / Hunyuan3D.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .compressor import SCALE_REF, axis_angle_to_matrix, decode_scale
from .template import fibonacci_ball_torch
from .config import Can3TokConfig, channel_budget, patch_layout
from .layers import (
    MLP,
    CrossAttentionBlock,
    FourierFeatures,
    SelfAttentionBlock,
    WindowSelfAttention,
    neighbor_gather,
    residual_head,
)
from .morton import grid_zorder_permutation


class GenerativeDecoder(nn.Module):
    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay = patch_layout(cfg)
        bud = channel_budget(cfg)
        self.G = lay["group_size"]
        self.M = lay["merge"]
        self.cells = lay["compact_cells"]
        self.num_groups = lay["num_groups"]
        self.points_per_cell = lay["points_per_cell"]
        self.per_group = bud["per_group"]
        self.c_cen = bud["centroid"]
        self.c_occ = bud["occupancy"]
        self.c_shape = bud["shape"]
        self.uniform = bool(cfg.uniform_budget)
        self.region_chunk = max(int(cfg.gen_region_chunk), 1)
        d = int(cfg.model_dim)

        h, w = int(cfg.compact_latent_hw[0]), int(cfg.compact_latent_hw[1])
        perm = grid_zorder_permutation(h, w) if cfg.latent_zorder else torch.arange(h * w)
        self.register_buffer("cell_to_grid", perm.long(), persistent=False)
        self.grid_h, self.grid_w = h, w

        self.from_compact = MLP(
            [int(cfg.compact_latent_channels), d, d], last_act=True, dropout=cfg.dropout,
            in_norm=True, hidden_norm=True,
        )
        self.window_blocks = nn.ModuleList(
            [
                WindowSelfAttention(
                    d, cfg.heads, (h, w), window=cfg.compress_window, dropout=cfg.dropout, shift=bool(i % 2)
                )
                for i in range(max(0, int(cfg.gen_window_layers)))
            ]
        )
        self.group_split = nn.Linear(d, self.M * d)
        self.group_emb = nn.Parameter(torch.zeros(1, 1, self.M, d))
        nn.init.normal_(self.group_emb, std=0.02)
        self.group_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
                for _ in range(max(0, int(cfg.gen_group_layers)))
            ]
        )
        self.slot_emb = nn.Embedding(self.G, d) if cfg.gen_use_slot_embedding else None
        self.expand_xyz = residual_head([d, d, 3], dropout=cfg.dropout)
        # Direct shape -> per-point offset (same unbroken path as StagedDecompressor).
        # Without this, gen only modulates a shared slot template and collapses.
        shape_in = self.c_shape if not self.uniform else self.per_group
        hid = max(128, 4 * self.G)
        self.folding = bool(getattr(cfg, "folding_decode", False))
        self.fold_frame = int(getattr(cfg, "folding_frame_channels", 6))
        if self.folding:
            self.register_buffer("unit_ball", fibonacci_ball_torch(self.G), persistent=False)
            self.d_ctx = int(cfg.model_dim)
            self.fold_head = MLP([shape_in + self.d_ctx, 128, self.fold_frame], dropout=0.0)
            nn.init.normal_(self.fold_head.net[-1].weight, std=1e-3)
            nn.init.zeros_(self.fold_head.net[-1].bias)
            self.shape_xyz = MLP([shape_in + self.d_ctx, hid, hid, self.G * 3], dropout=cfg.dropout)
        else:
            self.fold_head = None
            self.shape_xyz = MLP([shape_in, hid, hid, self.G * 3], dropout=cfg.dropout)
        self.xyz_pe = FourierFeatures(3, cfg.num_freqs_xyz, include_input=True)
        self.q_proj = MLP([self.xyz_pe.out_dim, d, d], last_act=True, dropout=cfg.dropout)
        n_refine = max(1, int(cfg.gen_cross_layers))
        # Point SA lets the merge*G slots coordinate before reading compact memory —
        # the main missing inductive bias vs independent per-point cross-attn.
        self.refine_self_blocks = nn.ModuleList(
            [SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout) for _ in range(n_refine)]
        )
        self.refine_blocks = nn.ModuleList(
            [CrossAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout) for _ in range(n_refine)]
        )
        self.xyz_residual = residual_head([d + self.xyz_pe.out_dim, d, d, 3])
        self.head_presence = nn.Linear(d, 1)

        # Attribute heads. Until now the student emitted `zeros(target_dim - 3)`
        # for everything except xyz, so the *deployment* path -- the one a world
        # model decodes -- could not produce a Gaussian at all. Same structure as
        # the codec's, for the same reason the folding basis is shared: a student
        # that builds attributes differently makes distillation compare surfaces
        # rather than solutions.
        self.c_app = int(bud.get("appearance", 0))
        self.attr_mlp = None
        self.head_scale = self.head_rot = self.head_opacity = None
        self.head_color = self.head_sh = None
        if int(cfg.target_dim) > 3:
            self.attr_mlp = MLP([d + self.xyz_pe.out_dim + self.c_app, d, d],
                                last_act=True, dropout=cfg.dropout)
            self.head_scale = nn.Linear(d, 3)
            self.head_rot = nn.Linear(d, 4)
            self.head_opacity = nn.Linear(d, 1)
            if int(cfg.target_dim) > 11:
                self.head_color = nn.Linear(d, 3)
            if int(cfg.sh_dim) > 0:
                self.head_sh = nn.Linear(d, int(cfg.sh_dim))
        self._init_heads()

    def _init_heads(self) -> None:
        if self.slot_emb is not None:
            nn.init.normal_(self.slot_emb.weight, std=0.02)
        nn.init.normal_(self.expand_xyz.net[-1].weight, std=5e-3)
        nn.init.zeros_(self.expand_xyz.net[-1].bias)
        if self.shape_xyz is not None:
            if self.folding:
                # near-zero, not zero: see StagedDecompressor for why
                nn.init.normal_(self.shape_xyz.net[-1].weight, std=1e-3)
                nn.init.zeros_(self.shape_xyz.net[-1].bias)
            else:
                nn.init.normal_(self.shape_xyz.net[-1].weight, std=0.02)
                nn.init.zeros_(self.shape_xyz.net[-1].bias)
        nn.init.zeros_(self.xyz_residual.net[-1].weight)
        nn.init.zeros_(self.xyz_residual.net[-1].bias)
        nn.init.constant_(self.head_presence.bias, 2.0)

    def _anchor(self, pg: torch.Tensor):
        """Read centroid / scale / occupancy straight out of the semantic channels."""
        if self.uniform:
            b, cells, m, _ = pg.shape
            return (
                pg.new_zeros(b, cells, m, 3),
                pg.new_full((b, cells, m, 1), SCALE_REF),
                pg.new_full((b, cells, m), float(self.G)),
            )
        centroid = pg[..., 0:3]
        scale = decode_scale(pg[..., 3:4])
        if self.c_occ > 0:
            occ0 = pg[..., self.c_cen]
            count = ((occ0 + 1.0) * 0.5 * float(self.G)).clamp(0.0, float(self.G))
        else:
            alive = (scale.squeeze(-1) > (SCALE_REF * 0.05)).to(dtype=scale.dtype)
            count = alive * float(self.G)
        return centroid, scale, count

    def _decode_range(self, group_tok, cell_vec, nb, centroid, scale, count, shape, r0: int, r1: int,
                      appear=None):
        b = group_tok.shape[0]
        chunk = r1 - r0
        d = int(self.cfg.model_dim)
        gt = group_tok[:, r0:r1]  # (B, chunk, M, d)
        cen = centroid[:, r0:r1]
        sc = scale[:, r0:r1]
        cnt = count[:, r0:r1]
        shp = shape[:, r0:r1]  # (B, chunk, M, c_shape)

        pt = gt.unsqueeze(3).expand(b, chunk, self.M, self.G, d)
        if self.slot_emb is not None:
            pt = pt + self.slot_emb.weight.view(1, 1, 1, self.G, d)
        # Primary geometry from shape channels; template expand is a small residual.
        if self.folding:
            # Must mirror StagedDecompressor._shape_to_offsets exactly: residual in
            # the whitened local frame, then per-axis scale, then rotation.
            # Same input as the codec's folding heads: raw shape channels *plus*
            # the contextualised group token. Reading the raw channels alone was
            # the remaining asymmetry between the two decode paths.
            hin = torch.cat([shp, gt], dim=-1)
            frame = self.fold_head(hin)
            aniso = F.softplus(frame[..., :3]) + 1e-3
            aniso = aniso / aniso.prod(dim=-1, keepdim=True).clamp(min=1e-6).pow(1.0 / 3.0)
            local = self.unit_ball.view(1, 1, 1, self.G, 3).to(shp.dtype)
            res = self.shape_xyz(hin).reshape(b, chunk, self.M, self.G, 3)
            cap = float(getattr(self.cfg, "folding_res_cap", 1.0))
            gain = float(getattr(self.cfg, "folding_res_gain", 1.0))
            if cap > 0.0:
                res = torch.tanh(res / cap) * cap
            local = local + res * gain
            # mirror StagedDecompressor: centre before the frame so the residual's
            # mean cannot move the group centroid
            local = local - local.mean(dim=3, keepdim=True)
            local = local * aniso.unsqueeze(3)
            if self.fold_frame >= 6:
                rot = axis_angle_to_matrix(frame[..., 3:6])
                local = torch.einsum("bcmij,bcmgj->bcmgi", rot, local)
            gen_basis = (frame, local)
            direct = local * sc.unsqueeze(3)
        else:
            gen_basis = None
            direct = self.shape_xyz(shp).reshape(b, chunk, self.M, self.G, 3) * sc.unsqueeze(3)
        # Gate the two free additive paths the way the codec's already are.
        #
        # Under folding, ``direct`` already produces the group. ``template`` can move a
        # point by gen_offset_scale (0.35) of the extent and ``residual`` below by
        # residual_scale (0.6) -- 0.95 of the extent between them, against a template
        # whose own per-axis std is 0.767. That is enough to rewrite the entire
        # within-group arrangement. Measured: gen's within-group chamfer went 0.509 at
        # init (both paths still near zero) to 0.805 once they trained, against 0.524
        # for putting every point at its group centroid, while gen's *global* chamfer
        # stayed fine (0.00216 vs the codec's 0.00180). The codec had the identical
        # disease and was fixed by capping and ramping all three of its additive
        # paths; gen never was.
        t_gain = float(getattr(self.cfg, "folding_res_gain", 1.0)) if self.folding else 1.0
        template = (
            torch.tanh(self.expand_xyz(pt))
            * sc.unsqueeze(3)
            * float(self.cfg.gen_offset_scale)
            * t_gain
        )
        coarse = cen.unsqueeze(3) + direct + template
        coarse = coarse.reshape(b * chunk, self.points_per_cell, 3)

        q = self.q_proj(self.xyz_pe(coarse)) + pt.reshape(b * chunk, self.points_per_cell, d)
        mem = [gt.reshape(b * chunk, self.M, d), cell_vec[:, r0:r1].reshape(b * chunk, 1, d)]
        if nb is not None:
            mem.append(nb[:, r0:r1].reshape(b * chunk, -1, d))
        mem = torch.cat(mem, dim=1)
        slot = torch.arange(self.G, device=q.device, dtype=q.dtype).view(1, 1, 1, self.G)
        occ = (cnt.unsqueeze(-1) - slot).clamp(0.0, 1.0)
        slot_pad = (occ <= 0.5).reshape(b * chunk, self.points_per_cell)
        h = q
        for sa, ca in zip(self.refine_self_blocks, self.refine_blocks):
            h = sa(h, key_padding_mask=slot_pad)
            h = ca(h, mem)

        residual = torch.tanh(self.xyz_residual(torch.cat([h, self.xyz_pe(coarse)], dim=-1)))
        s_flat = sc.unsqueeze(3).expand(b, chunk, self.M, self.G, 1).reshape(b * chunk, self.points_per_cell, 1)
        # gen's analogue of decoder_refine_alpha, which the codec's refine already has
        r_gain = float(getattr(self.cfg, "decoder_refine_alpha", 1.0)) if self.folding else 1.0
        xyz = coarse + residual * s_flat * float(self.cfg.residual_scale) * r_gain
        occ_prior = (occ - 0.5) * 8.0
        presence = self.head_presence(h).squeeze(-1) + occ_prior.reshape(b * chunk, self.points_per_cell)
        if gen_basis is None:
            zf = xyz.new_zeros(b, chunk, self.M, self.fold_frame)
            zl = xyz.new_zeros(b, chunk, self.M, self.G, 3)
            gen_basis = (zf, zl)

        pts = chunk * self.points_per_cell
        xyz_out = xyz.reshape(b, pts, 3)
        if self.attr_mlp is None:
            rest = xyz_out.new_zeros(b, pts, max(int(self.cfg.target_dim) - 3, 0))
            pred = torch.cat([xyz_out, rest], dim=-1) if rest.shape[-1] else xyz_out
        else:
            feat = [h, self.xyz_pe(xyz)]
            if self.c_app > 0 and appear is not None:
                a = appear[:, r0:r1].reshape(b * chunk, self.M, self.c_app)
                a = a.unsqueeze(2).expand(b * chunk, self.M, self.G, self.c_app)
                feat.append(a.reshape(b * chunk, self.points_per_cell, self.c_app))
            elif self.c_app > 0:
                feat.append(h.new_zeros(b * chunk, self.points_per_cell, self.c_app))
            attr = self.attr_mlp(torch.cat(feat, dim=-1))
            cap = float(getattr(self.cfg, "attr_scale_log_cap", 3.0))
            sc_raw = self.head_scale(attr)
            if cap > 0:                       # see CodecDecoder._bounded_log_scale
                sb = self.head_scale.bias.view(1, -1)
                sc_raw = sb + torch.tanh((sc_raw - sb) / cap) * cap
            parts = [xyz_out,
                     sc_raw.reshape(b, pts, 3),
                     F.normalize(self.head_rot(attr), dim=-1).reshape(b, pts, 4),
                     self.head_opacity(attr).reshape(b, pts, 1)]
            if self.head_color is not None:
                parts.append(self.head_color(attr).reshape(b, pts, 3))
            if self.head_sh is not None:
                parts.append(self.head_sh(attr).reshape(b, pts, int(self.cfg.sh_dim)))
            pred = torch.cat(parts, dim=-1)
        return (
            pred,
            presence.reshape(b, pts),
            gen_basis[0],
            gen_basis[1],
        )

    def forward(self, z_compact: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = z_compact.shape
        d = int(self.cfg.model_dim)
        grid = z_compact.reshape(b, c, h * w).transpose(1, 2)
        z_cells = grid[:, self.cell_to_grid]
        pg = z_cells.reshape(b, self.cells, self.M, self.per_group)
        centroid, scale, count = self._anchor(pg)
        if self.uniform:
            shape = pg
            appear = None
        else:
            s0 = self.c_cen + self.c_occ
            shape = pg[..., s0 : s0 + self.c_shape]
            appear = pg[..., s0 + self.c_shape : s0 + self.c_shape + self.c_app] if self.c_app else None

        cell_vec = self.from_compact(z_cells)
        if len(self.window_blocks) > 0:
            gv = cell_vec.new_zeros(b, self.grid_h * self.grid_w, d)
            gv[:, self.cell_to_grid] = cell_vec
            for block in self.window_blocks:
                gv = block(gv)
            cell_vec = gv[:, self.cell_to_grid]

        group_tok = self.group_split(cell_vec).reshape(b, self.cells, self.M, d) + self.group_emb.to(cell_vec.dtype)
        if len(self.group_blocks) > 0:
            hh = torch.cat([cell_vec.unsqueeze(2), group_tok], dim=2).reshape(b * self.cells, self.M + 1, d)
            for block in self.group_blocks:
                hh = block(hh)
            hh = hh.reshape(b, self.cells, self.M + 1, d)
            cell_vec = hh[:, :, 0]
            group_tok = hh[:, :, 1:]

        nb = None
        if self.cfg.gen_neighbor_context:
            if bool(getattr(self.cfg, "gen_neighbor_group_tokens", True)):
                # Neighbour *group* tokens (3x3 x M) — finer than cell summaries.
                flat = group_tok.reshape(b, self.cells, self.M * d)
                gridv = flat.new_zeros(b, self.grid_h * self.grid_w, self.M * d)
                gridv[:, self.cell_to_grid] = flat
                nb = neighbor_gather(gridv, self.grid_h, self.grid_w, radius=1)[:, self.cell_to_grid]
                nb = nb.reshape(b, self.cells, 9 * self.M, d)
            else:
                gridv = cell_vec.new_zeros(b, self.grid_h * self.grid_w, d)
                gridv[:, self.cell_to_grid] = cell_vec
                nb = neighbor_gather(gridv, self.grid_h, self.grid_w, radius=1)[:, self.cell_to_grid]

        use_ckpt = bool(self.training) and bool(self.cfg.checkpoint_gen)
        xs, ps, bfs, bls = [], [], [], []
        for r0 in range(0, self.cells, self.region_chunk):
            r1 = min(r0 + self.region_chunk, self.cells)
            if use_ckpt:
                chunk_pred, pres, bf, bl = checkpoint(
                    self._decode_range, group_tok, cell_vec, nb, centroid, scale, count, shape, r0, r1,
                    appear,
                    use_reentrant=False,
                )
            else:
                chunk_pred, pres, bf, bl = self._decode_range(
                    group_tok, cell_vec, nb, centroid, scale, count, shape, r0, r1, appear
                )
            xs.append(chunk_pred)
            ps.append(pres)
            bfs.append(bf)
            bls.append(bl)

        limit = self.num_groups * self.G
        # _decode_range already emits the full target width (xyz + attributes when
        # the heads exist), so nothing is zero-padded here any more.
        pred = torch.cat(xs, dim=1)[:, :limit]
        pres_all = torch.cat(ps, dim=1)[:, :limit]
        mp = int(self.cfg.max_points)
        out = {"pred": pred[:, :mp], "presence": pres_all[:, :mp]}
        if self.folding and bfs:
            ng = self.num_groups
            out["fold_frame"] = torch.cat(bfs, dim=1).reshape(pred.shape[0], -1, self.fold_frame)[:, :ng]
            out["fold_local"] = torch.cat(bls, dim=1).reshape(pred.shape[0], -1, self.G, 3)[:, :ng]
        return out
