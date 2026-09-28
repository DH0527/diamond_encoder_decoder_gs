"""Function-preserving joint refinement of a decoded Gaussian set.

The R8 family split every compact cell into ``shape`` and ``appearance`` and
then trained the two halves with disjoint objectives.  This module deliberately
reads *all* non-anchor channels and produces residuals for every Gaussian
parameter from one shared slot representation.

It is a refiner rather than a replacement decoder for one important experimental
reason: all output heads are zero initialised.  Loading an R8 checkpoint therefore
starts from exactly the R8 function while opening the missing joint gradient path.
Any improvement can be attributed to the new path, not to paying for a freshly
initialised decoder.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .compressor import axis_angle_to_matrix
from .config import Can3TokConfig, channel_budget, patch_layout
from .layers import CrossAttentionBlock, MLP, SelfAttentionBlock
from .template import fibonacci_ball_torch


class DirectJointGaussianDecoder(nn.Module):
    """Decode a complete Gaussian set from one shared non-anchor code.

    Unlike J1 this is not a residual on two already-separated decoders.  The
    same slot token emits position, scale, rotation, opacity, colour and
    presence, so both balanced geometry supervision and rendering update every
    one of the 27 learned channels.  Analytic centroid/extent anchors remain as
    a coordinate frame; they do not carry appearance or within-group shape.
    """

    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay, bud = patch_layout(cfg), channel_budget(cfg)
        self.G = int(lay["group_size"])
        self.structured_local_code = bool(getattr(cfg, "structured_local_code", False))
        self.global_dim = int(getattr(cfg, "compact_global_dim", 3))
        self.T = (int(getattr(cfg, "compact_local_tokens", 4))
                  if self.structured_local_code else int(lay["tokens_per_group"]))
        self.C = (int(getattr(cfg, "compact_local_dim", 6))
                  if self.structured_local_code else int(cfg.latent_channels))
        self.num_groups = int(lay["num_groups"])
        self.c_shared = int(bud["per_group"] - bud["centroid"] - bud["occupancy"])
        self.chunk = max(int(getattr(cfg, "joint_decoder_chunk", 256)), 1)
        self.nbr = max(int(getattr(cfg, "joint_nbr_window", 1)), 0)
        self.use_local_memory = (bool(getattr(cfg, "joint_local_memory", False))
                                 or self.structured_local_code)
        d = int(getattr(cfg, "joint_decoder_dim", 192))

        if self.structured_local_code:
            expected = self.global_dim + self.T * self.C
            if expected != self.c_shared:
                raise ValueError(
                    f"structured decoder expects {expected} learned channels, "
                    f"but compact budget provides {self.c_shared}")
        code_dim = self.global_dim if self.structured_local_code else self.c_shared

        self.code_proj = MLP([2 * code_dim, d, d], last_act=True,
                             dropout=cfg.dropout, in_norm=True, hidden_norm=True)
        self.anchor_proj = MLP([5, d, d], last_act=True, dropout=cfg.dropout,
                               in_norm=True)
        self.slot_emb = nn.Embedding(self.G, d)
        n_layers = max(int(getattr(cfg, "joint_decoder_layers", 2)), 0)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
            for _ in range(n_layers)
        ])
        if self.use_local_memory:
            self.local_proj = MLP([self.C, d, d], last_act=True,
                                  dropout=cfg.dropout, in_norm=True, hidden_norm=True)
            self.local_pos = nn.Parameter(torch.randn(1, self.T, d) * 0.02)
            self.memory_blocks = nn.ModuleList([
                SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
                for _ in range(max(int(getattr(cfg, "joint_memory_layers", 1)), 0))
            ])
            self.cross_blocks = nn.ModuleList([
                CrossAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
                for _ in range(n_layers)
            ])
            # Explicit geometric query coordinates stop 64 learned IDs from
            # degenerating into an arbitrary lookup table.  Cross-attention may
            # then deform each query differently as a function of the T local
            # tokens reconstructed from the compact latent.
            self.query_xyz = MLP([3, d, d], last_act=True, dropout=cfg.dropout)
        else:
            self.local_proj = None
            self.local_pos = None
            self.memory_blocks = nn.ModuleList()
            self.cross_blocks = nn.ModuleList()
            self.query_xyz = None
        self.norm = nn.LayerNorm(d)
        self.frame = nn.Linear(d, 6)
        self.head_translation = nn.Linear(d, 3)
        self.head_xyz = nn.Linear(d, 3)
        self.head_scale = nn.Linear(d, 3)
        self.head_rot = nn.Linear(d, 4)
        self.head_opacity = nn.Linear(d, 1)
        self.head_color = nn.Linear(d, 3) if cfg.target_dim > 11 else None
        self.head_sh = nn.Linear(d, int(cfg.sh_dim)) if cfg.target_dim > 14 else None
        self.head_presence = nn.Linear(d, 1)
        self.register_buffer("unit_ball", fibonacci_ball_torch(self.G), persistent=False)
        self.scale_base_a = nn.Parameter(torch.tensor(0.7872))

        nn.init.normal_(self.slot_emb.weight, std=0.02)
        nn.init.normal_(self.frame.weight, std=0.02)
        nn.init.zeros_(self.frame.bias)
        # A scene anchor is already a good coarse centre.  Randomly initialising
        # this head moved groups by ~0.13 normalised units in the F3B smoke run,
        # much farther than the measured ~0.04 target correction.  Start at the
        # anchor exactly; the absolute centroid loss gives this head a non-zero
        # weight gradient on the first backward pass.
        nn.init.zeros_(self.head_translation.weight)
        nn.init.zeros_(self.head_translation.bias)
        nn.init.normal_(self.head_xyz.weight, std=0.02)
        nn.init.zeros_(self.head_xyz.bias)
        for head, bias in ((self.head_scale, 0.0), (self.head_rot, 0.0),
                           (self.head_opacity, -2.13), (self.head_color, 0.0),
                           (self.head_sh, 0.0), (self.head_presence, 0.0)):
            if head is not None:
                nn.init.normal_(head.weight, std=1e-3)
                nn.init.constant_(head.bias, bias)
        with torch.no_grad():
            self.head_rot.bias[0] = 1.0

    def _neighbour_code(self, code: torch.Tensor) -> torch.Tensor:
        if self.nbr <= 0:
            return code
        x = F.pad(code.transpose(1, 2), (self.nbr, self.nbr), mode="replicate")
        return torch.stack(
            [x[:, :, i:i + code.shape[1]] for i in range(2 * self.nbr + 1)], dim=0
        ).mean(dim=0).transpose(1, 2)

    def _decode_chunk(self, code: torch.Tensor, centroid: torch.Tensor,
                      scale: torch.Tensor, count: torch.Tensor,
                      local_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Each row is one group.  The fixed ball supplies a non-collapsed basis;
        # the shared code predicts its frame and a bounded per-slot deformation.
        anchor = torch.cat([centroid, scale.clamp(min=1e-6).log(),
                            count.unsqueeze(-1) / float(self.G)], dim=-1)
        gh = self.code_proj(code) + self.anchor_proj(anchor)
        slot = self.slot_emb.weight.unsqueeze(0).to(gh.dtype)
        memory_std = gh.new_zeros(())
        if self.use_local_memory:
            if local_tokens.shape[1:] != (self.T, self.C):
                raise ValueError(
                    f"local token shape {tuple(local_tokens.shape)} does not match "
                    f"(*, {self.T}, {self.C})")
            memory = (self.local_proj(local_tokens)
                      + self.local_pos.to(gh.dtype)
                      + gh.unsqueeze(1))
            for block in self.memory_blocks:
                memory = block(memory)
            memory_std = memory.float().std(dim=1).mean().to(gh.dtype)
            basis = self.unit_ball.to(gh.dtype)
            h = gh.unsqueeze(1) + slot + self.query_xyz(basis).unsqueeze(0)
            for cross, block in zip(self.cross_blocks, self.blocks):
                h = cross(h, memory)
                h = block(h)
        else:
            h = gh.unsqueeze(1) + slot
            for block in self.blocks:
                h = block(h)
        h = self.norm(h)

        frame = self.frame(gh)
        cap = 1.5
        log_aniso = frame[:, :3] - frame[:, :3].mean(dim=-1, keepdim=True)
        aniso = torch.exp(torch.tanh(log_aniso / cap) * cap)
        base = self.unit_ball.to(h.dtype).unsqueeze(0) * aniso.unsqueeze(1)
        rot = axis_angle_to_matrix(frame[:, 3:6])
        base = torch.einsum("bij,bgj->bgi", rot, base)
        residual = torch.tanh(self.head_xyz(h)) * float(
            getattr(self.cfg, "joint_direct_xyz_cap", 1.5))
        local = base + residual
        local = local - local.mean(dim=1, keepdim=True)
        translation_cap = float(getattr(self.cfg, "joint_translation_cap", 0.25))
        tr_raw = self.head_translation(gh)
        # Unit slope at the origin: cap*tanh(raw/cap), unlike cap*tanh(raw),
        # keeps the centroid gradient well conditioned while still bounding bad
        # early updates to a quarter of the normalised scene.
        translation = translation_cap * torch.tanh(
            tr_raw / max(translation_cap, 1e-6))
        xyz = centroid.unsqueeze(1) + translation.unsqueeze(1) + local * scale.unsqueeze(1)

        # Group-relative scale base covers 99.9% of the measured range while the
        # asymmetric residual prevents the giant-splat rendering shortcut.
        scale_base = self.scale_base_a * scale.clamp(min=1e-6).log() - 2.559
        sr = self.head_scale(h)
        sr = torch.where(sr < 0, 3.0 * torch.tanh(sr / 3.0),
                         2.0 * torch.tanh(sr / 2.0))
        log_scale = scale_base.unsqueeze(1) + sr
        quat = F.normalize(self.head_rot(h), dim=-1)
        attrs = [log_scale, quat, self.head_opacity(h)]
        if self.head_color is not None:
            attrs.append(self.head_color(h))
        if self.head_sh is not None:
            attrs.append(self.head_sh(h))
        pred = torch.cat([xyz] + attrs, dim=-1)

        slot_index = torch.arange(self.G, device=h.device, dtype=h.dtype).view(1, self.G)
        count_prior = ((count.unsqueeze(-1) - slot_index - 0.5) * 2.0).clamp(-8.0, 8.0)
        presence = count_prior + self.head_presence(h).squeeze(-1)
        return {"pred": pred, "presence": presence,
                "translation_abs": translation.abs().mean(),
                "memory_token_std": memory_std}

    def forward(self, centroid: torch.Tensor, count: torch.Tensor,
                group_scale: torch.Tensor, shared_code: torch.Tensor,
                local_tokens: torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
        if shared_code is None:
            raise RuntimeError("joint_direct_decoder requires non-anchor shared_code")
        if self.use_local_memory and local_tokens is None and not self.structured_local_code:
            raise RuntimeError("joint_local_memory requires reconstructed local_tokens")
        b, ng, _ = shared_code.shape
        ng = min(ng, self.num_groups, centroid.shape[1])
        shared = shared_code[:, :ng]
        if self.structured_local_code:
            global_code = shared[..., :self.global_dim]
            local_tokens = shared[..., self.global_dim:].reshape(b, ng, self.T, self.C)
            code = torch.cat([global_code, self._neighbour_code(global_code)], dim=-1)
        else:
            code = torch.cat([shared, self._neighbour_code(shared)], dim=-1)
        pred_parts, presence_parts, translation_parts, memory_std_parts = [], [], [], []
        for s in range(0, ng, self.chunk):
            e, c = min(s + self.chunk, ng), min(s + self.chunk, ng) - s
            local = (local_tokens[:, s:e].reshape(b * c, self.T, self.C)
                     if local_tokens is not None
                     else shared_code.new_zeros(b * c, self.T, self.C))
            args = (code[:, s:e].reshape(b * c, -1),
                    centroid[:, s:e].reshape(b * c, 3),
                    group_scale[:, s:e].reshape(b * c, 1),
                    count[:, s:e].reshape(b * c), local)
            if self.training and torch.is_grad_enabled():
                part = torch.utils.checkpoint.checkpoint(
                    self._decode_chunk, *args, use_reentrant=False)
            else:
                part = self._decode_chunk(*args)
            pred_parts.append(part["pred"].reshape(b, c * self.G, -1))
            presence_parts.append(part["presence"].reshape(b, c * self.G))
            translation_parts.append(part["translation_abs"])
            memory_std_parts.append(part["memory_token_std"])
        return {"pred": torch.cat(pred_parts, dim=1),
                "presence": torch.cat(presence_parts, dim=1),
                "translation_abs": torch.stack(translation_parts).mean(),
                "memory_token_std": torch.stack(memory_std_parts).mean()}


class JointGaussianRefiner(nn.Module):
    """Refine xyz and attributes from one shared per-slot representation.

    ``geometry_base`` supplies the differentiable geometry-decoder positions.
    ``render_base`` supplies the old attribute decoder's deployable Gaussian.
    The same bounded xyz residual is applied to both, while all attribute
    residuals are applied to ``render_base``.  This keeps the old rendering
    function bit-for-bit at initialisation and still lets geometry losses train
    the original geometry decoder.
    """

    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay = patch_layout(cfg)
        bud = channel_budget(cfg)
        self.G = int(lay["group_size"])
        self.num_groups = int(lay["num_groups"])
        self.c_shared = int(bud["per_group"] - bud["centroid"] - bud["occupancy"])
        self.chunk = max(int(getattr(cfg, "joint_decoder_chunk", cfg.patch_chunk)), 1)
        self.nbr = max(int(getattr(cfg, "joint_nbr_window", 1)), 0)
        d = int(getattr(cfg, "joint_decoder_dim", 192))

        # Centre + local-neighbour mean.  Keeping the two separate lets the model
        # retain sharp cell-local information while repairing hard group seams.
        self.code_proj = MLP([2 * self.c_shared, d, d], last_act=True,
                             dropout=cfg.dropout, in_norm=True)
        self.xyz_proj = MLP([6, d, d], last_act=True, dropout=cfg.dropout,
                            in_norm=True)
        attr_dim = max(int(cfg.target_dim) - 3, 0)
        self.base_proj = (MLP([attr_dim, d, d], last_act=True, dropout=cfg.dropout,
                              in_norm=True) if attr_dim > 0 else None)
        self.slot_emb = nn.Embedding(self.G, d)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
            for _ in range(max(int(getattr(cfg, "joint_decoder_layers", 2)), 0))
        ])
        self.norm = nn.LayerNorm(d)

        self.head_xyz = nn.Linear(d, 3)
        self.head_scale = nn.Linear(d, 3)
        self.head_rot = nn.Linear(d, 4)
        self.head_opacity = nn.Linear(d, 1)
        self.head_color = nn.Linear(d, 3) if cfg.target_dim > 11 else None
        self.head_sh = nn.Linear(d, int(cfg.sh_dim)) if cfg.sh_dim > 0 else None
        self.head_presence = nn.Linear(d, 1)

        nn.init.normal_(self.slot_emb.weight, std=0.02)
        # Exact no-op at step zero.  Unlike widening AttributeDecoder.to_code_tok,
        # this does not discard any trained R8 projection.
        for head in (self.head_xyz, self.head_scale, self.head_rot,
                     self.head_opacity, self.head_color, self.head_sh,
                     self.head_presence):
            if head is not None:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def _neighbour_code(self, code: torch.Tensor) -> torch.Tensor:
        if self.nbr <= 0:
            return code
        x = F.pad(code.transpose(1, 2), (self.nbr, self.nbr), mode="replicate")
        return torch.stack(
            [x[:, :, i:i + code.shape[1]] for i in range(2 * self.nbr + 1)], dim=0
        ).mean(dim=0).transpose(1, 2)

    def _chunk(self, xyz: torch.Tensor, attrs: torch.Tensor,
               code: torch.Tensor, scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        # xyz: (Bc,G,3), code: (Bc,2*C), scale: (Bc,1,1)
        local = xyz - xyz.mean(dim=1, keepdim=True)
        local = local / scale.clamp(min=1e-6)
        h = self.slot_emb.weight.unsqueeze(0).expand(xyz.shape[0], -1, -1)
        h = h + self.xyz_proj(torch.cat([xyz, local], dim=-1))
        h = h + self.code_proj(code).unsqueeze(1)
        if self.base_proj is not None:
            h = h + self.base_proj(attrs)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)

        xyz_cap = float(getattr(self.cfg, "joint_xyz_cap", 0.10))
        scale_cap = float(getattr(self.cfg, "joint_scale_delta_cap", 1.0))
        opacity_cap = float(getattr(self.cfg, "joint_opacity_delta_cap", 2.0))
        color_cap = float(getattr(self.cfg, "joint_color_delta_cap", 0.5))
        rot_cap = float(getattr(self.cfg, "joint_rot_delta_cap", 0.25))

        return {
            "xyz": torch.tanh(self.head_xyz(h)) * xyz_cap * scale,
            "scale": torch.tanh(self.head_scale(h)) * scale_cap,
            "rot": torch.tanh(self.head_rot(h)) * rot_cap,
            "opacity": torch.tanh(self.head_opacity(h)) * opacity_cap,
            "color": (torch.tanh(self.head_color(h)) * color_cap
                      if self.head_color is not None else None),
            "sh": (torch.tanh(self.head_sh(h)) * color_cap
                   if self.head_sh is not None else None),
            "presence": self.head_presence(h).squeeze(-1),
        }

    def forward(self, geometry_base: torch.Tensor, render_base: torch.Tensor,
                presence_base: torch.Tensor, shared_code: torch.Tensor,
                group_scale: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, n, _ = geometry_base.shape
        ng = min(n // self.G, self.num_groups, shared_code.shape[1])
        n_used = ng * self.G
        center = shared_code[:, :ng]
        neigh = self._neighbour_code(center)
        code = torch.cat([center, neigh], dim=-1)

        geom_parts, render_parts, presence_parts = [], [], []
        delta_norms = []
        for s in range(0, ng, self.chunk):
            e = min(s + self.chunk, ng)
            c = e - s
            gx = geometry_base[:, s * self.G:e * self.G, :3].reshape(b * c, self.G, 3)
            rb = render_base[:, s * self.G:e * self.G].reshape(
                b * c, self.G, render_base.shape[-1])
            cd = code[:, s:e].reshape(b * c, -1)
            sc = group_scale[:, s:e].reshape(b * c, 1, 1)
            if self.training and torch.is_grad_enabled():
                dlt = torch.utils.checkpoint.checkpoint(
                    self._chunk, gx, rb[..., 3:], cd, sc, use_reentrant=False)
            else:
                dlt = self._chunk(gx, rb[..., 3:], cd, sc)

            gxyz = gx + dlt["xyz"]
            rxyz = rb[..., :3] + dlt["xyz"]
            attrs = [rb[..., 3:6] + dlt["scale"]]
            q = F.normalize(rb[..., 6:10] + dlt["rot"], dim=-1)
            attrs += [q, rb[..., 10:11] + dlt["opacity"]]
            if render_base.shape[-1] > 11:
                attrs.append(rb[..., 11:14] + dlt["color"])
            if render_base.shape[-1] > 14:
                attrs.append(rb[..., 14:] + dlt["sh"])
            refined_attrs = torch.cat(attrs, dim=-1)

            geom_parts.append(torch.cat([gxyz, refined_attrs], dim=-1).reshape(b, c * self.G, -1))
            render_parts.append(torch.cat([rxyz, refined_attrs], dim=-1).reshape(b, c * self.G, -1))
            pb = presence_base[:, s * self.G:e * self.G].reshape(b * c, self.G)
            presence_parts.append((pb + dlt["presence"]).reshape(b, c * self.G))
            delta_norms.append(dlt["xyz"].detach().abs().mean())

        geometry = torch.cat(geom_parts, dim=1)
        render = torch.cat(render_parts, dim=1)
        presence = torch.cat(presence_parts, dim=1)
        if n_used < n:
            geometry = torch.cat([geometry, geometry_base[:, n_used:]], dim=1)
            render = torch.cat([render, render_base[:, n_used:]], dim=1)
            presence = torch.cat([presence, presence_base[:, n_used:]], dim=1)
        return {
            "geometry": geometry[:, :n],
            "render": render[:, :n],
            "presence": presence[:, :n],
            "xyz_delta_abs": torch.stack(delta_norms).mean() if delta_norms else geometry.sum() * 0.0,
        }
