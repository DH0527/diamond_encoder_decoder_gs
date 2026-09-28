"""z_raw <-> z_compact codec.

Design notes
------------
* **Semantic channel budget.** Each compact cell holds ``merge`` groups; every
  group gets ``centroid | occupancy | shape`` channels instead of an even split
  across ``merge * tokens_per_group`` tokens. The centroid channels are *anchored*
  to the analytic group centroid and the occupancy channel to the valid-point
  count, so the coarse, irreplaceable part of the signal is carried explicitly and
  the learned capacity is spent on the residual shape. This is the residual /
  base+detail decomposition used by Gaussian-splat codecs and by DC-AE.
* **Staged bottleneck.** ``d -> mid -> budget`` with a nonlinearity between,
  rather than one linear projection from 32 channels/token straight down. TC-AE
  and COD-VAE both report that spreading the compression over stages is what
  keeps structure alive at aggressive ratios.
* **Masked attention.** With ``merge=2`` a cell can hold one real and one padded
  group; padded groups are excluded from every attention context instead of
  polluting it.
* **Cross-cell window attention** on the compact grid (Swin-style, shifted every
  other layer) so a cell can borrow context from its spatial neighbours, as in
  the SLat encoder of TRELLIS.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .template import fibonacci_ball_torch
import torch.nn.functional as F

from .config import Can3TokConfig, channel_budget, patch_layout
from .layers import (CrossAttentionBlock,
    MLP,
    FourierFeatures,
    ResBlock2d,
    SelfAttentionBlock,
    WindowSelfAttention,
    residual_head,
)
from .morton import grid_zorder_permutation, inverse_permutation

# Centroid nudge, in units of the group extent. The anchor written by the
# compressor is the analytic group centroid, i.e. already exact, and no active
# loss supervises this correction, so a wide bound is free error: at 0.5 the
# measured centroid rmse reached 0.00719 against 0.00113 at step 500.
CENTROID_DELTA = 0.05
SCALE_REF = 0.02  # reference group extent; see encode_scale
SCALE_LOG_SPAN = 3.0


def encode_scale(extent: torch.Tensor) -> torch.Tensor:
    """Per-group extent -> a normalised log-scale channel.

    Group extents span more than three orders of magnitude across a scene (p50
    0.006, p99 0.26), so every head that predicts a position *within* a group has
    to work in units of that group's extent -- otherwise it is asked to output a
    few tenths of a percent of its own range for a dense group, which is not
    something Adam can tune at any sane learning rate. Storing the scale costs
    one channel that was previously unused.
    """
    s = extent.amax(dim=-1).clamp(1e-5, 4.0)
    return torch.log(s / SCALE_REF) / SCALE_LOG_SPAN


def decode_scale(u: torch.Tensor) -> torch.Tensor:
    return (SCALE_REF * torch.exp(SCALE_LOG_SPAN * u.clamp(-3.0, 2.0))).clamp(1e-5, 4.0)


def axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` Rodrigues vector -> ``(..., 3, 3)`` rotation matrix.

    The group frame needs a rotation, not just per-axis scales: real groups are
    flat patches at arbitrary world orientations, and an axis-aligned ellipsoid
    has to fatten to cover a tilted one. Measured envelope-only chamfer,
    axis-aligned 0.336 vs rotated 0.301 -- 10% for three channels.
    """
    theta = v.norm(dim=-1, keepdim=True)
    k = v / theta.clamp(min=1e-8)
    kx, ky, kz = k.unbind(-1)
    o = torch.zeros_like(kx)
    K = torch.stack([o, -kz, ky, kz, o, -kx, -ky, kx, o], dim=-1).reshape(*v.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    st = torch.sin(theta).unsqueeze(-1)
    ct = (1.0 - torch.cos(theta)).unsqueeze(-1)
    return eye + st * K + ct * (K @ K)


class _GridMixin:
    """Cells are numbered in 3D-Morton order; the grid stores them in 2D Z-order.

    Anything spatial (window attention, neighbour gather) has to happen in *grid*
    order, so ``to_grid`` / ``from_grid`` wrap those stages.
    """

    def _register_grid(self, cfg: Can3TokConfig, cells: int) -> None:
        h, w = int(cfg.compact_latent_hw[0]), int(cfg.compact_latent_hw[1])
        perm = grid_zorder_permutation(h, w) if cfg.latent_zorder else torch.arange(h * w)
        self.register_buffer("cell_to_grid", perm.long(), persistent=False)
        self.register_buffer("grid_to_cell", inverse_permutation(perm.long()), persistent=False)
        self.grid_h, self.grid_w = h, w

    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(x.shape[0], self.grid_h * self.grid_w, x.shape[-1])
        out[:, self.cell_to_grid] = x
        return out

    def from_grid(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.cell_to_grid]

    def window_stack(self, blocks, x: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(blocks) == 0:
            return x
        g = self.to_grid(x)
        gv = None
        if valid is not None:
            gv = self.to_grid(valid.unsqueeze(-1)).squeeze(-1)
        for block in blocks:
            g = block(g, valid=gv)
        return self.from_grid(g)


class StagedCompressor(nn.Module, _GridMixin):
    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay = patch_layout(cfg)
        bud = channel_budget(cfg)
        self.G = lay["group_size"]
        self.T = lay["tokens_per_group"]
        self.C = int(cfg.latent_channels)
        self.M = lay["merge"]
        self.num_groups = lay["num_groups"]
        self.cells = lay["compact_cells"]
        self.slots = self.cells * self.M
        self.per_group = bud["per_group"]
        self.c_cen, self.c_occ, self.c_shape = bud["centroid"], bud["occupancy"], bud["shape"]
        self.c_app = int(bud.get("appearance", 0))
        self.uniform = bool(cfg.uniform_budget)
        self.shared_free_head = bool(getattr(cfg, "shared_free_head", False)) and not self.uniform
        self.structured_local_code = bool(
            getattr(cfg, "structured_local_code", False)) and not self.uniform
        self.compact_global_dim = int(getattr(cfg, "compact_global_dim", 3))
        self.compact_local_tokens = int(getattr(cfg, "compact_local_tokens", 4))
        self.compact_local_dim = int(getattr(cfg, "compact_local_dim", 6))
        if self.structured_local_code:
            expected = self.compact_global_dim + self.compact_local_tokens * self.compact_local_dim
            available = self.c_shape + self.c_app
            if expected != available:
                raise ValueError(
                    "structured local layout must exactly fill the non-anchor budget: "
                    f"global {self.compact_global_dim} + {self.compact_local_tokens}*"
                    f"{self.compact_local_dim} = {expected}, available={available}")
            if cfg.latent_mode != "deterministic":
                raise ValueError("structured_local_code currently requires deterministic latent mode")
        # Mirrors PatchPackEncoder: a learned pooler replaces the identity pack
        # exactly when the encoder reads more points than the decoder emits.
        _mip = int(getattr(cfg, "max_input_points", 0)) or int(cfg.max_points)
        self.pooled_input = (_mip // max(self.num_groups, 1)) > self.G
        d = int(cfg.model_dim)
        self._register_grid(cfg, self.cells)

        self.token_embed = nn.Linear(self.C, d)
        # Two staged layers with pre-activation norms, not one 8x linear crush.
        # Measured on the trained model: the input pack carries effective rank 20.48,
        # a single Linear(T*d -> d) passed 16.23 at init but training drove it to
        # **5.84**, and group_head then climbed back to 18.64 only because it also
        # receives cell_vec and the analytic anchors. In other words the point-data
        # path dies and the anchors carry the latent. Staging 3584 -> 2d -> d with
        # LayerNorm on every pre-activation gives that path room to hold what it
        # could already pass at init.
        # Measured on the trained model, effective rank along the point-data path:
        #   intra attention 34.26 -> token_merge 13.83  (width 448 both sides)
        # i.e. this 3584 -> 448 merge still throws away 60% of the directions that
        # reached it, even after the earlier fix that staged it from one Linear to
        # a two-layer MLP. Adding a stage costs one matmul per group and gives the
        # 8:1 squeeze somewhere to put what it currently drops.
        _ms = int(getattr(cfg, "compress_merge_stages", 0))
        _merge_dims = ([self.T * d] + ([4 * d] if _ms > 0 else []) + [2 * d, d])
        self.token_merge = MLP(_merge_dims, hidden_norm=True, dropout=cfg.dropout)
        # Attention pooling as an alternative to that concat. Concatenating the T
        # tokens fixes what each output coordinate reads before training starts;
        # a learned query decides it instead, which is the "attention-based
        # downsampling" COD-VAE (ICCV'25) puts in place of a direct point-to-latent
        # map, and 3DShape2VecSet / Can3Tok both form their latents the same way.
        # This is the measured collapse point: effective rank 34.26 -> 13.83 here.
        self.merge_attn = (CrossAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
                           if bool(getattr(cfg, "compress_merge_attn", False)) else None)
        self.merge_q = (nn.Parameter(torch.randn(1, 1, d) * 0.02)
                        if self.merge_attn is not None else None)
        self.token_pos = nn.Parameter(torch.zeros(1, 1, self.M * self.T, d))
        nn.init.normal_(self.token_pos, std=0.02)
        self.intra_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
                for _ in range(max(0, int(cfg.compress_intra_layers)))
            ]
        )
        self.structured_pool = (
            CrossAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
            if self.structured_local_code else None
        )
        self.structured_queries = (
            nn.Parameter(torch.randn(1, self.compact_local_tokens, d) * 0.02)
            if self.structured_local_code else None
        )
        self.cell_mix = MLP([d * self.M, d, d], last_act=True, dropout=cfg.dropout)
        self.window_blocks = nn.ModuleList(
            [
                WindowSelfAttention(
                    d,
                    cfg.heads,
                    (self.grid_h, self.grid_w),
                    window=cfg.compress_window,
                    dropout=cfg.dropout,
                    shift=bool(i % 2),
                )
                for i in range(max(0, int(cfg.compress_window_layers)))
            ]
        )
        self.anchor_pe = FourierFeatures(3, cfg.num_freqs_slot, include_input=True)
        self.group_head = MLP([d * 2 + self.anchor_pe.out_dim + 4, d, d], last_act=True, dropout=cfg.dropout)
        mid = max(int(cfg.compress_mid_channels), self.per_group)
        # Two-stage analysis MLP: d -> mid -> mid (extra nonlinearity before budget heads).
        self.mid = nn.Sequential(
            nn.Linear(d, mid),
            nn.LayerNorm(mid),
            nn.LeakyReLU(0.1),
            nn.Linear(mid, mid),
            nn.LayerNorm(mid),
            nn.LeakyReLU(0.1),
        )
        # Global N / fill: pad/truncate erases densify–prune scale from per-group
        # occupancy, so inject scene-level count stats into every group mid.
        self.global_to_mid = nn.Sequential(
            nn.Linear(2, mid),
            nn.LeakyReLU(0.1),
            nn.Linear(mid, mid),
        )
        nn.init.zeros_(self.global_to_mid[-1].weight)
        nn.init.zeros_(self.global_to_mid[-1].bias)
        # Staged budget heads. A single Linear(256 -> 16) is the last step of the
        # whole encoder, and it is where the latent stops being used: measured
        # effective rank goes mid 23.63 -> z_compact 10.91 of 32 channels, so two
        # thirds of the budget carries nothing. Going down in steps with a
        # normalised nonlinearity between them is the standard fix for exactly this
        # (DC-AE stages its bottleneck the same way) and is what makes the extra
        # channels reachable at all.
        _hs = list(getattr(cfg, "compress_head_stages", ()) or ())

        def _head(out_c):
            if not _hs:
                return nn.Linear(mid, out_c)
            return MLP([mid] + _hs + [out_c], hidden_norm=True, leaky=True, dropout=cfg.dropout)

        self.head_centroid = _head(self.c_cen)
        self.head_occ = _head(self.c_occ) if self.c_occ > 0 else None
        self.head_shape = _head(self.c_shape if not self.uniform else self.per_group)
        # Appearance rides the same `mid`, so the network -- not a hand-drawn split
        # -- decides how much of it describes geometry and how much describes
        # looks. Kept as its own head so the channels stay a contiguous suffix that
        # every consumer can slice, and so c_app = 0 is byte-identical to before.
        self.head_appear = _head(self.c_app) if (self.c_app > 0 and not self.uniform) else None
        self.head_shared = (_head(self.c_shape + self.c_app)
                            if self.shared_free_head else None)
        self.head_struct_global = (
            _head(self.compact_global_dim) if self.structured_local_code else None)
        self.head_struct_local = (
            MLP([d, d, self.compact_local_dim], hidden_norm=True,
                leaky=True, dropout=cfg.dropout)
            if self.structured_local_code else None)
        if self.head_shared is not None:
            # These legacy split heads are structurally absent in fix3.  Keeping
            # frozen modules avoids invasive checkpoint/layout special cases.
            for module in (self.head_shape, self.head_appear):
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad_(False)
        if self.structured_local_code:
            # F3D replaces the single shared head as well: leaving it trainable
            # would create another silent module that is never in the graph.
            for module in (self.head_shared, self.head_shape, self.head_appear):
                if module is not None:
                    for p in module.parameters():
                        p.requires_grad_(False)
        self.head_logvar = (
            nn.Linear(mid, self.c_shape if not self.uniform else self.per_group)
            if cfg.latent_mode == "vae"
            else None
        )
        # Optional wide mid on the compact spatial grid, then bottleneck to C.
        # CRITICAL: residual must NOT touch centroid/occupancy channels. Those are
        # semantic anchors the decompressor reads literally; an unstructured
        # residual on all 32ch (early wide runs) destroyed pose and made codec
        # eval look randomly broken while z_l1 still fell.
        self.wide_ch = int(cfg.compress_wide_channels)
        self.compact_ch = int(cfg.compact_latent_channels)
        n_wide = max(0, int(cfg.compress_wide_layers))
        if self.wide_ch > 0 and n_wide > 0:
            self.to_wide = nn.Conv2d(self.compact_ch, self.wide_ch, kernel_size=1)
            self.wide_blocks = nn.Sequential(*[ResBlock2d(self.wide_ch) for _ in range(n_wide)])
            self.to_compact = nn.Conv2d(self.wide_ch, self.compact_ch, kernel_size=1)
            nn.init.zeros_(self.to_compact.weight)
            nn.init.zeros_(self.to_compact.bias)
            shape_mask = torch.zeros(self.compact_ch)
            if not self.uniform:
                for m in range(self.M):
                    base = m * self.per_group
                    shape_mask[base + self.c_cen + self.c_occ : base + self.per_group] = 1.0
            else:
                shape_mask[:] = 1.0
            self.register_buffer("wide_shape_mask", shape_mask.view(1, -1, 1, 1), persistent=False)
        else:
            self.to_wide = None
            self.wide_blocks = None
            self.to_compact = None
            self.wide_shape_mask = None
        self._init_heads()

    @staticmethod
    def _last_linear(head):
        """The output layer of a head, whether it is a Linear or a staged MLP.

        The initialisation contract below is about what the head EMITS, so it has
        to land on the last layer. Reading `.weight` off the module worked only
        while every head was a bare Linear; with staged heads that silently became
        an AttributeError, and had the heads been Sequential instead it would have
        silently initialised the wrong layer.
        """
        if isinstance(head, nn.Linear):
            return head
        return [q for q in head.modules() if isinstance(q, nn.Linear)][-1]

    def _init_heads(self) -> None:
        heads = [self.head_centroid]
        if self.head_occ is not None:
            heads.append(self.head_occ)
        for head in heads:
            lin = self._last_linear(head)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # Start the shape channels inside the std band rather than near zero, so
        # the floor hinge does not have to lift them by two orders of magnitude
        # while everything downstream is trying to fit them.
        lin = self._last_linear(self.head_shared if self.head_shared is not None else self.head_shape)
        nn.init.normal_(lin.weight, std=0.06)
        nn.init.zeros_(lin.bias)
        if self.head_appear is not None and self.head_shared is None:
            lin = self._last_linear(self.head_appear)
            nn.init.normal_(lin.weight, std=0.06)
            nn.init.zeros_(lin.bias)
        if self.head_logvar is not None:
            lin = self._last_linear(self.head_logvar)
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, -6.0)
        if self.structured_local_code:
            for head in (self.head_struct_global, self.head_struct_local):
                lin = self._last_linear(head)
                nn.init.normal_(lin.weight, std=0.06)
                nn.init.zeros_(lin.bias)

    # ------------------------------------------------------------------
    def map_to_tokens(self, z_map: torch.Tensor) -> torch.Tensor:
        b, c, h, w = z_map.shape
        cells = z_map.reshape(b, c, h * w).transpose(1, 2)
        used = self.num_groups * self.T
        return cells[:, :used].reshape(b, self.num_groups, self.T, c)

    def _pad_groups(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[1] == self.slots:
            return t
        pad = self.slots - t.shape[1]
        return F.pad(t, [0, 0] * (t.dim() - 2) + [0, pad])

    def forward(self, z_map: torch.Tensor, anchors: Dict[str, torch.Tensor]):
        b = z_map.shape[0]
        d = int(self.cfg.model_dim)
        tokens = self._pad_groups(self.map_to_tokens(z_map))  # (B, slots, T, C)
        centroid = self._pad_groups(anchors["centroid"])  # (B, slots, 3)
        extent = self._pad_groups(anchors["extent"])
        count = self._pad_groups(anchors["count"].unsqueeze(-1)).squeeze(-1)
        valid = self._pad_groups(anchors["valid"].unsqueeze(-1)).squeeze(-1)

        # Put the compressor's input in units of the group extent, the same units
        # encode_scale's docstring already demands for "every head that predicts a
        # position within a group" -- the decoder does this (direct * scale), the
        # encoder never did. Group radii span 854x within a single scene, so in
        # absolute units the leading direction of the packed vectors across groups
        # is simply "how big is this group" and the shape sits four orders of
        # magnitude below it. Measured effective rank of the pack, width 192:
        #     (xyz - centroid)                      9.0    <- what the model saw
        #     (xyz - centroid) / group extent      26.9
        # and the collapse propagates straight through: pack 1.01 -> token_embed
        # 1.02 -> after intra-attention 4.60 -> after pooling 4.26 -> z_compact 5.41.
        # Lossless: the extent is already carried in its own anchored channel.
        # ...but ONLY when the pack's first 192 numbers really are xyz. With a
        # learned pooler the patch is a code with no metric meaning, and dividing
        # an arbitrary slice of it by a quantity that spans 854x across groups
        # injects exactly the scale variation this normalisation exists to remove.
        # The condition is the same one that decides whether the pooler is built.
        if not self.pooled_input:
            nxyz_in = self.G * 3
            s_in = decode_scale(encode_scale(extent)).unsqueeze(-1).to(tokens.dtype)
            flat_in = tokens.reshape(b, self.slots, self.T * self.C)
            tokens = torch.cat(
                [flat_in[..., :nxyz_in] / s_in.clamp(min=1e-5), flat_in[..., nxyz_in:]], dim=-1
            ).reshape(b, self.slots, self.T, self.C)

        x = self.token_embed(tokens).reshape(b, self.cells, self.M * self.T, d)
        x = x + self.token_pos.to(x.dtype)
        tok_valid = (
            valid.reshape(b, self.cells, self.M, 1).expand(b, self.cells, self.M, self.T).reshape(b, self.cells, -1)
        )
        if len(self.intra_blocks) > 0:
            h = x.reshape(b * self.cells, self.M * self.T, d)
            kpm = tok_valid.reshape(b * self.cells, -1) <= 0.5
            for block in self.intra_blocks:
                h = block(h, key_padding_mask=kpm)
            x = h.reshape(b, self.cells, self.M * self.T, d)

        # F3D takes K separate summaries here, before the historical T*d -> d
        # merge.  Each query therefore gets its own direct route from the encoder
        # token set into a fixed slice of z_compact; the decoder never has to
        # hallucinate K tokens back from one already-collapsed vector.
        structured_ctx = None
        if self.structured_pool is not None:
            hx = x.reshape(b, self.cells, self.M, self.T, d).reshape(-1, self.T, d)
            q = self.structured_queries.to(hx.dtype).expand(hx.shape[0], -1, -1)
            kpm = (tok_valid.reshape(b, self.cells, self.M, self.T)
                   .reshape(-1, self.T) <= 0.5)
            structured_ctx = self.structured_pool(q, hx, key_padding_mask=kpm)
            structured_ctx = structured_ctx.reshape(
                b, self.cells, self.M, self.compact_local_tokens, d)

        # Concat, not mean. The M axis is already merged by concat (cell_mix), and a
        # mean over the T tokens throws away 8*448 -> 448 with no learned mixing at
        # exactly the point where the group's whole pack has to survive.
        if self.merge_attn is not None:
            h = x.reshape(b * self.cells * self.M, self.T, d)
            q = self.merge_q.to(h.dtype).expand(h.shape[0], 1, d)
            group_vec = self.merge_attn(q, h)[:, 0].reshape(b, self.cells, self.M, d)
        else:
            group_vec = self.token_merge(x.reshape(b, self.cells, self.M, self.T * d))
        cell_vec = self.cell_mix(group_vec.reshape(b, self.cells, self.M * d))
        cell_valid = valid.reshape(b, self.cells, self.M).amax(dim=-1)
        cell_vec = self.window_stack(self.window_blocks, cell_vec, valid=cell_valid)

        anchor_feat = torch.cat(
            [
                self.anchor_pe(centroid.reshape(b, self.cells, self.M, 3)),
                extent.reshape(b, self.cells, self.M, 3),
                (count / float(self.G)).reshape(b, self.cells, self.M, 1),
            ],
            dim=-1,
        )
        gfeat = self.group_head(
            torch.cat(
                [group_vec, cell_vec.unsqueeze(2).expand(b, self.cells, self.M, d), anchor_feat], dim=-1
            )
        )
        mid = self.mid(gfeat)
        # Scene-level densify/prune signal (log N, fill fraction).
        n_used = count.reshape(b, -1)[:, : self.num_groups].sum(dim=-1).clamp(min=0.0)
        n_max = float(self.cfg.max_points)
        gstat = torch.stack(
            [
                torch.log1p(n_used) / math.log1p(n_max),
                (n_used / n_max).clamp(0.0, 1.0),
            ],
            dim=-1,
        ).to(dtype=mid.dtype)
        mid = mid + self.global_to_mid(gstat).view(b, 1, 1, -1)

        if self.uniform:
            pg = self.head_shape(mid)
            kl = pg.new_tensor(0.0)
            if self.head_logvar is not None:
                pg, kl = self._reparam(pg, self.head_logvar(mid))
        else:
            cen_res = self.head_centroid(mid)
            if self.structured_local_code:
                global_code = self.head_struct_global(mid)
                # Local codes must not see the collapsed group mid. Concatenating
                # it made the four queries copy one vector (cmem 0.034 -> 0.009).
                local_code = self.head_struct_local(structured_ctx)
                shared = torch.cat([
                    global_code, local_code.flatten(-2)
                ], dim=-1)
            else:
                local_code = None
                shared = self.head_shared(mid) if self.head_shared is not None else None
            shape = shared[..., :self.c_shape] if shared is not None else self.head_shape(mid)
            kl = shape.new_tensor(0.0)
            if self.head_logvar is not None:
                shape, kl = self._reparam(shape, self.head_logvar(mid))
            ext = extent.reshape(b, self.cells, self.M, 3)
            u_scale = encode_scale(ext).unsqueeze(-1)
            s = decode_scale(u_scale)
            cen = centroid.reshape(b, self.cells, self.M, 3) + torch.tanh(cen_res[..., :3]) * s * CENTROID_DELTA
            # channel 3 carries the group scale; anything beyond that stays free
            cen = torch.cat([cen, u_scale, cen_res[..., 4:]], dim=-1)
            if self.head_occ is not None and self.c_occ > 0:
                occ_res = self.head_occ(mid)
                occ_anchor = (2.0 * count / float(self.G) - 1.0).reshape(b, self.cells, self.M, 1)
                occ = torch.cat([occ_anchor + torch.tanh(occ_res[..., :1]) * 0.1, occ_res[..., 1:]], dim=-1)
                pg = torch.cat([cen, occ, shape], dim=-1)
            else:
                pg = torch.cat([cen, shape], dim=-1)
            if self.c_app > 0:
                appearance = (shared[..., self.c_shape:]
                              if shared is not None else self.head_appear(mid))
                pg = torch.cat([pg, appearance], dim=-1)

        z_cells = pg.reshape(b, self.cells, self.M * self.per_group)
        z_grid = z_cells.new_zeros(b, self.grid_h * self.grid_w, z_cells.shape[-1])
        z_grid[:, self.cell_to_grid] = z_cells
        z_compact = z_grid.transpose(1, 2).reshape(b, -1, self.grid_h, self.grid_w)
        # DC-AE-style staged bottleneck on *shape* channels only. Identity at init.
        if self.to_wide is not None:
            z_mid = self.wide_blocks(self.to_wide(z_compact))
            z_compact = z_compact + self.to_compact(z_mid) * self.wide_shape_mask.to(dtype=z_compact.dtype)
        empty_frac = 1.0 - (n_used / n_max).clamp(0.0, 1.0)
        aux = {
            "kl": kl,
            "cell_valid": cell_valid,
            "group_valid": valid,
            "n_used": n_used,
            "empty_frac": empty_frac,
            "global_cond": gstat.detach(),
        }
        if self.structured_local_code:
            aux["compact_local_std"] = local_code.float().std(dim=-2).mean()
        return z_compact, aux

    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor):
        logvar = logvar.clamp(-12.0, 4.0)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        if self.training:
            mu = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return mu, kl


class StagedDecompressor(nn.Module, _GridMixin):
    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay = patch_layout(cfg)
        bud = channel_budget(cfg)
        self.G = lay["group_size"]
        self.T = lay["tokens_per_group"]
        self.C = int(cfg.latent_channels)
        self.M = lay["merge"]
        self.num_groups = lay["num_groups"]
        self.cells = lay["compact_cells"]
        self.slots = self.cells * self.M
        self.patch_dim = lay["patch_dim"]
        self.token_dim = lay["token_dim"]
        self.raw_cells = lay["raw_cells"]
        self.per_group = bud["per_group"]
        self.c_cen, self.c_occ, self.c_shape = bud["centroid"], bud["occupancy"], bud["shape"]
        self.c_app = int(bud.get("appearance", 0))
        self.uniform = bool(cfg.uniform_budget)
        d = int(cfg.model_dim)
        self._register_grid(cfg, self.cells)

        self.anchor_pe = FourierFeatures(3, cfg.num_freqs_slot, include_input=True)
        in_dim = self.per_group + (self.anchor_pe.out_dim if not self.uniform else 0)
        # in_norm: the latent scale moves a lot early on (the std hinge lifts the
        # shape channels off ~0 within a few hundred steps) and the trunk must not
        # care. The exact centroid/occupancy path is read separately, so nothing
        # that has to stay metric passes through here.
        self.group_embed = MLP([in_dim, d, d], last_act=True, dropout=cfg.dropout, in_norm=True, hidden_norm=True)
        self.cell_mix = MLP([d * self.M, d, d], last_act=True, dropout=cfg.dropout)
        self.window_blocks = nn.ModuleList(
            [
                WindowSelfAttention(
                    d,
                    cfg.heads,
                    (self.grid_h, self.grid_w),
                    window=cfg.compress_window,
                    dropout=cfg.dropout,
                    shift=bool(i % 2),
                )
                for i in range(max(0, int(cfg.decompress_window_layers)))
            ]
        )
        self.intra_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
                for _ in range(max(0, int(cfg.decompress_intra_layers)))
            ]
        )
        # Width of the shared shape -> offset dictionary. `4 * G` was sized when
        # the only evidence was "G=64 needs more than the 128 that suited G=32";
        # nothing has measured where it saturates. Q16kD ran 64000 steps and its
        # `template_erank` -- the count of DISTINCT group shapes the decoder
        # actually emits -- was still climbing at the end, 28.4 against the ground
        # truth's 36.3, while every other geometry metric had gone flat. The
        # collapse the metric names lives in this map, not in the latent: the
        # shape channels can be fully diverse and this still repeats the same
        # stroke, because 1024 cells share one dictionary.
        hid = int(getattr(cfg, "shape_xyz_hidden", 0) or 0) or max(128, 4 * self.G)
        self.folding = bool(getattr(cfg, "folding_decode", False))
        self.fold_frame = int(getattr(cfg, "folding_frame_channels", 6))
        self.token_out = residual_head([d * 2, d, self.T * self.C])
        if self.folding:
            # Folding generates xyz from a shared template; a zero last layer keeps
            # that template from being cancelled at init.
            nn.init.zeros_(self.token_out.net[-1].weight)
            nn.init.zeros_(self.token_out.net[-1].bias)
        else:
            # Invert path: token_out *is* z_raw_hat. Zero-init made dL/d(z_compact)
            # vanish for xyz, so Chamfer was solved by sitting on the centroid.
            nn.init.normal_(self.token_out.net[-1].weight, std=1e-3)
            nn.init.zeros_(self.token_out.net[-1].bias)
        # Direct shape -> per-point offset. The deep token_out path alone can stay
        # at zero forever because its gradient w.r.t. the compact shape channels is
        # then also zero (chicken-and-egg with the centroid shortcut). This head
        # gives those channels an unbroken path to the residual loss from step 0.
        shape_in = self.c_shape if not self.uniform else self.per_group
        self.d_ctx = int(cfg.model_dim)
        # Width follows the group size: this head is a shape_in -> G*3 dictionary
        # shared by every group in the scene, so a G=64 layout needs more than the
        # 128 that was sized for G=32.
        if self.folding:
            self.register_buffer("unit_ball", fibonacci_ball_torch(self.G), persistent=False)
            # Both heads read the *whole* shape code. Splitting the channels by
            # hand ("first 6 are the frame") only removes information from each
            # head without saving any: the budget is the code width either way.
            # The heads read the contextualised group token *and* the raw channels.
            #
            # Reading only the raw 28 channels makes the folding path context-free and,
            # worse, leaves the compressor untrainable: with a near-zero last layer the
            # geometry gradient reaching z_compact is dL/d(shape) = W_last^T g, measured
            # at 1e-2 on the compressor against 1e+2 for any latent regulariser, so
            # w_latent_std owned 99.5% of the encode-side gradient and the shape block
            # sat at effective rank ~4 of 28. Concatenating the token gives geometry a
            # full-width (model_dim) path back through the attention stack, and lets the
            # frame and the residual use neighbourhood context at all.
            self.fold_head = MLP([shape_in + self.d_ctx, 128, self.fold_frame], dropout=0.0)
            self.shape_xyz = MLP([shape_in + self.d_ctx, hid, hid, self.G * 3], dropout=cfg.dropout)
            # Near-zero, NOT zero. Both heads read the shape channels, so an exactly
            # zero last layer makes dL/d(shape) = W_last^T g = 0 and the compressor
            # and encoder receive *no* geometry gradient at all -- measured: z_l1
            # gave |g| 1.9e-1 on the decompressor and exactly 0.0 on all 89
            # compressor tensors, leaving w_latent_std (a regulariser) responsible
            # for 92% of the encode-side gradient. std 1e-3 keeps the decoder within
            # ~0.1% of the bare template at init while keeping the path alive.
            for head in (self.fold_head, self.shape_xyz):
                nn.init.normal_(head.net[-1].weight, std=1e-3)
                nn.init.zeros_(head.net[-1].bias)
        else:
            self.fold_head = None
            self.shape_xyz = None
        self.register_buffer("slot_index", torch.arange(self.G, dtype=torch.float32), persistent=False)

    def _bounded_residual(self, res: torch.Tensor) -> torch.Tensor:
        """Bound and gate the folding residual.

        Unbounded, the 192-output residual head simply out-races the 6-output
        frame head and reproduces the group's anisotropy itself -- measured at
        step 500 of the first attempt: the decoder started as an exact ball
        (lam2 0.999) and by step 500 had been deformed to lam2 0.243 against
        0.579 in the GT, with the frame head still at its zero init. The
        envelope has to be the cheap path.

        The cap costs nothing: rank-22 oracle chamfer is 0.218 uncapped, 0.217
        at cap 1.2, 0.221 at 0.8, 0.236 at 0.4, against an aligned residual whose
        per-coordinate p90 is 0.881. ``folding_res_gain`` additionally holds the
        residual at zero for the first steps so the frame head trains first.
        """
        cap = float(getattr(self.cfg, "folding_res_cap", 1.0))
        gain = float(getattr(self.cfg, "folding_res_gain", 1.0))
        if gain <= 0.0:
            return res * 0.0
        if cap > 0.0:
            res = torch.tanh(res / cap) * cap
        return res * gain

    def _shape_to_offsets(self, shape: torch.Tensor, ctx: Optional[torch.Tensor] = None,
                          count: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, slots, c_shape) -> (B, slots, G, 3) in units of group extent."""
        if not self.folding:
            raise RuntimeError("shape_xyz generator is disabled when folding_decode is off")
        b, s, _ = shape.shape
        h = shape if ctx is None else torch.cat([shape, ctx], dim=-1)
        frame = self.fold_head(h)
        # Positive per-axis scales, geometric-mean-normalised so the analytic
        # group scale channel still owns the overall size.
        # Per-axis scales, geometric-mean-normalised so the analytic group scale
        # channel still owns the overall size.
        #
        # The normalisation fixes the PRODUCT but not the RATIO, so one axis can
        # run away as long as another shrinks. Measured on the J1 checkpoint:
        # aniso spanned 0.027 to 1411.5, and that stage inflated the offset
        # magnitude 0.620 -> 7.976 while DROPPING effective rank 5.80 -> 4.24. A
        # handful of needle-shaped groups dominate, which is exactly the huge
        # smeared blobs in the render. Bound the log-ratio: at cap 1.5 a single
        # axis may still stretch e^1.5 = 4.5x, which covers the anisotropy the
        # data actually has (GT lam2 0.757), without admitting 1411x.
        cap = float(getattr(self.cfg, "folding_aniso_log_cap", 0.0))
        if cap > 0.0:
            la = frame[..., :3]
            la = la - la.mean(dim=-1, keepdim=True)
            la = torch.tanh(la / cap) * cap
            aniso = torch.exp(la)
        else:
            aniso = F.softplus(frame[..., :3]) + 1e-3
            aniso = aniso / aniso.prod(dim=-1, keepdim=True).clamp(min=1e-6).pow(1.0 / 3.0)
        # Residual lives in the *whitened local* frame, i.e. it is added before
        # the scale and the rotation. data.py orders each group's points by an
        # optimal assignment to the same template in the same frame, so slot i
        # of the target really is the point this slot is meant to produce.
        local = self.unit_ball.view(1, 1, self.G, 3).to(shape.dtype)
        local = local + self._bounded_residual(self.shape_xyz(h).reshape(b, s, self.G, 3))
        # Mean-centre before the frame. The full 256-ball is centred, but a live
        # prefix is not. Partial cells (the common case on late snapshots) must
        # centre the predicted live subset, otherwise the decoder spends capacity
        # undoing a template bias. GT masks are not used here: count comes from
        # compact so latent-only decode stays valid.
        if (bool(getattr(self.cfg, "count_aware_template", False))
                and count is not None):
            k = count.reshape(b, s).clamp(0.0, float(self.G))
            idx = torch.arange(self.G, device=local.device, dtype=k.dtype)
            live = (idx.view(1, 1, self.G) < k.unsqueeze(-1)).to(local.dtype)
            denom = live.sum(dim=2, keepdim=True).clamp(min=1.0)
            mean = (local * live.unsqueeze(-1)).sum(dim=2) / denom
            local = local - mean.unsqueeze(2)
        else:
            local = local - local.mean(dim=2, keepdim=True)
        local = local * aniso.unsqueeze(2)
        rot = None
        if self.fold_frame >= 6:
            rot = axis_angle_to_matrix(frame[..., 3:6])
            local = torch.einsum("bsij,bsgj->bsgi", rot, local)
        # Keep the basis parameters around: the student decodes in the *same*
        # basis, so distilling frame and pre-scale offsets transfers the teacher's
        # actual solution rather than only its final points. `rot` is the cell
        # frame AttributeDecoder uses when attr_frame_needles is on.
        self._last_basis = {"frame": frame, "local": local, "rot": rot, "aniso": aniso}
        return local

    def _shortcut_tokens(self, centroid: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """Non-parametric token prior.

        Absolute-pack mode: broadcast group centroid into xyz (the old free path).
        Residual-pack mode: xyz prior is zeros; only occupancy/mask is free. That
        way ``z_l1`` and the codec coarse pose cannot be solved without shape.
        """
        b, slots = centroid.shape[0], centroid.shape[1]
        if bool(self.cfg.residual_pack):
            xyz = centroid.new_zeros(b, slots, self.G * 3)
        else:
            xyz = centroid.unsqueeze(2).expand(b, slots, self.G, 3).reshape(b, slots, self.G * 3)
        mask = (count.unsqueeze(-1) - self.slot_index.to(centroid.dtype)).clamp(0.0, 1.0)
        patch = torch.cat([xyz, mask], dim=-1)
        if self.token_dim > self.patch_dim:
            patch = F.pad(patch, (0, self.token_dim - self.patch_dim))
        return patch

    def forward(self, z_compact: torch.Tensor, shortcut_alpha: Optional[float] = None):
        b, c, h, w = z_compact.shape
        d = int(self.cfg.model_dim)
        grid = z_compact.reshape(b, c, h * w).transpose(1, 2)
        z_cells = grid[:, self.cell_to_grid]  # (B, cells, C)
        pg = z_cells.reshape(b, self.cells, self.M, self.per_group)

        if self.uniform:
            centroid = pg.new_zeros(b, self.slots, 3)
            count = pg.new_full((b, self.slots), float(self.G))
            scale = pg.new_full((b, self.slots, 1), SCALE_REF)
            feat = pg
        else:
            centroid = pg[..., 0 : 3].reshape(b, self.slots, 3)
            scale = decode_scale(pg[..., 3 : 4]).reshape(b, self.slots, 1)
            if self.c_occ > 0:
                occ0 = pg[..., self.c_cen : self.c_cen + 1].reshape(b, self.slots)
                count = ((occ0 + 1.0) * 0.5 * float(self.G)).clamp(0.0, float(self.G))
            else:
                # No per-group occ channel: empty groups keep near-min encode_scale
                # (extent≈0). Live groups get full slot budget; partial fills are
                # carried by the learned mask in z_raw / presence.
                alive = (scale.squeeze(-1) > (SCALE_REF * 0.05)).to(dtype=scale.dtype)
                count = alive * float(self.G)
            feat = torch.cat([pg, self.anchor_pe(pg[..., 0:3])], dim=-1)

        g = self.group_embed(feat)
        cell_vec = self.cell_mix(g.reshape(b, self.cells, self.M * d))
        cell_vec = self.window_stack(self.window_blocks, cell_vec)
        if len(self.intra_blocks) > 0:
            hh = torch.cat([cell_vec.unsqueeze(2), g], dim=2).reshape(b * self.cells, self.M + 1, d)
            for block in self.intra_blocks:
                hh = block(hh)
            hh = hh.reshape(b, self.cells, self.M + 1, d)
            cell_vec = hh[:, :, 0]
            g = hh[:, :, 1:]

        deep = self.token_out(
            torch.cat([g, cell_vec.unsqueeze(2).expand(b, self.cells, self.M, d)], dim=-1)
        ).reshape(b, self.slots, self.token_dim)
        nxyz = self.G * 3
        if self.uniform:
            shape = pg.reshape(b, self.slots, self.per_group)
        else:
            s0 = self.c_cen + self.c_occ
            shape = pg[..., s0 : s0 + self.c_shape].reshape(b, self.slots, self.c_shape)
        if self.folding:
            # Offsets in units of the group extent, then scaled to world units.
            ctx_tok = g.reshape(b, self.slots, d)
            direct = self._shape_to_offsets(shape, ctx_tok, count=count)
            deep_xyz = deep[..., :nxyz].reshape(b, self.slots, self.G, 3)
            deep_xyz = self._bounded_residual(deep_xyz)
            direct_xyz_full = (direct * scale.unsqueeze(-1)).reshape(b, self.slots, nxyz)
            learned_xyz_full = ((direct + deep_xyz) * scale.unsqueeze(-1)).reshape(b, self.slots, nxyz)
            learned = torch.cat([learned_xyz_full, deep[..., nxyz:]], dim=-1)
        else:
            # Invert the identity pack. z_raw xyz is (point - centroid) in scene
            # units. The folding/shape_xyz path instead *generated* 64 offsets from
            # a shared head and multiplied by group extent, so xyz was not an
            # inverse of the pack. Combined with a zero-init token_out, Chamfer's
            # cheapest answer was to leave every point on the cell centroid.
            learned_xyz_full = deep[..., :nxyz]
            direct_xyz_full = learned_xyz_full
            learned = deep
        shortcut = self._shortcut_tokens(centroid, count)
        if shortcut_alpha is None:
            alpha = float(self.cfg.shortcut_alpha_eval if not self.training else self.cfg.shortcut_alpha)
        else:
            alpha = float(shortcut_alpha)
        alpha = min(max(alpha, 0.0), 1.0)
        # Critical: alpha=0 removes the free centroid reconstruction path.
        tok = learned + alpha * shortcut
        learned_xyz = learned_xyz_full[:, : self.num_groups]
        direct_xyz = direct_xyz_full[:, : self.num_groups]
        # Diagnostic: learned magnitude vs group extent (not vs shortcut, which can
        # be ~0 under residual_pack and would make a ratio meaningless / gameable).
        ext = scale[:, : self.num_groups].detach().abs().mean().clamp(min=1e-8)
        basis = getattr(self, "_last_basis", None)
        res_ratio = learned_xyz.detach().abs().mean() / ext
        # Fraction of the learned offset that comes from the shape channels alone.
        # Collapses towards 0 when the deep path serves the residual from centroid
        # context and the shape code carries nothing.
        direct_frac = direct_xyz.detach().abs().mean() / learned_xyz.detach().abs().mean().clamp(min=1e-8)
        # Differentiable twin of the above, measured against the group extent rather
        # than against ``learned``. A floor on *this* cannot be satisfied by shrinking
        # the deep path, which a floor on direct_frac could. Mirrors res_ratio_floor.
        direct_ratio = direct_xyz.abs().mean() / ext

        tok = tok[:, : self.num_groups]
        z_used = tok.reshape(b, self.num_groups * self.T, self.C)
        if self.raw_cells > z_used.shape[1]:
            z_used = F.pad(z_used, (0, 0, 0, self.raw_cells - z_used.shape[1]))
        rh, rw = int(self.cfg.latent_hw[0]), int(self.cfg.latent_hw[1])
        z_raw_hat = z_used.transpose(1, 2).reshape(b, self.C, rh, rw)
        # The appearance suffix of z_compact, published untouched so the attribute
        # heads read the latent directly rather than whatever survived the geometry
        # path. Without this the heads can only ever be a function of shape, which
        # is measurably useless: predicting attributes from geometry alone scores a
        # held-out R^2 of -0.99 on opacity.
        appearance = None
        if self.c_app > 0 and not self.uniform:
            a0 = self.c_cen + self.c_occ + self.c_shape
            appearance = pg[..., a0 : a0 + self.c_app].reshape(b, self.slots, self.c_app)
            appearance = appearance[:, : self.num_groups]


        # A SEPARATE, wider code for the attribute decoder: the cell's shape
        # channels concatenated in front of its appearance channels. Published
        # alongside `appearance` rather than replacing it, because CodecDecoder
        # also reads `appearance` and was built at that width -- and it does not
        # need this, it already sees shape through z_raw_hat.
        #
        # Why it exists: with the position input detached, the attribute/render
        # objective puts EXACTLY 0.000e+00 gradient on the 19 shape channels
        # (measured). So shape is trained by the point-space terms and appearance
        # by the render -- two disjoint halves of every cell under two different
        # objectives -- while the channel-coalition ladder says xyz/scale/opacity/
        # colour only pay off when they move together. This carries render
        # gradient into shape through the ATTRIBUTE path, which is well
        # conditioned, instead of through position transport, which is not.
        attr_code = appearance
        if appearance is not None and bool(getattr(self.cfg, "attr_read_shape", False)):
            sh = pg[..., s0 : s0 + self.c_shape].reshape(b, self.slots, self.c_shape)
            attr_code = torch.cat([sh[:, : self.num_groups], appearance], dim=-1)

        # Joint decoder code: all learned non-anchor channels, without assigning
        # disjoint geometry/appearance semantics.  The explicit centroid/extent/
        # occupancy anchors remain outside this block.
        shared_code = None
        if not self.uniform:
            shared_code = pg[..., s0:].reshape(b, self.slots, self.per_group - s0)
            shared_code = shared_code[:, : self.num_groups]

        ctx = {
            "cell_vec": cell_vec,
            "attr_code": attr_code,
            "shared_code": shared_code,
            "group_vec": g,
            "appearance": appearance,
            "centroid": centroid[:, : self.num_groups],
            "count": count[:, : self.num_groups],
            "scale": scale[:, : self.num_groups],
            "res_ratio": res_ratio,
            "direct_frac": direct_frac,
            "direct_ratio": direct_ratio,
            "learned_xyz": learned_xyz,
            "direct_xyz": direct_xyz,
            "shortcut_alpha": tok.new_tensor(alpha),
        }
        if basis is not None:
            ctx["fold_frame"] = basis["frame"][:, : self.num_groups]
            ctx["fold_local"] = basis["local"][:, : self.num_groups]
            if basis.get("rot") is not None:
                ctx["fold_rot"] = basis["rot"][:, : self.num_groups]
            if basis.get("aniso") is not None:
                ctx["fold_aniso"] = basis["aniso"][:, : self.num_groups]
        return z_raw_hat, ctx
