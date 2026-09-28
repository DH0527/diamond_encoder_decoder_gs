"""Attribute decoder: everything a Gaussian is except where it sits.

Separate from the geometry decoder on purpose. The attribute heads used to hang
off the geometry decoder's refine token ``h``, which meant the render loss --
even with the position output detached -- still backpropagated through all ten
of the geometry stack's attention layers. That coupling was never intended, and
it is the one gradient path the measurements say should not exist: the render
gradient is badly conditioned for positions (cosine 0.13 with the true
correction, and only 24% of points receive any gradient at all) while being well
conditioned for appearance, which changes the pixels a Gaussian already covers.

So this module takes only detached inputs from the geometry side. Nothing it
learns can move the geometry decoder's weights, which means:

  * the render weight can be raised as far as the attributes need without
    risking the positions,
  * geometry and appearance can be trained in separate stages -- converge
    geometry, freeze it, then fit appearance -- which removes the
    reconstruction-vs-equivalence tug of war rather than balancing it by hand,
  * a failure is attributable to one module.

The one exception is deliberate: a **bounded position nudge**. Image gradients
are informative only within a Gaussian's own screen footprint, and past that
they point at the wrong place. Measured on this data the median Gaussian
projects to 1.40 px while one group radius is 12.0 px, so the useful range is
about 0.12 group extents. The nudge is tanh-capped at ``attr_nudge_cap`` (0.15
by default, i.e. a little over one footprint) and rides on top of the detached
geometry output, so the render loss can make the sub-footprint corrections it is
actually good at without being able to attempt transport it is bad at.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Can3TokConfig, channel_budget, patch_layout
from .layers import MLP, FourierFeatures, SelfAttentionBlock


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Hamilton product, wxyz, ``(..., 4)``."""
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def rotmat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3)`` -> ``(..., 4)`` wxyz. Shepperd; identity -> (1,0,0,0)."""
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    t0 = 1.0 + m00 + m11 + m22
    q0 = torch.stack([t0, m21 - m12, m02 - m20, m10 - m01], dim=-1)
    t1 = 1.0 + m00 - m11 - m22
    q1 = torch.stack([m21 - m12, t1, m01 + m10, m02 + m20], dim=-1)
    t2 = 1.0 - m00 + m11 - m22
    q2 = torch.stack([m02 - m20, m01 + m10, t2, m12 + m21], dim=-1)
    t3 = 1.0 - m00 - m11 + m22
    q3 = torch.stack([m10 - m01, m02 + m20, m12 + m21, t3], dim=-1)
    qs = torch.stack([q0, q1, q2, q3], dim=-2)
    ts = torch.stack([t0, t1, t2, t3], dim=-1)
    one_hot = F.one_hot(ts.argmax(dim=-1), 4).to(dtype=qs.dtype).unsqueeze(-1)
    return F.normalize((qs * one_hot).sum(dim=-2), dim=-1)


class AttributeDecoder(nn.Module):
    def __init__(self, cfg: Can3TokConfig):
        super().__init__()
        self.cfg = cfg
        lay = patch_layout(cfg)
        bud = channel_budget(cfg)
        self.G = int(lay["group_size"])
        self.num_groups = int(lay["num_groups"])
        self.c_app = int(bud.get("appearance", 0))
        # The code this module reads. With attr_read_shape the decompressor
        # prepends the cell's shape channels, so every projection below has to be
        # built at the wider width -- and the render gradient then reaches shape.
        # Shape channels reach this module through a SEPARATE, zero-initialised
        # projection that is ADDED to the appearance one -- not by widening
        # to_code_tok's input. Widening forces to_code_tok to be reinitialised,
        # which resets the whole appearance->slot conditioning path: measured on
        # a previous attempt, held-out PSNR fell from 15.39 to 9.55 at step 100
        # and was still 12.11 at step 250, so the run measured "shape connected"
        # and "appearance path relearned" mixed together.
        #
        # The additive zero-init branch is function-preserving: at step 0 the model
        # is bit-identical to its checkpoint, and the branch's own weight still
        # gets gradient (d out / d W = shape, non-zero), so it opens by itself.
        # This is NOT the zero-product saddle this codebase hit before -- that was
        # tanh(gate) * residual(feat) with BOTH factors zero.
        #
        # Why shape at all: with the position input detached, the render objective
        # puts EXACTLY 0.000e+00 gradient on the 19 shape channels (measured), so
        # shape is trained by point-space losses and appearance by the render --
        # two disjoint halves of every cell under two different objectives. And a
        # Gaussian's proper scale depends on its group's SHAPE (a thin plate needs
        # one small axis), which this module could not see.
        self.c_shape_read = (int(bud.get("shape", 0))
                             if bool(getattr(cfg, "attr_read_shape", False)) else 0)
        self.c_code = self.c_shape_read + self.c_app
        self.chunk = max(int(cfg.patch_chunk), 1)
        d = int(cfg.attr_decoder_dim)

        self.xyz_pe = FourierFeatures(3, cfg.num_freqs_xyz, include_input=True)
        # The same encoding again, on group-local coordinates. Positions arrive
        # scene-normalised and a group's radius is ~0.5% of the scene, so the
        # encoding above is mostly a group *address*: measured, its within-group
        # variation is 0.27 of its between-group variation. The attributes it has
        # to explain are the other way round -- 52-78% of their variance is inside
        # a group. Recentring on the group and dividing by its radius turns that
        # ratio into 3.52, a 13x gain in the resolution that matters, and this is
        # the decoder's only per-point input so nothing else can supply it.
        # Added rather than substituted: the scene-level encoding still says which
        # part of the scene this is, which is what makes appearance predictable at
        # all.
        self.use_local_pe = bool(getattr(cfg, "attr_local_pe", True))
        self.loc_pe = (FourierFeatures(3, cfg.num_freqs_xyz, include_input=True)
                       if self.use_local_pe else None)
        pe_dim = self.xyz_pe.out_dim + (self.loc_pe.out_dim if self.loc_pe else 0)
        self._pe_dim = pe_dim
        # Per-slot basis, the appearance analogue of the geometry decoder's fixed
        # fibonacci template: one appearance code per group has to explain
        # variation *within* the group, and 52-78% of the attribute variance lives
        # there. A learned per-slot embedding modulated by the group code gives
        # that variation somewhere to come from.
        self.slot_emb = nn.Embedding(self.G, d)
        # How the group code reaches the slots. FiLM was the first choice -- it is
        # the standard way to inject a global vector into a set of tokens -- but
        # `gam` and `bet` broadcast across all 64 slots, so the code can only
        # rescale and shift a basis that is otherwise identical in every scene. It
        # cannot say "slot 5 is red, slot 6 is blue", and 60% of the attribute
        # variance is exactly that within-group part.
        #
        # Measured as an autodecoder, 8 scenes through one shared network, same
        # width/depth/optimiser, only this path changed (tools/oracle_conditioning.py):
        #     none      0.3021 nrmse   0.3830 within-group
        #     film      0.3055          0.3876     <- worse than ignoring the code
        #     concat    0.2983          0.3759
        #     xattn     0.2783          0.3547     <- 8.9% better than film
        #     film+cat  0.2969          0.3741
        # Same ordering on a single scene. FiLM scoring below `none` is the part
        # that decided it: the mechanism was not merely weak, it was costing more
        # in optimisation than the code was worth, which would have made the 16
        # appearance channels dead weight.
        self.cond = str(getattr(cfg, "attr_cond", "xattn"))
        self.n_code_tok = 4
        # How many neighbouring groups' codes this decoder may read, each side.
        #
        # Why this exists. The pack partitions the scene into hard 64-point Morton
        # groups and gives each one 16 appearance channels, so a point's appearance
        # was determined by exactly 16 numbers and nothing else. That puts a hard
        # ceiling on how much a group's appearance can vary *inside* the group:
        # the within-group pattern lives in at most a 16-dimensional space while
        # the ground truth's is 64x3 = 192-dimensional. Measured on the trained
        # model, the within-group share of colour variation is 14.9% for the
        # encoder's own codes and 23.3% for random codes of the same magnitude --
        # the encoder's codes are smoother than noise -- and even codes with 16x
        # the variance cap at 36.5%, against 74.7% for the ground truth. That 36.5%
        # is this hard partition, measured.
        #
        # Can3Tok, which this model is based on, does not partition at all: its
        # learnable canonical queries cross-attend to every Gaussian in the scene,
        # so a point's appearance can draw on the whole latent. That is O(N^2) and
        # impossible at 262144 points, but a *local* window recovers most of it:
        # Morton order keeps consecutive group indices spatially adjacent, which is
        # the same assumption the compressor's own windowed attention already makes.
        self.nbr = int(getattr(cfg, "attr_nbr_window", 0))
        use_film = self.cond in ("film", "film+cat") and self.c_app > 0
        self.film = nn.Linear(self.c_app, 2 * d) if use_film else None
        if self.cond == "xattn" and self.c_app > 0:
            self.to_code_tok = nn.Linear(self.c_app, self.n_code_tok * d)
            self.to_shape_tok = (nn.Linear(self.c_shape_read, self.n_code_tok * d)
                                 if self.c_shape_read > 0 else None)
            if self.to_shape_tok is not None:
                nn.init.zeros_(self.to_shape_tok.weight)
                nn.init.zeros_(self.to_shape_tok.bias)
            # Which neighbour a code came from, so the window is an ordered
            # sequence rather than a bag. Without it the decoder cannot tell its
            # own code from the one two groups away.
            self.nbr_emb = (nn.Parameter(torch.randn(1, 2 * self.nbr + 1, 1, d) * 0.02)
                            if self.nbr > 0 else None)
            self.xattn = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
            self.xnorm = nn.LayerNorm(d)
            # One scalar on the code branch. Measured inside this module, the
            # within-group / between-group std ratio of the slot representation is
            # 1.44 at in_proj's output and 0.27 after the code is added -- the
            # within-group part is not erased, the between-group part explodes
            # 0.364 -> 1.194 because `ct` is identical for all 64 slots. So the
            # group code out-weighs slot identity ~3:1 and the heads see a group,
            # not 64 Gaussians. The consequence is measurable at the output: the
            # within-group log-scale residual is 0.301 against the ground truth's
            # 1.660, and the within-group orientation spread is 3% of the GT's.
            #
            # Init 1.0 keeps the checkpoint's function exactly. It is a degree of
            # freedom, not a prescribed reduction: the optimiser can lower it if
            # slot identity is worth more than code magnitude, which it could not
            # do before.
            self.xattn_gate = nn.Parameter(torch.ones(1))
        else:
            self.to_code_tok = self.xattn = self.xnorm = None
            self.to_shape_tok = None
        self.cat_code = self.cond in ("concat", "film+cat") and self.c_app > 0
        self.in_proj = MLP([d + pe_dim + (self.c_app if self.cat_code else 0), d, d],
                           last_act=True, dropout=cfg.dropout)
        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(d, heads=cfg.heads, dropout=cfg.dropout)
             for _ in range(max(0, int(cfg.attr_decoder_layers)))]
        )

        self.head_scale = nn.Linear(d, 3)
        self.head_rot = nn.Linear(d, 4)
        self.head_opacity = nn.Linear(d, 1)
        self.head_color = nn.Linear(d, 3) if cfg.target_dim > 11 else None
        self.head_sh = nn.Linear(d, int(cfg.sh_dim)) if cfg.sh_dim > 0 else None
        self.head_nudge = nn.Linear(d, 3)
        # log_scale base = scale_base_a * log(group extent) + head_scale.bias.
        # Both are learnable; the initialisation is the measured least-squares fit
        # of a group's median log-scale against its own extent (a = 0.787,
        # b = -2.559, R^2 = 0.759 over 5 snapshots x 4096 groups).
        self.scale_base_a = nn.Parameter(torch.tensor(0.7872))
        self._cap_hit = None
        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.normal_(self.slot_emb.weight, std=0.02)
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if self.xattn is not None:
            # Deliberately NOT zero-initialised. Zeroing out_proj makes the branch
            # a no-op, which also makes d(output)/d(code tokens) exactly zero, so
            # `to_code_tok` receives no gradient at all on the first step and the
            # code path activates only indirectly. The variant that won the
            # measurement used ordinary initialisation; the heads are near-zero
            # anyway, so the branch cannot disturb the output much. Only the bias
            # is zeroed.
            nn.init.zeros_(self.xattn.out_proj.bias)
        nn.init.normal_(self.head_scale.weight, std=1e-3)
        # Measured on the data, not guessed: the log-scale channel has mean -7.58
        # and spans -11.18..-0.70. The old -5.5 started every Gaussian ~8x too
        # large, which the scale anchor then spent its whole budget undoing --
        # it was the loudest attribute term in the gradient audit at 41%.
        if float(getattr(self.cfg, "attr_scale_cap_down", 0.0)) > 0.0:
            # intercept of the group-relative fit, not an absolute log-scale
            nn.init.constant_(self.head_scale.bias, -2.559)
        else:
            nn.init.constant_(self.head_scale.bias, -7.58)
        nn.init.normal_(self.head_rot.weight, std=1e-3)
        nn.init.zeros_(self.head_rot.bias)
        with torch.no_grad():
            self.head_rot.bias[0] = 1.0
        # -2.13 is the measured mean of the opacity channel (alpha 0.106); colour
        # sits at dc=0, which is mid-grey once the SH DC term is converted.
        for h, b in ((self.head_opacity, -2.13), (self.head_color, 0.0), (self.head_sh, 0.0)):
            if h is not None:
                nn.init.normal_(h.weight, std=1e-3)
                nn.init.constant_(h.bias, b)
        # Start with no nudge at all, so early training is exactly the geometry
        # decoder's own answer and the nudge has to earn every bit of movement.
        nn.init.zeros_(self.head_nudge.weight)
        nn.init.zeros_(self.head_nudge.bias)

    def _bounded_log_scale(self, raw: torch.Tensor,
                           group_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Asymmetric tanh band around a GROUP-RELATIVE base, in log units.

        The band has to stay -- unbounded, `exp` lets one Gaussian grow 148x and
        the render loss pushes exactly that way, because enlarging a splat is the
        cheapest way to cover a hole. What was wrong was the band's CENTRE, not
        its width. It was ``head_scale.bias``: one global scalar for a scene whose
        Gaussians span log-scale -18.4 to +0.8. Measured over 3.5M GT values,
        centred at -7.58 with cap 3 the band cannot reach **5.94%** of them
        (5.29% too small, 0.66% too large), and the model duly under-disperses --
        R8 emitted scale at 0.739 of the ground truth's spread.

        Re-centring per group fixes it at the source rather than by widening.
        A group's median log-scale is predicted by its own extent:

            median log_scale  ~  0.787 * log(extent) - 2.559     R^2 = 0.759

        and once that base is removed the residual the head still has to emit is
        small -- within-group p0.1 = -2.54, p99.9 = +2.66, and the base fit's own
        residual is p1 = -1.33, p99 = +1.29. So a +-3 band that was missing 5.94%
        of the data reaches 99.93% of it once it is centred correctly, and the
        caps below are set from those percentiles with margin.

        The asymmetry is deliberately NOT data-driven -- the two tails are nearly
        equal once group-relative. The upper cap is kept tighter than the lower
        one as a safety margin against the render loss's giant-splat shortcut,
        which is the failure the band exists to prevent.
        """
        down = float(getattr(self.cfg, "attr_scale_cap_down", 0.0))
        up = float(getattr(self.cfg, "attr_scale_cap_up", 0.0))
        if down <= 0.0 or up <= 0.0:
            cap = float(getattr(self.cfg, "attr_scale_log_cap", 3.0))
            if cap <= 0:
                return raw
            b = self.head_scale.bias.view(*([1] * (raw.dim() - 1)), -1)
            return b + torch.tanh((raw - b) / cap) * cap

        b = self.head_scale.bias.view(*([1] * (raw.dim() - 1)), -1)
        if group_scale is not None and bool(getattr(self.cfg, "attr_scale_group_base", True)):
            g = group_scale.reshape(group_scale.shape[0], 1, 1).clamp(min=1e-6).log()
            base = self.scale_base_a * g + b
        else:
            base = b
        # Residual around the linear's own bias, NOT around `base`.
        # `raw = W h + b`, so `raw - base = W h - a log(extent)`: at the zero
        # init (W~0) every group with extent < 1 saturates the UPPER cap and
        # every Gaussian starts ~e^1.5 too large. `raw - b = W h` is 0 at init,
        # so the output starts at the group-relative median the band is for.
        r = raw - b
        # tanh on each side with its own cap. Continuous and C1 at r = 0: both
        # branches pass through 0 with unit slope, so no gradient discontinuity.
        r = torch.where(r < 0, down * torch.tanh(r / down), up * torch.tanh(r / up))
        self._cap_hit = ((r.detach().abs() > 0.9 * min(down, up)).float().mean())
        return base + r

    def _chunk(self, p: torch.Tensor, app: Optional[torch.Tensor],
               sc: torch.Tensor, slot_mask: Optional[torch.Tensor] = None,
               grot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """One chunk of groups: (Bc, G, 3) -> (Bc, G, 3 + attrs)."""
        tok = self.slot_emb.weight.view(1, self.G, -1).expand(p.shape[0], -1, -1)
        # The group's OWN code drives modulation and concatenation; the window is
        # extra context for cross-attention only, so those paths keep their meaning.
        # code layout is [shape | appearance]; the historical paths use appearance
        # only, so film / cat_code / `own` keep their exact meaning.
        shp = app[..., : self.c_shape_read] if (app is not None and self.c_shape_read) else None
        app_o = app[..., self.c_shape_read :] if app is not None else None
        own = app_o[:, self.nbr : self.nbr + 1] if app_o is not None else None
        if self.film is not None and own is not None:
            gam, bet = self.film(own).chunk(2, -1)
            tok = tok * (1.0 + gam) + bet
        feats = [tok, self.xyz_pe(p)]
        pad = None
        if slot_mask is not None:
            pad = slot_mask <= 0.5
        if self.loc_pe is not None:
            if slot_mask is not None:
                w = (slot_mask > 0.5).to(p.dtype).unsqueeze(-1)
                # Empty groups keep a dummy live slot so the mean stays finite.
                w = torch.where(w.sum(1, keepdim=True) > 0, w, torch.ones_like(w))
                mean = (p * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1e-6)
                loc = (p - mean) * (slot_mask > 0.5).to(p.dtype).unsqueeze(-1)
                rad = loc.norm(dim=-1, keepdim=True).sum(1, keepdim=True)
                rad = rad / w.sum(1, keepdim=True).clamp(min=1e-6)
            else:
                loc = p - p.mean(1, keepdim=True)
                rad = loc.norm(dim=-1, keepdim=True).mean(1, keepdim=True)
            loc = loc / rad.clamp(min=1e-6)
            feats.append(self.loc_pe(loc))
        if self.cat_code and own is not None:
            feats.append(own.expand(-1, self.G, -1))
        h = self.in_proj(torch.cat(feats, dim=-1))
        # Cross-attention lets each slot read a different part of the code, which
        # a broadcast modulation cannot do.
        if self.xattn is not None and app is not None:
            # (Bc, W, c_app) -> (Bc, W, n_code_tok, d) -> (Bc, W*n_code_tok, d)
            ct = self.to_code_tok(app_o)
            if self.to_shape_tok is not None and shp is not None:
                ct = ct + self.to_shape_tok(shp)
            ct = ct.reshape(p.shape[0], app.shape[1], self.n_code_tok, -1)
            if self.nbr_emb is not None:
                ct = ct + self.nbr_emb
            ct = ct.reshape(p.shape[0], -1, ct.shape[-1])
            h = h + self.xattn_gate * self.xattn(self.xnorm(h), ct, ct, need_weights=False)[0]
        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad)

        log_s = self._bounded_log_scale(self.head_scale(h), sc)
        q_delta = F.normalize(self.head_rot(h), dim=-1)
        if bool(getattr(self.cfg, "attr_frame_needles", False)) and grot is not None:
            # Scale channels are the three FRAME axes (size μ = mean, anisotropy
            # = deviation). Rotation is the cell frame times a residual that is
            # identity at init, so a sphere starts aligned and stretching a
            # channel has a 3D meaning. grot is detached: folding still belongs
            # to the xyz losses (H2 smeared when orientation was a free match).
            q_cell = rotmat_to_quat(grot).unsqueeze(1).expand(-1, self.G, -1)
            q = F.normalize(quat_mul(q_cell, q_delta), dim=-1)
        else:
            q = q_delta
        parts = [
            log_s,
            q,
            self.head_opacity(h),
        ]
        if self.head_color is not None:
            parts.append(self.head_color(h))
        if self.head_sh is not None:
            parts.append(self.head_sh(h))

        # Bounded nudge, in units of the group's own extent. 0.15 is a little
        # over one projected footprint (1.40 px against a 12.0 px group
        # radius), i.e. the range where an image gradient still points at the
        # right place.
        cap = float(getattr(self.cfg, "attr_nudge_cap", 0.15))
        d_xyz = torch.tanh(self.head_nudge(h)) * cap * sc
        return torch.cat([p + d_xyz] + parts, dim=-1)

    def forward(self, xyz: torch.Tensor, appear: Optional[torch.Tensor],
                group_scale: torch.Tensor,
                use_checkpoint: Optional[bool] = None,
                slot_mask: Optional[torch.Tensor] = None,
                group_rot: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """(B, N, 3) positions + (B, groups, c_app) code -> full Gaussian tensor.

        ``xyz`` and ``group_scale`` must already be detached by the caller; this
        module deliberately has no path back into the geometry decoder.

        ``group_rot`` is ``(B, num_groups, 3, 3)``, the folding cell frame.
        Detached here so attribute losses cannot rotate the xyz template.
        """
        b, n, _ = xyz.shape
        g = self.G
        ng = min(n // g, self.num_groups)
        if group_rot is not None:
            group_rot = group_rot.detach()
        # Window of neighbouring group codes, built once per forward.
        # (B, ng, 2*nbr+1, c_app); replicate-padded at the two ends.
        if appear is not None and self.c_app > 0:
            w = self.nbr
            a = appear[:, :ng]
            if w > 0:
                a = F.pad(a.transpose(1, 2), (w, w), mode="replicate").transpose(1, 2)
                win = torch.stack([a[:, i : i + ng] for i in range(2 * w + 1)], dim=2)
            else:
                win = a.unsqueeze(2)
        else:
            win = None
        ck = (self.training if use_checkpoint is None else bool(use_checkpoint))
        outs = []
        for s in range(0, ng, self.chunk):
            e = min(s + self.chunk, ng)
            c = e - s
            p = xyz[:, s * g : e * g].reshape(b * c, g, 3)
            # Gated on the code existing, not on FiLM existing. When the
            # conditioning switched to cross-attention `self.film` became None,
            # this still read it, and the code was silently dropped for every
            # chunk -- the appearance path was disconnected with no error.
            app = (win[:, s:e].reshape(b * c, win.shape[2], self.c_code)
                   if (self.c_app > 0 and appear is not None) else None)
            sc = group_scale[:, s:e].reshape(b * c, 1, 1)
            sm = None
            if slot_mask is not None:
                sm = slot_mask[:, s * g : e * g].reshape(b * c, g)
            if group_rot is not None:
                grot = group_rot[:, s:e].reshape(b * c, 3, 3)
            else:
                grot = torch.eye(3, device=p.device, dtype=p.dtype).expand(b * c, 3, 3).contiguous()
            if ck and torch.is_grad_enabled():
                o = torch.utils.checkpoint.checkpoint(
                    self._chunk, p, app, sc, sm, grot, use_reentrant=False)
            else:
                o = self._chunk(p, app, sc, sm, grot)
            outs.append(o.reshape(b, c * g, -1))

        pred = torch.cat(outs, dim=1)
        if pred.shape[1] < n:                       # tail slots the loop skipped
            pad = xyz[:, pred.shape[1] :]
            rest = pred.new_zeros(b, pad.shape[1], pred.shape[-1] - 3)
            pred = torch.cat([pred, torch.cat([pad, rest], dim=-1)], dim=1)
        return {"pred": pred[:, :n]}
