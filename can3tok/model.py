"""Can3TokAE: identity pack -> compressor -> z_compact -> decompressor -> codec decoder.

    points (max_points x 3)
      | identity Morton patch pack (+ optional learned residual)
    z_raw            latent_channels x latent_hw          (internal only)
      | staged compressor: masked intra-cell attn -> windowed cell attn -> budget heads
    z_compact        32 x 64 x 64                          (the world-model interface)
      | staged decompressor -> z_raw_hat -> identity-unpack codec decoder

``decode_compact(z_compact)`` is both the training decode and the world-model
decode. A separate gen / joint_direct path that emits a new Fibonacci set is not
invertible and is not what a later WM should call.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .compressor import StagedCompressor, StagedDecompressor
from .config import Can3TokConfig, channel_budget, describe_layout, patch_layout, validate_layout
from .decoder import CodecDecoder
from .encoder import PatchPackEncoder
from .attr_decoder import AttributeDecoder
from .joint_decoder import DirectJointGaussianDecoder, JointGaussianRefiner
from .gen_decoder import GenerativeDecoder


def _presence_or_count_mask(presence: torch.Tensor, count: torch.Tensor,
                            n_points: int, group_size: int) -> torch.Tensor:
    """Deployable live-slot mask from predicted presence, with count as fallback."""
    if presence is not None and presence.numel() == n_points * presence.shape[0]:
        m = (presence > 0).to(presence.dtype)
        if int(m.sum()) >= 1:
            return m[:, :n_points]
    g = int(group_size)
    ng = max(int(count.shape[1]), 1)
    idx = torch.arange(g, device=count.device)
    live = (idx.view(1, 1, g) < count.clamp(0, g).unsqueeze(-1)).reshape(count.shape[0], ng * g)
    if live.shape[1] < n_points:
        live = torch.nn.functional.pad(live, (0, n_points - live.shape[1]))
    return live[:, :n_points].to(count.dtype)


class Can3TokAE(nn.Module):
    def __init__(self, cfg: Can3TokConfig, allow_padded_tokens: bool = False):
        super().__init__()
        validate_layout(cfg, allow_padded_tokens=allow_padded_tokens)
        self.cfg = cfg
        self.layout = patch_layout(cfg)
        self.budget = channel_budget(cfg)

        self.encoder = PatchPackEncoder(cfg)
        self.compressor = StagedCompressor(cfg)
        self.decompressor = StagedDecompressor(cfg)
        self.decoder = CodecDecoder(cfg)
        self.gen_decoder = GenerativeDecoder(cfg) if cfg.use_gen_branch else None
        # One shared attribute decoder for both branches. Both consume the same
        # z_compact and produce positions in the same space, so appearance is the
        # same function in both cases; giving them separate copies would only make
        # the student's version worse for no reason.
        self.attr_decoder = (
            AttributeDecoder(cfg)
            if (cfg.target_dim > 3 and int(getattr(cfg, "attr_decoder_layers", 0)) > 0)
            else None
        )
        self.joint_decoder = (
            JointGaussianRefiner(cfg)
            if bool(getattr(cfg, "joint_shared_decoder", False)) else None
        )
        self.direct_joint_decoder = (
            DirectJointGaussianDecoder(cfg)
            if bool(getattr(cfg, "joint_direct_decoder", False)) else None
        )
        if self.direct_joint_decoder is not None:
            if self.joint_decoder is not None:
                raise ValueError("joint_direct_decoder and joint_shared_decoder are mutually exclusive")
            # The replacement decoder never executes the legacy point decoder.
            # In the hierarchical variant the decompressor is *not* legacy: its
            # T reconstructed local tokens are the cross-attention memory and
            # must receive geometry gradients.  F3B froze it and then ignored
            # z_raw_hat, cutting the only multi-token group representation out
            # of the deployable path.
            frozen = [self.decoder]
            if (not bool(getattr(cfg, "joint_local_memory", False))
                    or bool(getattr(cfg, "structured_local_code", False))):
                frozen.append(self.decompressor)
            for module in frozen:
                for p in module.parameters():
                    p.requires_grad_(False)

        # Running latent scale, mirroring the ``scale_factor`` that Hunyuan3D hands
        # to its DiT: downstream diffusion should consume z_compact / latent_scale.
        self.register_buffer("latent_scale", torch.tensor(1.0))
        self.register_buffer("latent_scale_count", torch.tensor(0.0))

    # ------------------------------------------------------------------
    def describe(self) -> str:
        return describe_layout(self.cfg)

    def set_encoder_residual(self, flag: bool) -> None:
        self.encoder.set_residual_enabled(flag)

    def set_decoder_refine_trainable(self, flag: bool) -> None:
        if self.direct_joint_decoder is not None:
            return
        self.decoder.set_refine_trainable(flag)

    @torch.no_grad()
    def _update_latent_scale(self, z: torch.Tensor) -> None:
        m = float(self.cfg.latent_scale_momentum)
        std = z.detach().float().std()
        if not torch.isfinite(std):
            return
        if float(self.latent_scale_count) < 0.5:
            self.latent_scale.fill_(float(std))
        else:
            self.latent_scale.mul_(1.0 - m).add_(m * std)
        self.latent_scale_count.add_(1.0)

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, mask: torch.Tensor,
               group_anchor: Optional[torch.Tensor] = None):
        z_raw, anchors, patch = self.encoder(x, mask, group_anchor=group_anchor)
        return z_raw, anchors, patch

    def compress(self, z_raw: torch.Tensor, anchors: Dict[str, torch.Tensor]):
        return self.compressor(z_raw, anchors)

    def compact_from_points(self, x: torch.Tensor, mask: torch.Tensor,
                            group_anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encoder + compressor only. Used by the equivariance regulariser."""
        z_raw, anchors, _ = self.encoder(x, mask, group_anchor=group_anchor)
        z_compact, _ = self.compressor(z_raw, anchors)
        return z_compact

    def _group_rot_from_ctx(self, ctx: Dict[str, torch.Tensor]):
        """Folding cell-frame rotation for AttributeDecoder, or None."""
        if not bool(getattr(self.cfg, "attr_frame_needles", False)):
            return None
        if ctx.get("fold_rot") is not None:
            return ctx["fold_rot"]
        fr = ctx.get("fold_frame")
        if fr is not None and fr.shape[-1] >= 6:
            from .compressor import axis_angle_to_matrix
            return axis_angle_to_matrix(fr[..., 3:6])
        return None

    def _decode_gaussians(
        self,
        z_compact: torch.Tensor,
        shortcut_alpha: Optional[float] = None,
        attr_xyz: Optional[torch.Tensor] = None,
        slot_mask: Optional[torch.Tensor] = None,
        teacher_force: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Shared deployable decode: decompressor -> codec -> attributes.

        ``forward`` and ``decode_compact`` must return the same Gaussians for the
        same raw ``z_compact``. Teacher forcing is opt-in and only used in train.
        """
        alpha = 0.0 if shortcut_alpha is None else float(shortcut_alpha)
        z_raw_hat, ctx = self.decompressor(z_compact, shortcut_alpha=alpha)
        out: Dict[str, torch.Tensor] = {
            "z_raw_hat": z_raw_hat,
            "decoded_centroid": ctx["centroid"],
            "decoded_count": ctx["count"],
            "decoded_scale": ctx["scale"],
            "res_ratio": ctx["res_ratio"],
            "direct_frac": ctx["direct_frac"],
            "direct_ratio": ctx["direct_ratio"],
            "learned_xyz": ctx["learned_xyz"],
            "direct_xyz": ctx["direct_xyz"],
            "shortcut_alpha": ctx["shortcut_alpha"],
        }
        if "fold_local" in ctx:
            out["teacher_fold_frame"] = ctx["fold_frame"]
            out["teacher_fold_local"] = ctx["fold_local"]

        if self.direct_joint_decoder is not None:
            local_tokens = None
            if (bool(getattr(self.cfg, "joint_local_memory", False))
                    and not bool(getattr(self.cfg, "structured_local_code", False))):
                b = z_raw_hat.shape[0]
                local_tokens = z_raw_hat.flatten(2).transpose(1, 2)
                used = self.layout["num_groups"] * self.layout["tokens_per_group"]
                local_tokens = local_tokens[:, :used].reshape(
                    b, self.layout["num_groups"], self.layout["tokens_per_group"],
                    int(self.cfg.latent_channels))
            joint = self.direct_joint_decoder(
                ctx["centroid"], ctx["count"], ctx["scale"], ctx.get("shared_code"),
                local_tokens=local_tokens)
            pred, presence = joint["pred"], joint["presence"]
            out["group_translation_abs"] = joint["translation_abs"]
            if "memory_token_std" in joint:
                out["memory_token_std"] = joint["memory_token_std"]
            out["pred"] = pred
            out["presence"] = presence
            out["attr_pred"] = pred
            out["_ctx"] = ctx
            return out

        pred, presence = self.decoder(
            z_raw_hat,
            ctx["cell_vec"],
            ctx["scale"],
            centroid=ctx["centroid"],
            count=ctx["count"],
            group_vec=ctx.get("group_vec"),
            attr_xyz=attr_xyz if teacher_force else None,
            appear=ctx.get("appearance"),
        )
        out["pred"] = pred
        out["presence"] = presence

        if self.attr_decoder is not None:
            ap = ctx.get("appearance")
            geo = pred[..., 0:3]
            if bool(getattr(self.cfg, "attr_detach_geometry", True)):
                geo = geo.detach()
            p_tf = float(getattr(self.cfg, "attr_teacher_prob", 0.0)) if teacher_force else 0.0
            if attr_xyz is not None and p_tf > 0.0:
                true_xyz = attr_xyz[..., 0:3][:, : geo.shape[1]]
                if p_tf >= 1.0:
                    geo = true_xyz
                else:
                    keep = torch.rand(geo.shape[0], geo.shape[1], 1, device=geo.device) < p_tf
                    geo = torch.where(keep, true_xyz, geo)
            sm = slot_mask
            if sm is None and bool(getattr(self.cfg, "attr_slot_mask", False)):
                sm = _presence_or_count_mask(presence, ctx["count"], pred.shape[1],
                                             int(self.layout["group_size"]))
            attr_kwargs = {}
            if bool(getattr(self.cfg, "attr_slot_mask", False)):
                attr_kwargs["slot_mask"] = sm
            grot = self._group_rot_from_ctx(ctx)
            if grot is not None:
                attr_kwargs["group_rot"] = grot
            out["attr_pred"] = self.attr_decoder(
                geo,
                ctx.get("attr_code", ap),
                ctx["scale"].detach(),
                **attr_kwargs,
            )["pred"]

        if self.joint_decoder is not None:
            if "attr_pred" not in out:
                raise RuntimeError("joint_shared_decoder requires attr_decoder_layers > 0")
            joint = self.joint_decoder(
                out["pred"], out["attr_pred"], out["presence"],
                ctx.get("shared_code"), ctx["scale"].detach(),
            )
            out["base_pred"] = out["pred"]
            out["base_attr_pred"] = out["attr_pred"]
            out["pred"] = joint["geometry"]
            out["attr_pred"] = joint["render"]
            out["presence"] = joint["presence"]
            out["joint_xyz_delta_abs"] = joint["xyz_delta_abs"]
        out["_ctx"] = ctx
        return out

    def decode_compact(self, z_compact: torch.Tensor, shortcut_alpha: Optional[float] = None,
                       normalized: bool = False):
        """World-model decode: z_compact -> the same Gaussians training renders.

        Pass ``normalized=True`` when the tensor came from ``encode_compact()``.
        Raw ``forward()['z_compact']`` uses the default ``normalized=False``.
        """
        z = z_compact
        if normalized:
            z = z * self.latent_scale.clamp(min=1e-6)
        alpha = 0.0 if shortcut_alpha is None else float(shortcut_alpha)
        out = self._decode_gaussians(z, shortcut_alpha=alpha, teacher_force=False)
        out.pop("_ctx", None)
        return out

    @torch.no_grad()
    def encode_compact(self, x: torch.Tensor, mask: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        z = self.compact_from_points(x, mask)
        return z / self.latent_scale.clamp(min=1e-6) if normalized else z

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        run_decode: bool = True,
        run_gen: bool = True,
        gen_noise_std: float = 0.0,
        attr_xyz: Optional[torch.Tensor] = None,
        enc_x: Optional[torch.Tensor] = None,
        enc_mask: Optional[torch.Tensor] = None,
        group_anchor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not bool(getattr(self.cfg, "use_fixed_anchor_center", False)):
            group_anchor = None
        # The encoder may read a larger point set than the decoder emits. `x`/`mask`
        # stay the decoder-sized tensors every loss is written against; `enc_x` is
        # only the encoder's input when the loader supplies a bigger one.
        z_raw, anchors, patch = self.encode(
            x if enc_x is None else enc_x, mask if enc_mask is None else enc_mask,
            group_anchor=group_anchor)
        z_compact, aux = self.compress(z_raw, anchors)
        if self.training:
            self._update_latent_scale(z_compact)

        out: Dict[str, torch.Tensor] = {
            "z_raw": z_raw,
            "z_compact": z_compact,
            "patch": patch,
            "kl": aux["kl"],
            "group_valid": aux["group_valid"],
            "cell_valid": aux["cell_valid"],
        }
        if "compact_local_std" in aux:
            out["compact_local_std"] = aux["compact_local_std"]
        if "n_used" in aux:
            out["n_used"] = aux["n_used"]
            out["empty_frac"] = aux["empty_frac"]
            out["global_cond"] = aux["global_cond"]

        # codec teacher: always the clean latent, so the z_raw target stays honest
        alpha = float(self.cfg.shortcut_alpha if self.training else self.cfg.shortcut_alpha_eval)
        out["residual_pack"] = bool(self.cfg.residual_pack)
        if run_decode:
            decoded = self._decode_gaussians(
                z_compact, shortcut_alpha=alpha, attr_xyz=attr_xyz,
                teacher_force=self.training,
            )
            ctx = decoded.pop("_ctx")
            out.update(decoded)
        else:
            ctx = {}
            pred = x.new_zeros(x.shape[0], int(self.cfg.max_points), int(self.cfg.target_dim))
            presence = x.new_zeros(x.shape[0], int(self.cfg.max_points))
            out["pred"] = pred
            out["presence"] = presence
            z_raw_hat, ctx = self.decompressor(z_compact, shortcut_alpha=alpha)
            out["z_raw_hat"] = z_raw_hat
            out["decoded_centroid"] = ctx["centroid"]
            out["decoded_count"] = ctx["count"]
            out["decoded_scale"] = ctx["scale"]
            out["res_ratio"] = ctx["res_ratio"]
            out["direct_frac"] = ctx["direct_frac"]
            out["direct_ratio"] = ctx["direct_ratio"]
            out["learned_xyz"] = ctx["learned_xyz"]
            out["direct_xyz"] = ctx["direct_xyz"]
            out["shortcut_alpha"] = ctx["shortcut_alpha"]

        # Teacher-cycle branch. The question is whether z_compact preserves what
        # the teacher *decoder* needs, so the gradient must reach the compressor
        # and decompressor -- and must NOT reach the decoder's own weights. If it
        # does, the cheapest way to make D(z_hat) match D(z) is for D to become
        # insensitive to its input, which is the opposite of what the term is for.
        #
        # `out["pred"]` cannot be used for this: it is the same tensor the
        # reconstruction losses train the decoder through. So both sides of the
        # cycle are decoded with the decoder's parameters *detached*, via
        # functional_call: gradient flows to z_raw_hat and stops at the weights.
        # The reference side additionally runs under no_grad.
        #
        # Cost: two extra decodes. Gated behind cfg.teacher_cycle, off by default.
        if run_decode and bool(getattr(self.cfg, "teacher_cycle", False)):
            frozen = (
                {k: v.detach() for k, v in self.decoder.named_parameters()},
                {k: v.detach() for k, v in self.decoder.named_buffers()},
            )
            dec_args = (ctx["cell_vec"], ctx["scale"])
            dec_kwargs = dict(centroid=ctx["centroid"], count=ctx["count"],
                              group_vec=ctx.get("group_vec"), attr_xyz=attr_xyz,
                              appear=ctx.get("appearance"),
                              use_checkpoint=False)
            if bool(getattr(self.cfg, "teacher_cycle_freeze", True)):
                cp, _ = torch.func.functional_call(
                    self.decoder, frozen, (out["z_raw_hat"],) + dec_args, dec_kwargs
                )
                out["pred_cycle"] = cp
            with torch.no_grad():
                tp, _ = torch.func.functional_call(
                    self.decoder, frozen, (z_raw,) + dec_args, dec_kwargs
                )
            out["pred_from_true_pack"] = tp

        if run_gen and self.gen_decoder is not None:
            z_in = z_compact.detach() if bool(self.cfg.gen_detach_latent) else z_compact
            if gen_noise_std > 0:
                z_in = z_in + torch.randn_like(z_in) * float(gen_noise_std)
            gen = self.gen_decoder(z_in)
            out["gen_pred"] = gen["pred"]
            out["gen_presence"] = gen["presence"]
            if "fold_local" in gen:
                out["gen_fold_frame"] = gen["fold_frame"]
                out["gen_fold_local"] = gen["fold_local"]
            if self.attr_decoder is not None:
                ap = ctx.get("appearance")
                gen_kwargs = {}
                grot = self._group_rot_from_ctx(ctx)
                if grot is not None:
                    gen_kwargs["group_rot"] = grot
                out["gen_attr_pred"] = self.attr_decoder(
                    gen["pred"][..., 0:3].detach(),
                    ap.detach() if ap is not None else None,
                    ctx["scale"].detach(),
                    **gen_kwargs,
                )["pred"]
        return out

    # ------------------------------------------------------------------
    def param_groups(self, lr: float, lr_scales: Optional[Dict[str, float]] = None):
        lr_scales = lr_scales or {}
        named = {
            "encoder": self.encoder,
            "compressor": self.compressor,
            "decompressor": self.decompressor,
            "decoder": self.decoder,
        }
        if self.gen_decoder is not None:
            named["gen"] = self.gen_decoder
        if self.attr_decoder is not None:
            named["attr"] = self.attr_decoder
        if self.joint_decoder is not None:
            named["joint"] = self.joint_decoder
        if self.direct_joint_decoder is not None:
            named["joint_direct"] = self.direct_joint_decoder
        # The encoder's attribute path gets its OWN group. It has to: train.py
        # zeroes the "encoder" group's learning rate whenever the residual branch
        # is off (`if not flags["encoder_residual"]`), which is correct for a
        # zero-init gated residual but silently froze the appearance encoder once
        # that lived in the same module. Measured on two checkpoints 500 steps
        # apart, every encoder tensor was bit-identical while compressor and
        # attr_decoder moved normally, and the pack's aux block sat at its
        # initialisation value (std 0.0163) for the whole run.
        attr_enc = getattr(self.encoder, "attr_encoder", None)
        skip = {id(p) for p in attr_enc.parameters()} if attr_enc is not None else set()
        if attr_enc is not None:
            named["attr_enc"] = attr_enc
        # The per-slot basis gets its own group. Measured, it is the ONLY
        # group-independent per-slot signal the attribute decoder has -- zeroing it
        # and collapsing the position input together take within-group parameter
        # diversity to exactly 0.0 -- and it moved 1.47x in 14000 steps at the
        # shared rate while the code branch it competes with sits at |w| 1.8e-1.
        # A separate group is the only way to give it a rate matched to how far it
        # has to travel without moving the rest of the module with it.
        attr_slot = None
        if self.attr_decoder is not None:
            attr_slot = getattr(self.attr_decoder, "slot_emb", None)
        slot_skip = {id(p) for p in attr_slot.parameters()} if attr_slot is not None else set()
        if attr_slot is not None:
            named["attr_slot"] = attr_slot
        groups = []
        for name, module in named.items():
            params = [p for p in module.parameters() if p.requires_grad]
            if name == "encoder":
                params = [p for p in params if id(p) not in skip]
            if name == "attr":
                params = [p for p in params if id(p) not in slot_skip]
            if not params:
                continue
            groups.append(
                {
                    "params": params,
                    "lr": lr * float(lr_scales.get(name, 1.0)),
                    "base_lr": lr * float(lr_scales.get(name, 1.0)),
                    "name": name,
                }
            )
        # Every trainable parameter must belong to exactly one group. This list is
        # written by hand, so adding a submodule and forgetting to name it here
        # hands the optimiser a model with a frozen component -- and the failure is
        # silent, because the gradients are still computed, just never applied.
        # That is what happened to attr_decoder: 3.325M parameters sat at their
        # initialisation for a full run while the losses were logged as if they
        # were training, and the only visible symptom was eval metrics identical
        # to three decimal places between checkpoints.
        owned = {id(p) for grp in groups for p in grp["params"]}
        orphan = sorted(n for n, p in self.named_parameters()
                        if p.requires_grad and id(p) not in owned)
        if orphan:
            raise RuntimeError(
                f"{len(orphan)} trainable parameters are in no optimiser group, "
                f"so they would never be updated: {orphan[:6]}"
                + (" ..." if len(orphan) > 6 else "")
                + " -- add the owning submodule to param_groups()."
            )
        return groups


def build_model(cfg: Can3TokConfig, allow_padded_tokens: bool = False) -> Can3TokAE:
    return Can3TokAE(cfg, allow_padded_tokens=allow_padded_tokens)
