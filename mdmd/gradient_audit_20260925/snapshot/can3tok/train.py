"""Training entry point for Can3TokAE (DDP via torchrun).

Fixes that were latent bugs in the previous pipeline
---------------------------------------------------
* ``encoder_warmup_*`` is actually applied (per-module learning-rate groups).
* best checkpoint is codec-only until ``gen_start``, then ``0.5*(codec+gen)``.
* generative metrics are logged during training, not only at eval milestones.
* frozen phases are implemented by zeroing a param group's learning rate instead
  of toggling ``requires_grad``, which keeps DDP gradient buckets stable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from contextlib import contextmanager, nullcontext
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import Can3TokConfig, channel_budget, describe_layout, patch_layout
from .data import ReplayGaussianDataset, list_npz_files
from .eval_utils import (
    EvalView,
    compute_eval_metrics,
    eval_fingerprint,
    filter_eval_views,
    group_error_breakdown,
    latent_diagnostics,
    render_eval_metrics,
    save_eval_outputs,
)
from .io_utils import save_json
from .losses import equivariance_loss, render_loss, responsibility_target, total_loss
from .model import build_model
from .schedule import (
    apply_eval_schedule,
    effective_weights,
    global_lr,
    lr_scales,
    phase_of,
    schedule_flags,
    sinkhorn_epsilon_at,
    detail_gains,
)


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("can3tok encoder/decoder training")
    # data
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--split_path", type=str, default="")
    p.add_argument("--num_val_files", type=int, default=300)
    p.add_argument("--min_snapshot_step", type=int, default=0,
                   help="exclude immature trajectory snapshots (fix3 uses 12000)")
    p.add_argument("--stats_path", type=str, default="")
    p.add_argument("--stats_stride", type=int, default=50)
    p.add_argument("--stats_quantile", type=float, default=0.02)
    p.add_argument("--max_points", type=int, default=262144)
    p.add_argument("--drop_outside", action="store_true")
    p.add_argument("--sample_mode", type=str, default="stratified",
                   choices=["stratified", "importance_even", "morton", "first"])
    p.add_argument("--crop_prob", type=float, default=0.35)
    p.add_argument("--density_aware_sample", action="store_true", default=True,
                   help="boost sparse regions in stratified importance sampling")
    p.add_argument("--no_density_aware_sample", action="store_true")
    p.add_argument("--density_importance_weight", type=float, default=0.35)
    # Off by default: spreading points over all 8192 groups buys a better coarse
    # fit from denser centroids but flatlines within-group learning (A/B in
    # scripts/ab_layout.sh -- z_res -5% versus -21% for prefix packing).
    p.add_argument("--slot_redistribute", action="store_true", default=False,
                   help="spread real points across all groups (no empty-prefix waste)")
    p.add_argument("--no_slot_redistribute", action="store_true")
    p.add_argument("--partition_mode", type=str, default="morton", choices=["morton", "kd"],
                   help="how to cut selected points into groups after redistribute")
    p.add_argument("--partition_block", type=int, default=0,
                   help="local kd inside runs of N Morton-consecutive groups (0 = plain "
                        "Morton-contiguous). Keeps prefix packing and grid locality while "
                        "removing the Morton-jump groups; 32 measured best")
    p.add_argument("--slot_sort", type=str, default="morton", choices=["morton", "template"],
                   help="slot order inside a group. 'template' assigns each group's points to the "
                        "fixed folding template by optimal transport, so slot i of the target is "
                        "the point that slot i of the decoder is built to emit. Measured: mean "
                        "residual 1.010 -> 0.408, reachable chamfer 0.261 -> 0.225")
    p.add_argument("--augment", action="store_true")
    p.add_argument("--aug_rot_deg", type=float, default=8.0)
    p.add_argument("--aug_scale_jitter", type=float, default=0.03)
    p.add_argument("--aug_shift", type=float, default=0.01)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)

    # geometry / latent shape
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--local_tokens_per_group", type=int, default=0,
                   help="0 = derive the lossless value patch_dim / latent_channels")
    p.add_argument("--latent_channels", type=int, default=32)
    p.add_argument("--latent_hw", type=int, nargs=2, default=[0, 0],
                   help="0 0 = pick the smallest power-of-two grid that fits")
    p.add_argument("--compact_latent_channels", type=int, default=32)
    p.add_argument("--compact_latent_hw", type=int, nargs=2, default=[64, 64])
    p.add_argument("--budget_centroid", type=int, default=4)
    p.add_argument("--budget_occupancy", type=int, default=1,
                   help="1 channel for per-group count (required once groups are partially filled)")
    p.add_argument("--budget_shape", type=int, default=11)
    p.add_argument("--budget_appearance", type=int, default=0,
                   help="compact channels for appearance, taken out of budget_shape. 0 keeps the "
                        "geometry-only latent. Measured knee is 8: an 8-dim appearance code lifts "
                        "rank-28 geometry 17.05 -> 21.43 dB on held-out poses, 64 buys 2.9 dB more")
    p.add_argument("--attr_decoder_layers", type=int, default=0,
                   help="0 = attribute heads live inside the geometry decoder and share its "
                        "refine token, so the render loss reaches the geometry stack even with "
                        "the position output detached. >0 builds a SEPARATE AttributeDecoder "
                        "whose inputs from the geometry side are all detached")
    p.add_argument("--attr_decoder_dim", type=int, default=256)
    p.add_argument("--attr_cond", type=str, default="xattn",
                   choices=["xattn", "film", "concat", "film+cat"],
                   help="how the group appearance code reaches the slot tokens. Measured "
                        "as an autodecoder over 8 scenes, attribute nrmse: none 0.3021, "
                        "film 0.3055, concat 0.2983, xattn 0.2783, film+cat 0.2969 -- FiLM "
                        "broadcasts one scale/shift to all 64 slots and scored below "
                        "ignoring the code entirely")
    p.add_argument("--attr_local_pe", type=int, default=1,
                   help="1 = the attribute decoder also gets group-local coordinates. Its "
                        "only per-point input is xyz, and scene-normalised xyz resolves "
                        "within-group differences at 0.27 of between-group ones while "
                        "52-78%% of the attribute variance is within-group. Recentring on "
                        "the group turns that ratio into 3.52; concatenated, 1.00. Measured "
                        "output response to a 0.1-radius jitter: 0.018 without, 0.436 with")
    p.add_argument("--attr_nudge_cap", type=float, default=0.15,
                   help="tanh cap on the render-driven position nudge, in group extents. The "
                        "median Gaussian projects to 1.40 px against a 12.0 px group radius, so "
                        "~0.12 is one footprint -- the range where the image gradient still "
                        "points the right way (cos 0.134 at 0.6 px, 0.054 at 6 px)")
    p.add_argument("--attr_scale_log_cap", type=float, default=3.0)
    p.add_argument("--attr_scale_cap_down", type=float, default=0.0,
                   help="lower half-width of the log-scale band, around a GROUP-RELATIVE "
                        "base. >0 (with --attr_scale_cap_up) replaces the single global "
                        "band, which measured 5.94%% of GT log-scale unreachable")
    p.add_argument("--attr_scale_cap_up", type=float, default=0.0,
                   help="upper half-width. Kept tighter than the lower one on purpose: the "
                        "band exists to stop the render loss covering holes with one giant "
                        "splat, and that failure is one-sided")
    p.add_argument("--attr_scale_group_base", type=int, default=1)
    p.add_argument("--attr_read_shape", type=int, default=0,
                   help="1 = the attribute decoder also reads the cell's SHAPE channels, not "
                        "just its appearance channels. Measured: with the position input "
                        "detached, the attribute/render objective puts EXACTLY 0.000e+00 "
                        "gradient on the 19 shape channels, so shape is trained by the "
                        "point-space terms and appearance by the render -- two disjoint halves "
                        "of every cell under two different objectives, while the coalition "
                        "ladder says xyz/scale/opacity/colour only pay when they move "
                        "together. This is the one path that carries render gradient into "
                        "shape without going through position transport. Costs 0 latent "
                        "channels (8 -> 27 is the decoder's read width, not the latent)")
    p.add_argument("--joint_shared_decoder", type=int, default=0,
                   help="function-preserving shared refiner over all non-anchor compact "
                        "channels; jointly emits xyz/scale/rotation/opacity/color/presence")
    p.add_argument("--joint_direct_decoder", type=int, default=0,
                   help="replace the split legacy decoders with one shared-27 decoder")
    p.add_argument("--joint_local_memory", type=int, default=0,
                   help="decode Gaussian point queries by cross-attending to the reconstructed "
                        "per-group local z_raw tokens instead of broadcasting one group vector")
    p.add_argument("--joint_memory_layers", type=int, default=1)
    p.add_argument("--joint_direct_xyz_cap", type=float, default=1.5)
    p.add_argument("--joint_translation_cap", type=float, default=0.25)
    p.add_argument("--joint_decoder_dim", type=int, default=192)
    p.add_argument("--joint_decoder_layers", type=int, default=2)
    p.add_argument("--joint_decoder_chunk", type=int, default=256)
    p.add_argument("--joint_nbr_window", type=int, default=1)
    p.add_argument("--joint_xyz_cap", type=float, default=0.10)
    p.add_argument("--joint_scale_delta_cap", type=float, default=1.0)
    p.add_argument("--joint_opacity_delta_cap", type=float, default=2.0)
    p.add_argument("--joint_color_delta_cap", type=float, default=0.5)
    p.add_argument("--joint_rot_delta_cap", type=float, default=0.25)
    p.add_argument("--joint_detach_render_xyz", type=int, default=1,
                   help="detach only xyz inside the rasterizer while render gradients still "
                        "train the shared representation through all attribute heads")
    p.add_argument("--render_presence", type=int, default=0,
                   help="1 = the render loss selects the PREDICTION's Gaussians by the "
                        "presence head instead of the GT mask. Until now presence never "
                        "entered the render objective, so the model was scored with "
                        "information it lacks at inference, and emitting a Gaussian into an "
                        "empty region cost nothing -- which is the precondition a "
                        "variable-count decoder needs")
    p.add_argument("--w_cov3d", type=float, default=0.0,
                   help="supervise the 3D covariance R diag(s^2) R^T instead of scale and "
                        "rotation separately. The render only sees the covariance, so the "
                        "separate rot term spends gradient on a factorisation choice the "
                        "image cannot observe -- measured attr_rot_nrmse 1.089, i.e. no "
                        "better than the dataset mean. Scale-free (normalised by the "
                        "target's own Frobenius norm) so one weight covers fine structures "
                        "and background splats alike")
    p.add_argument("--w_aniso", type=float, default=0.0,
                   help="L1 on the sorted, mean-removed log-scale triple: a Gaussian's shape "
                        "with size AND orientation divided out. w_cov3d does not cover this. "
                        "Measured on T16k step 70000, giving covariance3d_loss the GROUND-TRUTH "
                        "rotation changes it by 0.0% while giving it the ground-truth scale "
                        "removes 99.1%, and trace-normalising leaves only 1.6% rotation "
                        "sensitivity -- because R S S^T R^T loses R when S is spherical and the "
                        "model emits near-spheres (anisotropy p50 3.73 vs GT 15.51). Rotation "
                        "is not learnable until the ellipsoids elongate, and this is the term "
                        "that elongates them: it keeps a gradient on a sphere. Sorted so the "
                        "axis-permutation ambiguity that killed the separate rot term cannot "
                        "reach it")
    p.add_argument("--w_splat_area", type=float, default=0.0,
                   help="hinge on opacity * (projected radius / GT quantile radius - 1)^2. "
                        "Blocks the shortcut a wider upper cap would otherwise open")
    p.add_argument("--splat_area_quantile", type=float, default=0.80,
                   help="GT projected-radius quantile the splat-area hinge compares against. "
                        "0.99 was a no-op on R8H (logged splat=0.00000 every step): scene-wide "
                        "p99 is the legitimate giant tail, so hole-covering blobs never tripped "
                        "the hinge")
    # Width of the appearance encoder, which had no CLI knob at all. It is the
    # module that must compress a group's 64x11 = 704 attribute numbers into the
    # pack's aux block, and at hidden=64 it holds 59,073 parameters -- against
    # 3.6M in the attribute decoder and 17.0M in the compressor. Measured, the
    # code that reaches 21.01 dB exists inside the 16-channel-per-group space and
    # this model's own frozen decoder can decode it, but the encoder produces one
    # worth 15.49 dB. That 5.52 dB is what this width is for.
    p.add_argument("--attr_nbr_window", type=int, default=0,
                   help="neighbouring groups' codes the attribute decoder may read, each "
                        "side (Morton order). 0 = own code only, the measured 36.5%% "
                        "within-group ceiling.")
    p.add_argument("--attr_pack_hidden", type=int, default=64,
                   help="hidden width of encoder.attr_encoder (appearance compression)")
    p.add_argument("--attr_pack_dim", type=int, default=0,
                   help="attribute channels per point entering the pack, replacing the mask block. "
                        "0 = the historical xyz-only pack (appearance never reaches z_compact). "
                        "11 = log_scale 3 + quat 4 + logit opacity 1 + SH DC 3")
    p.add_argument("--uniform_budget", action="store_true")
    p.add_argument("--allow_padded_tokens", action="store_true")
    p.add_argument("--no_latent_zorder", action="store_true")

    # widths
    p.add_argument("--model_dim", type=int, default=384)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--map_blocks", type=int, default=0,
                   help="ResBlock2d stem depth on z_raw before compress")
    p.add_argument("--compress_intra_layers", type=int, default=3)
    p.add_argument("--compress_window_layers", type=int, default=2)
    p.add_argument("--compress_window", type=int, default=8)
    p.add_argument("--compress_mid_channels", type=int, default=32)
    p.add_argument("--compress_wide_channels", type=int, default=0,
                   help="optional z_mid width on compact grid; 0 disables. "
                        "Final z_compact channels stay compact_latent_channels.")
    p.add_argument("--compress_wide_layers", type=int, default=0,
                   help="ResBlock2d depth on z_mid before project-to-compact")
    p.add_argument("--decompress_intra_layers", type=int, default=3)
    p.add_argument("--decompress_window_layers", type=int, default=2)
    p.add_argument("--decoder_layers", type=int, default=4)
    p.add_argument("--residual_scale", type=float, default=0.6,
                   help="point refinement budget, as a fraction of the group extent")
    p.add_argument("--patch_chunk", type=int, default=512)
    p.add_argument("--gen_window_layers", type=int, default=3)
    p.add_argument("--gen_group_layers", type=int, default=1)
    p.add_argument("--gen_cross_layers", type=int, default=3)
    p.add_argument("--gen_region_chunk", type=int, default=256)
    p.add_argument("--gen_offset_scale", type=float, default=1.0,
                   help="coarse spread of generated points, in units of the group extent")
    p.add_argument("--no_gen_branch", action="store_true")
    p.add_argument("--no_gen_detach", action="store_true",
                   help="let the gen losses backprop into z_compact (the old behaviour). "
                        "Off by default: the gen decoder is weaker than the codec, so "
                        "co-optimising the latent for it smooths away the codec's detail")
    p.add_argument("--no_decoder_neighbor_context", action="store_true")
    p.add_argument("--no_gen_neighbor_context", action="store_true")
    p.add_argument("--encoder_residual", action="store_true")
    p.add_argument("--pack_trainable", action="store_true",
                   help="let the patch pack learn instead of staying the frozen identity "
                        "matrix. The freeze only ever made sense while w_z_raw supervised "
                        "z_raw against the pack itself -- with that term off, a frozen "
                        "identity means the encoder's geometry path has zero trainable "
                        "parameters (measured: ||dp||=0 on the encoder group for every step "
                        "of every run so far) and the 'encoder' is really Morton sort plus "
                        "Hungarian slot assignment. Identity stays the initialisation")
    p.add_argument("--anchor_spill", type=int, default=8,
                   help="how many nearest anchors a point may fall through to when the "
                        "nearer ones are full. 1 = strict nearest-anchor, which drops the "
                        "surplus of every over-subscribed anchor: measured 26.1%% of the "
                        "selected points lost on step_005070 and 15.9%% on step_018020. "
                        "8 recovers most of it (15.6%% / 6.1%%)")
    p.add_argument("--anchor_assignment", type=str, default="spill",
                   choices=["spill", "capacitated"])
    p.add_argument("--scene_anchors", type=str, default="",
                   help="npy of (num_groups, 3) fixed k-means anchors. Assigns each point to "
                        "its nearest anchor instead of cutting the Morton order into equal "
                        "chunks, so latent cell j means the same region in every snapshot. "
                        "Required for the world model: with Morton chunks, densification "
                        "shifts every boundary and the measured cell-centre drift between "
                        "adjacent snapshots is 13.9x the group radius (0.13x with anchors)")
    p.add_argument("--latent_mode", type=str, default="deterministic", choices=["deterministic", "vae"])
    p.add_argument("--denoise_std", type=float, default=0.02)
    p.add_argument("--no_checkpoint_decode", action="store_true")
    p.add_argument("--checkpoint_gen", action="store_true")
    p.add_argument("--folding_decode", action="store_true",
                   help="fixed unit-shell + anisotropic frame + zero-init residual "
                        "(FoldingNet / Scaffold-GS style); avoids shape_xyz rank collapse")
    p.add_argument("--folding_aniso_log_cap", type=float, default=1.5,
                   help="tanh bound on the folding frame's per-axis log-scale ratio. The "
                        "geometric-mean normalisation constrains the PRODUCT but not the "
                        "RATIO, so measured aniso ran 0.027..1411.5 -- inflating offsets 13x "
                        "and dropping group-shape effective rank 5.80 -> 4.24. 1.5 allows a "
                        "4.5x stretch per axis, which covers the data's anisotropy")
    p.add_argument("--folding_res_cap", type=float, default=1.0,
                   help="tanh bound on the folding residual, in template units. Free: rank-22 "
                        "oracle chamfer 0.218 uncapped vs 0.217 at 1.2 and 0.221 at 0.8")
    p.add_argument("--folding_res_start", type=int, default=800,
                   help="step at which the folding residual starts to open up")
    p.add_argument("--folding_res_ramp_steps", type=int, default=700)
    p.add_argument("--folding_frame_channels", type=int, default=6,
                   help="3 per-axis scales + 3 axis-angle rotation. 3 = scale only "
                        "(measured envelope chamfer 0.336 vs 0.301 with rotation)")
    p.add_argument("--no_residual_pack", action="store_true",
                   help="pack absolute xyz (legacy). Default is residual_pack=(xyz-centroid).")
    p.add_argument("--shortcut_alpha_init", type=float, default=0.0,
                   help="decompressor free-shortcut mix at the start of training (0 = no free centroid)")
    p.add_argument("--shortcut_alpha_final", type=float, default=0.0,
                   help="optional late assist; keep 0 unless shape has already learned")
    p.add_argument("--shortcut_alpha_start", type=int, default=10**9,
                   help="step when shortcut_alpha begins ramping toward final")
    p.add_argument("--shortcut_alpha_ramp_steps", type=int, default=3000)
    p.add_argument("--shortcut_alpha_eval", type=float, default=0.0)
    p.add_argument("--decoder_refine_start", type=int, default=-1,
                   help="step when codec refine residual begins to ramp; default=latent_end")
    p.add_argument("--decoder_refine_ramp_steps", type=int, default=1500)

    # curriculum
    p.add_argument("--max_steps", type=int, default=60000)
    p.add_argument("--stop_at", type=int, default=0,
                   help="exit after this completed step (0 = run to max_steps). "
                        "Does not change the cosine which still uses max_steps.")
    p.add_argument("--latent_end", type=int, default=3000)
    p.add_argument("--late_decode_steps", type=int, default=1000)
    p.add_argument("--geo_end", type=int, default=9000)
    p.add_argument("--geo_weak_scale", type=float, default=0.15)
    p.add_argument("--gen_start", type=int, default=5000)
    p.add_argument("--gen_ramp_steps", type=int, default=3000)
    p.add_argument("--gen_end", type=int, default=15000)
    p.add_argument("--polish_start", type=int, default=0,
                   help="step at which polish LR multipliers override the regular "
                        "curriculum; 0 disables polish")
    p.add_argument("--polish_encoder_scale", type=float, default=0.0,
                   help="encoder LR multiplier in polish")
    p.add_argument("--polish_attr_encoder_scale", type=float, default=0.0,
                   help="appearance-encoder LR multiplier in polish; non-zero lets "
                        "rendering improve the appearance code without fully reopening geometry")
    p.add_argument("--polish_compressor_scale", type=float, default=0.0,
                   help="compressor LR multiplier in polish")
    p.add_argument("--polish_decompressor_scale", type=float, default=0.1,
                   help="decompressor LR multiplier in polish")
    p.add_argument("--polish_decoder_scale", type=float, default=1.0,
                   help="geometry decoder LR multiplier in polish")
    p.add_argument("--polish_attr_scale", type=float, default=1.0,
                   help="attribute decoder LR multiplier in polish")
    p.add_argument("--polish_attr_slot_scale", type=float, default=-1.0,
                   help="LR multiplier for attr_decoder.slot_emb alone. -1 follows "
                        "--polish_attr_scale. The per-slot basis is the only "
                        "group-independent per-slot signal the module has and it moved "
                        "1.47x in 14000 steps at the shared rate.")
    p.add_argument("--latent_decay_steps", type=int, default=6000)
    p.add_argument("--diversity_decay_steps", type=int, default=8000)
    p.add_argument("--encoder_residual_start", type=int, default=12000)
    p.add_argument("--attr_start", type=int, default=10 ** 9)
    p.add_argument("--attr_loss_ramp_steps", type=int, default=1,
                   help="ramp attribute parameter losses (scale/rot/opacity/colour/sh/cov3d) "
                        "after attr_start; prevents a hard shared-latent gradient shock after "
                        "geometry pretraining. THE DEFAULT OF 1 IS A HARD SWITCH and defeats "
                        "the mechanism: at attr_start these weights go 0 -> full in one step "
                        "while the render side ramps over render_ramp_steps and starts at "
                        "1/render_downscale_start resolution. Measured at the switch, cov3d "
                        "alone took 35-46%% of the objective (raw c3d 56-84 against ~0.9 a "
                        "hundred steps later), and in P16k 95%% of batches between 1350 and "
                        "2300 exceeded the 25x spike guard and were dropped. Set this to "
                        "render_ramp_steps so both sides reach full strength together.")
    p.add_argument("--no_teacher_cycle_freeze", action="store_true",
                   help="let the teacher-cycle term reach the decoder's weights (A/B only; "
                        "the decoder can then satisfy it by going deaf to its input)")
    p.add_argument("--attr_force_steps", type=int, default=2000,
                   help="steps of pure teacher forcing on the attribute heads' xyz input")
    p.add_argument("--attr_anneal_steps", type=int, default=4000,
                   help="scheduled-sampling window from teacher forcing to inference conditioning")
    p.add_argument("--stage", type=str, default="xyz", choices=["xyz", "geometry", "full"])

    # optimisation
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_min", type=float, default=1e-5)
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--loss_spike_mult", type=float, default=25.0,
                   help="drop a batch whose loss exceeds this multiple of the running "
                        "median; 0 disables. Non-finite losses are always dropped. "
                        "grad_clip bounds the update but not what AdamW's second "
                        "moment absorbs, so a clipped spike still distorts the "
                        "effective lr for tens of steps afterwards")
    p.add_argument("--grad_spike_mult", type=float, default=3.0,
                   help="skip the optimizer step when the pre-clip grad norm exceeds "
                        "this multiple of the median over the first 32 accepted steps. "
                        "The reference is frozen, and it is collected again once "
                        "detail_start is reached, because full-res / edge / VGG raise "
                        "the normal grad norm. 0 disables")
    p.add_argument("--grad_spike_abs", type=float, default=6000.0,
                   help="always skip a step whose pre-clip grad norm exceeds this, "
                        "including while the anchor is still being collected. 0 disables. "
                        "B1's collapse was 6k-1e5; the detail phase sits around 2k-3k")
    p.add_argument("--near_detach_frac", type=float, default=0.1,
                   help="in each render view, points closer than this fraction of the "
                        "median depth stay in the image but their parameters are "
                        "detached, so 1/z rasterizer gradients cannot dominate the "
                        "step. 0 disables. The fraction is relative to that view, so "
                        "the same value applies on every dataset")
    p.add_argument("--latent_scale_sync_every", type=int, default=50,
                   help="all-reduce the latent_scale buffer across ranks every N "
                        "steps. DDP runs with broadcast_buffers=False, so without "
                        "this each rank fits its own scale and only rank 0's is "
                        "checkpointed")
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--encoder_warmup_steps", type=int, default=800)
    p.add_argument("--encoder_warmup_decoder_lr_scale", type=float, default=0.5)
    p.add_argument("--encoder_warmup_gen_lr_scale", type=float, default=0.25)
    p.add_argument("--late_codec_lr_scale", type=float, default=0.4)
    p.add_argument("--late_codec_start", type=int, default=-1,
                   help="step to start applying late_codec_lr_scale to encoder/compressor. "
                        "-1 = gen_end (delay until shape is more stable)")

    # codec weights
    p.add_argument("--w_xyz", type=float, default=40.0)
    p.add_argument("--w_xyz_mse", type=float, default=4.0)
    p.add_argument("--w_xyz_hard", type=float, default=14.0)
    p.add_argument("--w_chamfer", type=float, default=3.0)
    p.add_argument("--w_coverage", type=float, default=0.0,
                   help="GT→pred onesided chamfer; fills holes that set-chamfer alone under-penalizes")
    p.add_argument("--w_voxel_occ", type=float, default=0.0,
                   help="soft 3D voxel occupancy miss on GT-occupied cells")
    p.add_argument("--w_plane_chamfer", type=float, default=2.5)
    p.add_argument("--w_proj_hist", type=float, default=2.5)
    p.add_argument("--w_dispersion", type=float, default=3.0)
    p.add_argument("--w_latent_decorr", type=float, default=0.0,
                   help="off-diagonal penalty on the shape-channel correlation matrix "
                        "(VICReg/Barlow-Twins covariance term). w_latent_std is per-channel "
                        "and blind to correlation: every checkpoint measured healthy std "
                        "0.25-0.57 with the shape block stuck at effective rank ~4, so "
                        "widening the budget 12 -> 28 bought nothing")
    p.add_argument("--w_z_intra_chamfer", type=float, default=0.0,
                   help="permutation-invariant within-group set distance on the PACKED "
                        "reconstruction. Unlike --w_intra_chamfer it needs no decoder, so it "
                        "is active during the latent phase, where the collapse to the "
                        "conditional mean actually happens (measured aniso_lam2 0.959 -> "
                        "0.082 between steps 500 and 1000 with only w_z_residual on)")
    p.add_argument("--w_intra_chamfer", type=float, default=0.0,
                   help="permutation-invariant within-group set distance, extent-normalised")
    p.add_argument("--w_intra_sinkhorn", type=float, default=0.0,
                   help="balanced within-group transport; unlike Chamfer, duplicated "
                        "predictions must cover distinct target points")
    p.add_argument("--w_group_centroid", type=float, default=0.0,
                   help="extent-normalised centroid loss for stable fixed-anchor group pose")
    p.add_argument("--geometry_target_scale", type=int, default=0,
                   help="normalise local set losses by detached target-group extent")
    p.add_argument("--geometry_center_local", type=int, default=0,
                   help="remove pred/target group centres before local Chamfer/Sinkhorn")
    p.add_argument("--geometry_centroid_absolute", type=int, default=0,
                   help="train group translation with an absolute centroid loss, separate "
                        "from target-normalised local shape")
    p.add_argument("--sinkhorn_epsilon", type=float, default=0.08,
                   help="entropic regularisation of the within-group transport. NOT a "
                        "free knob: at 0.08 the plan is not an assignment -- measured "
                        "exp(H)=4.7-6.1 targets per prediction, peak 0.34-0.41 against "
                        "1.0 for a hard match -- so the anti-duplication gradient is "
                        "averaged away and nn_unique sits at 0.45 no matter the weight.")
    p.add_argument("--sinkhorn_epsilon_final", type=float, default=0.0,
                   help="anneal target for sinkhorn_epsilon; 0 disables the anneal. "
                        "0.005 measured exp(H)=1.04-1.13, i.e. an actual permutation.")
    p.add_argument("--sinkhorn_anneal_start", type=int, default=0)
    p.add_argument("--sinkhorn_anneal_steps", type=int, default=1)
    p.add_argument("--sinkhorn_iterations", type=int, default=6)
    p.add_argument("--sinkhorn_chunk", type=int, default=256)
    p.add_argument("--w_intra_spacing", type=float, default=0.0,
                   help="Hinge on the within-group nearest-neighbour SPACING. The "
                        "measured failure it targets: each group emits 64 points onto "
                        "~32 distinct places (within-group unique 50.9%, NN spacing "
                        "0.261 of the group radius against ~0.5 for a well-spread set), "
                        "so scene coverage is 49.3% and rendering half the surface "
                        "costs 28.5 dB. Symmetric chamfer scores duplication as "
                        "perfect and the one-sided coverage term is in the wrong units, "
                        "so no active term forbids it.")
    p.add_argument("--intra_spacing_ratio", type=float, default=1.0,
                   help="Fraction of the GT group's own median NN spacing to demand. "
                        "1.0 asks for the ground truth's density; below 1.0 leaves slack.")
    p.add_argument("--w_p2g", type=float, default=0.0,
                   help="Within-group PRECISION: predicted -> nearest GT. Measured, the "
                        "median predicted point sits 0.315 group radii off the GT "
                        "surface, which the symmetric intra_chamfer averages away.")
    p.add_argument("--w_radius", type=float, default=0.0,
                   help="Within-group group-RADIUS match. Measured, the predicted "
                        "group radius is 0.670x the GT group's -- the group is shrunk "
                        "toward its centroid. Also disambiguates the spacing hinge, "
                        "which could otherwise be satisfied by pushing apart the "
                        "near-coincident pairs the GT genuinely uses (its median "
                        "within-group NN distance is 0.007 group radii).")
    p.add_argument("--w_attr_spread", type=float, default=0.0,
                   help="셀 내부 속성 표준편차를 GT 에 맞춘다 (표준화 채널). 측정: "
                        "logscale 비율 0.39, opacity 0.50, rot 0.53, sh_dc 0.66 "
                        "인 반면 xyz 는 0.98 -- 위치는 정상적으로 퍼지는데 속성만 "
                        "셀 평균으로 뭉친다. 난간·모서리는 이웃과 크기가 다른 "
                        "길쭉한 가우시안 몇 개라 이 붕괴가 곧 디테일 소실이다. "
                        "std 는 집합 함수라 슬롯 순서에 무관하다.")
    p.add_argument("--w_attr_slope", type=float, default=0.0,
                   help="셀 로컬 좌표 -> 속성의 11x3 최소제곱 기울기를 GT 에 맞춘다. "
                        "측정: 셀 내부 위치가 속성을 설명하는 정도가 GT 0.234 인데 "
                        "예측은 0.045 (logscale, R^2). 속성 헤드가 attr_read_shape=1 "
                        "로 위치를 읽으면서도 공간 변조를 거의 하지 않는다.")
    p.add_argument("--w_attr_hung_scale", type=float, default=0.0,
                   help="Per-cell xyz Hungarian then standardised L1 on log-scale. "
                        "Unlike sinkhorn/slot targets this does not blend a thin and "
                        "a fat Gaussian that share a position.")
    p.add_argument("--w_attr_hung_opacity", type=float, default=0.0,
                   help="Same matching as w_attr_hung_scale, on logit opacity.")
    p.add_argument("--w_attr_hung_rot", type=float, default=0.0,
                   help="Same matching as w_attr_hung_scale, on sign-fixed quaternion.")
    p.add_argument("--w_attr_set", type=float, default=0.0,
                   help="Permutation-invariant SET distance between a group's 64 "
                        "predicted attribute vectors and the GT's 64. Every other "
                        "attribute target here is keyed on position -- the slot "
                        "assignment ignores attributes (co-located GT pairs: the lower "
                        "slot holds the larger Gaussian 47.0% of the time, a coin flip) "
                        "and the sinkhorn plan costs cdist(pred_xyz, gt_xyz) only, so a "
                        "co-located pair yields a BLENDED target. Measured, the model's "
                        "attribute set sits 5.7% from a total collapse to its group mean "
                        "and right on top of the GT's group mean, and no term sees it.")
    p.add_argument("--w_attr_local_set", type=float, default=0.0,
                   help="attr_set on a k-NN neighbourhood inside the cell, not the "
                        "whole 256-slot bag. Cell-wide attr_set at group 256 can be "
                        "met by medium spheres; a local bag at a rail cannot. Still a "
                        "set distance — not a GT quaternion copy. xyz is detached.")
    p.add_argument("--attr_local_set_k", type=int, default=16,
                   help="Neighbourhood size for w_attr_local_set. Must stay below "
                        "typical occupancy (~71) or the term equals cell-wide attr_set. "
                        "k=1 is the closed nearest-neighbour copy.")
    p.add_argument("--attr_local_set_chunk", type=int, default=32)
    p.add_argument("--attr_set_chunk", type=int, default=512)
    p.add_argument("--sinkhorn_attr_weight", type=float, default=0.0,
                   help="Weight on standardised attribute channels inside the sinkhorn "
                        "transport cost. 0 keeps the historical position-only plan. "
                        "Above 0 lets the plan prefer the GT Gaussian a prediction "
                        "already resembles, turning a blended target into a choice.")
    p.add_argument("--intra_chamfer_chunk", type=int, default=1024)
    p.add_argument("--detail_ramp_steps", type=int, default=500,
                   help="ramp for the permutation-invariant within-group terms, "
                        "measured from when the decoder starts running")
    p.add_argument("--w_presence", type=float, default=0.05)
    p.add_argument("--w_scale", type=float, default=2.0)
    p.add_argument("--w_rot", type=float, default=2.0)
    p.add_argument("--w_opacity", type=float, default=1.0)
    p.add_argument("--w_color", type=float, default=0.5)
    p.add_argument("--w_sh", type=float, default=0.1)
    p.add_argument("--xyz_beta", type=float, default=0.002,
                   help="smooth-L1 knee; the median coordinate error is ~7e-4, so a large\n                         beta parks the geometry losses in their quadratic region")
    p.add_argument("--hard_frac", type=float, default=0.05)
    p.add_argument("--hard_min_points", type=int, default=2048)
    p.add_argument("--chamfer_scales", type=str, default="1024,4096,16384")
    p.add_argument("--chamfer_scale_weights", type=str, default="0.2,0.3,0.5")
    p.add_argument("--balanced_chamfer", action="store_true")
    p.add_argument("--coverage_samples", type=int, default=16384)
    p.add_argument("--voxel_occ_bins", type=int, default=24)
    p.add_argument("--plane_chamfer_samples", type=int, default=16384)
    p.add_argument("--proj_hist_samples", type=int, default=8192)
    p.add_argument("--proj_hist_bins", type=int, default=64)
    p.add_argument("--proj_hist_sigma", type=float, default=0.025)
    p.add_argument("--proj_hist_scale", type=float, default=100.0)
    p.add_argument("--dispersion_group_size", type=int, default=32)
    p.add_argument("--dispersion_margin", type=float, default=0.001,
                   help="absolute floor on within-group std (was 0.01 = always-on blur)")
    p.add_argument("--dispersion_margin_frac", type=float, default=0.35,
                   help="also require std >= frac * GT group std")
    p.add_argument("--spatial_bins", type=int, default=32)
    p.add_argument("--spatial_weight_power", type=float, default=0.5)
    p.add_argument("--spatial_weight_max", type=float, default=8.0)

    # latent weights
    p.add_argument("--w_z_raw", type=float, default=10.0)
    p.add_argument("--w_z_raw_mse", type=float, default=1.5)
    p.add_argument("--w_z_hard", type=float, default=3.0)
    p.add_argument("--w_z_token_var", type=float, default=1.5)
    p.add_argument("--w_z_std_ratio", type=float, default=1.5)
    p.add_argument("--pretrain_w_z_raw", type=float, default=30.0)
    p.add_argument("--pretrain_w_z_raw_mse", type=float, default=3.0)
    p.add_argument("--pretrain_w_z_hard", type=float, default=10.0)
    p.add_argument("--pretrain_w_z_token_var", type=float, default=6.0)
    p.add_argument("--pretrain_w_z_std_ratio", type=float, default=3.0)
    p.add_argument("--z_hard_frac", type=float, default=0.12)
    p.add_argument("--z_hard_min_tokens", type=int, default=256)
    p.add_argument("--z_std_ratio_target", type=float, default=1.0)
    p.add_argument("--w_kl", type=float, default=0.0)
    p.add_argument("--w_latent_std", type=float, default=1.0,
                   help="band hinge keeping shape-channel std inside [floor, ceil]")
    p.add_argument("--latent_std_floor", type=float, default=0.25)
    p.add_argument("--latent_std_ceil", type=float, default=1.5,
                   help="0 disables the upper bound; otherwise stops latent scale drift")
    # within-group detail: without these the model settles on "all points at centroid"
    p.add_argument("--w_z_residual", type=float, default=40.0)
    p.add_argument("--w_z_residual_hard", type=float, default=20.0)
    p.add_argument("--pretrain_w_z_residual", type=float, default=80.0)
    p.add_argument("--pretrain_w_z_residual_hard", type=float, default=40.0)
    p.add_argument("--w_learned_residual", type=float, default=20.0)
    p.add_argument("--pretrain_w_learned_residual", type=float, default=40.0)
    p.add_argument("--w_shape_direct", type=float, default=10.0,
                   help="supervise the shape->offset MLP alone; stops the deep context "
                        "path from serving the whole residual and collapsing shape channels")
    p.add_argument("--pretrain_w_shape_direct", type=float, default=20.0)
    p.add_argument("--w_shape_sinkhorn", type=float, default=0.0,
                   help="permutation-invariant one-to-one supervision on the direct "
                        "shape->offset path; use instead of shape_direct for anchors")
    p.add_argument("--w_res_ratio", type=float, default=0.0,
                   help="optional magnitude hinge on learned/extent; prefer residual_pack over this")
    p.add_argument("--res_ratio_floor", type=float, default=0.0,
                   help="floor for w_res_ratio (disabled by default; magnitude-only and brittle)")
    p.add_argument("--residual_hard_frac", type=float, default=0.2)
    p.add_argument("--w_xyz_residual", type=float, default=30.0)
    p.add_argument("--w_xyz_residual_hard", type=float, default=15.0)
    p.add_argument("--w_equiv", type=float, default=1.0)
    p.add_argument("--equiv_start", type=int, default=0,
                   help="delay equivariance until anchor geometry is established")
    p.add_argument("--equiv_ramp_steps", type=int, default=1,
                   help="linear ramp length for w_equiv after --equiv_start")
    p.add_argument("--equiv_every", type=int, default=4)
    p.add_argument("--equiv_rot_deg", type=float, default=10.0)
    p.add_argument("--equiv_shape_weight", type=float, default=0.0,
                   help="0 = centroid-only equivariance (shape invariance kills detail)")

    # render loss -- the only non-proxy term. Off by default; needs the CUDA
    # rasteriser (diff_gaussian_rasterization) and a per-frame camera in the npz.
    p.add_argument("--w_render", type=float, default=0.0,
                   help="NOT comparable to the point-space weights. A rasteriser gradient is "
                        "~1000x a chamfer gradient per unit weight at 262k points (measured, "
                        "tools/audit_render_scale.py), so useful values are ~1e-3, not ~10. "
                        "Measure before changing it")
    p.add_argument("--render_start", type=int, default=6000,
                   help="before this the point set is too coarse for the image gradient to help")
    p.add_argument("--render_ramp_steps", type=int, default=1000)
    # Deep-feature term on the attribute render. L1 and SSIM are both distortion
    # metrics minimised by the conditional mean, which is a blurred low-contrast
    # image; measured, the model's within-group colour variation is 14.9% of total
    # against 74.7% for the ground truth, and raising lam_dssim 0.2 -> 0.45 did not
    # move it. A VGG feature loss scores texture statistics instead.
    # Path to a {npz basename: image path} map. With it the attribute render is
    # compared against the actual photograph rather than a render of the target
    # Gaussians. Measured: model 14.55 dB vs photo, target Gaussians 21.75 dB vs
    # photo, model 16.02 dB vs the target render -- i.e. the current reference is
    # itself 7.2 dB short of the thing we actually want to match.
    p.add_argument("--photo_map", type=str, default="",
                   help="JSON mapping npz basename -> ground-truth image path")
    # Extra REAL views per sample. The dataset is 3000 snapshots of one scene, so
    # all 301 photographs are valid views of every snapshot. Without this each
    # sample carries exactly one real image, and the jittered views only re-render
    # the target Gaussians, adding no information. Measured consequence: fitting a
    # code to 4 views gains +3.71 dB on those views and loses 0.93 dB on 4 held-out
    # ones -- appearance is under-determined, so the loss-minimising answer is the
    # conditional mean, i.e. the washed-out render. Per-scene methods use ~300
    # images per scene; this was using 1.
    p.add_argument("--compress_head_stages", type=int, nargs="*", default=[],
                   help="widths between mid and the budget heads, e.g. 128 64. "
                        "Empty = the old single Linear, which measured effective rank "
                        "10.91 of 32 channels at z_compact.")
    p.add_argument("--max_input_points", type=int, default=0,
                   help="points the ENCODER may read (0 = same as max_points). The decoder "
                        "still emits max_points. A larger independent sample made z_compact "
                        "describe a different set than the decoder target, so a world-model "
                        "cell had no stable Gaussian identity. Keep 0 unless you explicitly "
                        "want extra encoder capacity.")
    p.add_argument("--cache_slots", action="store_true",
                   help="reuse the first (selected, slot_src) packing of each npz. Combined "
                        "with no augment and crop_prob=0 this keeps slot j the same Gaussian "
                        "across epochs, which invertibility losses need.")
    p.add_argument("--pool_dim", type=int, default=128,
                   help="width of the per-group learned pooling that replaces the identity pack")
    p.add_argument("--pool_chunk", type=int, default=-1,
                   help="Groups per pooling chunk. -1 keeps pool_chunk*group_in "
                        "constant, so raising the per-cell cap costs no extra memory. "
                        "The cap matters: the 144 implied by max_input_points 589824 "
                        "drops 14-23%% of the points of this dataset's densest snapshots "
                        "before the encoder sees them (max cell count 768); 512 drops "
                        "0.0-0.2%%.")
    p.add_argument("--pool_queries", type=int, default=16,
                   help="internal learned set queries before projection to the fixed patch")
    p.add_argument("--allow_low_shape_budget", action="store_true",
                   help="override the shape-channel floor in validate_layout. Only correct "
                        "when the split was chosen on measured rendered dB: at 16384 the "
                        "cell layout is fixed at 1024x16, and appearance was worth more per "
                        "channel than shape (appearance 4->8 gained more than shape 3->7), "
                        "so intra-cell geometry is deliberately left to the centroid.")
    p.add_argument("--shape_xyz_hidden", type=int, default=0,
                   help="hidden width of the shared shape -> per-point offset map. "
                        "0 keeps max(128, 4*group_size). Raise it when template_erank "
                        "is still climbing at the end of a run: Q16kD finished 64000 "
                        "steps at 28.4 against the GT's 36.3 with every other geometry "
                        "metric flat, and 1024 cells share this one dictionary.")
    p.add_argument("--pool_blocks", type=int, default=1,
                   help="staged cross-attention reductions in the pooler. 1 = the original "
                        "single collapse of group_in points into pool_queries. >1 adds query "
                        "self-attention between reductions so the queries can divide the group "
                        "up instead of independently summarising the same points.")
    p.add_argument("--pool_feedback", type=int, default=1,
                   help="with pool_blocks>1, let the point features re-read the running summary "
                        "between reductions (COD-VAE style). Costs one extra cross-attention per "
                        "block on the wide point side.")
    p.add_argument("--use_fixed_anchor_center", type=int, default=0,
                   help="use transformed scene anchors as canonical group origins")
    # New-run defaults are 1. build_config / dataset getattr default 0 so an
    # old args.json (S/T/E1) stays on the historical contract when re-evaluated.
    p.add_argument("--normalize_pooler_xyz", type=int, default=1,
                   help="divide pooler local xyz by the analytic cell extent")
    p.add_argument("--count_aware_template", type=int, default=1,
                   help="mean-center the folding template on the predicted live prefix")
    p.add_argument("--attr_slot_mask", type=int, default=1,
                   help="AttributeDecoder local PE / self-attention ignore inactive slots")
    p.add_argument("--shared_cell_owner", type=int, default=1,
                   help="one cell-ownership plan for encoder input and decoder target")
    p.add_argument("--holdout_own_photo", type=int, default=1,
                   help="apply the held-out exclude set to a snapshot's own photograph")
    p.add_argument("--keep_extra_fullres", type=int, default=1,
                   help="keep extra training photographs at native resolution")
    p.add_argument("--reuse_prev_vis", type=int, default=0,
                   help="reuse previous-batch visibility for attribute anchors (unsafe under shuffle)")
    p.add_argument("--eval_pred_mask", type=int, default=1,
                   help="render eval with predicted presence (the deployable set)")
    p.add_argument("--shared_free_head", type=int, default=0,
                   help="one non-anchor compact head instead of separate shape/appearance heads")
    p.add_argument("--structured_local_code", type=int, default=0,
                   help="store learned compact channels as global + K local tokens pooled "
                        "directly from the encoder token set")
    p.add_argument("--compact_global_dim", type=int, default=3)
    p.add_argument("--compact_local_tokens", type=int, default=4)
    p.add_argument("--compact_local_dim", type=int, default=6)
    p.add_argument("--compress_merge_attn", action="store_true",
                   help="pool a group's tokens with a learned query (COD-VAE style) "
                        "instead of concatenating them; the measured rank collapse "
                        "34.26 -> 13.83 happens at that concat")
    p.add_argument("--compress_merge_stages", type=int, default=0,
                   help="1 adds a 4*d stage to token_merge (measured rank 34.26 -> 13.83 there)")
    p.add_argument("--view_pool", type=str, default="",
                   help="JSON list of {image, cam[16]} real views of the scene")
    p.add_argument("--extra_real_views", type=int, default=0,
                   help="real photographs sampled per training sample, on top of its own")
    p.add_argument("--view_downscale", type=int, default=2,
                   help="downscale applied when loading pool images (I/O and memory)")
    p.add_argument("--w_render_perc", type=float, default=0.0,
                   help="weight of the VGG feature loss inside the attribute render")
    p.add_argument("--render_lam_dssim", type=float, default=0.2,
                   help="3DGS's own L1/D-SSIM blend. NOT worth ramping late: raising it "
                        "0.2 -> 0.45 was measured not to move the within-group colour "
                        "variation (14.9%% against the GT's 74.7%%), because L1 and SSIM "
                        "are both distortion metrics minimised by the same blurred mean.")
    p.add_argument("--detail_start", type=int, default=0,
                   help="step at which the late detail phase begins. 0 disables it and "
                        "every knob below is inert, so existing configs are unchanged. "
                        "Late on purpose: a sparse high-resolution image gradient is the "
                        "wrong signal while the geometry is still moving.")
    p.add_argument("--detail_phase_ramp", type=int, default=2000,
                   help="linear ramp for edge_gain / w_render_perc and the log ramp for "
                        "render_downscale_final, both starting at detail_start. Named apart "
                        "from --detail_ramp_steps, which is the within-group term ramp and "
                        "has nothing to do with this phase.")
    p.add_argument("--render_downscale_final", type=int, default=0,
                   help="second-leg render downscale reached during the detail phase. "
                        "0 = stay at --render_downscale. The first leg stops at 2, i.e. "
                        "488x272 of a 977x544 photograph, so structures under two source "
                        "pixels are below Nyquist and never supervised at all.")
    p.add_argument("--w_sobel", type=float, default=0.0,
                   help="L1 between the Sobel gradients of the predicted and reference "
                        "renders, added to the render L1. edge_gain only decides where "
                        "the L1 is spent, which a blur that keeps the local mean still "
                        "satisfies; this asks for the edge itself. Measured on the "
                        "attribute decoder with 14 supervised views, scored on 5 views "
                        "that were never supervised: the handrail crop goes 21.89 -> "
                        "21.15 without it and 21.89 -> 25.61 at 1.0, lettering 21.92 -> "
                        "21.01 vs 21.92 -> 24.08. At 5.0 the crops go further (27.14 / "
                        "26.56) but the held-out mean stops moving, so 1.0 is the "
                        "operating point. Ramped with the other detail gains.")
    p.add_argument("--edge_gain", type=float, default=0.0,
                   help="w = 1 + edge_gain * |grad(ref)| / mean|grad(ref)| on the L1 term. "
                        "Measured on the step_028630 own-camera render: the flat half of "
                        "the pixels carries 28%% of the L1 at mean error 0.035 while the "
                        "top 5%% by gradient carries 11%% at 0.132. At gain 4 those become "
                        "6%% and 34%%. Reached over detail_phase_ramp from detail_start.")
    p.add_argument("--render_downscale", type=int, default=2,
                   help="977x544 -> 488x272; halves rasteriser time and backward memory")
    p.add_argument("--render_downscale_start", type=int, default=0,
                   help="coarse-to-fine: start rendering at this downscale and step down to "
                        "--render_downscale over --render_downscale_steps. 0 disables. This is "
                        "the mechanism that replaces the geometry detach: an image gradient is "
                        "badly conditioned for position TRANSPORT (measured cosine 0.13 with "
                        "the true correction, because the median Gaussian covers 1.4 px while "
                        "the error is ~6 px), and the fix is to make a pixel cover more world "
                        "-- at 8x downscale one pixel is ~4 group radii, so the gradient points "
                        "the right way over the range the error actually spans")
    p.add_argument("--render_downscale_steps", type=int, default=2000)
    p.add_argument("--attr_detach_release", type=int, default=-1,
                   help="step at which the attribute decoder stops detaching its position "
                        "input, i.e. when the render loss is allowed to refine geometry. -1 "
                        "never releases. Use with --attr_detach_geometry 1 to get the "
                        "geometry-first curriculum: positions learned by the set-level "
                        "geometry terms first, then the image allowed to correct them once "
                        "they are close enough for its gradient to point the right way")
    p.add_argument("--attr_detach_geometry", type=int, default=1,
                   help="1 = the attribute decoder's position input is detached, so the render "
                        "loss trains attributes only (the historical behaviour). 0 = the render "
                        "gradient also reaches the geometry decoder, compressor and encoder. "
                        "With it at 1 the ONLY objective aligned with held-out PSNR touches 3.6M "
                        "of the model's 118M parameters, and geometry is left to the "
                        "reconstruction terms whose joint optimum at this channel budget renders "
                        "at 9.95 dB (measured: 12 shape + 16 appearance channels, per-group PCA, "
                        "8 held-out photographs)")
    p.add_argument("--attr_frame_needles", type=int, default=0,
                   help="1 = AttributeDecoder scale is along the folding cell-frame axes and "
                        "rotation is a residual around that frame. Load a C1 checkpoint and "
                        "the rot head is re-inited to identity so the default Gaussian is a "
                        "sphere oriented with the cell.")
    p.add_argument("--attr_detach_scale_rot", type=int, default=0,
                   help="1 = photometric / VGG / Sobel do not flow to scale or rotation. "
                        "Colour and opacity still do. C1 Sobel 2000 steps lowered aniso p50 "
                        "3.88 -> 3.48; this blocks that path.")
    p.add_argument("--render_max_points", type=int, default=0,
                   help="0 = all valid points; a cap subsamples pred and target identically")
    p.add_argument("--render_gen_scale", type=float, default=1.0,
                   help="multiplier on w_render for the gen branch")
    p.add_argument("--render_min_coverage", type=float, default=0.25,
                   help="skip samples whose reference render fills less of the frame than this "
                        "(random crops leave both sides nearly black, giving a free zero loss)")
    p.add_argument("--render_views", type=int, default=1,
                   help="synthetic viewpoints per sample. 1 = the frame's own camera only, which "
                        "cannot certify 3D equivalence: what the camera misses is unconstrained "
                        "and scale/opacity can absorb geometric error. >1 costs one extra "
                        "reference + one extra gradient render each")
    p.add_argument("--render_view_jitter_deg", type=float, default=8.0,
                   help="orbit amplitude for the extra viewpoints")
    p.add_argument("--w_distill_attr", type=float, default=0.0,
                   help="distil the teacher's ATTRIBUTES to the student, normalised per channel "
                        "by the teacher's own spread. w_distill covers xyz only, so without this "
                        "the student's attribute heads receive no teacher signal at all")
    p.add_argument("--w_render_attr", type=float, default=0.0,
                   help="render loss with the POSITION path detached, so it drives only "
                        "scale/rotation/opacity/colour. Needs its own weight: measured at "
                        "w_render=0.004 (set from the xyz gradient) the render term was 0.0%% of "
                        "the attribute gradient while the parameter losses were 99.99%%, and the "
                        "model duly landed on the reconstruction ceiling (12.53 dB) instead of "
                        "the equivalence one (21.43 dB at the same 8-channel budget)")
    p.add_argument("--attr_param_decay_steps", type=int, default=0,
                   help="steps over which w_scale/w_rot/w_opacity/w_color decay to "
                        "attr_param_floor once the attribute render loss is live. Reconstruction "
                        "needs ~8 channels/point and the budget is 0.125, so holding the "
                        "parameter losses at full strength pins the model to a target it cannot "
                        "reach and fights the equivalence objective. 0 = no decay")
    p.add_argument("--attr_anchor_covered", type=float, default=1.0,
                   help="fraction of the attribute anchor kept on points the render "
                        "actually covers. 1.0 = the anchor applies everywhere (old "
                        "behaviour). Lower it to let the render own the points it can "
                        "see: on those the two objectives disagree and the anchor is "
                        "worse by a measured 10 dB (16.74 dB vs 26.77 dB on identical "
                        "positions), while off the covered set the render contributes "
                        "nothing at all and the anchor is the only signal")
    p.add_argument("--attr_responsibility", type=float, default=1.0,
                   help="1 = the attribute parameter loss targets the GT points each "
                        "prediction is actually the closest one to (union extent for scale, "
                        "composited opacity), 0 = the old slot-to-slot pairing. The slot "
                        "partner is the nearest GT point only 7%% of the time and renders "
                        "1.8 dB worse; the responsibility target is also what tells a "
                        "prediction standing in for several Gaussians to grow to cover them")
    p.add_argument("--attr_match_mode", type=str, default="responsibility",
                   choices=["responsibility", "sinkhorn", "slot"],
                   help="attribute correspondence. sinkhorn reuses the balanced xyz transport "
                        "plan and does not grow scale to merge several GT Gaussians")
    p.add_argument("--attr_param_floor", type=float, default=0.15,
                   help="fraction of the original attribute weights kept after the decay -- not "
                        "0, because they are what keeps the heads in a sane range while the "
                        "render loss reshapes them")

    # gen weights
    p.add_argument("--w_gen_xyz", type=float, default=28.0)
    p.add_argument("--w_gen_xyz_mse", type=float, default=3.0)
    p.add_argument("--w_gen_xyz_hard", type=float, default=10.0)
    p.add_argument("--w_gen_chamfer", type=float, default=3.5)
    p.add_argument("--w_gen_coverage", type=float, default=0.0)
    p.add_argument("--w_gen_voxel_occ", type=float, default=0.0)
    p.add_argument("--w_gen_plane_chamfer", type=float, default=3.0)
    p.add_argument("--w_gen_proj_hist", type=float, default=3.0)
    p.add_argument("--w_gen_dispersion", type=float, default=1.5)
    p.add_argument("--w_gen_intra_chamfer", type=float, default=0.0)
    p.add_argument("--w_gen_presence", type=float, default=0.05)
    p.add_argument("--w_gen_xyz_residual", type=float, default=24.0)
    p.add_argument("--w_gen_xyz_residual_hard", type=float, default=12.0)
    p.add_argument("--w_gen_p2g", type=float, default=0.0,
                   help="one-sided pred->GT precision on the student. Symmetric chamfer is "
                        "satisfiable by spreading; measured gen p->g 1.294 vs g->p 0.521")
    p.add_argument("--w_gen_radius", type=float, default=0.0,
                   help="|pred group radius / GT group radius - 1|. Measured gen radius 1.83x GT")
    p.add_argument("--w_teacher_cycle", type=float, default=0.0,
                   help="match decode(z_raw_hat) to decode(true z_raw). Asks whether z_compact "
                        "keeps what the decoder needs, which the pack-space losses do not")
    p.add_argument("--w_gen_basis", type=float, default=0.0,
                   help="distil the teacher's folding basis (frame + pre-scale offsets) "
                        "into the student. Matching only the final points let gen reach a similar set through a worse basis: radius 1.83x GT, precision 1.294 vs coverage 0.521")
    p.add_argument("--w_distill", type=float, default=3.0)

    # logging / eval
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_milestones", type=str, default="500,1000,2000,3000,5000,8000,9000,12000,15000,20000,30000")
    p.add_argument("--eval_val_indices", type=str, default="0,4,7,63,74,138,183,288")
    p.add_argument("--eval_chamfer_samples", type=int, default=32768)
    p.add_argument("--score_metric", type=str, default="intra_chamfer",
                   choices=["rmse", "rel_offset", "chamfer", "intra_chamfer", "psnr",
                            "psnr_teacher", "psnr_gap_worst"],
                   help="what ckpt_best is selected on. Prefer 'psnr' (held-out photographs): "
                        "every point-space choice here is a proxy, and the proxies were "
                        "measured to disagree with it -- over steps 2000-6000 of the Q1 run "
                        "intra_chamfer improved 0.280 -> 0.254 while xyz_rmse went 0.0143 -> "
                        "0.0163 and every attribute error diverged")
    p.add_argument("--eval_view_count", type=int, default=8,
                   help="photographs held out of training and used only for eval PSNR. They "
                        "are removed from the pool the loader draws --extra_real_views from, "
                        "so the number is a generalisation measure rather than a fit measure")
    p.add_argument("--eval_view_force", type=str, default="",
                   help="comma-separated view-pool indices to ADD to the held-out set on "
                        "top of the seeded draw. Use it to pin a specific photograph out "
                        "of training. Note the other half: `view_exclude` only gates the "
                        "extra_real_views draw, while a snapshot's OWN photo comes through "
                        "photo_map and is never gated -- npz_to_image maps 6000 snapshots "
                        "onto 482 photographs, so a photograph pinned here is still trained "
                        "on through every snapshot that owns it unless those snapshots are "
                        "also kept out of the split.")
    p.add_argument("--eval_view_seed", type=int, default=1234,
                   help="which photographs become the held-out set; fixed across runs so "
                        "PSNR is comparable between them")
    p.add_argument("--ema_decay", type=float, default=0.0,
                   help="weight EMA for the eval copy (0 disables). Reported alongside the raw "
                        "model; ckpt_best is always selected on the raw score")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="",
                   help="continue a run: weights + optimizer + step + best_score")
    p.add_argument("--init_rescale", type=str, default="",
                   help="Comma list of tensor:multiplier applied right after --init_from, "
                        "e.g. 'attr_decoder.slot_emb.weight:5.0'. For parameters that "
                        "are trainable but too small to influence the output.")
    p.add_argument("--init_skip", type=str, default="",
                   help="comma-separated state_dict prefixes to leave at their fresh "
                        "initialisation when using --init_from. Needed when a tensor keeps "
                        "its shape but changes MEANING -- attr_decoder.head_scale.bias went "
                        "from an absolute log-scale to the intercept of a group-relative "
                        "fit, and loading it would silently mis-centre the band")
    p.add_argument("--init_from", type=str, default="",
                   help="warm-start a DIFFERENT stage from a checkpoint's weights only. Step, "
                        "optimizer and schedule all start from 0, and tensors the new model has "
                        "but the checkpoint does not (the attribute heads, when going xyz -> "
                        "geometry) keep their init. Reports exactly what was loaded and what was "
                        "not -- a silent partial load is worse than a crash")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny CPU-friendly consistency run")
    return p


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank0() -> bool:
    return (not is_dist()) or dist.get_rank() == 0


def log(msg: str) -> None:
    if rank0():
        print(msg, flush=True)


def recenter_scale_band(core, optim, args) -> bool:
    """Re-interpret attr_decoder.head_scale when the group-relative band turns on.

    R8H stored ``head_scale.bias`` as an absolute log-scale (~-7.58). The
    group-relative band uses it as the intercept of ``a * log(extent) + b``
    (~-2.559). Loading the old value recentres every Gaussian ~150x too small.
    Reset when the loaded bias still looks like the absolute centre, and drop
    any Adam state for those tensors so a --resume cannot drag them back.
    """
    down = float(getattr(args, "attr_scale_cap_down", 0.0))
    up = float(getattr(args, "attr_scale_cap_up", 0.0))
    if down <= 0.0 or up <= 0.0:
        return False
    dec = getattr(core, "attr_decoder", None)
    if dec is None or not hasattr(dec, "head_scale"):
        return False
    bmean = float(dec.head_scale.bias.detach().float().mean())
    if bmean >= -4.0:
        return False
    torch.nn.init.constant_(dec.head_scale.bias, -2.559)
    torch.nn.init.normal_(dec.head_scale.weight, std=1e-3)
    if optim is not None:
        for p in (dec.head_scale.weight, dec.head_scale.bias):
            if p in optim.state:
                del optim.state[p]
    log(f"re-centred attr_decoder.head_scale for group-relative band "
        f"(bias {bmean:.3f} -> -2.559; weight re-inited)")
    return True


def reset_frame_needle_rot_head(core, optim, args) -> bool:
    """Identity residual around the cell frame.

    C1's head_rot is a free quaternion, not a residual. Interpreting those
    weights as R_cell * R_delta would scramble orientation. Re-init to
    identity so the first step's Gaussians are spheres on the folding frame,
    and drop Adam state so a resume cannot drag the old rot back.
    """
    if not bool(int(getattr(args, "attr_frame_needles", 0))):
        return False
    dec = getattr(core, "attr_decoder", None)
    if dec is None or not hasattr(dec, "head_rot"):
        return False
    torch.nn.init.zeros_(dec.head_rot.weight)
    torch.nn.init.zeros_(dec.head_rot.bias)
    with torch.no_grad():
        dec.head_rot.bias[0] = 1.0
    if optim is not None:
        for p in (dec.head_rot.weight, dec.head_rot.bias):
            if p in optim.state:
                del optim.state[p]
    log("reset attr_decoder.head_rot to identity residual (attr_frame_needles)")
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_shape_args(args) -> None:
    """Fill in the lossless token count and the smallest z_raw grid that fits."""
    patch_dim = int(args.group_size) * 4
    if int(args.local_tokens_per_group) <= 0:
        if patch_dim % int(args.latent_channels) != 0:
            raise SystemExit(
                f"group_size*4={patch_dim} is not divisible by latent_channels={args.latent_channels}; "
                "pick latent_channels that divides it or pass --local_tokens_per_group explicitly"
            )
        args.local_tokens_per_group = patch_dim // int(args.latent_channels)
    num_groups = int(math.ceil(args.max_points / float(args.group_size)))
    need = num_groups * int(args.local_tokens_per_group)
    if int(args.latent_hw[0]) <= 0 or int(args.latent_hw[1]) <= 0:
        side = int(2 ** math.ceil(math.log2(math.sqrt(max(need, 1)))))
        other = int(2 ** math.ceil(math.log2(need / float(side))))
        args.latent_hw = [side, max(other, 1)]
    if int(args.latent_hw[0]) * int(args.latent_hw[1]) < need:
        raise SystemExit(f"latent_hw={args.latent_hw} too small, need {need} cells")


def build_config(args, target_dim: int, sh_dim: int) -> Can3TokConfig:
    # The three stages are prefixes of the 59-channel layout
    # (xyz 3 | log_scale 3 | quat 4 | logit_opacity 1 | SH DC 3 | SH rest 45), so
    # each one is just a truncation:
    #   xyz       3   geometry only
    #   geometry 14   + scale, rotation, opacity, DC colour -- everything the
    #                 rasteriser needs at sh_degree=0, which is what the render
    #                 loss can actually see
    #   full     59   + the 45 view-dependent SH channels
    #
    # `geometry` used to set target_dim=59 while the decoder emitted 14 (its SH
    # head is None when sh_dim=0), so the stage could not run at all. Deriving it
    # as target_dim - sh_dim keeps the two in step by construction.
    if args.stage == "xyz":
        t_dim = 3
    elif args.stage == "geometry":
        t_dim = int(target_dim) - int(sh_dim)
    else:
        t_dim = int(target_dim)
    return Can3TokConfig(
        max_points=args.max_points,
        target_dim=t_dim,
        sh_dim=sh_dim if args.stage == "full" else 0,
        input_dim=target_dim + 4,
        group_size=args.group_size,
        local_tokens_per_group=args.local_tokens_per_group,
        latent_channels=args.latent_channels,
        latent_hw=tuple(args.latent_hw),
        compact_latent_channels=args.compact_latent_channels,
        compact_latent_hw=tuple(args.compact_latent_hw),
        budget_centroid=args.budget_centroid,
        budget_occupancy=args.budget_occupancy,
        budget_shape=args.budget_shape,
        budget_appearance=int(getattr(args, "budget_appearance", 0)),
        attr_pack_dim=int(getattr(args, "attr_pack_dim", 0)),
        attr_pack_hidden=int(getattr(args, "attr_pack_hidden", 64)),
        attr_nbr_window=int(getattr(args, "attr_nbr_window", 0)),
        max_input_points=int(getattr(args, "max_input_points", 0)),
        pool_dim=int(getattr(args, "pool_dim", 128)),
        pool_queries=int(getattr(args, "pool_queries", 16)),
        allow_low_shape_budget=bool(getattr(args, "allow_low_shape_budget", False)),
        shape_xyz_hidden=int(getattr(args, "shape_xyz_hidden", 0)),
        pool_blocks=int(getattr(args, "pool_blocks", 1)),
        pool_feedback=bool(int(getattr(args, "pool_feedback", 1))),
        pool_chunk=int(getattr(args, "pool_chunk", -1)),
        use_fixed_anchor_center=bool(int(getattr(args, "use_fixed_anchor_center", 0))),
        normalize_pooler_xyz=bool(int(getattr(args, "normalize_pooler_xyz", 0))),
        count_aware_template=bool(int(getattr(args, "count_aware_template", 0))),
        attr_slot_mask=bool(int(getattr(args, "attr_slot_mask", 0))),
        compress_head_stages=tuple(getattr(args, "compress_head_stages", ()) or ()),
        compress_merge_stages=int(getattr(args, "compress_merge_stages", 0)),
        compress_merge_attn=bool(getattr(args, "compress_merge_attn", False)),
        shared_free_head=bool(int(getattr(args, "shared_free_head", 0))),
        structured_local_code=bool(int(getattr(args, "structured_local_code", 0))),
        compact_global_dim=int(getattr(args, "compact_global_dim", 3)),
        compact_local_tokens=int(getattr(args, "compact_local_tokens", 4)),
        compact_local_dim=int(getattr(args, "compact_local_dim", 6)),
        attr_decoder_layers=int(getattr(args, "attr_decoder_layers", 0)),
        attr_decoder_dim=int(getattr(args, "attr_decoder_dim", 256)),
        attr_local_pe=bool(int(getattr(args, "attr_local_pe", 1))),
        attr_cond=str(getattr(args, "attr_cond", "xattn")),
        attr_nudge_cap=float(getattr(args, "attr_nudge_cap", 0.15)),
        attr_scale_log_cap=float(getattr(args, "attr_scale_log_cap", 3.0)),
        attr_scale_cap_down=float(getattr(args, "attr_scale_cap_down", 0.0)),
        attr_scale_cap_up=float(getattr(args, "attr_scale_cap_up", 0.0)),
        attr_scale_group_base=bool(int(getattr(args, "attr_scale_group_base", 1))),
        attr_read_shape=bool(int(getattr(args, "attr_read_shape", 0))),
        joint_shared_decoder=bool(int(getattr(args, "joint_shared_decoder", 0))),
        joint_direct_decoder=bool(int(getattr(args, "joint_direct_decoder", 0))),
        joint_local_memory=bool(int(getattr(args, "joint_local_memory", 0))),
        joint_memory_layers=int(getattr(args, "joint_memory_layers", 1)),
        joint_direct_xyz_cap=float(getattr(args, "joint_direct_xyz_cap", 1.5)),
        joint_translation_cap=float(getattr(args, "joint_translation_cap", 0.25)),
        joint_decoder_dim=int(getattr(args, "joint_decoder_dim", 192)),
        joint_decoder_layers=int(getattr(args, "joint_decoder_layers", 2)),
        joint_decoder_chunk=int(getattr(args, "joint_decoder_chunk", 256)),
        joint_nbr_window=int(getattr(args, "joint_nbr_window", 1)),
        joint_xyz_cap=float(getattr(args, "joint_xyz_cap", 0.10)),
        joint_scale_delta_cap=float(getattr(args, "joint_scale_delta_cap", 1.0)),
        joint_opacity_delta_cap=float(getattr(args, "joint_opacity_delta_cap", 2.0)),
        joint_color_delta_cap=float(getattr(args, "joint_color_delta_cap", 0.5)),
        joint_rot_delta_cap=float(getattr(args, "joint_rot_delta_cap", 0.25)),
        joint_detach_render_xyz=bool(int(getattr(args, "joint_detach_render_xyz", 1))),
        attr_detach_geometry=bool(int(getattr(args, "attr_detach_geometry", 1))),
        attr_frame_needles=bool(int(getattr(args, "attr_frame_needles", 0))),
        attr_detach_scale_rot=bool(int(getattr(args, "attr_detach_scale_rot", 0))),
        pack_trainable=bool(getattr(args, "pack_trainable", False)),
        uniform_budget=args.uniform_budget,
        model_dim=args.model_dim,
        heads=args.heads,
        dropout=args.dropout,
        encoder_residual=args.encoder_residual,
        map_blocks=args.map_blocks,
        compress_intra_layers=args.compress_intra_layers,
        compress_window_layers=args.compress_window_layers,
        compress_window=args.compress_window,
        compress_mid_channels=args.compress_mid_channels,
        compress_wide_channels=args.compress_wide_channels,
        compress_wide_layers=args.compress_wide_layers,
        decompress_intra_layers=args.decompress_intra_layers,
        decompress_window_layers=args.decompress_window_layers,
        decoder_layers=args.decoder_layers,
        decoder_neighbor_context=not args.no_decoder_neighbor_context,
        residual_scale=args.residual_scale,
        patch_chunk=args.patch_chunk,
        use_gen_branch=not args.no_gen_branch,
        gen_window_layers=args.gen_window_layers,
        gen_group_layers=args.gen_group_layers,
        gen_cross_layers=args.gen_cross_layers,
        gen_region_chunk=args.gen_region_chunk,
        gen_offset_scale=args.gen_offset_scale,
        gen_neighbor_context=not args.no_gen_neighbor_context,
        gen_detach_latent=not bool(getattr(args, "no_gen_detach", False)),
        latent_mode=args.latent_mode,
        latent_zorder=not args.no_latent_zorder,
        denoise_std=args.denoise_std,
        checkpoint_decode=not args.no_checkpoint_decode,
        checkpoint_gen=args.checkpoint_gen,
        residual_pack=not args.no_residual_pack,
        shortcut_alpha=float(args.shortcut_alpha_init),
        shortcut_alpha_eval=float(args.shortcut_alpha_eval),
        folding_decode=bool(getattr(args, "folding_decode", False)),
        folding_frame_channels=int(getattr(args, "folding_frame_channels", 6)),
        folding_res_cap=float(getattr(args, "folding_res_cap", 1.0)),
        folding_aniso_log_cap=float(getattr(args, "folding_aniso_log_cap", 1.5)),
        teacher_cycle=float(getattr(args, "w_teacher_cycle", 0.0)) != 0.0,
        teacher_cycle_freeze=not bool(getattr(args, "no_teacher_cycle_freeze", False)),
    )


def make_datasets(args):
    # Multi-scene: --root may be a comma-separated list. The file list here must
    # be built exactly the way ReplayGaussianDataset builds it (roots in order,
    # each sorted) or the split indices point at the wrong files.
    _roots = [r.strip() for r in str(args.root).split(",") if r.strip()]
    files = []
    for _r in _roots:
        files.extend(list_npz_files(_r))
    n = len(files)
    if args.split_path and os.path.exists(args.split_path):
        with open(args.split_path) as f:
            split = json.load(f)
        train_idx = split.get("train_indices", split.get("train"))
        val_idx = split.get("val_indices", split.get("val"))
        if train_idx and isinstance(train_idx[0], str):
            name_to_idx = {os.path.basename(p): i for i, p in enumerate(files)}
            train_idx = [name_to_idx[os.path.basename(x)] for x in train_idx]
            val_idx = [name_to_idx[os.path.basename(x)] for x in val_idx]
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n)
        val_idx = sorted(perm[: args.num_val_files].tolist())
        train_idx = sorted(perm[args.num_val_files :].tolist())

    # Keep the established split, then select the mature part of the trajectory.
    # ReplayGaussianDataset expects indices into the original sorted file list.
    min_step = int(getattr(args, "min_snapshot_step", 0))
    if min_step > 0:
        import re

        def _mature(i):
            m = re.search(r"step_(\d+)", os.path.basename(files[int(i)]))
            return m is not None and int(m.group(1)) >= min_step

        train_idx = [int(i) for i in train_idx if _mature(i)]
        val_idx = [int(i) for i in val_idx if _mature(i)]
        if not train_idx or not val_idx:
            raise ValueError(
                f"min_snapshot_step={min_step} leaves train={len(train_idx)}, val={len(val_idx)}"
            )
        log(f"[data] mature snapshot filter >= {min_step}: "
            f"train={len(train_idx)} val={len(val_idx)}")

    photo_map = {}
    pmp = str(getattr(args, "photo_map", "") or "")
    if pmp and os.path.exists(pmp):
        with open(pmp) as f:
            photo_map = json.load(f)
    # One stats file per scene. Without this a two-scene run would normalise both
    # scenes by scene 0's center/scale, which silently puts the second scene
    # outside the unit cube and drops most of its points at `drop_outside`.
    if args.stats_path:
        stats_path = args.stats_path
    else:
        _rs = [r.strip() for r in str(args.root).split(",") if r.strip()]
        stats_path = ",".join(os.path.join(args.out_dir, f"stats_scene{i}.json")
                              for i in range(len(_rs)))
    dens_aware = bool(getattr(args, "density_aware_sample", True)) and not bool(
        getattr(args, "no_density_aware_sample", False)
    )
    slot_redist = bool(getattr(args, "slot_redistribute", True)) and not bool(
        getattr(args, "no_slot_redistribute", False)
    )
    common = dict(
        root=args.root,
        max_points=args.max_points,
        max_input_points=int(getattr(args, "max_input_points", 0)),
        cache_slots=bool(getattr(args, "cache_slots", False)),
        stats_path=stats_path,
        stats_stride=args.stats_stride,
        stats_quantile=args.stats_quantile,
        drop_outside=args.drop_outside,
        chunk_size=args.chunk_size,
        group_size=args.group_size,
        seed=args.seed,
        density_aware_sample=dens_aware,
        density_importance_weight=float(getattr(args, "density_importance_weight", 0.35)),
        slot_redistribute=slot_redist,
        partition_mode=str(getattr(args, "partition_mode", "kd")),
        partition_block=int(getattr(args, "partition_block", 0)),
        slot_sort=str(getattr(args, "slot_sort", "morton")),
        scene_anchors=str(getattr(args, "scene_anchors", "") or ""),
    )

    train_ds = ReplayGaussianDataset(
        file_indices=train_idx,
        sample_mode=args.sample_mode,
        crop_prob=args.crop_prob,
        augment=args.augment,
        aug_rot_deg=args.aug_rot_deg,
        aug_scale_jitter=args.aug_scale_jitter,
        aug_shift=args.aug_shift,
        **common,
    )
    val_ds = ReplayGaussianDataset(
        file_indices=val_idx, sample_mode=args.sample_mode, crop_prob=0.0, augment=False, **common
    )
    train_ds.photo_map = photo_map
    val_ds.photo_map = photo_map
    vp = []
    vpp = str(getattr(args, "view_pool", "") or "")
    if vpp and os.path.exists(vpp):
        with open(vpp) as f:
            vp = json.load(f)
    for ds in (train_ds, val_ds):
        ds.anchor_spill = int(getattr(args, "anchor_spill", 8))
        ds.anchor_assignment = str(getattr(args, "anchor_assignment", "spill"))
        ds.view_pool = vp
        ds.extra_real_views = int(getattr(args, "extra_real_views", 0))
        ds.view_downscale = int(getattr(args, "view_downscale", 2))
        ds.shared_cell_owner = bool(int(getattr(args, "shared_cell_owner", 0)))
        ds.holdout_own_photo = bool(int(getattr(args, "holdout_own_photo", 0)))
        ds.keep_extra_fullres = bool(int(getattr(args, "keep_extra_fullres", 0)))
        ds.count_aware_template = bool(int(getattr(args, "count_aware_template", 0)))
        if isinstance(vp, dict):
            ds.scene_names = list(vp.keys())
    return train_ds, val_ds


class ModelEMA:
    """Shadow weights evaluated alongside the raw model.

    Reported only; ``ckpt_best`` is still selected on the raw score, so a bad EMA
    horizon cannot silently pick a worse checkpoint.
    """

    def __init__(self, model, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if torch.is_floating_point(v)
        }

    @torch.no_grad()
    def update(self, model) -> None:
        sd = model.state_dict()
        for k, buf in self.shadow.items():
            buf.mul_(self.decay).add_(sd[k].detach().float(), alpha=1.0 - self.decay)

    @contextmanager
    def swapped(self, model):
        sd = model.state_dict()
        backup = {k: sd[k].detach().clone() for k in self.shadow}
        try:
            for k, buf in self.shadow.items():
                sd[k].copy_(buf.to(sd[k].dtype))
            yield
        finally:
            for k, v in backup.items():
                sd[k].copy_(v)


def load_eval_views(args):
    """Photographs reserved for scoring, and the pool indices to keep out of training.

    Held out rather than merely "a few of the training views": appearance from a
    per-group code is badly under-determined from one view, so a model can score
    well on views it was fitted to and badly everywhere else -- measured, fitting
    a code to 4 views gains +3.71 dB on those 4 and *loses* 0.93 dB on 4 unseen
    ones. A fit-set PSNR would therefore have reported success for exactly the
    failure this run is trying to avoid.
    """
    vpp = str(getattr(args, "view_pool", "") or "")
    k = int(getattr(args, "eval_view_count", 0))
    if not vpp or not os.path.exists(vpp) or k <= 0:
        return [], set()
    with open(vpp) as f:
        vp = json.load(f)
    # MULTI-SCENE. The old form only unwrapped a dict of exactly one pool, so two
    # scenes silently produced "eval views: 0" -- and with --score_metric psnr that
    # means ckpt_best is chosen on a metric that was never computed. Hold out k
    # views PER SCENE and exclude the same indices from training in every scene.
    #
    # The same index set across scenes, not an independent draw per scene: the
    # loader's `view_exclude` is one set applied to whichever pool the sample's
    # scene selects, so per-scene draws would leak scene A's held-out index into
    # scene B's training pool.
    if isinstance(vp, dict):
        pools = list(vp.values())
    elif isinstance(vp, list):
        pools = [vp]
    else:
        pools = []
    pools = [p for p in pools if p]
    if not pools:
        return [], set()
    n_min = min(len(p) for p in pools)
    sel = np.random.default_rng(int(getattr(args, "eval_view_seed", 1234))).choice(
        n_min, size=min(k, n_min), replace=False)
    held = set(int(i) for i in sel)
    for _tok in str(getattr(args, "eval_view_force", "") or "").split(","):
        _tok = _tok.strip()
        if _tok:
            held.add(int(_tok))
    held = sorted(i for i in held if 0 <= i < n_min)
    views = []
    try:
        from PIL import Image
    except Exception:
        return [], set(held)
    if isinstance(vp, dict):
        named_pools = [(str(k), p) for k, p in vp.items() if p]
    else:
        named_pools = [("scene0", pools[0])] if pools else []
    for scene_id, pool in named_pools:
        for i in held:
            if i >= len(pool):
                continue
            e = pool[i]
            try:
                im = np.asarray(Image.open(e["image"]).convert("RGB"), np.float32) / 255.0
            except Exception:
                continue
            image_id = str(e.get("image", i))
            views.append(EvalView(
                scene_id=scene_id,
                image_id=image_id,
                camera=np.asarray(e["cam"], np.float32),
                photo=torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))),
                pool_index=int(i),
            ))
    return views, set(held)


def random_rotation(deg: float, device, dtype) -> torch.Tensor:
    a = (torch.rand(3, device=device) * 2 - 1) * math.radians(deg)
    cx, sx = torch.cos(a[0]), torch.sin(a[0])
    cy, sy = torch.cos(a[1]), torch.sin(a[1])
    cz, sz = torch.cos(a[2]), torch.sin(a[2])
    one = torch.ones((), device=device)
    zero = torch.zeros((), device=device)
    Rx = torch.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx]).reshape(3, 3)
    Ry = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy]).reshape(3, 3)
    Rz = torch.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one]).reshape(3, 3)
    return (Rz @ Ry @ Rx).to(dtype)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_eval(model, val_ds, args, cfg, step: int, device, amp_ctx, save_outputs: bool,
             tag: str = "", eval_views=None) -> Dict[str, float]:
    core = model.module if isinstance(model, DDP) else model
    core.eval()
    # Match the training refine / residual schedule so early evals are not
    # inflated by a fully-open refine head.
    apply_eval_schedule(core, step, args)
    run_gen = bool(cfg.use_gen_branch) and int(step) >= int(args.gen_start)
    indices = [int(i) for i in str(args.eval_val_indices).split(",") if i != ""]
    indices = [i for i in indices if i < len(val_ds)]
    agg: Dict[str, List[float]] = {}
    # Per-scene aggregation alongside the pooled one. With two scenes sharing one
    # latent grid and no scene conditioning, the pooled mean hides the case where
    # the model is winning on one scene and losing on the other -- measured on
    # L16k at step 58000, held-out PSNR beat the GT Gaussians by 1.2-3.0 dB on the
    # truck and lost by up to 2.2 dB on the locomotive, and the reported single
    # number was +0.45 dB. A regression confined to one scene was invisible.
    agg_scene: Dict[int, Dict[str, List[float]]] = {}
    out_dir = os.path.join(args.out_dir, "eval", f"step{step:08d}", "val")
    for i in indices:
        item = val_ds[i]
        scene_id = int(item.get("scene", 0))
        scene_key = item.get("scene_key") or f"scene{scene_id}"
        if torch.is_tensor(scene_key):
            scene_key = f"scene{int(scene_key)}"
        agg_before = {k: len(v) for k, v in agg.items()}
        x = item["input"].unsqueeze(0).to(device).float()
        target = item["target"].unsqueeze(0).to(device).float()
        mask = item["mask"].unsqueeze(0).to(device).float()
        # The layout is prefix-ordered, so every stage is a truncation of the
        # 59-channel target. Slicing to cfg.target_dim covers xyz (3), geometry
        # (14) and full (59, a no-op) with one line.
        target = target[..., : cfg.target_dim]
        # The encoder's input is NOT the decoder's slot layout once
        # --max_input_points exceeds --max_points: the loader packs a second,
        # larger tensor and the training step passes it as enc_x. Omitting it here
        # let the encoder reshape a 262,144-slot tensor into (4096, 144) groups --
        # a completely different grouping from the one it was trained on, padded
        # with zeros for the missing 327,680 slots. Nothing errors; eval simply
        # reports a different model than the one being trained (measured: eval
        # xyz_rmse 0.675 against a training render L1 of 0.138, i.e. better than
        # the run whose eval said it was 20x worse).
        ex = item.get("enc_input")
        em = item.get("enc_mask")
        ga = item.get("group_anchor")
        with amp_ctx():
            out = core(x, mask, run_decode=True, run_gen=run_gen, gen_noise_std=0.0,
                       enc_x=None if ex is None else ex.unsqueeze(0).to(device).float(),
                       enc_mask=None if em is None else em.unsqueeze(0).to(device).float(),
                       group_anchor=None if ga is None else ga.unsqueeze(0).to(device).float())
        scale = float(item["scale"])
        # Geometry metrics stay on out["pred"] -- the geometry decoder's own xyz,
        # so every number stays comparable with the runs before the split. The
        # attributes come from the module that actually learns them.
        ap = out.get("attr_pred")
        gap = out.get("gen_attr_pred")
        layout_for_eval = getattr(val_ds, "layout", None) or {
            "xyz": 0, "scale": 3, "rot": 6, "opacity": 10, "color": 11, "sh": 14}
        a_tgt = (responsibility_target(ap[..., 0:3].float(), target, mask,
                                       layout_for_eval, cfg.group_size)
                 if ap is not None else None)
        m_codec = compute_eval_metrics(
            out["pred"].float(), out["presence"].float(), target, mask, scale,
            args.eval_chamfer_samples,
            attr_pred=None if ap is None else ap.float(), attr_target=a_tgt,
            group_size=int(cfg.group_size),
        )
        for k, v in m_codec.items():
            agg.setdefault(f"codec_{k}", []).append(v)
        for k, v in group_error_breakdown(out["pred"].float(), target, mask, cfg.group_size).items():
            agg.setdefault(f"codec_{k}", []).append(v)
        m_gen = None
        if "gen_pred" in out:
            g_tgt = (responsibility_target(gap[..., 0:3].float(), target, mask,
                                           layout_for_eval, cfg.group_size)
                     if gap is not None else None)
            m_gen = compute_eval_metrics(
                out["gen_pred"].float(), out["gen_presence"].float(), target, mask, scale,
                args.eval_chamfer_samples,
                attr_pred=None if gap is None else gap.float(), attr_target=g_tgt,
                group_size=int(cfg.group_size),
            )
            for k, v in m_gen.items():
                agg.setdefault(f"gen_{k}", []).append(v)
            for k, v in group_error_breakdown(
                out["gen_pred"].float(), target, mask, cfg.group_size
            ).items():
                agg.setdefault(f"gen_{k}", []).append(v)
        # The target metric. Everything above is a proxy; this is what the model
        # is for, and it is measured on views no training step ever saw.
        # `attr_pred` is the deployable Gaussian (learned attributes + nudged
        # positions), so that is what gets rendered -- rendering out["pred"] would
        # score the geometry decoder's own untrained attribute channels.
        if eval_views and int(cfg.target_dim) >= 14:
            views_for = filter_eval_views(eval_views, scene_key)
            if eval_views and isinstance(eval_views[0], EvalView) and not views_for:
                raise RuntimeError(
                    f"no held-out views for scene {scene_key!r}; "
                    f"have {sorted({v.scene_id for v in eval_views if isinstance(v, EvalView)})}"
                )
            use_pred_mask = bool(int(getattr(args, "eval_pred_mask", 0)))
            for br, key, apk, pkey in (("codec", "pred", ap, "presence"),
                                       ("gen", "gen_pred", gap, "gen_presence")):
                if key not in out:
                    continue
                pm = out[pkey].float()[0] if (use_pred_mask and pkey in out) else None
                rendered = render_eval_metrics(
                    (out[key] if apk is None else apk).float()[0],
                    target[0], mask[0], item["center"], scale, views_for,
                    layout_for_eval, downscale=int(args.render_downscale),
                    scene_id=scene_key, pred_mask=pm,
                )
                view_ids = rendered.pop("_eval_view_ids", None)
                if view_ids:
                    agg.setdefault(f"{br}_eval_view_ids", []).extend(view_ids)
                for k, v in rendered.items():
                    agg.setdefault(f"{br}_{k}", []).append(v)
        bud = channel_budget(cfg)
        for k, v in latent_diagnostics(
            out["z_compact"], per_group=bud["per_group"], c_anchor=bud["centroid"] + bud["occupancy"]
        ).items():
            agg.setdefault(k, []).append(v)
        agg.setdefault("res_ratio", []).append(float(out["res_ratio"]))
        if "direct_frac" in out:
            agg.setdefault("direct_frac", []).append(float(out["direct_frac"]))
        if "direct_ratio" in out:
            agg.setdefault("direct_ratio", []).append(float(out["direct_ratio"]))
        n_used = float(mask.sum().item())
        n_max = float(mask.numel())
        agg.setdefault("n_used", []).append(n_used)
        agg.setdefault("empty_frac", []).append(1.0 - n_used / max(n_max, 1.0))
        if "num_points_original" in item:
            agg.setdefault("n_original", []).append(float(item["num_points_original"]))
        if "num_points_valid" in item:
            agg.setdefault("n_valid", []).append(float(item["num_points_valid"]))
        if save_outputs and rank0():
            raw_name = os.path.splitext(item["name"])[0]
            safe_sk = str(scene_key).replace("/", "_").replace(" ", "_")
            name = f"{safe_sk}_{raw_name}"
            # The saved point cloud is the deployable Gaussian, so it carries the
            # learned attributes and the nudged positions, not the geometry
            # decoder's attribute channels which nothing trains.
            deploy_mask = (out["presence"].float()[0]
                           if bool(int(getattr(args, "eval_pred_mask", 0))) else mask[0])
            save_eval_outputs(out_dir, name, "codec",
                              (out["pred"] if ap is None else ap).float()[0],
                              target[0], mask[0], item["center"], scale, m_codec,
                              pred_mask=deploy_mask)
            if m_gen is not None:
                save_eval_outputs(out_dir, name, "gen",
                                  (out["gen_pred"] if gap is None else gap).float()[0],
                                  target[0], mask[0], item["center"], scale, m_gen,
                                  pred_mask=out["gen_presence"].float()[0]
                                  if bool(int(getattr(args, "eval_pred_mask", 0))) else mask[0])
        # Whatever this sample appended to `agg` also belongs to its scene. Reading
        # the tail rather than threading a scene argument through every metric
        # helper keeps the two aggregations impossible to get out of sync.
        sagg = agg_scene.setdefault(scene_id, {})
        for k, v in agg.items():
            new = v[agg_before.get(k, 0):]
            if new:
                sagg.setdefault(k, []).extend(new)
    def _mean_numeric(vals):
        if not vals:
            return None
        if isinstance(vals[0], (str, bytes)):
            return None
        try:
            return float(np.mean(np.asarray(vals, dtype=np.float64)))
        except (TypeError, ValueError):
            return None

    metrics = {}
    for k, v in agg.items():
        if k.endswith("eval_view_ids"):
            metrics[k] = sorted(set(str(x) for x in v))
            continue
        m = _mean_numeric(v)
        if m is not None:
            metrics[k] = m
    # Flattened as `scene<N>/<key>` so the JSON stays a flat float map and every
    # existing reader keeps working.
    for sid, sagg in sorted(agg_scene.items()):
        for k, v in sagg.items():
            if not v:
                continue
            if k.endswith("eval_view_ids"):
                metrics[f"scene{sid}/{k}"] = sorted(set(str(x) for x in v))
                continue
            m = _mean_numeric(v)
            if m is not None:
                metrics[f"scene{sid}/{k}"] = m
    metrics["step"] = step
    metrics["eval_refine_alpha"] = float(core.cfg.decoder_refine_alpha)
    # Logged beside nn_unique so the anneal and the metric it targets are read
    # together: a flat uniq while eps is still high says "not sharp yet", a flat
    # uniq at the final eps says the assignment did not help.
    metrics["sinkhorn_eps"] = float(sinkhorn_epsilon_at(int(step), args))
    if rank0():
        save_json(os.path.join(args.out_dir, "eval", f"step{step:08d}", f"metrics{tag}.json"), metrics)
    core.train()
    return metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    resolve_shape_args(args)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0 and torch.cuda.is_available():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Chamfer NN is a big GEMM; TF32 speeds the no_grad search without changing
    # sample counts. Autograd distances stay in the amp dtype as before.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    set_seed(args.seed + (dist.get_rank() if is_dist() else 0))
    if int(args.decoder_refine_start) < 0:
        args.decoder_refine_start = int(args.latent_end)

    os.makedirs(args.out_dir, exist_ok=True)
    if rank0():
        save_json(os.path.join(args.out_dir, "args.json"), vars(args))

    train_ds, val_ds = make_datasets(args)
    # Held-out photographs: loaded once, kept on CPU, and *removed from the pool
    # the loader samples training views from*. Both halves matter -- scoring on
    # views the optimiser also fitted would report the fit, not the model.
    eval_views, held_view_idx = load_eval_views(args)
    for ds in (train_ds, val_ds):
        ds.view_exclude = set(held_view_idx)
    n_scenes = len({v.scene_id for v in eval_views if isinstance(v, EvalView)})
    log(f"eval views: {len(eval_views)} photographs held out of training "
        f"across {n_scenes} scene(s) "
        f"(pool indices {sorted(held_view_idx)[:8]}{' ...' if len(held_view_idx) > 8 else ''})")
    if not eval_views:
        log("  WARNING: no held-out views -- eval will report no PSNR and "
            "--score_metric psnr would select on +inf for every checkpoint")
    cfg = build_config(args, train_ds.target_dim, train_ds.sh_dim)
    log(describe_layout(cfg, allow_padded_tokens=args.allow_padded_tokens))
    # z_raw reconstruction targets only mean something when z_raw is the identity
    # xyz pack. Once max_input_points > max_points the pooler makes z_raw a learned
    # code with no fixed scale: in L17k it reached std 315 and w_z_raw_mse alone
    # took 97.2% of the objective (total = 2.199 * z_l1^2, R^2 = 0.9974). The term
    # is RMS-normalised now so it can no longer explode, but it is still asking the
    # code to hold still while it is being learned.
    if int(getattr(args, "max_input_points", 0) or 0) > int(args.max_points):
        _zw = {k: float(getattr(args, k, 0.0)) for k in
               ("w_z_raw", "w_z_raw_mse", "w_z_hard", "w_z_token_var", "w_z_std_ratio",
                "w_z_residual", "w_z_residual_hard", "w_z_intra_chamfer")}
        _on = {k: v for k, v in _zw.items() if v != 0.0}
        if _on:
            log(f"[layout] WARNING: the pooler is active (max_input_points "
                f"{args.max_input_points} > max_points {args.max_points}), so z_raw is a "
                f"learned code, not an identity xyz pack. These z_raw reconstruction "
                f"weights are non-zero and pin a moving target: {_on}")
    log(f"dataset: train={len(train_ds)} val={len(val_ds)} target_dim={cfg.target_dim}")

    model = build_model(cfg, allow_padded_tokens=args.allow_padded_tokens).to(device)
    # zero-initialised, so enabling it from step 0 keeps the graph static while
    # the encoder learning rate stays at 0 until args.encoder_residual_start
    model.set_encoder_residual(cfg.encoder_residual)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_par / 1e6:.2f}M | device={device}")
    core = model
    if is_dist():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)
        core = model.module

    groups = core.param_groups(args.lr)
    # eps well below the post-clip gradient scale. grad_clip=1.0 fires on 100% of
    # steps here (measured gn p05 1746 / p50 2259 on L17k), so the gradient AdamW
    # sees always has unit norm -- per-coordinate RMS 1/sqrt(101.1e6) = 9.9e-5.
    # At the default eps=1e-8 the attenuation |g|/(|g|+eps) drops to 0.50 for a
    # coordinate at 1e-4 of that RMS and 0.09 at 1e-5, so the tail of parameters
    # driven only by the small non-z terms is silently damped. Pre-clip those same
    # coordinates sit ~2000x higher and eps never bites; the clip alone moves them
    # into the eps-dominated regime.
    optim = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay,
                              betas=(0.9, 0.95), eps=1e-15)
    group_names = [g["name"] for g in optim.param_groups]
    # Does the "encoder" group hold anything besides the zero-init residual branch?
    # If it does, the residual gate must not be allowed to zero its learning rate.
    encoder_has_core = bool(
        getattr(core.encoder, "pooler", None) is not None
        or next(core.encoder.pack.parameters()).requires_grad
        or len(core.encoder.map_blocks) > 0
    )
    log(f"encoder group: trainable core = {encoder_has_core} "
        f"(pack_trainable={next(core.encoder.pack.parameters()).requires_grad}, "
        f"pooler={getattr(core.encoder, 'pooler', None) is not None}, "
        f"map_blocks={len(core.encoder.map_blocks)})")
    # --- silent-freeze guard -------------------------------------------------
    # This project has now shipped FIVE instances of the same failure: a module in
    # the graph, whose losses are logged as if it were training, whose parameters
    # never move. attr_decoder's 3.325M params sat at init for a full run;
    # encoder.attr_encoder sat at init because a residual gate zeroed the whole
    # encoder group's learning rate; the encoder residual branch could not leave
    # zero because tanh(gate)*residual(feat) is a product of two zero inits. Each
    # time the only symptom was eval metrics equal to several decimals between
    # checkpoints -- which nobody reads until weeks later.
    #
    # `touch` further down guarantees p.grad is never None, so a frozen path shows
    # up as grad EXACTLY 0.0 and no existing check can see it. The invariant that
    # matters is not "has a gradient" but "moved", so that is what this measures.
    _snap = {g["name"]: torch.cat([q.detach().reshape(-1) for q in g["params"]]).clone()
             for g in optim.param_groups}
    _frozen_streak = {n: 0 for n in group_names}

    def _param_movement():
        """Per-group ||delta p|| since the last call, plus groups that look frozen."""
        moved, suspect = {}, []
        for g in optim.param_groups:
            n = g["name"]
            cur = torch.cat([q.detach().reshape(-1) for q in g["params"]])
            moved[n] = float((cur - _snap[n]).norm())
            _snap[n] = cur.clone()
            if float(g["lr"]) > 0.0 and moved[n] == 0.0:
                _frozen_streak[n] += 1
                if _frozen_streak[n] >= 3:
                    suspect.append(n)
            else:
                _frozen_streak[n] = 0
        return moved, suspect
    # ------------------------------------------------------------------------
    ema = ModelEMA(core, args.ema_decay) if float(args.ema_decay) > 0 else None

    if args.amp == "bf16" and device.type == "cuda":
        amp_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    elif args.amp == "fp16" and device.type == "cuda":
        amp_ctx = lambda: torch.autocast("cuda", dtype=torch.float16)
    else:
        amp_ctx = nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))

    step = 0
    best_score = float("inf")
    best_gen_score = float("inf")
    score_fp = eval_fingerprint(args, held_view_idx)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        core.load_state_dict(ck["model"])
        if "optim" in ck:
            optim.load_state_dict(ck["optim"])
        step = int(ck.get("step", 0))
        saved_fp = ck.get("eval_fingerprint")
        if saved_fp != score_fp:
            log(f"best_score reset: eval fingerprint changed {saved_fp} -> {score_fp}")
            best_score = float("inf")
            best_gen_score = float("inf")
        else:
            best_score = float(ck.get("best_score", float("inf")))
            best_gen_score = float(ck.get("best_gen_score", float("inf")))
        log(f"resumed from {args.resume} at step {step}")
    elif args.init_from and os.path.exists(args.init_from):
        # Stage transfer, not continuation. Geometry is budget-bound and already at
        # ~94% of its ceiling, so the attribute stage should start from a trained
        # geometry rather than spend its first few thousand steps re-deriving it.
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        have = core.state_dict()
        src = ck["model"]
        skip_pref = tuple(x for x in str(getattr(args, "init_skip", "") or "").split(",") if x)
        take = {k: v for k, v in src.items()
                if k in have and have[k].shape == v.shape and not k.startswith(skip_pref)}
        if skip_pref:
            log(f"  init_skip {skip_pref}: left at fresh init")
        skipped = [k for k in src if k not in take]
        fresh = [k for k in have if k not in take]
        core.load_state_dict(take, strict=False)
        log(f"init_from {args.init_from} (step {ck.get('step')}): "
            f"loaded {len(take)}/{len(have)} tensors")
        if fresh:
            log(f"  left at init ({len(fresh)}): {', '.join(sorted(fresh)[:6])}"
                + (" ..." if len(fresh) > 6 else ""))
        if skipped:
            log(f"  in ckpt but unused ({len(skipped)}): {', '.join(sorted(skipped)[:6])}"
                + (" ..." if len(skipped) > 6 else ""))
        if not take:
            raise SystemExit(f"--init_from matched 0 tensors; wrong checkpoint for this config?")
        # Rescale named tensors after loading. This exists because a tensor can be
        # trainable, receiving gradient, and still be irrelevant: slot_emb moved
        # only 0.0265 -> 0.0295 over 14000 steps while the features it is added to
        # have magnitude ~1, so it contributed 11% of a within-group diversity that
        # was itself 9.8% of the ground truth's. Its gradient does not grow when it
        # does -- d(out)/d(slot_emb) depends on the downstream weights -- so a
        # higher learning rate alone starts from the same negligible influence.
        # Multiplying it up puts the per-slot signal on the same scale as the
        # positional features, and the dedicated learning rate then lets the
        # optimiser tune it from there. Prefer this to re-initialising: the pattern
        # across slots is already high-rank (effective rank 33 of 64), it is only
        # too quiet to matter.
        resc = [x for x in str(getattr(args, "init_rescale", "") or "").split(",") if x]
        for spec in resc:
            name, _, mult = spec.partition(":")
            name, mult = name.strip(), float(mult)
            sd = core.state_dict()
            if name not in sd:
                raise SystemExit(f"--init_rescale: no such tensor {name!r}")
            with torch.no_grad():
                before = float(sd[name].float().std())
                sd[name].mul_(mult)
                log(f"  init_rescale {name} x{mult:g}: std {before:.4f} -> "
                    f"{float(sd[name].float().std()):.4f}")
    recenter_scale_band(core, optim, args)
    reset_frame_needle_rot_head(core, optim, args)

    sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_dist() else None
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=int(os.environ.get("DATALOADER_PREFETCH", "4")) if args.workers > 0 else None,
    )

    milestones = {int(s) for s in str(args.eval_milestones).split(",") if s != ""}
    if args.eval_only:
        run_eval(model, val_ds, args, cfg, step, device, amp_ctx, save_outputs=True,
                 eval_views=eval_views)
        return

    model.train()
    layout = train_ds.layout
    budget = channel_budget(cfg)
    model_layout = dict(patch_layout(cfg))
    model_layout.update(
        per_group=budget["per_group"], c_centroid=budget["centroid"],
        c_occupancy=budget["occupancy"],
        # Width of the learned shape block. The latent regularisers used to infer
        # it as "everything after the anchors", which swept the appearance channels
        # in with it; they now need the upper bound explicitly.
        c_shape=budget["shape"], c_appearance=budget.get("appearance", 0),
        # Non-zero only when the pack's aux block carries a learned appearance
        # code. Tells the z-space losses to compare the deterministic xyz prefix
        # only, and every mask consumer to stop reading the aux block.
        # Gated on the pack path, not just on attr_pack_dim. The prefix is only
        # "the deterministic xyz block" on the identity path, where the pack is
        # literally cat([xyz, aux]). When the encoder's GroupPointPooler is active
        # -- which it is whenever max_input_points exceeds max_points, i.e. on
        # every config in this tree -- the pack is one Linear's output and has no
        # per-slice meaning, so slicing [:group_size*3] cuts an arbitrary 768 of a
        # homogeneous 1024-dim code. Measured on L17k: the slice's L1 (278.62) is
        # indistinguishable from the whole code's absmean (278.60), which is what
        # an arbitrary cut of a homogeneous vector looks like and not what a
        # metric xyz block would look like (codec_xyz there is 0.0129).
        geom_pack_dim=(cfg.group_size * 3
                       if (int(getattr(cfg, "attr_pack_dim", 0)) > 0
                           and getattr(core, "encoder", None) is not None
                           and getattr(core.encoder, "pooler", None) is None)
                       else 0),
    )
    t_last = time.time()
    epoch = 0
    running: Dict[str, float] = {}
    # Spike-guard state. `loss_ref` stays None until there is enough history for a
    # stable median, so warmup -- where the loss legitimately falls by orders of
    # magnitude -- is never treated as a spike.
    ddp_active = is_dist()
    loss_spike_mult = float(getattr(args, "loss_spike_mult", 0.0))
    grad_spike_mult = float(getattr(args, "grad_spike_mult", 0.0))
    loss_hist: Deque[float] = deque(maxlen=256)
    gnorm_hist: Deque[float] = deque(maxlen=256)
    gnorm_anchor: Optional[float] = None
    gnorm_detail_reset = False
    grad_spike_abs = float(getattr(args, "grad_spike_abs", 0.0))
    gskipped = 0
    # Second history that DOES include skipped batches, plus a trailing record of
    # the decisions themselves. `loss_hist` deliberately excludes skips, which is
    # right while spikes are a minority and fatal when they are not: P16k crossed
    # the attribute/render turn-on at step 1200, 95% of batches went over 25x, and
    # because none of them reached `loss_hist` the reference froze at 175.4148 from
    # step 1362 to 2213 and the guard could not observe that the loss floor had
    # legitimately moved. 950 steps were dropped before the surviving 5% dragged it
    # back. A guard whose reference sits three orders below the typical batch is not
    # a spike guard, it is a stop switch, so detect that state and re-reference.
    loss_hist_all: Deque[float] = deque(maxlen=256)
    skip_win: Deque[int] = deque(maxlen=200)
    stall_note = -10 ** 9
    loss_ref: Optional[float] = None
    # Anchored once at step 2000 and never updated -- the fixed reference the
    # trailing-median spike guard cannot provide.
    loss_ref_slow: Optional[float] = None
    skipped = 0

    # Per-point render visibility from the previous step. The attribute render is
    # computed after total_loss, so the anchor cannot see this step's mask; a
    # one-step-old one is equivalent in practice because visibility is a geometric
    # property that moves slowly and the cameras are re-jittered every step
    # anyway, so a "fresh" mask would come from different poses regardless.
    prev_vis = None
    stop_at = int(getattr(args, "stop_at", 0) or 0)
    while step < args.max_steps and not (stop_at and step >= stop_at):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        for batch in loader:
            if step >= args.max_steps:
                break
            if stop_at and step >= stop_at:
                break
            flags = schedule_flags(step, args)
            weights = effective_weights(step, args)
            weights["residual_pack"] = bool(cfg.residual_pack)
            scales = lr_scales(step, args)
            if not flags["run_decode"]:
                scales["decoder"] = 0.0
            if not flags["run_gen"]:
                scales["gen"] = 0.0
            # Zeroing the whole "encoder" group when the residual branch is off is
            # only correct while that branch is the ONLY thing in the group. It
            # stopped being true twice: once when attr_encoder lived here (fixed by
            # splitting it out), and again now that the pack can be trainable and a
            # learned pooler can exist. Both are core encoder parameters that must
            # train from step 0, and freezing them is invisible -- the losses still
            # fall, driven entirely by the compressor.
            if not flags["encoder_residual"] and not encoder_has_core:
                scales["encoder"] = 0.0
            base_lr = global_lr(step, args)
            for g, name in zip(optim.param_groups, group_names):
                g["lr"] = base_lr * float(scales.get(name, 1.0))

            # Structural schedule: keep free centroid / refine paths off early.
            core.cfg.shortcut_alpha = float(flags["shortcut_alpha"])
            core.cfg.folding_res_gain = float(flags["folding_res_gain"])
            core.cfg.shortcut_alpha_eval = float(args.shortcut_alpha_eval)
            core.cfg.decoder_refine_alpha = float(flags["decoder_refine_alpha"])
            core.cfg.attr_detach_geometry = bool(flags["attr_detach_geometry"])

            x = batch["input"].to(device, non_blocking=True).float()
            target = batch["target"].to(device, non_blocking=True).float()
            mask = batch["mask"].to(device, non_blocking=True).float()
            # The render loss needs every channel even in the xyz stage: it takes
            # the target's scale/rot/opacity/colour so the image difference is due
            # to geometry alone. Keep the full tensor before the slice.
            target_full = target
            target = target[..., : cfg.target_dim]

            # Geometry-first factorisation p(X, A | Z) = p(X | Z) p(A | X, Z): the
            # attribute heads are conditioned on a position, and early in training
            # the predicted position is wrong by more than a point spacing, so
            # conditioning on it teaches the heads to compensate for geometry error
            # instead of learning A(x). Feed the true xyz first and hand over
            # gradually. The blend happens inside the decoder, per point, against
            # its own prediction -- one forward, and it degrades cleanly to
            # inference conditioning when the probability reaches 0.
            attr_xyz = target_full[..., 0:3] if cfg.target_dim > 3 else None
            core.cfg.attr_teacher_prob = float(flags["attr_teacher_prob"])

            # Resolve the encoder input once and reuse it for both the ordinary
            # and rotated forwards.  R6 encoded the 589,824-point enc_input in the
            # ordinary branch but rotated the 262,144-point decoder target for the
            # equivariance branch.  Those tensors have different selections and
            # group memberships, so forcing their latents to match collapses the
            # code instead of teaching equivariance.
            enc_x = (batch["enc_input"].to(device, non_blocking=True).float()
                     if "enc_input" in batch else x)
            enc_mask = (batch["enc_mask"].to(device, non_blocking=True).float()
                        if "enc_mask" in batch else mask)
            group_anchor = (batch["group_anchor"].to(device, non_blocking=True).float()
                            if "group_anchor" in batch else None)

            with amp_ctx():
                out = model(
                    x, mask,
                    run_decode=flags["run_decode"],
                    run_gen=flags["run_gen"],
                    gen_noise_std=args.denoise_std,
                    attr_xyz=attr_xyz,
                    enc_x=enc_x,
                    enc_mask=enc_mask,
                    group_anchor=group_anchor,
                )
                if (bool(int(getattr(args, "reuse_prev_vis", 0)))
                        and prev_vis is not None and prev_vis.shape == mask.shape):
                    out["render_visible"] = prev_vis
                loss, logs = total_loss(
                    out, target, mask, weights, layout, model_layout, cfg.sh_dim,
                    with_attrs=flags["with_attrs"],
                    run_decode=flags["run_decode"],
                    run_gen=flags["run_gen"],
                )
                # The fixed-anchor decoder has two geometrically distinct jobs:
                # move each group to its target centre and recover its centred
                # shape.  Keep the translation magnitude visible so a saturated
                # translation head or a shape-only collapse cannot hide behind
                # the aggregate loss.
                if "group_translation_abs" in out:
                    logs["group_translation_abs"] = out[
                        "group_translation_abs"
                    ].detach()
                if "memory_token_std" in out:
                    logs["memory_token_std"] = out["memory_token_std"].detach()
                if "compact_local_std" in out:
                    logs["compact_local_std"] = out["compact_local_std"].detach()
                if (
                    float(weights.get("w_equiv", 0.0)) > 0
                    and args.equiv_every > 0
                    and step % args.equiv_every == 0
                    and step >= args.latent_end
                ):
                    R = random_rotation(args.equiv_rot_deg, device, x.dtype)
                    enc_x_rot = enc_x.clone()
                    enc_x_rot[..., 0:3] = enc_x[..., 0:3] @ R.transpose(0, 1)
                    ga_rot = (None if group_anchor is None
                              else group_anchor @ R.transpose(0, 1))
                    z_rot = core.compact_from_points(
                        enc_x_rot, enc_mask, group_anchor=ga_rot)
                    eq = equivariance_loss(
                        out["z_compact"],
                        z_rot,
                        R,
                        budget["merge"],
                        budget["per_group"],
                        budget["centroid"],
                        shape_weight=float(getattr(args, "equiv_shape_weight", 0.0)),
                    )
                    loss = loss + float(weights["w_equiv"]) * eq
                    logs["equiv"] = eq.detach()
                # keep every parameter in the autograd graph so DDP buckets stay
                # stable across phase switches (cheaper than find_unused_parameters)
                touch = sum((p * 0.0).sum() for p in core.parameters() if p.requires_grad)
                loss = loss + touch

            # Rasterisation runs outside autocast: the CUDA rasteriser takes fp32
            # buffers and a bf16 tensor reaches it as garbage rather than as an
            # error, so the cast has to be explicit and the block has to be here.
            w_render = float(weights.get("w_render", 0.0))
            _dsr = bool(int(getattr(args, "attr_detach_scale_rot", 0)))
            if w_render > 0 and flags["run_decode"] and step >= int(args.render_start):
                for br, key in (("", "pred"), ("gen_", "gen_pred")):
                    if br and not flags["run_gen"]:
                        continue
                    wb = w_render * (float(args.render_gen_scale) if br else 1.0)
                    if wb <= 0 or key not in out:
                        continue
                    r = render_loss(
                        out.get(key.replace("pred", "attr_pred")
                                if key == "pred" else "gen_attr_pred", out[key]).float(),
                        target_full, mask,
                        batch["camera"], batch["center"], batch["scale"], layout,
                        lam_dssim=float(args.render_lam_dssim),
                        downscale=int(flags["render_downscale"]),
                        max_points=int(args.render_max_points),
                        min_coverage=float(args.render_min_coverage),
                        views=int(args.render_views),
                        view_jitter_deg=float(args.render_view_jitter_deg),
                        seed=step,
                        detach_scale_rot=_dsr,
                    )
                    loss = loss + wb * r["render"]
                    for k, v in r.items():
                        logs[br + k] = v.detach()

            # Attribute-only render: same rasteriser, position path cut. Kept as a
            # separate term rather than a bigger w_render because the two paths
            # need weights three orders of magnitude apart.
            _edge_g, _perc_g, _sobel_g = detail_gains(step, args)
            w_ra = float(weights.get("w_render_attr", 0.0))
            if (w_ra > 0 and flags["run_decode"] and flags["with_attrs"]
                    and step >= int(args.render_start)):
                # Route to the separate attribute decoder when it exists: that is
                # the module the render loss is meant to train, and its inputs
                # from the geometry side are detached so nothing leaks back.
                # detach_xyz only when there is no separate attribute decoder. It
                # exists to keep the render gradient out of the geometry stack,
                # and with the separate module that is already guaranteed at the
                # module boundary -- every geometry input is detached, and a
                # backward through this term puts exactly 0.0 on decoder.* and
                # decompressor.* either way (measured).
                #
                # Detaching here as well had one further effect, which was not
                # intended: it zeroed the bounded nudge. head_nudge is downstream
                # of the position output, so detach_xyz=True left it with no
                # gradient at all, and the only term still reaching it was
                # w_render at 0.004 -- 1/17000 of this weight. The nudge was dead.
                # Measured on head_nudge: 0.000 detached, 3.3e-4 via w_render,
                # 5.56 here.
                has_attr_dec = "attr_pred" in out
                ra = render_loss(
                    out.get("attr_pred", out["pred"]).float(), target_full, mask,
                    batch["camera"], batch["center"], batch["scale"], layout,
                    lam_dssim=float(args.render_lam_dssim),
                    edge_gain=_edge_g,
                    w_sobel=_sobel_g,
                    downscale=int(flags["render_downscale"]),
                    max_points=int(args.render_max_points),
                    min_coverage=float(args.render_min_coverage),
                    views=int(args.render_views),
                    view_jitter_deg=float(args.render_view_jitter_deg),
                    seed=step,
                    detach_xyz=((bool(getattr(cfg, "joint_shared_decoder", False))
                                 or bool(getattr(cfg, "joint_direct_decoder", False)))
                                and bool(getattr(cfg, "joint_detach_render_xyz", True)))
                               or not has_attr_dec,
                    detach_scale_rot=_dsr,
                    w_perc=_perc_g,
                    w_splat=float(getattr(args, "w_splat_area", 0.0)),
                    splat_quantile=float(getattr(args, "splat_area_quantile", 0.80)),
                    near_detach_frac=float(getattr(args, "near_detach_frac", 0.1)),
                    pred_presence=(out.get("presence")
                                   if int(getattr(args, "render_presence", 0)) else None),
                    # batch tensors are moved individually, so this one must be too
                    ref_image=(batch["photo"].to(device, non_blocking=True).float()
                               if "photo" in batch else None),
                    extra_cams=(batch["extra_cams"].to(device, non_blocking=True).float()
                                if "extra_cams" in batch else None),
                    extra_imgs=(batch["extra_imgs"].to(device, non_blocking=True).float()
                                if "extra_imgs" in batch else None),
                )
                loss = loss + w_ra * ra["render"]
                logs["render_attr"] = ra["render"].detach()
                w_sa = float(getattr(args, "w_splat_area", 0.0))
                if w_sa > 0.0 and "splat_area" in ra:
                    loss = loss + w_sa * ra["splat_area"]
                    logs["splat"] = ra["splat_area"].detach()
                prev_vis = ra.get("visible")
                logs["rcover"] = ra["render_covered"]
                logs["rattr"] = ra["render_l1"].detach()
                logs["rattr_dssim"] = ra["render_dssim"].detach()
                if "render_perc" in ra:
                    logs["rperc"] = ra["render_perc"].detach()
                if "render_sobel" in ra:
                    logs["rsobel"] = ra["render_sobel"].detach()

                # The same term on the student. gen is what a world model decodes,
                # so it is the branch whose rendered attributes actually ship; the
                # codec's are a diagnostic. Scaled by render_gen_scale because the
                # student is further from convergence and its geometry is worse
                # (nn_unique 0.29 against the codec's 0.48), so an equal weight
                # would have it chasing an image its point set cannot form.
                if flags["run_gen"] and "gen_pred" in out:
                    rg = render_loss(
                        out.get("gen_attr_pred", out["gen_pred"]).float(), target_full, mask,
                        batch["camera"], batch["center"], batch["scale"], layout,
                        lam_dssim=float(args.render_lam_dssim),
                        downscale=int(flags["render_downscale"]),
                        max_points=int(args.render_max_points),
                        min_coverage=float(args.render_min_coverage),
                        views=int(args.render_views),
                        view_jitter_deg=float(args.render_view_jitter_deg),
                        seed=step, detach_xyz="gen_attr_pred" not in out,
                        detach_scale_rot=_dsr,
                    )
                    loss = loss + w_ra * float(args.render_gen_scale) * rg["render"]
                    logs["gen_rattr"] = rg["render_l1"].detach()

            # `logs["total"]` is set inside total_loss, before the render, equiv,
            # splat-area and gen-render terms are added above. The printed `loss`
            # was therefore not the quantity being differentiated, which is a bad
            # property in general and a disqualifying one while chasing a spike --
            # the log could look calm on the step that blew up. Overwritten with
            # the real thing.
            logs["total"] = loss.detach()
            with torch.no_grad():
                if "extra_cams" in batch:
                    fx = batch["extra_cams"][..., 0]
                    logs["n_extra"] = (fx != 0).float().sum() / max(fx.shape[0], 1)
                if flags["run_decode"] and flags["with_attrs"]:
                    ap = out.get("attr_pred", out.get("pred"))
                    if ap is not None and ap.shape[-1] > 6:
                        sl = int(layout.get("scale", 3))
                        m = mask[0] > 0.5
                        if int(m.sum()) >= 64:
                            def _aniso(t):
                                ls = t[m][:, sl: sl + 3].float().clamp(-15.0, 5.0)
                                return (ls.max(-1).values - ls.min(-1).values).exp()
                            pa, ta = _aniso(ap[0]), _aniso(target_full[0])
                            logs["aniso_p50"] = torch.quantile(pa, 0.5)
                            logs["aniso_gt_p50"] = torch.quantile(ta, 0.5)
                            logs["aniso_lt4"] = (pa < 4).float().mean()
                            logs["aniso_gt_lt4"] = (ta < 4).float().mean()
                            rq = int(layout.get("rot", 6))
                            from .losses import _quat_to_R
                            live = ap[0][m]
                            tgt = target_full[0][m]
                            Rp = _quat_to_R(live[:, rq: rq + 4].float())
                            Rt = _quat_to_R(tgt[:, rq: rq + 4].float())
                            kp = live[:, sl: sl + 3].float().argmax(-1)
                            kt = tgt[:, sl: sl + 3].float().argmax(-1)
                            ar = torch.arange(Rp.shape[0], device=Rp.device)
                            axp = Rp[ar, :, kp]
                            axt = Rt[ar, :, kt]
                            axp = axp / axp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                            axt = axt / axt.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                            ang = torch.acos((axp * axt).sum(-1).abs().clamp(0, 1)) * (180.0 / math.pi)
                            hi = ta > 8
                            logs["rot_hi_p50"] = (ang[hi].median() if int(hi.sum()) >= 8
                                                  else ang.median())
            logs["egain"] = torch.as_tensor(float(_edge_g), device=loss.device)
            logs["wperc"] = torch.as_tensor(float(_perc_g), device=loss.device)
            logs["wsobel"] = torch.as_tensor(float(_sobel_g), device=loss.device)

            # Loss guard. There was none, and the L16k run needed one: the logged
            # total sat at ~17 and jumped to 725 / 929 / 5405 on isolated batches.
            # grad_clip bounds the *update*, but AdamW still folds the clipped
            # gradient's square into exp_avg_sq, so one spike distorts the
            # effective learning rate for the ~1/(1-beta2) = 20 steps that follow
            # -- and at beta2=0.95 with spikes every few hundred steps the second
            # moment never fully recovers. Skipping the batch outright is cheaper
            # than the recovery.
            #
            # DDP: every rank must agree, or the ranks that skip never reach the
            # gradient all-reduce and the job deadlocks. The decision is reduced
            # before it is acted on.
            _lv = float(loss.detach())
            if math.isfinite(_lv):
                loss_hist_all.append(_lv)
            # Stall breaker, evaluated before the all-reduce so every rank still
            # agrees through the MAX below. A majority-skip window means the
            # reference is stale, not that the batches are bad.
            if (len(skip_win) == skip_win.maxlen and sum(skip_win) * 2 > len(skip_win)
                    and len(loss_hist_all) >= 32):
                _new = float(np.median(loss_hist_all))
                if loss_ref is None or _new > loss_ref:
                    if rank0() and step - stall_note >= 200:
                        stall_note = step
                        print(f"  [STALL] step {step}: {sum(skip_win)}/{len(skip_win)} of the "
                              f"last batches were skipped against a frozen reference "
                              f"{loss_ref:.4g}. The loss floor moved; re-referencing to the "
                              f"median over all batches, {_new:.4g}.", flush=True)
                    loss_ref = _new
            bad = (not torch.isfinite(loss)) or (
                loss_spike_mult > 0.0 and loss_ref is not None
                and _lv > loss_ref * loss_spike_mult)
            if ddp_active:
                flag = torch.tensor([1.0 if bad else 0.0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                bad = bool(flag.item() > 0.5)
            skip_win.append(1 if bad else 0)
            if bad:
                skipped += 1
                optim.zero_grad(set_to_none=True)
                if rank0() and skipped % max(int(args.log_every), 1) == 1:
                    # Which term actually blew up. Without this the log says only
                    # that the total was large, and the cause has to be guessed.
                    _sh = []
                    for _k, _v in logs.items():
                        if _k == "total":
                            continue
                        _w = weights.get(f"w_{_k}")
                        if _w is None and _k.startswith("codec_"):
                            _w = weights.get(f"w_{_k[6:]}")
                        if _w is None:
                            continue
                        _c = abs(float(_w) * float(_v))
                        if _c > 0.0:
                            _sh.append((_c, _k))
                    _sh.sort(reverse=True)
                    _tot = max(abs(_lv), 1e-9)
                    _top = " ".join(f"{n}:{c / _tot:.0%}" for c, n in _sh[:4])
                    print(f"  [SKIP] step {step}: loss {_lv:.4g} vs "
                          f"reference {loss_ref if loss_ref is None else round(loss_ref, 4)} "
                          f"-- batch dropped, optimiser state untouched "
                          f"({skipped} skipped so far) TOP[{_top}]", flush=True)
                step += 1
                continue
            # Median-tracking reference, not a mean: a mean is dragged up by the
            # very spikes this is meant to catch.
            loss_hist.append(float(loss.detach()))
            if len(loss_hist) >= 32:
                loss_ref = float(np.median(loss_hist))
                # DRIFT GUARD, separate from the spike guard above. The spike guard
                # compares against a 256-step trailing median, which is blind by
                # construction to monotone growth: L17k climbed 4,388 -> 222,319
                # over 30000 steps, a 2.7% rise per 256-step window against a 25x
                # threshold, and it never fired once in 53000 steps. Anchor a second
                # reference early and never move it, so a slow runaway is reported
                # at the point it becomes obvious rather than at the end of the run.
                if loss_ref_slow is None and step >= 2000:
                    loss_ref_slow = loss_ref
                elif (loss_ref_slow is not None and rank0()
                      and step % 1000 == 0 and loss_ref > loss_ref_slow * 10.0):
                    print(f"  [DRIFT] step {step}: trailing median {loss_ref:.4g} is "
                          f"{loss_ref / loss_ref_slow:.0f}x the step-2000 median "
                          f"{loss_ref_slow:.4g}. A term is running away -- read TOP[..] "
                          f"in the log line above.", flush=True)

            optim.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                gnorm = torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
            else:
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
            gn = float(gnorm)
            # The anchor is frozen so a run of moderately large steps cannot
            # walk the threshold up. detail_start changes the normal grad size
            # (full resolution, edge weight, VGG), so the anchor is collected
            # again there. An absolute ceiling still drops the 6k-1e5 rasterizer
            # spikes while that second median is being measured.
            detail_at = int(getattr(args, "detail_start", 0) or 0)
            if detail_at > 0 and step >= detail_at and not gnorm_detail_reset:
                gnorm_hist.clear()
                gnorm_anchor = None
                gnorm_detail_reset = True
                if rank0():
                    print(f"  [GSPAN] step {step}: detail phase, re-collecting "
                          f"grad-norm anchor", flush=True)
            if gnorm_anchor is None and len(gnorm_hist) >= 32:
                gnorm_anchor = float(np.median(list(gnorm_hist)))
                if rank0():
                    print(f"  [GSPAN] step {step}: frozen grad-norm anchor "
                          f"{gnorm_anchor:.4g} x{grad_spike_mult:g}", flush=True)
            over_abs = grad_spike_abs > 0.0 and (not math.isfinite(gn) or gn > grad_spike_abs)
            over_rel = (grad_spike_mult > 0.0 and gnorm_anchor is not None
                        and (not math.isfinite(gn) or gn > gnorm_anchor * grad_spike_mult))
            gbad = over_abs or over_rel
            if ddp_active:
                gflag = torch.tensor([1.0 if gbad else 0.0], device=device)
                dist.all_reduce(gflag, op=dist.ReduceOp.MAX)
                gbad = bool(gflag.item() > 0.5)
            if gbad:
                gskipped += 1
                optim.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.update()
                if rank0() and gskipped % max(int(args.log_every), 1) == 1:
                    _anch = "unset" if gnorm_anchor is None else f"{gnorm_anchor:.4g}"
                    print(f"  [GSKIP] step {step}: gnorm {gn:.4g} vs anchor {_anch} "
                          f"x{grad_spike_mult:g} abs {grad_spike_abs:g} -- step dropped, "
                          f"optimiser state untouched ({gskipped} grad-skips so far)",
                          flush=True)
                step += 1
                continue
            if math.isfinite(gn) and gnorm_anchor is None:
                gnorm_hist.append(gn)
            if scaler.is_enabled():
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            logs["gnorm"] = gnorm.detach() if torch.is_tensor(gnorm) else torch.as_tensor(gnorm)

            # The compressor's running latent std is a buffer, and DDP is built
            # with broadcast_buffers=False, so without this each rank carries its
            # own scale and only rank 0's reaches the checkpoint. Everything that
            # consumes `encode_compact(normalized=True)` -- i.e. the world model --
            # then sees a scale that was fitted on a third of the data.
            if ddp_active and step % max(int(args.latent_scale_sync_every), 1) == 0:
                ls = core.latent_scale.detach().clone()
                dist.all_reduce(ls, op=dist.ReduceOp.SUM)
                core.latent_scale.copy_(ls / float(dist.get_world_size()))

            if ema is not None:
                ema.update(core)

            for k, v in logs.items():
                prev = running.get(k)
                running[k] = v if prev is None else prev + v
            step += 1

            if step % args.log_every == 0:
                _moved, _suspect = _param_movement()
                _mv_str = "d[" + " ".join(f"{n[:4]} {_moved[n]:.0e}" for n in group_names) + "] "
                if _suspect:
                    print(f"  [FROZEN] nonzero lr but zero parameter movement for 3 "
                          f"consecutive logs: {', '.join(_suspect)} -- that module is in "
                          f"the graph and reporting losses while not training", flush=True)
                dt = (time.time() - t_last) / args.log_every
                t_last = time.time()
                avg = {k: float(v) / args.log_every for k, v in running.items()}
                running = {}
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
                gs = float(weights.get("w_gen_scale", 1.0))
                # TOP-K BY WEIGHTED CONTRIBUTION. The L17k run spent 53000 steps
                # with `w_z_raw_mse * z_mse` at 97% of the objective while every
                # printed term looked normal -- z_mse was simply not in the line.
                # Curating which terms to print cannot prevent that: the next term
                # to blow up is by definition the one nobody thought to print. So
                # rank every logged scalar by weight*value and print the top few,
                # with each one's share. A term at 97% is then impossible to miss,
                # and a term at 1e-6 (the latent_std floor on that same run) is
                # visible as dead rather than merely absent.
                _shares = []
                for _k, _v in avg.items():
                    for _wn in (f"w_{_k}", f"w_{_k[6:]}" if _k.startswith("codec_") else None):
                        if not _wn:
                            continue
                        _w = weights.get(_wn)
                        if _w is None:
                            continue
                        _c = abs(float(_w) * float(_v))
                        if _c > 0.0:
                            _shares.append((_c, _k))
                        break
                _tot_abs = max(abs(float(avg.get("total", 0.0))), 1e-9)
                _shares.sort(reverse=True)
                _top = " ".join(f"{n}:{c / _tot_abs:.0%}" for c, n in _shares[:4])
                msg = (
                    f"[{flags['phase']}] step {step}/{args.max_steps} "
                    f"loss {avg.get('total', 0):.4f} "
                    f"z_l1 {avg.get('z_l1', 0):.4f} "
                    f"codec_xyz {avg.get('codec_xyz', 0):.4f} "
                    f"gen_xyz {avg.get('gen_xyz', 0):.4f} "
                    f"distill {avg.get('gen_distill', 0):.4f} "
                    f"lat_std {avg.get('latent_shape_std', 0):.2f}/{avg.get('latent_std_max', 0):.2f} "
                    f"lrank {avg.get('latent_erank', 0):.1f} "
                    f"res {avg.get('res_ratio', 0):.4f} "
                    f"dfrac {avg.get('direct_frac', 0):.2f} "
                    f"z_res {avg.get('z_residual', 0):.4f} "
                    f"G[ch {avg.get('codec_chamfer', 0):.4f} "
                    f"cov {avg.get('codec_coverage', 0):.4f} "
                    f"ich {avg.get('codec_intra_chamfer', 0):.4f} "
                    f"spc {avg.get('codec_intra_spacing', 0):.4f} "
                    f"ot {avg.get('codec_intra_sinkhorn', 0):.4f} "
                    f"cen {avg.get('codec_group_centroid', 0):.4f} "
                    f"tr {avg.get('group_translation_abs', 0):.4f} "
                    f"mem {avg.get('memory_token_std', 0):.3f} "
                    f"cmem {avg.get('compact_local_std', 0):.3f}] "
                    # The terms that were in the total but not in this line. The
                    # L16k run spiked from ~17 to 5405 with every printed term
                    # flat, which made the spike unattributable from the log
                    # alone -- these four are the ones that can do that, and
                    # `rad` in particular divides by a GT group radius.
                    + (f"U[rad {avg.get('codec_radius', 0):.3f} "
                       f"c3d {avg.get('codec_cov3d', 0):.3f} "
                       f"aset {avg.get('codec_attr_set', 0):.4f} "
                       f"aloc {avg.get('codec_attr_local_set', 0):.4f} "
                       f"aspr {avg.get('codec_attr_spread', 0):.4f} "
                       f"aslp {avg.get('codec_attr_slope', 0):.4f} "
                       f"ahsc {avg.get('codec_attr_hung_scale', 0):.4f} "
                       f"ahop {avg.get('codec_attr_hung_opacity', 0):.4f} "
                       f"ahrt {avg.get('codec_attr_hung_rot', 0):.4f} "
                       f"dec {avg.get('latent_decorr', 0):.4f} "
                       f"p2g {avg.get('codec_p2g', 0):.3f} "
                       f"eq {avg.get('equiv', 0):.4f} "
                       f"gn {avg.get('gnorm', 0):.2f}"
                       f"{f' skip {skipped}' if skipped else ''}] "
                       f"TOP[{_top}] ")
                    + (f"jdx {avg.get('joint_dxyz', 0):.6f} "
                       if bool(getattr(cfg, "joint_shared_decoder", False)) else "")
                    + (f"rend {avg.get('render_l1', 0):.4f}/{avg.get('render_dssim', 0):.3f}"
                       f"@{avg.get('render_used', 0):.2f} "
                       if float(weights.get("w_render", 0.0)) > 0 else "")
                    # Gated on w_render_attr, not w_render. The position render is 0
                    # now that both terms route to the same tensor, and this block
                    # used to hang off w_render -- so the only photometric term still
                    # running was invisible in the log. `cov` is the fraction of
                    # points the render reached, which is what decides how much of
                    # the anchor is gated off.
                    + (f"rattr {avg.get('rattr', 0):.4f}/{avg.get('rattr_dssim', 0):.3f} "
                       f"cov {avg.get('rcover', 0):.2f} "
                       + (f"splat {avg.get('splat', 0):.5f} "
                          if float(getattr(args, 'w_splat_area', 0.0)) > 0 else "")
                       + (f"rperc {avg.get('rperc', 0):.3f} " if avg.get('rperc', 0) else "")
                       + (f"rsobel {avg.get('rsobel', 0):.4f} " if avg.get('rsobel', 0) else "")
                       if float(weights.get("w_render_attr", 0.0)) > 0 else "")
                    + _mv_str
                    + (f"tf {float(flags['attr_teacher_prob']):.2f} "
                       f"A[sc {avg.get('codec_scale', 0):.3f} ani {avg.get('codec_aniso', 0):.3f} "
                       f"rot {avg.get('codec_rot', 0):.3f} "
                       f"op {avg.get('codec_opacity', 0):.3f} col {avg.get('codec_color', 0):.3f}] "
                       if flags["with_attrs"] else "")
                    # The gen terms, weighted. Without these a 15x rise in the total
                    # cannot be attributed from the log at all: every term that was
                    # printed stayed flat while w_gen_basis quietly ramped to 57% of
                    # the objective. Printed as contributions (value x weight), which
                    # is the only form in which terms of different units compare.
                    + ("".join(
                        # `w_gen_scale` is the gen ramp and is applied inside
                        # total_loss, not by effective_weights -- but only to the
                        # branch_geometry terms. Leaving it out here overstated
                        # those by 1/ramp (3.7x at the start of the ramp) and made
                        # the printed contributions sum past the printed total.
                        f"{n} {float(weights.get(w, 0.0)) * avg.get(k, 0.0) * (gs if ramped else 1.0):.2f} "
                        for n, k, w, ramped in (
                            ("Gbas", "gen_basis", "w_gen_basis", False),
                            ("Gres", "gen_xyz_residual", "w_gen_xyz_residual", True),
                            ("Gich", "gen_intra_chamfer", "w_gen_intra_chamfer", True),
                            ("Gdst", "gen_distill", "w_distill", False),
                            ("Gp2g", "gen_p2g", "w_gen_p2g", True),
                        ))
                       if flags["run_gen"] else "")
                    + f"n {avg.get('n_used', 0):.0f} "
                    f"empty {avg.get('empty_frac', 0):.2f} "
                    f"a_sc {float(flags['shortcut_alpha']):.2f} "
                    f"a_rf {float(flags['decoder_refine_alpha']):.2f} "
                    f"dtch {int(flags['attr_detach_geometry'])} "
                    f"fn {int(getattr(args, 'attr_frame_needles', 0))} "
                    f"dsr {int(getattr(args, 'attr_detach_scale_rot', 0))} "
                    f"det[eg {avg.get('egain', 0):.2f} perc {avg.get('wperc', 0):.3f} "
                    f"sob {avg.get('wsobel', 0):.3f}] "
                    f"shp[p50 {avg.get('aniso_p50', 0):.2f}/{avg.get('aniso_gt_p50', 0):.2f} "
                    f"<4 {100*avg.get('aniso_lt4', 0):.0f}/{100*avg.get('aniso_gt_lt4', 0):.0f}% "
                    f"rot {avg.get('rot_hi_p50', 0):.0f}°] "
                    f"nx {avg.get('n_extra', 0):.0f} "
                    f"lr {base_lr:.2e} {dt:.2f}s/it {mem:.1f}GB"
                )
                log(msg)
                if rank0():
                    rec = {
                        "step": step,
                        "loss": avg.get("total", 0),
                        "lr": float(base_lr),
                        "rattr": avg.get("rattr", 0),
                        "rsobel": avg.get("rsobel", 0),
                        "rperc": avg.get("rperc", 0),
                        "egain": avg.get("egain", 0),
                        "wperc": avg.get("wperc", 0),
                        "wsobel": avg.get("wsobel", 0),
                        "aniso_p50": avg.get("aniso_p50", 0),
                        "aniso_gt_p50": avg.get("aniso_gt_p50", 0),
                        "aniso_lt4": avg.get("aniso_lt4", 0),
                        "aniso_gt_lt4": avg.get("aniso_gt_lt4", 0),
                        "rot_hi_p50": avg.get("rot_hi_p50", 0),
                        "codec_scale": avg.get("codec_scale", 0),
                        "codec_aniso": avg.get("codec_aniso", 0),
                        "codec_rot": avg.get("codec_rot", 0),
                        "codec_attr_set": avg.get("codec_attr_set", 0),
                        "codec_attr_local_set": avg.get("codec_attr_local_set", 0),
                        "n_extra": avg.get("n_extra", 0),
                        "s_per_it": dt,
                    }
                    with open(os.path.join(args.out_dir, "train_metrics.jsonl"), "a") as _jf:
                        _jf.write(json.dumps(rec) + "\n")

            do_eval = (step in milestones) or (args.eval_every > 0 and step % args.eval_every == 0)
            if do_eval:
                metrics = run_eval(model, val_ds, args, cfg, step, device, amp_ctx,
                                   save_outputs=True, eval_views=eval_views)
                codec_r = metrics.get("codec_xyz_rmse_norm", float("inf"))
                gen_r = metrics.get("gen_xyz_rmse_norm", None)
                # ckpt_best is selected on score_metric. Plain rmse concentrates on
                # the handful of Morton-jump groups that hold most of the squared
                # error and is only weakly related to how detailed the renders look.
                key = {
                    "rmse": "xyz_rmse_norm",
                    "rel_offset": "rel_offset_p50",
                    "chamfer": "chamfer_norm",
                    "intra_chamfer": "intra_chamfer_rel",
                    "psnr": "psnr_photo",
                    "psnr_teacher": "psnr_teacher",
                    # Handled specially in `_score`; the key is unused but has to
                    # exist so the lookup below does not KeyError.
                    "psnr_gap_worst": "psnr_gap",
                }[str(args.score_metric)]
                # PSNR is better when larger; every other choice here is better when
                # smaller, and the comparison below is a strict "<". Negating is the
                # whole conversion -- getting it wrong would silently save the WORST
                # checkpoint, which is exactly the class of failure this file keeps
                # hitting, so it is done in one place for both branches.
                sgn = -1.0 if str(args.score_metric).startswith("psnr") else 1.0

                def _score(branch: str):
                    if str(args.score_metric) == "psnr_gap_worst":
                        # Worst scene's deficit against its OWN GT Gaussians.
                        # Selecting on pooled psnr_photo saved a checkpoint that was
                        # 1.2-3.0 dB ahead on the truck and up to 2.2 dB behind on
                        # the locomotive, because the mean of the two was still
                        # positive. The gap (gt - pred) is used rather than raw PSNR
                        # so that the intrinsically harder scene is not permanently
                        # the argmin regardless of how the model is doing on it.
                        gaps = [float(v) for k, v in metrics.items()
                                if k.startswith("scene") and k.endswith(f"/{branch}_psnr_gap")]
                        if gaps:
                            return max(gaps)
                        v = metrics.get(f"{branch}_psnr_gap")
                        return float("inf") if v is None else float(v)
                    v = metrics.get(f"{branch}_{key}")
                    return float("inf") if v is None else sgn * float(v)

                codec_s = _score("codec")
                gen_s = None if f"gen_{key}" not in metrics else _score("gen")
                # Two separate bests instead of one average. The average froze
                # ckpt_best the moment gen switched on: at gen_start the gen branch
                # has had exactly zero training (m_gen = 0 there), so 0.5*(codec+gen)
                # jumped from 0.245 to 0.386 and could never beat any earlier
                # codec-only score again. It also let the path that is thrown away
                # decide half of the selection.
                score = codec_s
                log(
                    f"  eval step {step}: "
                    # The target metric first, because it is the one that decides
                    # whether the run is working. gt_photo is the ceiling this
                    # snapshot allows and moves with it, so a bare prediction PSNR
                    # cannot be read without it.
                    f"PSNR photo={metrics.get('codec_psnr_photo', float('nan')):.2f} "
                    f"(gt {metrics.get('codec_psnr_gt_photo', float('nan')):.2f}, "
                    f"gap {metrics.get('codec_psnr_gap', float('nan')):.2f}) "
                    f"teacher={metrics.get('codec_psnr_teacher', float('nan')):.2f} "
                    f"ssim={metrics.get('codec_ssim_photo', float('nan')):.3f} "
                    # Per-scene PSNR and its gap against that scene's own GT
                    # Gaussians. The pooled number above averages scenes that can
                    # move in opposite directions.
                    + "".join(
                        f"| s{sid} {metrics[f'scene{sid}/codec_psnr_photo']:.2f}"
                        f"({metrics.get(f'scene{sid}/codec_psnr_gap', float('nan')):+.2f}) "
                        for sid in sorted(
                            int(k.split("/")[0][5:]) for k in metrics
                            if k.startswith("scene") and k.endswith("/codec_psnr_photo"))
                    )
                    + (f"| gen PSNR={metrics.get('gen_psnr_photo', float('nan')):.2f} "
                       if "gen_psnr_photo" in metrics else "")
                    + f"| codec_rmse={codec_r:.5f} "
                    f"gen_rmse={(gen_r if gen_r is not None else float('nan')):.5f} "
                    f"score[{args.score_metric}]={score:.5f} "
                    # The coverage barrier. Computed into `agg` since the anchor
                    # fix but never printed, so a run could sit at 0.45 for 32000
                    # steps without it being visible in the log. It is the metric
                    # the sinkhorn anneal is meant to move, and 28.5 dB of render
                    # error rides on it, so it goes next to the render numbers.
                    f"uniq={metrics.get('codec_nn_unique', float('nan')):.3f} "
                    f"eps={metrics.get('sinkhorn_eps', float('nan')):.4f} "
                    f"ich={metrics.get('codec_intra_chamfer_rel', float('nan')):.3f} "
                    f"thr={metrics.get('codec_thread_pred', float('nan')):.3f}/"
                    f"{metrics.get('codec_thread_gt', float('nan')):.3f} "
                    f"relq={metrics.get('codec_rel_offset_p50', float('nan')):.3f} "
                    f"off={metrics.get('codec_offset_rmse_norm', float('nan')):.5f} "
                    f"cen={metrics.get('codec_centroid_rmse_norm', float('nan')):.5f} "
                    f"chf={metrics.get('codec_chamfer_norm', float('nan')):.5f} "
                    f"empty={metrics.get('empty_frac', 0):.2f} "
                    f"a_rf={metrics.get('eval_refine_alpha', 0):.2f} "
                    f"hf={metrics.get('latent_hf_ratio', 0):.3f}/"
                    f"{metrics.get('shape_hf_ratio', 0):.3f} "
                    f"sstd={metrics.get('shape_chan_std_min', 0):.2f}-"
                    f"{metrics.get('shape_chan_std_max', 0):.2f} "
                    f"rad999={metrics.get('codec_group_radius_p999', 0):.3f} "
                    f"lam2={metrics.get('codec_aniso_lam2_pred', float('nan')):.3f}/"
                    f"{metrics.get('codec_aniso_lam2_gt', float('nan')):.3f} "
                    f"tmpl={metrics.get('codec_template_erank_pred', float('nan')):.1f}/"
                    f"{metrics.get('codec_template_erank_gt', float('nan')):.1f}"
                )
                if ema is not None:
                    with ema.swapped(core):
                        em = run_eval(model, val_ds, args, cfg, step, device, amp_ctx,
                                      save_outputs=False, tag="_ema")
                    log(f"    ema: codec_rmse={em.get('codec_xyz_rmse_norm', float('nan')):.5f} "
                        f"relq={em.get('codec_rel_offset_p50', float('nan')):.3f}")

                if rank0() and math.isfinite(score) and score < best_score:
                    best_score = score
                    torch.save(
                        {"model": core.state_dict(), "step": step, "best_score": best_score,
                         "cfg": cfg.as_dict(), "args": vars(args),
                         "eval_fingerprint": score_fp},
                        os.path.join(args.out_dir, "ckpt_best.pt"),
                    )
                    log(f"  new best score {best_score:.5f} -> ckpt_best.pt")

                # gen is the path a world model actually consumes, so it needs its
                # own selection rather than riding the codec's.
                if (
                    rank0()
                    and gen_s is not None
                    and math.isfinite(gen_s)
                    and step >= int(args.gen_start)
                    and gen_s < best_gen_score
                ):
                    best_gen_score = gen_s
                    torch.save(
                        {"model": core.state_dict(), "step": step, "best_gen_score": best_gen_score,
                         "cfg": cfg.as_dict(), "args": vars(args),
                         "eval_fingerprint": score_fp},
                        os.path.join(args.out_dir, "ckpt_best_gen.pt"),
                    )
                    log(f"  new best GEN score {best_gen_score:.5f} -> ckpt_best_gen.pt")

            if rank0() and args.save_every > 0 and step % args.save_every == 0:
                payload = {
                    "model": core.state_dict(), "optim": optim.state_dict(), "step": step,
                    "best_score": best_score, "cfg": cfg.as_dict(), "args": vars(args),
                    "eval_fingerprint": score_fp,
                }
                torch.save(payload, os.path.join(args.out_dir, f"ckpt_step{step:08d}.pt"))
                torch.save(payload, os.path.join(args.out_dir, "ckpt_latest.pt"))
        epoch += 1

    log("training done")
    if is_dist():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
