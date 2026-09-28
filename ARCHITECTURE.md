# can3tok: architecture, training, and limits

Current run: `runs/v4_262k_g64_20260808_083806`.
Fixed constraints: `max_points = 262144`, `z_compact = 32 x 64 x 64`, stage `xyz`.

Everything is scored on the **permutation-invariant within-group chamfer,
normalised by the group's mean radius** (`intra_chamfer_rel`). The GT cloud's own
nearest-neighbour spacing is **0.191** in those units, so 0.191 is "1.0x" -- the
resolution the data itself carries.

| | chamfer | x spacing |
|---|---|---|
| centroid only, no shape at all | 0.524 | 2.74 |
| best this codebase reached before this work (`_fix` @8000) | 0.339 | 1.77 |
| folding envelope alone, zero training | 0.294 | 1.54 |
| **current, @2000** | **0.242 - 0.249** | **1.27 - 1.30** |
| folding + rank-22 residual on aligned targets (oracle) | 0.218 | 1.14 |

---

## 1. Data flow

```
262,144 points (xyz, scene-normalised)
    |
    |  (1) data.py -- no learned parameters
    |      Morton sort -> local kd (block 32) -> template-aligned slot order
    v
4096 groups x 64 points
    |
    |  (2) encoder (0.27M) -- identity pack, effectively a reshape
    |      64*3 xyz + 64 mask = 256 numbers/group
    |      residual_pack: stores (xyz - group centroid)
    v
z_raw : 32ch x 256x128   (32768 cells = 4096 groups x 8 tokens)
    |
    |  (3) compressor (17.0M)
    |      input normalised by group extent
    |      8 tokens -> token_merge MLP -> group token (d=448)
    |      + cell_vec (window attention) + analytic anchors
    |      -> mid 448 -> 256 -> 256 -> heads
    v
z_compact : 32 x 64x64 = 131,072 numbers          <-- the diffusion interface
    |    per group: centroid 3 + log-extent 1  (analytic anchors, written literally)
    |                shape 28                   (learned)
    |    0.5 compact channels per point
    |
    +--> (4) decompressor (13.6M) + decoder (49.3M)   codec teacher
    +--> (5) gen_decoder (25.7M)                      student, z_compact only,
                                                      gradients detached
```

Total **105.8M**. The "encoder" is not a network: 0.27M is an optional residual
head frozen until step 12000, and the pack itself is deterministic. The real
encoder is the compressor.

Both decode paths need **only z_compact** at inference -- neither reads the true
`z_raw`. The codec reconstructs `z_raw_hat` from `z_compact` and decodes that; the
gen branch goes straight from `z_compact` to points. The codec exists to build the
information through a large intermediate (1,048,576 numbers); gen is the path a
world model uses, because it consumes a `z_compact` the world model updated.

### The folding decoder (both paths)

```
shape (28ch) + contextualised group token (448)
    |-> fold_head  -> 3 log-scales + 3 axis-angle  ->  R(a) diag(s)
    |-> shape_xyz  -> residual, tanh-capped at 1.0, gain-ramped from step 500

local   = fibonacci_ball(64) + bounded_residual   <- fixed template, never trained
local   = local - local.mean(slots)               <- residual cannot move the centroid
offsets = R(a) diag(s) local
points  = centroid_anchor + scale * offsets
    v
codec: 10 cross-attention refine layers, gated by decoder_refine_alpha
gen:   template expand + cross-attention refine, both gated the same way
```

| piece | the measurement that justifies it |
|---|---|
| fixed template instead of a free MLP | the old `shape_xyz` had output effective rank **4.67** (an untrained random 28->192 linear map gives 26.3); the bare template beats the trained map, 0.294 vs 0.345 |
| filled **ball**, not a shell | 0.336 vs 0.367 |
| **rotated** frame, not axis-aligned | 0.301 vs 0.336 |
| template scale sqrt(3), not sqrt(5) | initial chamfer 0.299 vs 0.331 |
| residual tanh-capped at 1.0 | free (rank-22 oracle 0.218 uncapped, 0.217 at cap 1.2); uncapped, the 192-output residual out-races the 6-output frame head and reproduces the anisotropy itself (`aniso_lam2` 0.999 -> 0.243) |
| every additive path capped **and** ramped | with the residual and refine off, `direct_frac` sat at **30.6**: the deep path was subtracting 97% of the folding output |
| mean-centre before the frame | nothing else supervises the group centroid; `cen` 0.00719 -> 0.00143 |
| proper rotation (det = +1) | eigh's basis plus a free per-axis sign flip lands on a reflection for **50%** of groups, which `R(axis-angle) diag(softplus)` can never produce |
| template-aligned slots | mean residual 1.010 -> 0.408, reachable chamfer 0.261 -> **0.218** |
| compressor input normalised by group extent | pack effective rank **1.01 -> 20.48** (below) |
| 8 tokens merged by MLP, not by mean | 4.26 -> 16.23 at init |

---

## 2. Training

Phases, 20000 steps, 4-GPU DDP, batch 1 per rank, AdamW lr 2e-4 (500 warmup,
cosine to 1e-5), wd 1e-4, grad clip 1.0, bf16. ~3.3 s/it, 15.6 GB per rank.

| step | what turns on |
|---|---|
| 0-500 | compressor <-> decompressor round trip only. Residual gain 0, so the output is the bare template times the learned frame |
| 500-1000 | folding residual gain 0 -> 1 |
| 2500 | **decode on** at `geo_weak_scale` 0.15; `w_intra_chamfer` starts its own 500-step ramp |
| 3000 | `w_intra_chamfer` at full weight |
| 4000 | `latent_end`. geo ramps 0.15 -> 1.0 (to 10000); latent weights decay (`w_z_raw` 30->10, `w_z_residual` 80->40); `decoder_refine_alpha` 0 -> 1 |
| 5000 | gen branch starts, 1500-step ramp; `gen_detach_latent=True` |
| 10000 / 12000 / 15000 | geo full / encoder residual unfreezes / codec lr x0.4 |

### Active losses (12, down from 19)

Chosen by measuring each term's gradient on three parameter groups and the
pairwise cosine between terms (`tools/audit_losses.py`). 19 terms collapsed into
4 independent directions, clusters with pairwise cosine 0.75-1.00.

| codec | weight (pretrain) | represents |
|---|---|---|
| `w_z_residual` | 40 (80) | the 7-term ordered cluster, 51.8% of the within-group gradient |
| `w_z_intra_chamfer` | 25 | permutation-invariant, on the **packed** reconstruction so it needs no decoder and covers the latent phase |
| `w_intra_chamfer` | 14 | same on the decoder output; 99.3% of the decoder's gradient |
| `w_z_raw` | 10 (30) | the 4-term latent/mask cluster, pairwise cosine 1.00 |
| `w_chamfer` | 20 | the 5-term global cluster, 0.3% of the decoder gradient |
| `w_latent_std` | 2 | per-channel std band [0.5, 3.0] |
| `w_latent_decorr` | 0.1 | off-diagonal of the shape-channel correlation matrix |

gen: `gen_chamfer` 8, `gen_intra_chamfer` 10, `gen_xyz_residual` 24,
`gen_presence` 0.2, `distill` 30. Plus `equiv` 1.0 every 4 steps.

**Removed with evidence:** `group_cov` (moment matching is satisfiable by noise:
precision 0.238 -> 0.439, coverage 0.540 -> 0.345, symmetric chamfer unchanged);
`direct_ratio`; `slot_sort=pca` (an axis-monotone order makes the leading
principal component a straight line, `sqrt(lam2/lam1)` 0.036 against 0.544).

**Never gate on ordered metrics.** `rmse` is dominated by the ~1% of groups with
the largest extent, `rel_offset` is slot-ordered, and both move when the decoder
merely permutes its output: at one point `rel_offset` went 0.758 -> 1.011 and
`rmse` 0.01571 -> 0.01453 while the point *set* was unchanged (0.393 -> 0.396).

---

## 3. Limits

### 3.1 Information-theoretic (hard under the current constraints)

**Half a channel per point** -- but that is *not* an information-theoretic wall.
Corrected arithmetic (an earlier version of this file said "1 bit per point",
which was the figure for a 16,384 latent and was carried over by mistake):

```
fp16 capacity : 131072 * 16 / 262144 = 8.0 bits/point
requirement, ordered 1:1              = 10.2 bits/point
requirement, permutation-invariant    =  5.5 bits/point   (log2(64!)/64 = 4.6 saved)
```

8.0 > 5.5, so the budget is sufficient in principle for point-spacing accuracy on
an unordered set. The 0.218 / 1.14x figure is an **empirical** bound for one
decomposition -- fixed template plus a rank-22 *linear* correction on a
template-aligned target -- not a limit of the representation. The gap between
0.242 and 0.191 is architectural, and the effective capacity actually in use is
far below 8 bits/point: the shape block's measured effective rank is 11-14 of 28,
and 42% of cells are empty.

**Radius-normalised PCA over 19131 GT groups** -- the within-group arrangement is
close to incompressible:

| rank k | residual (group radii) |
|---|---|
| 5 | 0.849 |
| 12 | 0.725 |
| 28 (the budget) | 0.593 |
| 64 | 0.454 |

lam1/lam28 is only 16.9: the spectrum is nearly flat, so there is no small set of
templates to learn. This is why the permutation-invariant framing matters -- an
unordered 64-point set costs log2(64!) ~ 296 bits less than the ordered list.

**Attributes do not compress the way xyz does.** Measured over 5 scenes, ratio of
within-group std to global std:

| block | ch | per-channel std | within/global | PCA 95% dims |
|---|---|---|---|---|
| xyz | 3 | 0.213 | **0.311** | 2.0 |
| log_scale | 3 | 1.509 | 0.830 | 2.8 |
| rot | 4 | 0.182 | 0.928 | 3.8 |
| opacity | 1 | 2.914 | 0.903 | 1.0 |
| color | 3 | 1.282 | 0.934 | 1.8 |
| SH | 45 | 0.038 | 0.933 | 18.6 |

Only xyz is spatially smooth. Everything else is essentially per-point
independent, so the group -> centroid + shape idea that the whole architecture is
built on does **not** transfer. Storing opacity + color alone needs 1,048,576
numbers -- **8x the entire current latent**.

Also note `patch_dim = g * 4` is hardcoded to xyz + mask, so `--stage geometry`
or `full` today would train the decoder to emit 59 channels from a latent that
carries only xyz.

### 3.2 Wasted latent capacity (fixable, partly fixed)

| | before | now |
|---|---|---|
| shape-channel effective rank | 5.4 / 28 (19%) | **11-14 / 28 (~45%)** |
| cell occupancy | 42% of 4096 cells empty | unchanged |

Effective utilisation is roughly 0.58 x 0.45 = **26%** of `z_compact`, up from
11%. Two known levers remain:

* **Rank.** The input pack now carries rank 20.48, but `token_merge` passed 16.23
  at init and training drove it to **5.84**; `group_head` recovers to 18.64 only
  because it also receives `cell_vec` and the analytic anchors. The point-data
  path dies and the anchors carry the latent. The current run replaces the single
  `Linear(3584 -> 448)` with a staged MLP to test this; **unverified**.
  Raising `w_latent_decorr` 0.1 -> 0.3 was measured neutral (rank 11.8 -> 14.0
  moved chamfer 0.245 -> 0.243) and harmful at the transition (0.274 -> 0.309):
  decorrelating channels is not the same as making them informative.
* **Occupancy.** Scenes hold 79k-262k points against `max_points` 262144. The
  absolute-chamfer oracle for using all 4096 cells is -17 to -20%, but a
  2500-step A/B flatlined within-group learning. Untested retry: duplicate-pad
  each group to 64 slots so the occupancy mask stays constant.

### 3.3 Sparse groups: fixed as far as the representation allows

192 groups (12,288 points, 4.7% of `max_points`) hold 46% of the absolute squared
error. But this is no longer a learning failure -- the model now **beats** the
zero-training ellipsoid oracle everywhere, by the largest margin exactly there:

| radius bin | before (rank 5) | now (rank 14) | model / oracle |
|---|---|---|---|
| p0-50 (dense) | 1.68x | **1.36x** | 1.15 -> 0.93 |
| p50-95 | 2.24x | **1.50x** | 1.12 -> 0.75 |
| p95-99 | 2.97x | **1.63x** | 1.11 -> 0.61 |
| p99-100 | 3.47x | **1.76x** | 1.13 -> **0.57** |

Cause of the residual gap: multi-modality. Spacing over radius falls 0.215
(dense) to 0.103 (sparsest), so a single ellipsoid puts points in the voids
between clusters. Three partition schemes were measured and the top-1% share
stays 71-75% in all of them, so re-partitioning is not the lever; latent rank is.

### 3.4 The generative branch does not work yet

gen is the path a world model consumes, and it gets **worse with training** in
every configuration measured:

| | untrained | trained |
|---|---|---|
| `_fix` (pre-folding) | 0.638 | 0.892 |
| rankfix | 0.509 | 0.747 |
| gendetach_off (`--no_gen_detach`) | 0.509 | 0.798 |
| genfix (free paths gated) | 0.509 | 0.792 |

against **0.524** for putting every point at its group centroid. Its *global*
chamfer is fine (0.00216 vs the codec's 0.00180) -- the failure is entirely
within-group, and the mode is **inflation**: radius 1.83x GT, p->g 1.294 against
g->p 0.521, thread 3.377 against 1.551. That asymmetry is the signature of
over-coverage, the same way `group_cov` failed earlier.

Two hypotheses tested and refuted:

* **`gen_detach_latent`** -- turning it off collapsed `hf_ratio` 0.272 -> 0.030
  (confirming the detach's original justification) and gen still degraded.
* **ungated free paths** (`expand_xyz` at 0.35 x extent, `xyz_residual` at
  0.6 x extent) -- gating both moved gen 0.805 -> 0.792.

Remaining hypothesis, now under test: the loss balance. The codec's group shape
is pinned by `z_residual` (40-80) acting on the exact, template-aligned packed
target; gen has no packed intermediate, and until this run its teacher signal was
`w_distill` **3** against 32 of symmetric chamfer -- 5% of its supervision, in a
design whose stated intent is that gen is built *from* what the codec produced.
This run sets `w_distill` 30 and `w_gen_chamfer` 8, and moves `gen_start`
3000 -> 5000 so the teacher is past its worst point (`ich` 0.320 at step 3000).

### 3.4b Coverage: 46% of the point budget is spent twice

Measured once the rasteriser existed (see `PLAN.md` Milestone G for the full
tables). Matching each predicted point to its nearest GT point:

```
distinct GT points hit / predictions = 0.540   model
                                       0.991   the selected GT subset (control)
```

so nearly half the predictions land where another already is, and the matching
surface goes uncovered. Snapping every prediction onto its nearest GT point --
which removes all positional error -- moves the render only 17.2 -> 19.2 dB, so
this is the dominant term, not fine accuracy.

Against isotropic noise on the GT points, the model's error is **more clustered
than random noise twice its size**: at `intra_chamfer` 0.246 the noise control
gives `nn_unique` 0.61, the model gives 0.475, and even 0.45-radius noise
(chamfer 0.383) gives 0.538. Points are on the right surface, arranged wrongly.
`template_erank_pred` 7.2 against 22.5 in the GT says the same thing from the
other side: ~7 distinct group shapes.

Cause: the decoder output carries **no one-to-one supervision**. `w_xyz`,
`w_xyz_mse`, `w_xyz_hard` and `w_xyz_residual` are all 0; only `w_chamfer` and
`w_intra_chamfer` remain, both permutation-invariant and both blind to
duplication by construction. `w_z_residual` (40) is 1:1 but acts on the *packed*
intermediate -- everything after the unpack is unconstrained, and
`rel_offset_p50` sits at 0.499, half the group radius. Present already in the
coarse output, so not the refine stack.

### 3.5 Never measured

* **Render metrics need converged frames.** `eval_val_indices` includes three
  frames from before 3DGS step 530 (opacity max 0.140 on one of them, SH all
  zero); their *originals* render as fog. Point-space metrics are unaffected
  (0.6-2.1%, inside the scene spread, because they are radius-normalised and
  ignore opacity/SH) but PSNR/SSIM on them is meaningless -- `codec_snap` comes
  out *below* `codec` there, the opposite sign from every converged frame.
* **World-model latents.** Validation covers "encode a held-out npz, decode it"
  (train 2700 / val 300, zero overlap, `crop_prob=0`, no augmentation; 24 val
  scenes never used in eval score **0.286** against 0.270 for the 8 eval scenes,
  i.e. +5.9%, inside the scene-to-scene spread). It does **not** cover decoding a
  `z_compact` produced by a diffusion model, which is the actual deployment path.
* **Attributes.** Only `--stage xyz` has ever been run at scale. The heads, the
  geometry-first conditioning curriculum and the render loss are now wired and
  smoke-tested end to end, but no full run has used them.
* **Rendering (no longer unmeasured).** `can3tok/render.py` +
  `tools/render_compare.py`. Best checkpoint on 7 held-out frames: original ->
  canonical 30.3 dB / SSIM 0.961, original -> reconstruction 17.2 dB / SSIM
  0.518. The reconstruction is recognisable -- layout, structure and colour
  survive; detail and dark regions collapse. See 3.4b for what that revealed.

### 3.6 Scale relative to the literature

Feed-forward autoencoders over Gaussians that feed a latent diffusion model
(GaussianCube, TRELLIS/SLat, L3DG, DiffGS) work at roughly **32k Gaussians**, set
by their voxel grid. This design carries **262,144** -- 8x more -- into a latent
of the same order. That ratio is where the 0.5 channels/point comes from, and
`pruning_scores` is already present in every npz (min 0, max 351,976) if the
count is ever allowed to drop.

---

## Priorities

| | action | target |
|---|---|---|
| 1 | gen at step 6500 of the live run: `< 0.51` means the loss balance was the cause | 3.4 |
| 2 | `scripts/launch_coverage_262k_ddp.sh` -- render loss + 1:1 decoder-output term. Gate: `nn_unique` > 0.60 @4000, > 0.70 @8000 with `intra_chamfer` < 0.25 | 3.4b |
| 3 | check `token_merge` rank at step 4000 (>10 = the staged MLP held; <5.84 = revert) | 3.2 |
| 4 | `--stage full` with the attribute curriculum + render loss, once `nn_unique` clears its gate | 3.5 |
| 5 | decide the point budget by render ablation: 32k / 64k / 128k / 262k. `pruning_scores` is unusable (stored wrong), so importance must be learned | 3.1, 3.6 |
| 6 | `patch_dim = g * (target_dim + 1)` and a re-split channel budget for `--stage geometry` | 3.1 |
| 7 | frame-to-frame latent permutation stability -- Morton order is recomputed per frame, so cell *i* can hold a wall at *t* and a chair at *t+1* | 3.5 |
