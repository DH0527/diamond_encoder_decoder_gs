# Plan

Pre-committed gates. A milestone that fails its gate does not advance; the next
milestone is only allowed to start once the previous one's gate is met, because
seven restarts in the previous session showed that stacking changes destroys
attribution.

Metric throughout: **permutation-invariant within-group chamfer normalised by the
group's mean radius**. GT's own nearest-neighbour spacing is 0.191, so 0.191 is
1.0x.

---

## Milestone C -- fix the student (IMPLEMENTED, not yet run)

The blocker. gen is the path a world model consumes and it gets **worse with
training** in every configuration measured:

| configuration | untrained | trained |
|---|---|---|
| `_fix` (pre-folding) | 0.638 | 0.892 |
| rankfix | 0.509 | 0.747 |
| `--no_gen_detach` | 0.509 | 0.798 |
| free paths gated | 0.509 | 0.792 |

against **0.524** for putting every point at its group centroid. The failure mode
is inflation, not collapse: radius 1.83x GT, precision (p->g) 1.294 against
coverage (g->p) 0.521, thread 3.377 against 1.551. Its *global* chamfer is fine
(0.00216 vs the codec's 0.00180), so only the within-group arrangement is wrong.

Three hypotheses have already been tested and refuted: `gen_detach_latent`
(turning it off collapsed `hf_ratio` 0.272 -> 0.030 and gen still degraded),
ungated free paths (0.805 -> 0.792), and loss reweighting alone (in progress in
the previous folder).

### What changed here

**1. The student reads what the teacher reads.** `fold_head` and `shape_xyz` in
`GenerativeDecoder` took the raw 28 shape channels; the codec's take
`cat([shape, contextualised group token])`. That was the last input asymmetry
between two decoders that are otherwise built on the same basis.

**2. Basis distillation.** Both decoders build a group as
`R(a) diag(s) (fixed_template + residual)`. Only the final points were being
distilled (`w_distill`), which leaves the student free to reach a similar point
set through a different basis -- and the basis it found inflates. The teacher's
`frame` (6 params) and `local` (pre-scale offsets, whose magnitude *is* the group
radius) are now exposed on both paths and distilled directly:

```
w_gen_basis * ( |gen_frame - sg(teacher_frame)| + |gen_local - sg(teacher_local)| )
```

This is the proposal's #34 and #48 made specific: not "share the basis" (they
already did) but "supervise the basis parameters the teacher actually chose".

### Gate

| step | requirement |
|---|---|
| any | gen chamfer **< 0.509** (its own untrained value) |
| 6500 | gen **< 0.45**, gen radius/GT **< 1.2** (from 1.83), gen `nn_unique` **> 0.65** |

`nn_unique` was added to this gate after Milestone G measured it. The original two
conditions cannot see duplication: a student can reach the right radius and the
right chamfer while stacking points on the same surface patches, which is exactly
the failure the codec turned out to have.

**Reading it on the run that is already going.** `runs/mBC_262k_g64_20260808_100309`
started before the metric existed, so its own `eval/*/metrics.json` have no
`nn_unique` field — the process had already imported `eval_utils`. Restarting to
pick it up would throw away 3450 steps and the clean attribution this milestone
exists for, so the gate is read **offline from the checkpoints** instead:

```
tools/diag_duplication.py --ckpt runs/<run>/ckpt_step00006000.pt   # codec + gen + control
tools/render_compare.py   --ckpt runs/<run>/ckpt_step00006000.pt   # + render PSNR
```

Both compute the same quantity from the same weights, so the number is comparable
to any later run that logs it inline. Checkpoints land every 2000 steps, so the
6500 gate is evaluated at 6000.

If gen still rises above its untrained value, the teacher/student split itself is
the problem and Milestone C has failed -- not D-H. In that case the next test is
a single decoder (drop the codec refine stack, keep decompressor + `z_residual`)
rather than more student tuning.

---

## Milestone B -- make the bottleneck teacher explicit

Partly exists already: `z_compact -> decompressor -> z_raw_hat` **is** the
H_rich_hat cycle and `z_l1`/`z_residual`/`z_z_intra_chamfer` are its
reconstruction loss. What is missing:

* `H_rich` is not learned -- it is the deterministic identity pack, and the
  "encoder" is 0.27M and frozen. A real `G -> H_rich` encoder is new work.
* nothing checks `D(D_T(H_hat), D_T(H))`, i.e. that the compact latent preserves
  what the teacher *decoder* needs rather than what the pack happens to contain.

Add the second loss first -- it is cheap and directly diagnostic. Only build a
learned `H_rich` if that loss shows the pack is the limiting factor.

**Gate:** codec chamfer must not regress past 0.25 while the cycle loss falls.

---

## Milestone A' -- encoder information path

Measured: the input pack carries effective rank 20.48, `token_merge` passed 16.23
at init and training drove it to **5.84**, and `group_head` recovers to 18.64
only because it also receives `cell_vec` and the analytic anchors. The point-data
path dies and the anchors carry the latent.

Two changes, tested one at a time:

1. `token_merge` staged MLP instead of one `Linear(3584 -> 448)` -- **already
   running in the previous folder as `v4`**; its step-4000 `token_merge` rank
   decides this (>10 held, <5.84 revert).
2. context must be **residual**, not a shortcut: `h = h_local + alpha * h_context`
   with alpha ramped, and the anchors entering *after* the point path rather than
   beside it.

**Gate:** `token_merge` rank after training > 10 **and** codec chamfer <= 0.242.
Rank alone is not sufficient -- raising `w_latent_decorr` 0.1 -> 0.3 moved rank
11.8 -> 14.0 and chamfer 0.245 -> 0.243, i.e. nothing.

---

## Ordering (revised after Milestone G)

Milestone letters no longer describe the order. The measured sequence is:

1. **Milestone C** (running) -- finish it, but read `gen_nn_unique` alongside its
   original gate.
2. **Geometry coverage** (`launch_coverage_262k_ddp.sh`) -- the goal is `nn_unique`,
   not chamfer. A model at chamfer 0.26 / `nn_unique` 0.90 is worth more than one
   at 0.24 / 0.50, because attributes attach per slot.
3. **Render-aware canonicaliser** (Milestone D) -- **before** `--stage full`, not
   after. The current `canon` is a sampler baseline that already costs 25-30 dB;
   training an attribute decoder against it would make a damaged representation
   the ground truth.
4. **Full Gaussian attributes** (Milestone E/F) at 262k -- entry condition
   `nn_unique` > 0.85.
5. Full-SH renderer, then DIAMOND-produced latents (Milestone H).

### Why not attributes now

At `nn_unique` 0.5, slots A and B carry GT A's and GT B's attributes while their
xyz sit on the same spot. The attribute decoder would learn a correspondence that
does not exist. Geometry coverage first.

---

## Milestone D -- canonicaliser

Arbitrary N -> K <= 262144 with render equivalence. **Blocked on a rasteriser**,
so it cannot start before Milestone G. `pruning_scores` is present in every npz
but the stored values are invalid, so importance must be learned:

```
gate m_i = sigmoid(a_i),  alpha'_i = m_i * alpha_i
L = L_render(R(G_gated), R(G_orig)) + lam_c * count + lam_b * binarisation
then hard top-K, then fixed-count fine-tune of all parameters
```

Must be a **separate offline stage** that writes canonical npz files. Training it
jointly with the AE makes failure unattributable.

The render term must be **multi-view** here, not single-view. "Equivalent under K
Gaussians" means nothing if it only holds from the one pose the frame was captured
at: a gate that removes a Gaussian hidden behind another from that view costs
nothing, and reappears as a hole from any other. `render.perturb_camera_vector` +
`render_loss(views=M)` already provide this; the canonicaliser should use a larger
`M` and a wider orbit than the autoencoder does, because it is deciding what to
*delete*.

It also gets a direct baseline to beat, now that `canon` is provably an exact GT
subset (`canon_src_unique` 1.000, `canon_xyz_err` 0.0): **sampler top-K at 27.8 dB
mean over held-out frames** (25.1 dB on a 338k-point scene, 30.5 dB on a
166k-point one). Any canonicaliser that does not clear that is not worth its
complexity.

Also decide K by ablation (32k / 64k / 128k / 262k) rather than assuming 262144.
Published Gaussian autoencoders that feed latent diffusion (GaussianCube,
TRELLIS/SLat, L3DG) operate near 32k, set by their voxel grid.

---

## Milestone E/F -- attributes (IMPLEMENTED, not yet run at scale)

The decoder already had attribute heads (`attr_mlp` + separate scale/rot/opacity/
colour/SH heads) and an `attr_xyz` conditioning hook, but nothing ever drove the
hook and `--stage full` would have trained the heads on their own wrong
positions. Now wired:

* **Geometry-first conditioning.** `p(X, A | Z) = p(X | Z) p(A | X, Z)` with the
  attribute heads reading a position, blended per point inside the decoder
  between the true xyz and the decoder's own -- teacher forcing
  (`attr_force_steps`), then scheduled sampling (`attr_anneal_steps`), then the
  inference condition. One forward, not two. Eval always uses the inference
  condition, so the reported number is never teacher-forced.
* **Render loss** so attributes are supervised by appearance rather than by
  parameter reconstruction, which is what makes the budget close: storing them
  costs 8-112x the compact latent, matching their appearance costs nothing.

The storage arithmetic below is unchanged and still decides `K`; the render loss
is what makes the first option (decode, don't store) testable.

### Open questions to settle when this stage actually runs

* **SH curriculum.** The renderer is `sh_degree=0` today, which is right for
  geometry diagnosis and wrong for the final model: a degree-0 render loss cannot
  check view-dependent appearance at all. Sequence: geometry at degree 0 ->
  appearance at DC + low SH -> full SH. Raising the degree changes what the loss
  can even see, so it is a curriculum step, not a config change.
* **Scheduled-sampling granularity.** Mixing GT and predicted xyz **per point** is
  correct while the attribute heads are per-point MLPs. If they later attend
  within a group, a group containing an arbitrary GT/predicted interleaving is a
  distribution that never occurs at inference, and group-wise or sample-wise
  mixing should be A/B'd against it. Not a problem for the current heads.
* **Attribute compensation control.** Run the render loss twice, once with
  predicted attributes and once with attributes frozen at GT. The difference
  separates a geometry gain from opacity/scale absorbing geometric error. In the
  xyz stage this is already blocked (both sides get the target's attributes,
  detached), so the control only becomes necessary here.
* **`w_gen_p2g` / `w_gen_radius` are failure-specific regularisers**, aimed at one
  measured pathology (radius 1.83x, p->g 1.294 vs g->p 0.521). Once the student is
  stable they must be ablated back down. Left at full strength they would make the
  method look hand-tuned to its own metric.

`patch_dim = g * 4` is hardcoded to xyz + mask, so the latent carries no
attribute information today; `--stage full` would train the decoder to invent 59
channels. Extending to `g * (target_dim + 1)` is mechanical. The budget is not:

| stored per point | numbers | vs z_compact |
|---|---|---|
| opacity + color (4) | 1,048,576 | **8x over** |
| + log_scale + rot (11) | 2,883,584 | 22x over |
| + SH (56) | 14,680,064 | 112x over |

and measured within-group/global std is 0.83-0.93 for every attribute against
**0.311** for xyz, so grouping does not reduce them. Two ways out, and one must
be chosen before writing code:

* **decode attributes from the group code + view direction**, supervised by
  rendering (Scaffold-GS). Attribute latent cost -> 0; the target becomes render
  equivalence rather than parameter reconstruction.
* **reduce K** until the 11 non-SH channels fit (~32k).

Geometry-first factorisation `p(X, A | Z) = p(X | Z) p(A | X, Z)` is right either
way, as is the teacher-forcing -> scheduled-sampling -> inference curriculum on
the xyz that conditions the attribute decoder, and the ablation
`latent-only / GT-xyz / latent+GT-xyz / latent+pred-xyz`.

---

## Milestone G -- rasteriser (DONE, and it changed the diagnosis)

`can3tok/render.py` wraps `diff_gaussian_rasterization`; `tools/render_compare.py`
loads a checkpoint and renders a held-out scene four ways. The npz carries full
per-frame intrinsics and extrinsics (`state_t['camera']`: fx, fy, cx, cy, R, T →
977x544, 80.2 deg horizontal FoV) but no photograph, so the reference is a render
of the original Gaussians. Two conventions had to be got right: `color` is the DC
spherical-harmonic coefficient, not RGB (`rgb = 0.2821 * dc + 0.5`; feeding it
raw renders flat grey fog), and `opacity`/`scaling` in the npz are already
activated.

Best checkpoint (v4 @2000, `intra_chamfer` 0.249), 7 held-out frames:

| | PSNR | SSIM |
|---|---|---|
| original -> canonical (262k selection) | 30.3 | 0.961 |
| original -> reconstruction | 17.2 | 0.518 |
| original -> reconstruction, snapped to nearest GT | 19.2 | |

The reconstruction is **recognisable** -- scene layout, structure and colour all
survive; detail and dark low-texture regions collapse. That is the first visual
evidence the model works at all.

### The number this milestone actually produced

Snapping every predicted point onto its nearest GT point removes *all* positional
error and buys only 2 dB. So the error is not "points are slightly off". Matching
each prediction to its nearest GT point:

```
distinct GT points hit / predictions = 0.540   model
                                       0.991   the selected GT subset (control)
```

**46% of the 262k budget lands where another predicted point already is**, and the
corresponding surface goes uncovered. A symmetric chamfer scores a duplicate as
perfect, which is exactly why 1.23x point spacing and 17.9 dB coexist. Now
tracked every eval as `nn_unique`.

Calibration against isotropic noise on the GT points (same scene):

| sigma / radius | intra_chamfer | nn_unique |
|---|---|---|
| 0.05 | 0.070 | 0.828 |
| 0.20 | 0.208 | 0.636 |
| 0.30 | 0.280 | 0.587 |
| 0.45 | 0.383 | 0.538 |
| **model** | **0.246** | **0.475** |

The model's error is *more clustered than random noise twice its size*: it puts
points on the right surface (chamfer far better than the control) but arranges
them wrongly. Consistent with `template_erank_pred` 7.2 against 22.5 in the GT --
the shape->offset map produces ~7 distinct group shapes. Across 45 evals,
`rel_offset_p50` correlates with `intra_chamfer` at r=+0.73 and
`template_erank_pred` at r=-0.61, so both point the same way.

The duplication is already present in the **coarse** output (refine gated off
gives the identical 0.499/0.253), so it is not the refine stack.

### Why it happens

The decoder output has **no one-to-one supervision at all**: `w_xyz`,
`w_xyz_mse`, `w_xyz_hard` and `w_xyz_residual` are all 0, leaving only
`w_chamfer` and `w_intra_chamfer`, both permutation-invariant and both blind to
duplication. `w_z_residual` (40) does supervise 1:1 but on the *packed*
intermediate; everything after the unpack is free to collapse, and
`rel_offset_p50` sits at 0.499 -- half the group radius.

### Next run: `scripts/launch_coverage_262k_ddp.sh`

Two levers, both aimed at `nn_unique`: `--w_render` (photometric — the term
closest to the task space, and the one most directly sensitive to duplication)
and `--w_xyz_residual 12` (the extent-normalised 1:1 term on the decoder output).

**`w_render` is 0.004, not 6.** The first guess (6, by analogy with
`w_chamfer 20`) was wrong by three orders of magnitude. Measured with
`tools/audit_render_scale.py` — `|dL/dxyz|` on one perturbed 262k cloud, every
other term zeroed:

| term | weight | \|w·dL/dxyz\| | share |
|---|---|---|---|
| `w_render` | 6.00 | 1.234e+02 | **99.9%** |
| `w_chamfer` | 20.00 | 1.182e-01 | 0.1% |
| `w_intra_chamfer` | 14.00 | 3.515e-02 | 0.0% |
| `w_xyz_residual` | 12.00 | 1.353e-02 | 0.0% |

At weight 6 the render term owns the whole step and the three geometry terms are
effectively off — which is exactly the "render loss replacing geometry loss"
failure the review warned against. The ratio holds across scenes (505x, 1027x,
1044x) and is insensitive to the perturbation size (999x at 0.1 group radii,
1044x at 0.5), so it is a property of the rasteriser rather than of the error.
Solving for a ~30% share gives 0.0035.

General rule this establishes: **any term whose units differ from the rest needs
this measurement before its weight is chosen.** Weighing by analogy loses orders
of magnitude.

**Gate — two different bars, deliberately.**

*Does the lever work?* `nn_unique` > 0.60 @4000, > 0.70 @8000, with
`intra_chamfer` < 0.25 (it must not be bought with chamfer) and render PSNR
> 20 dB.

*Is geometry good enough to hang attributes on?* **`nn_unique` > 0.85.** At 0.70,
30% of the output still shares a nearest GT point with another prediction, and
attributes are per-slot: training them on that teaches the attribute decoder a
correspondence that does not exist. 0.70 clears this run; it does not clear
`--stage full`.

If `nn_unique` does not move while the render loss falls, the render term is
being satisfied by opacity/scale compensation rather than by geometry — though in
the xyz stage that path is already largely blocked, since both sides get the
target's attributes detached, so this diagnosis matters mainly once `--stage full`
runs. There the check is explicit: render with predicted attributes vs render
with GT-frozen attributes, which separates geometry gain from attribute
compensation.

If `nn_unique` still does not move with the slot term at full weight, the next
candidate is a **local one-to-one assignment** — Sinkhorn or Hungarian on the
64x64 cost matrix `C_ij = |x_i - x_j|^2` within each group, minimising
`sum_ij P_ij C_ij`. 64x64 per group is tractable where a global 262k matching is
not. Explicitly **not** a repulsion term: repulsion pushes points apart
regardless of where the GT density is, distorting the distribution being
reconstructed.

---

## Milestone H -- world-model latents

Validation today is encode-then-decode on held-out npz (train 2700 / val 300,
zero overlap; 24 val scenes never used in eval score 0.286 against 0.270 for the
8 eval scenes, i.e. +5.9%, inside the scene-to-scene spread). It does **not**
cover decoding a `z_compact` produced by a diffusion model.

A cheap lower bound is available now: inject noise into `z_compact` (the
`gen_noise_std` path already exists) and measure where reconstruction breaks.
That sets the accuracy the world model has to hit.

**Additional risk, raised by the review and not previously tracked:** Morton
order is recomputed per frame from the point set, so cell *i* can hold a wall at
*t* and a chair at *t+1*. Harmless for an autoencoder, potentially fatal for a
world model learning temporal transitions. Needs a deterministic spatial origin
and scale, and a direct measurement of latent correspondence across consecutive
frames. Untested.

---

## Corrected in this folder

`ARCHITECTURE.md` claimed "1 bit per point" for the 131,072 latent. It is
**8.0 bits/point** at fp16 (that figure was computed for a 16,384 latent and
carried over by mistake). Against ~5.5 bits/point for permutation-invariant
point-spacing accuracy, the budget is sufficient in principle, so the "hard
information-theoretic ceiling" framing is withdrawn: 0.218 is an empirical bound
for one decomposition, and the remaining gap is architectural.
