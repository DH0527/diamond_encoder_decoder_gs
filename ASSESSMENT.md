# Assessment of the redesign proposal

Base: `can3tok_encoder_decoer_fix_2` at run `v4_262k_g64_20260808_083806`.
Everything marked **measured** is a number from that codebase, not an assertion.

---

## 1. The proposal is right about the framing

The single most important claim -- that this is not "the decoder is bad, fix the
decoder" but a whole-system problem -- is supported by what was actually measured:

* the compressor's **input** carried effective rank 1.01 of 256 until the packing
  was renormalised (fixed: 20.48)
* `token_merge` passed 16.23 at init and training drove it to **5.84**, while
  `group_head` climbed back to 18.64 only because it also receives `cell_vec` and
  the analytic anchors
* the gen branch gets **worse with training** in every configuration tried
* the latent carries **xyz only** (`patch_dim = g * 4` is hardcoded), so no
  attribute path exists at all

Four independent failure sites. Decoder-only work could not have fixed any of
them.

---

## 2. The proposal caught a real error in ARCHITECTURE.md

> "131,072 numbers for 262,144 points. At fp16 that is 1 bit per point."

Wrong. Verified:

```
fp16 : 131072 * 16 / 262144 = 8.0 bits/point
requirement, ordered 1:1     = 10.2 bits/point
requirement, permutation-inv =  5.5 bits/point   (log2(64!)/64 = 4.6 saved)
```

**8.0 > 5.5**, so the budget is sufficient in principle for point-spacing
accuracy on an unordered set. The 1-bit figure was computed for the 16,384 latent
discussed separately and was wrongly carried over.

Consequence: **"1.14x spacing is an information-theoretic hard ceiling" is
withdrawn.** 0.218 is an empirical bound for one specific decomposition (fixed
template + rank-22 linear correction on a template-aligned target), not a limit
of the representation. The gap is architectural and therefore attackable.

This correction is why the redesign is worth doing rather than accepting the
current numbers as near-optimal.

---

## 3. Point-by-point

### Accepted, already supported by measurement

| # | claim | evidence |
|---|---|---|
| 5 | point information collapses in the encoder | 20.48 -> 5.84 at `token_merge` |
| 6 | high effective rank != useful information | `w_latent_decorr` 0.1 -> 0.3 moved rank 11.8 -> 14.0 and chamfer 0.245 -> 0.243 (nothing), and hurt at the transition (0.274 -> 0.309) |
| 7 | context must be residual, not a shortcut | `group_head` recovers rank only via `cell_vec` + anchors, i.e. the point path is being bypassed |
| 9 | gen fails by inflation, not collapse | radius 1.83x GT, p->g 1.294 vs g->p 0.521, thread 3.377 vs 1.551 |
| 10 | teacher and student see the **same** `z_compact` | confirmed in `model.py`: both decode paths take only `z_compact`; the teacher merely has a larger intermediate (1,048,576) and a bigger decoder |
| 25 | attributes cannot ride the xyz grouping | within/global std: xyz **0.311**, log_scale 0.830, rot 0.928, opacity 0.903, color 0.934, SH 0.933 |
| 27 | keep template-aligned slots | mean residual 1.010 -> 0.408, reachable chamfer 0.261 -> 0.218 |
| 33 | keep the folding prior | zero-training envelope 0.294 beats the trained free MLP 0.345 |
| 36 | separate evaluation metric from training metric | `rel_offset` 0.758 -> 1.011 and `rmse` 0.01571 -> 0.01453 while the point set was unchanged (0.393 -> 0.396) |
| 43 | no rasteriser exists | confirmed; every number in this repo is point-space |
| 55 | world-model latents never tested | validation is encode-then-decode on held-out npz only |

### Accepted on the user's word, not measurable here

| # | claim | note |
|---|---|---|
| 17 | `pruning_scores` is wrongly stored and must not be used | the field exists (min 0, max 351,976) but the user states the values are invalid. Removed from every plan; an importance signal must be learned instead |
| 16, 18-21 | arbitrary N > 262144 needs a render-aware canonicaliser, not truncation | correct in principle. Requires a rasteriser, so it cannot start before Milestone G |

### Accepted as sound engineering, untested

| # | claim |
|---|---|
| 14 | distil from the **bottleneck** teacher, not the rich teacher -- the rich teacher sees information the student cannot have |
| 22, 50, 72 | canonicalisation and the AE must be trained separately, and phases must not be stacked |
| 45 | teacher forcing -> scheduled sampling -> inference conditioning for the attribute decoder's xyz input |
| 48 | distil intermediate features, not only the final points |
| 69 | explicit learned EMPTY token + attention mask for unused cells |
| 71 | rebalance parameters toward the encoder rather than growing the student decoder |

### Excellent point that was **not** in my analysis

**#70 -- latent permutation instability across frames.** Morton ordering is
recomputed per frame from the point set, so cell *i* can hold a wall at *t* and a
chair at *t+1*. Harmless for an autoencoder, potentially fatal for a world model
that must learn temporal transitions in this latent. Nothing in this repo tests
it. This is now a first-class risk in the plan.

---

## 4. Where the proposal needs correction or is incomplete

### 4.1 "Student should share the teacher's geometry basis" -- they already do

Both `StagedDecompressor` and `GenerativeDecoder` run the same folding path with
the same fixed `unit_ball` template, the same `R(axis-angle) diag(softplus)`
frame, and the same tanh-capped residual. The basis is already shared.

The actual asymmetries, from the code:

| | codec | gen |
|---|---|---|
| frame/residual head input | `cat([shape, contextualised group token])` | **raw `shape` channels only** |
| extra additive paths | `deep_xyz` (capped + ramped) | `expand_xyz` at 0.35 x extent, `xyz_residual` at 0.6 x extent |
| within-group anchor | `z_residual` 40-80 on the **exact template-aligned packed target** | none |

The third row is the substantive one. The codec's group shape is pinned by an
exact ordered target on the packed intermediate; gen has no packed intermediate,
so its strongest within-group signal was `w_gen_xyz_residual` 24 against 32 of
symmetric chamfer -- and symmetric chamfer is satisfiable by spreading, which is
exactly the inflation measured.

So the fix is more specific than "share the basis": **give the student the same
contextualised input, delete its redundant free paths, and give it an anchor of
comparable strength to `z_residual`.** Gating the free paths alone was already
tried and moved gen 0.805 -> 0.792, i.e. nearly nothing -- consistent with the
anchor, not the paths, being the cause.

### 4.2 The bottleneck-teacher proposal partly already exists

`z_compact -> decompressor -> z_raw_hat` **is** the H_rich_hat cycle, and
`z_l1` / `z_residual` / `z_z_intra_chamfer` **are** the reconstruction loss on it.
What is missing is not the structure but:

* `H_rich` is not a learned representation -- it is the deterministic identity
  pack. There is no `G -> H_rich` encoder to speak of (encoder = 0.27M, frozen).
* there is no loss of the form `D(D_T(H_hat), D_T(H))`, i.e. nothing checks that
  the compact latent preserves what the teacher *decoder* needs, as opposed to
  what the pack contains.

Renaming the current pack to `H_rich` would be misleading. Building a real
learned rich representation is a genuine change and belongs in the plan, but it
should be stated as new work, not as a relabel.

### 4.3 The attribute budget is not addressed

The proposal routes attributes through a separate encoder and a geometry-first
decoder, which is right. But the arithmetic still has to close:

| | numbers needed | vs `z_compact` (131,072) |
|---|---|---|
| opacity + color (4 ch/point) | 1,048,576 | **8x over** |
| + log_scale + rot (11 ch) | 2,883,584 | 22x over |
| + SH (56 ch) | 14,680,064 | 112x over |

With within/global at 0.90-0.93 these are close to per-point independent, so
grouping does not reduce them. A 12-channel attribute slice of the latent cannot
carry them by reconstruction alone. Two ways out, and the plan must pick one:

* **Scaffold-GS style**: do not store attributes. Decode them from the group code
  **and the view direction**, supervised by rendering. Attribute latent cost goes
  to zero; the target becomes render equivalence, not parameter reconstruction.
* **Reduce the count**: at ~32k Gaussians the 11 non-SH channels fit. This is the
  regime published Gaussian autoencoders for latent diffusion actually operate in
  (GaussianCube, TRELLIS/SLat, L3DG all sit near a 32^3 grid).

The proposal's #41 ("parameter loss + render loss") implicitly assumes the first
without saying the storage problem is what forces it.

### 4.4 Scope

The proposal contains ~8 milestones and its own advice (#50, #72) is not to stack
them. Building all of it at once would repeat the mistake this session already
made seven times: bundling changes and losing attribution. This folder therefore
implements **Milestone C only** (the student fix), keeps everything else as a
written plan with pre-committed gates, and touches nothing that Milestone C does
not need.

---

## 4.5 What the rasteriser changed (added after Milestone G ran)

Two of the judgements above have to be revised now that renders exist.

**#43 "no rasteriser exists" was accepted as a gap. It was the *main* gap.**
Building it took one file and immediately produced the largest single finding of
the project: 46% of predicted points duplicate another's nearest GT point
(`nn_unique` 0.540 against 0.991 for the GT subset), which no point-space metric
in this repo could see. Every loss on the decoder output is permutation-invariant
and a symmetric chamfer is far too weakly sensitive to it: the duplicate itself
scores a perfect p->g, and the uncovered GT point's cost enters only through the
g->p half, averaged over the whole group. See `PLAN.md` Milestone G.

**The proposal's ordering was wrong, and so was mine.** Both put rendering late
(its letter G, its own section 43 of 76). It should have been first: it is cheap,
it needs no retraining, and it reclassified the problem from "detail accuracy" to
"coverage". The rule this supports is the proposal's own #36 -- separate the
evaluation metric from the training metric -- taken further than either of us
took it: the evaluation metric has to be in the space the system is *for*.

**#25 (attributes cannot ride the xyz grouping) survives, but its consequence
changes.** The measured within/global std ratios still say attributes are close
to per-point independent, so they cannot be *stored*. The render loss makes them
decodable instead, because view equivalence is a far weaker requirement than
parameter reconstruction. That is what makes #41's "parameter loss + render loss"
work; the proposal asserted the pairing without saying the storage arithmetic is
what forces it.

---

## 5. What this folder does

| | |
|---|---|
| base | full `fix_2` tree, all verified work intact |
| changed | student decoder (Milestone C); rasteriser + render loss (G); attribute conditioning curriculum (E/F) |
| added | `can3tok/render.py`, `tools/render_compare.py`, `tools/diag_duplication.py`, `nn_unique` in eval, camera in the dataset |
| corrected | the bits/point error in `ARCHITECTURE.md` |
| unchanged | canonicaliser, learned `H_rich` encoder, hierarchical cell attention, EMPTY token, Gaussian-budget ablation -- all planned, none started |

The camera has to move with the augmentation or the render loss is wrong on 100%
of training samples. Augmentation maps `x -> s R x + t_w`; the matching camera is
`R_wc' = R_wc R^T`, `T' = s T - R_wc R^T t_w`, which scales camera-space
coordinates uniformly by `s` -- and perspective projection is invariant to that,
while the Gaussians' own scales were multiplied by the same `s`. Verified: the
augmented cloud rendered through the transformed camera matches the original
render at 98.2 dB.

The gate is unchanged and pre-committed: **gen's within-group chamfer must fall
below its own untrained value of 0.509.** Every configuration measured so far
goes the other way (0.747, 0.792, 0.798, 0.892). If this one does too, the
teacher/student split itself is the problem and the plan's Milestone C fails,
not its Milestone D-H.
