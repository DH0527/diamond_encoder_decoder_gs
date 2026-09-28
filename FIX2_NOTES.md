# Within-group detail: what was wrong and what replaced it

Constraints held throughout: `max_points` 262144, `z_compact` 32x64x64.

The metric everything below is scored on is the **permutation-invariant
within-group chamfer, normalised by the group's mean radius** (`intra_chamfer_rel`
in `metrics.json`). The GT cloud's own nearest-neighbour spacing is **0.191** in
those units, so 0.191 is "1.0x" -- indistinguishable at the resolution the data
itself carries.

| decoder | chamfer | x spacing |
|---|---|---|
| `_fix` @6000, plain `shape_xyz` MLP (best this repo ever reached) | 0.345 | 1.81 |
| first folding attempt, @2000 | 0.382 | 2.00 |
| folding envelope alone, **zero training** | 0.294 | 1.54 |
| folding + rank-22 residual on a template-aligned target | **0.218** | **1.14** |

## Why the old decoder could not get there

`shape_xyz` was an MLP mapping the 28 shape channels straight to `G*3`
coordinates under a slot-ordered L1/L2 loss. Three measurements say that is the
wrong object, not a matter of capacity or loss weights:

1. **The target is nearly incompressible.** Radius-normalised PCA over 19131 GT
   groups: rank-28 leaves 0.593 residual, rank-64 still leaves 0.454, and
   lam1/lam28 is only 16.9. The spectrum is flat -- there is no small set of
   templates to learn.
2. **So the L2 optimum is the conditional mean, and it collapses.** Probing the
   trained map with random codes spanning all 28 dimensions gives an output
   effective rank of **4.67**; an *untrained* random 28->192 linear map gives
   26.3. Predicted group radius was 0.706 of GT -- textbook regression-to-mean
   shrinkage. `template_erank` 5.1 against 59.2 in the GT is the same fact, and
   it is what "the same pattern repeating everywhere" looked like in the renders.
3. **A low-rank correction is worse than none at all.** Fixed template plus a
   rank-k oracle correction: k=0 gives 0.524, k=8 gives 0.332, and only at k=22
   (0.263) does it beat the 0.302 of the bare template. The old decoder sat at
   rank 4.67, i.e. inside the region where learning actively hurt.

## The decoder now

```
offsets_g = R(a_g) diag(s_g) (template + residual_g)
```

* `template` -- `can3tok/template.py`, a fixed Fibonacci-spiral **filled ball**
  (not a shell), shared by every group, never trained.
* `R diag(s)` -- 3 axis-angle + 3 per-axis scale from `fold_head`, zero-init.
* `residual_g` -- `shape_xyz`, last layer zero-init, so the decoder *starts* as
  the bare template and can only improve on it.

Both heads read the whole 28-channel code; splitting the channels by hand only
removes information from each head without saving any.

Measured, envelope only:

| template | envelope | chamfer |
|---|---|---|
| fibonacci **shell** x axis-aligned scale (first attempt) | 3 ch | 0.367 |
| fibonacci shell x rotated frame | 6 ch | 0.320 |
| filled **ball** x axis-aligned scale | 3 ch | 0.336 |
| filled ball x rotated frame | 6 ch | **0.301** |

Template scale is `sqrt(3)`, not the `sqrt(5)` that matches the whitened target's
per-axis std. Converged chamfer is flat in this constant (the residual absorbs
it) but the *initial* value is not -- 0.299 at sqrt(3) against 0.331 at sqrt(5) --
and starting above the previous best is the entire point of a fixed template.

## Template-aligned slot order (`--slot_sort template`)

Which GT point lands in slot *i* is our choice: the reconstruction is scored as a
set. `data.py` therefore centres each group, whitens it in the covariance
eigenframe, and solves the 64x64 assignment against the same template the decoder
uses.

| slot order | mean\|residual\| | rank-22 bound | chamfer |
|---|---|---|---|
| Morton (previous) | 1.010 | 1.176 | 0.261 |
| optimal assignment | 0.408 | 0.679 | **0.218** |

Without it the residual head spends most of its 22 usable dimensions undoing an
arbitrary permutation. That is why the first folding run sat at 0.382 -- *worse
than its own 0.367 envelope oracle*.

**The sign convention must yield a proper rotation.** eigh's basis plus a free
per-axis sign flip lands on a reflection for **50%** of groups, and the decoder's
`R(axis-angle) diag(softplus(.))` has det > 0 always -- half the groups would have
been unreproducible. `_template_slot_order` forces det = +1 by flipping the axis
whose skew is closest to zero.

Cost: 0.94 s/scene against 0.36 s/scene for plain Morton. Not a bottleneck at
6 workers per rank and ~4.6 s/it. Greedy nearest-slot was measured too (rank-22
0.751, |res| 0.600) and is both worse and no faster than the exact solver.

## Removed

* **`group_cov`** -- matching each group's 3x3 second moment is satisfiable by
  noise. At step 3000 of the run that used it: predicted radius 0.706 -> 0.965,
  precision (p->g) 0.238 -> 0.439, coverage (g->p) 0.540 -> 0.345, and the
  symmetric chamfer unchanged at 0.393 -> 0.396. It bought nothing and scrambled
  the slot order (`thread` 0.050 -> 1.038 against 0.576 in the GT). The envelope
  is now structural, so no loss has to ask for it.
* **`w_direct_ratio` / `direct_ratio_floor`** -- guarded a direct/deep split that
  folding removes.
* **`slot_sort=pca`** -- sorting along the major axis makes the target monotone,
  so the leading principal component *is* a straight line. The rank-k PCA bound
  duly improved 23.6% while every group collapsed to a streak
  (`sqrt(lam2/lam1)` 0.036 against 0.544).

## Do not gate on ordered metrics

`rmse` is dominated by the ~1% of Morton-jump groups; `rel_offset` is
slot-ordered. Both move when the decoder merely permutes its output. At step 3000
of the `group_cov` run `rel_offset` went 0.758 -> 1.011 and `rmse` went
0.01571 -> 0.01453 while the point *set* was unchanged (0.393 -> 0.396). Earlier,
`fix2`@2000 beat `_fix`@2000 on rmse (0.01571 vs 0.01922) while the renders were
visibly worse. `--score_metric intra_chamfer` is the default for that reason.

## Measured and rejected

* **Full cell utilisation** (spread the real points over all 4096 cells instead of
  leaving 30-70% empty). The absolute-chamfer oracle is genuinely better
  (-17 to -20%), but the 2500-step A/B flatlined within-group learning
  (`z_res` -7% against -25% for prefix). Likely cause, untested: a 64-slot group
  holding ~27 points turns a free occupancy mask (full-or-empty) into a
  high-entropy one, paid for out of the shape budget. Retry only as
  duplicate-padding (repeat each group's own points to fill 64 slots, `count`
  constant, mask cost zero), and gate on absolute chamfer, not `z_res`.
* **Canonical re-orderings to flatten the spectrum** -- PCA-aligned + local Morton
  moves rank-28 from 0.593 to 0.555 (-6%). Ordering cannot fix a flat spectrum;
  it is only useful for *aligning the target to the decoder's prior*, which is
  what `--slot_sort template` does.
* **Cross-group context** (Scaffold-GS / HAC style). A plane fitted from a group's
  4/8/16 nearest neighbour groups explains it at 0.516/0.550/0.624 radii, against
  0.228 for its own best-fit plane. This cloud is not a smooth surface
  (`sqrt(lam3/lam1)` 0.233, thickness 0.228 radii > the 0.191 point spacing), so
  surface-prior methods have little to exploit here.
* **Channel-wise bit sweep** concluded "shape carries no information" because
  2-bit shape quantisation cost only 2.6% chamfer. It was run on the collapsed
  (erank 4.67) checkpoint, so it measured the pathology, not the data. Re-run it
  once the residual head is actually using its rank.

## Remaining headroom

Rank-40 oracle reaches 0.200 (1.04x) but needs 46 channels, over the 28 budget.
Within the constraints the untried lever is duplicate-padded full cell
utilisation, whose oracle is another -17 to -20% on top.
