# Verification: full cell utilization (2026-08-07)

## Claim under test
Most scenes use only ~30–87% of the 4096 compact cells under prefix
packing with `group_size=64`. Spreading real points across all 4096 groups
raises channels/point ~2.3× and removes dead cells for diffusion.

Earlier `slot_redistribute` failed for *learning* even with Morton slots
(not only kd-axis sort). This run re-tests that claim with G=64 / 28ch and
permutation-invariant metrics.

## Gates (must use these, not ordered PCA alone)

| check | pass |
|---|---|
| Oracle live-cell fraction | treatment ≈ 1.0 on typical val scenes |
| Oracle perm-invariant chamfer ceiling | treatment ≤ control −10% |
| Train 2500 steps: `z_res` slope | treatment descends ≥ half of control |
| Train 2500: perm chamfer (eval) | treatment ≤ control |
| PNG / radius ratio | no streak collapse, radius ratio not ≪ 0.7 or ≫ 1.1 |

If oracle wins but train fails → utilization helps diffusion only; keep
prefix for AE or change decoder (folding) before retrying redistribute.
If both win → adopt full-fill + occupancy=1, then folding + bit sweep.

## Results (2026-08-07)

### Oracle (`tools/verify_utilization_oracle.py`)
Mean over 8 val scenes (G=64, rank-28 ordered PCA — reference only):

| arm | util | pca_abs | vs prefix |
|---|---:|---:|---:|
| A prefix+localkd | 0.605 | 0.00329 | — |
| B fullfill morton | 1.000 | 0.00268 | −18.5% mean (sparse scenes −36%, full scenes **+10%**) |
| C fullfill kd | 1.000 | 0.00246 | −25% mean |

Utilization claim: **confirmed**. Oracle ceiling: **partially confirmed** (helps sparse, hurts full).

### Train A/B 2500 steps (`scripts/ab_utilization.sh`)
Latent-only, `w_group_cov=0`, Morton slots, shape_direct on.

| arm | z_res 50→2500 | Δ | z_l1@2500 |
|---|---|---:|---:|
| A prefix | 0.151→**0.114** | **−24.8%** | 0.0014 |
| B fullfill morton + occ1 | 0.166→0.154 | −7.1% (flat after ~500) | 0.016 |
| C fullfill kd + occ1 | 0.153→0.140 | −8.7% | 0.0058 |
| D fullfill morton occ0 | 0.167→0.154 | −7.6% | 0.022 |

**Decision: reject full-cell redistribute for the current decoder.**
Same failure mode as the earlier G=32 redistribute A/B: PCA/utilization ↑, within-group learning flatlines.
`np.sort` Morton slots and occupancy=1 do **not** fix it.

### Bit sweep (`tools/verify_latent_bits.py`, fix2@2000 ckpt)
Group chamfer / radius after quantizing channel blocks:

| bits | all | shape-only | centroid-only |
|---:|---:|---:|---:|
| 16 | 0.387 | 0.387 | 0.387 |
| 8 | 1.36 | **0.387** | 1.36 |
| 4 | 27.1 | **0.387** | 27.1 |
| 2 | 170 | **0.397** | 170 |

Shape channels are almost insensitive to quantization → they carry ~no reconstruction
signal at step 2000. Centroid/scale carry essentially everything. Confirms rank-collapse
diagnosis independently of the utilization A/B.

### Folding A/B (`scripts/ab_folding.sh`, 2000 latent steps, prefix packing)

| arm | z_res 50→2000 | Δ | end |
|---|---|---:|---:|
| baseline `shape_xyz` | 0.151→0.118 | −21.9% | 0.118 |
| **folding** | 0.211→**0.113** | **−46.4%** | **0.113** |

Folding starts worse (fixed shell ≠ GT) but overtakes baseline and keeps
`dfrac≈0.9`. **Adopt `--folding_decode` on prefix packing for the next full run.**

### Final decision
| lever | verdict |
|---|---|
| Full 4096-cell redistribute | **Reject** for current decoder (train fails gate) |
| Prefix + local kd | **Keep** |
| Folding decode | **Adopt** |
| Bit/quantization of shape | Not the bottleneck (shape already ~empty) |
| Diffusion dead cells | Separate: validity mask / empty-token — later |

## Arms
- **A prefix**: `--no_slot_redistribute --partition_block 32` (current fix2)
- **B fullfill_morton**: `--slot_redistribute --partition_mode morton --partition_block 0 --budget_occupancy 1 --budget_shape 27`
- **C fullfill_kd**: same as B but `--partition_mode kd` (Morton slot order inside leaf)

`w_group_cov=0` for all arms (known noise-spray failure). Latent-heavy
2500 steps so `z_res` is visible before geo noise.
