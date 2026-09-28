# can3tok — Gaussian-splat autoencoder with a fixed latent

> **현재 상태 (2026-09-28) — 새 환경에서 시작한다면 [`SETUP_KR.md`](SETUP_KR.md) 부터 보세요.**
>
> - 현재 latent 는 **`16 × 32 × 32` = 16,384** (셀 1024 × 16채널 = centroid 4 · occupancy 1 · shape 3 · appearance 8),
>   출력 가우시안 262,144. 아래 본문은 초기 `32 × 64 × 64` 설계 기록입니다.
> - 데이터: vanilla 3DGS replay (Tanks&Temples train + truck). 만드는 법 `SETUP_KR.md` §4.
> - 학습 설정: `configs/B1_bgcap.args.json` (처음부터), `configs/B1g_stable.args.json` (재개).
>   실행은 `scripts/launch_from_config.sh`.
> - 체크포인트 · 데이터 · 그림은 크기 때문에 저장소에 없습니다 (`SETUP_KR.md` §9).

---

Rewrite of the CoordFirstGSAE codec/generative autoencoder for **262,144 input
Gaussians** under a hard constraint: the world-model interface stays exactly

```
z_compact ∈ R^{32 × 64 × 64}
```

The goal is therefore not a bigger latent but a *better used* latent: maximum
recoverable detail, plus statistics that a continuous-diffusion world model
(DIAMOND) can actually learn.

---

## 1. The bug that motivated the rewrite

The identity patch pack writes `group_size * 4 = 128` numbers per group
(32 xyz triplets + 32 mask flags). The old run configured
`local_tokens_per_group = 8`, i.e. `token_dim = 8 * 32 = 256`, so **half of
`z_raw` was provably zero**, verified on `ckpt_step0008000.pt`:

```
encoder.pack.weight  shape=(256, 128)
  rows   0:128  exact identity
  rows 128:256  all zeros
```

Because the compact budget was split evenly across `merge * tokens_per_group = 16`
tokens, **16 of the 32 channels in every cell were compressing constants**:

| configuration | useful channels / cell | points / cell | channels / point |
|---|---:|---:|---:|
| old 131k run (`tpg=8`) | 16 | 32 | 0.50 |
| old 262k run (`tpg=8`) | 16 | 64 | **0.25** |
| this code (`tpg=4`) | 32 | 64 | **0.50** |

`local_tokens_per_group` now defaults to `group_size * 4 / latent_channels` and
`validate_layout` raises if a configuration would pad zeros, so this class of
mistake cannot recur silently. `describe_layout` prints the resulting
channels-per-point on the first line of every run.

---

## 2. Where each design decision lives

| Concern | Fix | File |
|---|---|---|
| Half the latent held zeros | lossless `tokens_per_group`, hard validation | `config.py` |
| Even channel split wastes budget | `centroid | occupancy | shape` budget, centroid/occupancy anchored to analytic group statistics | `config.py`, `compressor.py` |
| Single-shot 8x compression | staged `d -> mid -> budget` heads (TC-AE / COD-VAE) | `compressor.py` |
| Cells see no context | Swin-style shifted window attention over the cell grid (TRELLIS SLat) | `layers.py`, `compressor.py` |
| Padded groups pollute attention | key-padding masks derived from group validity | `compressor.py` |
| Encoder not learnable | zero-init residual branch on top of the identity shortcut (DC-AE Residual Autoencoding), unlocked only once geometry loss is active | `encoder.py` |
| Patch seams in the codec output | cross-attention memory includes the 3x3 cell window | `decoder.py` |
| Generative branch too weak (1 vector -> 64 points, 2 KV tokens) | cell -> group -> point hierarchy, coarse xyz anchored on the centroid channels, KV = groups + cell + 3x3 window | `gen_decoder.py` |
| 2D neighbours were not 3D neighbours | cells stored in 2D Z-order (bit interleaved), a free permutation that improves conv locality | `morton.py`, `compressor.py` |
| Latent had no scale discipline | running `latent_scale` (Hunyuan3D `scale_factor`), two-sided per-channel std band `[--latent_std_floor, --latent_std_ceil]`, optional weak VAE | `model.py`, `losses.py` |
| Pretraining never looked at geometry | decode switches on late in the latent phase with weak weights | `schedule.py` |
| Regularisation cliff at the phase switch | every weight is ramped, none stepped | `schedule.py` |
| Noise blocked codec reconstruction | codec always uses the clean latent, noise only on the generative path | `model.py` |
| `encoder_warmup_*` was dead code | per-module learning-rate groups | `model.py`, `schedule.py` |
| Best checkpoint ignored the generative branch | score = `0.5 * (codec_rmse + gen_rmse)` | `train.py` |
| Only half of a dense scene was ever seen | Morton-bin stratified sampling + random Morton-window multi-crop | `data.py` |
| No augmentation / no equivariance | rotation/scale/shift jitter with quaternion bookkeeping + EQ-VAE style latent equivariance term | `data.py`, `losses.py` |
| Single-scale chamfer at 6% of the points | multi-scale chamfer (1k / 4k / 16k) on both branches, hard mining on both | `losses.py` |
| No generative metric during training | `gen_xyz`, `distill`, latent diagnostics in the train log | `train.py` |

---

## 3. Architecture

```
points (262144 x 3, Morton ordered)
  │  identity Morton patch pack  (+ zero-init learned residual)
  ▼
z_raw           32 x 256 x 128        internal only, exactly the input patches
  │  masked intra-cell attention -> shifted window attention -> staged heads
  ▼
z_compact       32 x 64 x 64          the only thing DIAMOND ever sees
  ├─ staged decompressor -> z_raw_hat -> codec decoder      (teacher, clean latent)
  └─ generative decoder                                      (student, noisy latent)
```

Per compact cell (`merge = 2` groups, 64 points, 32 channels):

```
group 0 : centroid[4] | occupancy[1] | shape[11]
group 1 : centroid[4] | occupancy[1] | shape[11]
```

`centroid[0:3]` is the analytic group centroid (the network may nudge it by
±0.05), `occupancy[0]` encodes the valid-point count, and everything else is
learned. Both decoders read these channels directly, which is why they start
from a sane coarse reconstruction instead of from noise.

---

## 4. Curriculum

| steps | phase | what trains | notes |
|---|---|---|---|
| 0 – 2000 | `latent` | compressor + decompressor | z_raw reconstruction only, strong diversity terms |
| 2000 – 3000 | `latent` (late) | + codec decoder | decode on, geometry at 15% weight |
| 3000 – 9000 | `geo` | codec | geometry ramps to full, latent terms glide down |
| 5000 – 15000 | `joint` | + generative | gen weights and distillation ramp in |
| 15000 – end | `gen` | both | codec LR reduced to 0.4x |
| optional | `polish` | decoder heads only | DC-AE style final stage, `--polish_start` |

The encoder residual branch stays at learning rate 0 until
`--encoder_residual_start` (default 12000): a trainable encoder combined with a
`z_raw` reconstruction target is degenerate while no geometry loss is active.

---

## 5. Running

```bash
# 2-minute wiring check on CPU with real data
bash scripts/smoke_cpu.sh

# layout / identity / gradient assertions
python -m tests.test_layout

# the real run: 4 GPUs, 262k points, from step 0
bash scripts/launch_can3tok_262k_ddp.sh --detached
tail -f runs/can3tok_262k_c32_zorder_*.log
```

Useful environment overrides: `PATCH_CHUNK`, `GEN_REGION_CHUNK`,
`CHAMFER_PAIR_CHUNK` (all trade VRAM for speed), `WORKERS`, `NPROC`,
`MAX_STEPS`.

`assets/stats_replay_seg.json` is the normalisation used by the previous runs
(quantile 0.02, `scale = 47.418`), so `xyz_rmse_norm` is directly comparable to
them. `assets/replay_seg_2700_300_seed42.json` is the same 2700/300 split.

## 6. Targets

| metric | old 262k run | target |
|---|---|---|
| codec `xyz_rmse_norm` @ 8–10k | ~0.016 | ≤ 0.013 |
| gen `xyz_rmse_norm` | 0.48 @ 8k (decode was off) | ≤ 0.02 |
| shape-channel std | unconstrained | ≥ 0.25, no dead channels |
| `latent_hf_ratio` | not measured | monitored, lower is easier to diffuse |
| dense-scene coverage | ~0.53 of the frame | uniform via stratified + multi-crop |

## 7. Exporting latents for a world model

```python
z = model.encode_compact(x, mask)          # already divided by latent_scale
```

`latent_scale` is an EMA of the training-time latent standard deviation and is
stored in the checkpoint, so encode/decode stay consistent across runs.

Without an upper bound the compressor is free to inflate its output as long as
the decompressor divides it back out: reconstruction is unaffected but the latent
distribution keeps drifting, which is exactly what a diffusion prior cannot
follow. `--latent_std_ceil` (default 1.5) bounds it. The training log prints
`lat_std <mean>/<max>` over the learned shape channels; both should settle inside
the band.
