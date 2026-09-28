# 데이터 파이프라인과 모듈별 전방 계산

작성: 2026-09-15  
코드: `can3tok/data.py`, `encoder.py`, `compressor.py`, `decoder.py`, `attr_decoder.py`,
`model.py`, `io_utils.py`, `render.py`.  
숫자는 E1/S16k/T16k `args.json`.

---

## 1. 입력 npz

경로 (`E1 args.root`, 쉼표로 씬 두 개):

```
.../speedy-splat/output/train_colmap_seg_prune_scores/replay_seg
.../speedy-splat/output/truck_colmap_seg_prune_scores/replay_seg
```

`io_utils.load_npz_state`가 xyz, scaling, rot, opacity, color, 선택적 sh,
`pruning_scores`, camera를 읽는다.

`gaussian_target_channels`가 한 행을 만든다:

```
[ xyz_norm 3 | log(scale/scene_scale) 3 | quat 4 | logit(opacity) 1 | rgb 3 | sh? ]
```

지금 SH는 DC만 쓰므로 `target_dim=14`, `sh_dim=0`. 원본 3DGS 59채널이 아니다.

정규화: 씬별 `assets/stats_speedy_{train,truck}.json`의 center/scale.
`drop_outside=true`면 `|xyz_norm|>1` 제거.

`min_snapshot_step=12000`. 미성숙 스냅샷의 GT 사진 천장이 낮아서 필터한다.

---

## 2. `ReplayGaussianDataset.__getitem__`

순서 (`data.py`):

1. 파일의 `scene_idx`로 center/scale/앵커를 고른다. 씬 0을 쓰면 안 된다.
2. 증강 (`augment=true`): 회전 ±8°, 스케일 ±3%, 이동 ±1%. **카메라와 앵커를
   같은 변환으로 옮긴다.** 이 3줄이 없던 시절 슬롯 81%가 비었다.
3. `drop_outside`.
4. `_importance`: opacity + 0.25×zscore(`pruning_scores`) + 밀도 항
   (`density_importance_weight=0.35`). pruning_scores는 **샘플링에만**.
   인코더 피처에 안 넣음 (`decode(z_compact)`가 못 보는 정보).
5. Morton 정렬 후 `_select`: 디코더용 `max_points=262144`.
6. `_pack_slots`: `scene_anchors`가 있고 개수가 그룹 수와 같으면
   **k-means 배정** (`_anchor_assign`, `anchor_spill=8`, `anchor_assignment=spill`).
   `partition_mode=morton`은 이 분기에서 실행되지 않는다.
7. 칸이 가득 차면 `_template_reorder_full_groups`: 화이트닝 고유프레임에서
   Fibonacci 공에 Hungarian. det=+1. 부분 칸은 live prefix 유지.
8. `target` (262144×14), `mask`, `source_index` (슬롯→npz 행, 빈 칸 -1).
9. 인코더 입력이 더 크면 (`max_input_points=2097152`): 디코더 선택을 **포함하고**
   나머지를 중요도로 채운다. 상위집합. 칸 용량은 인코더 쪽이 256이 아니라
   2048 (`2097152/1024`).

배치 크기 1. `workers=6`. 카메라 벡터는 렌더용으로 같이 나간다.

멀티씬: `root` / `stats_path` / `scene_anchors`가 같은 개수의 쉼표 리스트.
E1:

```
scene_anchors=assets/anchors_speedy_train_n1024.npy,assets/anchors_speedy_truck_n1024.npy
```

각 npy는 (1024, 3) 월드(또는 정규화) 좌표. 학습 파라미터가 아니다.

`split_path=assets/split_speedy_both_blockB.json`. S16k가 홀드아웃 누수를 줄인 분할.
`--eval_view_force`와 학습 스냅샷 7개 제거가 여기 있다 (`ANALYSIS` §13.6).

---

## 3. 인코더 `PatchPackEncoder`

### 3.1 풀러가 켜지는 조건

```
(max_input_points // num_groups) > group_size
2097152 / 1024 = 2048 > 256  → pooled_input = True
```

identity pack(xyz 256개 복사)은 정의가 안 된다. `GroupPointPooler`:

```
feats = MLP( [그룹로컬 xyz 3 + 표준화 attr 11] )     # in_dim ≈ 14
q = 학습 쿼리 16개, dim 128
for i in pool_blocks:          # 지금 1
    q = CrossAttend(q, points, pad=empty)
    # blocks>1 이면 q self-attn, points가 q를 다시 읽음 (pool_feedback)
out = Linear(16*128 → patch_dim=1024)
```

`pool_chunk=-1`이면 그룹축을 메모리에 맞게 자른다.

`GroupAttributeEncoder`는 풀러 경로에서 **호출되지 않고** `requires_grad_(False)`.
체크포인트 호환용.

### 3.2 pack과 residual

`pack = Linear(1024, 1024)` identity 초기화. `pack_trainable=true`.
`w_z_raw`가 0인 지금, 동결하면 인코더 기하 경로 학습 파라미터가 사실상 풀러뿐.

```
z_tok = pack(patch) + tanh(residual_gate) * residual(feat)
```

gate 초기값 1e-2 (둘 다 0이면 영구 안장점). `encoder_residual_start=0`이면
처음부터 켜짐. lr은 `schedule.lr_scales`가 모듈별로 곱한다.

`z_raw`는 토큰을 32채널 × 256×128 격자에 심는다. 남는 셀은 0.

인코더가 쓰는 `centroid/extent/count`는 **인코더가 본 점**의 통계.
`use_fixed_anchor_center=0`이면 이것이 compact cen의 기준이다.

---

## 4. `StagedCompressor`

`pooled_input`이면 **extent로 xyz를 나누지 않는다.** 풀러 출력은 미터 단위가 아님.
identity pack일 때만 `(xyz-centroid)/extent` (랭크 9.0→26.9 측정은 그 경로).

이후 (`model_dim=448`, `heads=8`):

1. `token_embed`: 32 → 448 + `token_pos`
2. intra `SelfAttention` × `compress_intra_layers=3`, 빈 칸 key-padding
3. `token_merge`: `32*448 → 2*448 → 448` (`compress_merge_stages=0`).
   stages>0이면 앞에 `4*448` 층. `compress_merge_attn`이면 쿼리 1개 cross-attn
   (지금 false).
4. `cell_mix`: merge=1이라 `448→448`
5. `WindowSelfAttention` ×2, 격자 32×32, window 8, 홀수 층 shift
6. `group_head`: `[group_vec, cell_vec, Fourier(centroid), extent, count/G] → 448`
7. `mid`: 448→256→256, LN+LeakyReLU 0.1
8. `global_to_mid(log N, fill)` zero-init 가산
9. 헤드: Linear(256→4/1/3/8). `compress_head_stages`가 비어 있으면 단층.

z-order로 compact 격자에 기록. `compress_wide_channels=0`이라 격자 ResBlock 없음.

`latent_mode=deterministic`, KL 0. `latent_scale` EMA는 `model._update_latent_scale`.
DDP는 `broadcast_buffers=False`라 fix_7이 `--latent_scale_sync_every`로 평균한다.

---

## 5. `StagedDecompressor`

compact를 다시 그룹 축으로 펼친 뒤 prefix 슬라이스:

```
cen 4 | occ 1 | shape 3 | app 8
```

count는 occ에서, scale은 cen[3]에서 `decode_scale`.

`shortcut_alpha=0`, `residual_pack=true`: 토큰 prior의 xyz는 0. 모양은 shape가
만들어야 한다.

folding (`folding_decode=true`):

- `unit_ball`: `template.fibonacci_ball(256)`, 학습 안 함
- `fold_head`: shape(+ctx) → 6 (log-aniso 3 + axis-angle 3)
- `shape_xyz`: shape(+ctx) → 256×3, 마지막 층 zero-init, hidden `max(128, 4*256)=1024`
  (`shape_xyz_hidden=0`이면 이 기본)

aniso: 축 평균을 빼고 `tanh(·/1.5)*1.5` 후 exp. 곱만 고정하던 softplus는
비율 1411배로 랭크를 죽여서 캡을 넣음.

잔차: `folding_res_gain`이 0이면 전부 0. cap 1.0 tanh.

그다음 접힌 좌표를 identity unpack 형식의 `z_raw_hat`으로 다시 쓴다.
CodecDecoder는 이 토큰을 정제한다.

ctx에 `appearance`, `centroid`, `scale`, `count`, `cell_vec`, `group_vec`,
`attr_code`(shape를 앞에 붙일 수 있음)를 넣는다.

---

## 6. `CodecDecoder`

coarse: `unpack` identity로 `z_raw_hat` → 칸 안 xyz.

refine (`decoder_layers=10`):

- 슬롯 쿼리 = Fourier(슬롯) + 선택적 `slot_embedding` + Fourier(coarse xyz)
- 메모리 = 칸 토큰 + (옵션) 3×3 이웃 그룹 토큰
- 각 층: **그룹 내 self-attn 후** cross-attn
- `xyz_residual` zero-init, 크기는 `residual_scale * extent`
- `decoder_refine_alpha`가 0이면 이 잔차가 꺼짐 (E1: 8000까지)

presence 헤드. 속성 헤드도 모듈 안에 있으나 `attr_decoder_layers>0`이면
학습·렌더·eval이 읽는 것은 `attr_pred`라 **여기 속성 헤드는 죽은 출력**.

`patch_chunk=192`로 그룹축 체크포인트.

---

## 7. `AttributeDecoder`

기하 스택과 분리. 의도: 렌더 그래디언트가 10층 기하 attention을 관통하지 않게.

입력:

- `geo` xyz — `attr_detach_geometry=true`(기본)면 **detach**. 스케줄
  `attr_detach_release` 이후 풀릴 수 있음
- appearance 8, 선택적으로 shape 3을 **zero-init 가산 분기**로 (`attr_read_shape`).
  to_code 폭을 8→11로 넓히면 경로 전체가 리셋됨 (T5 실패)
- 그룹 scale (detach)

출력: scale, quat, opacity, color + `tanh` nudge × `attr_nudge_cap=0.15` × extent.

scale 밴드: `attr_scale_cap_down/up=3`, `attr_scale_group_base=1` — 그룹 상대 중심.

`attr_nbr_window=1`: 이웃 칸 appearance를  Cond로 봄.

`attr_teacher_prob`: 학습 중 진짜 xyz를 확률적으로 넣음. eval은 0.
`attr_decoder_layers>0`이면 예전 CodecDecoder teacher forcing은 무효였고,
지금은 이 모듈에 걸려 있다.

렌더·파라미터 손실·배포가 쓰는 가우시안은 `attr_pred`.
기하 chamfer 등은 보통 `pred`의 xyz (디태치 정책에 따름).

---

## 8. `Can3TokAE.forward`에서 빠지는 것

```python
if not use_fixed_anchor_center:
    group_anchor = None
```

npy 앵커는 **데이터 배정에만** 쓰인다. 디코더 원점은 compact cen.

`run_gen=false` (`no_gen_branch`). gen 노이즈는 안 넣음. codec은 항상 깨끗한 compact.

`encode_compact(..., normalized=True)`는 `z / latent_scale`. WM이 이 스케일을 써야 함.

`decode_compact`는 attr를 안 붙인다. compact-only 그림 게이트를 쓰려면 여기와
학습 렌더 경로의 차이를 적을 것.

---

## 9. 렌더 `render.py`

`render_gaussians`: 예측/GT 가우시안 + `Camera`(16벡터).
`photometric_loss`: L1 + λ DSSIM (`render_lam_dssim`, DSSIM만 올리는 실험은 닫힘).
`edge_gain`은 `detail_start` 이후 `detail_gains`가 램프.
`vgg_perceptual`도 그때 `w_render_perc`.

해상도: `schedule.render_downscale_at`. 먼저 거칠게 (조건수), `detail_start` 이후
최종 배율로. E1은 `detail_start=20000` > `max_steps=16000`이라 **이 런에서 VGG/edge 없음**.

`render_presence=1`: 예측 presence로 점을 고름. GT 마스크만 쓰면 oracle 편향 0.13 dB.

`joint_detach_render_xyz`: 렌더가 xyz를 직접 안 당김. 손잡이는 scale/opacity/nudge.

---

## 10. 타깃 레이아웃 인덱스 (`io_utils`)

```
xyz 0:3 | scale 3:6 | rot 6:10 | opacity 10:11 | color 11:14 | sh 14:
```

손실·헤드가 이 슬라이스를 쓴다. `w_rot=0`, `w_cov3d=2` — 축 이름 대신
Σ = R diag(s²) Rᵀ.
