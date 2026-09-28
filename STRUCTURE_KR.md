# can3tok 인코더/디코더 · 학습 구조 전체 정리

작성 2026-08-18 · 대상 `can3tok_encoder_decoder_new_fix_3` · 실측 기준 `runs/R8I_20260818_035640` (16.35 dB @4500) 및 진행중 M1/M3/M4

이 문서는 **코드에 실제로 있는 것만** 적습니다. 계획·가설은 `HISTORY_KR.md`, 일자별 기록은 `SESSION_LOG_KR.md`에 있습니다.

---

## 0. 한 장 요약

```
npz (81k~565k Gaussian)
  │  data.py      선택 K=262,144 · 고정 k-means anchor 4096개에 배정 · 슬롯 패킹
  ▼
enc_in (B, 589824, 15)   ← 인코더는 디코더보다 2.25배 많이 읽음
  │  encoder.py    GroupPointPooler(그룹당 144점 → 256차원) → pack → z_raw
  ▼
z_raw (B, 32, 256, 128)  = 1,048,576   (중간 표현, 병목 아님)
  │  compressor.py  intra attn ×3 → token_merge → cell_mix → window attn ×2 → mid → budget head
  ▼
z_compact (B, 32, 64, 64) = 131,072    ★ 병목. world model이 쓰는 latent
  │                                       Gaussian 1개당 0.5개 숫자
  ├─ decompressor ──→ z_raw_hat ──→ CodecDecoder      ──→ pred      (xyz 담당)
  │                        │                                  │ detach
  └─ attr_code (shape19+app8) ────→ AttributeDecoder ──→ attr_pred (scale/rot/opacity/color + nudge)
                                                              │
                                                       render_loss / eval / 배포
```

- 파라미터 **84.99M** — encoder 1.07M / compressor 17.04M / decompressor 13.58M / CodecDecoder 49.68M / AttributeDecoder 3.63M
- 셀 예산 32채널 = **centroid 4 | occupancy 1 | shape 19 | appearance 8**
- `merge = 4096/4096 = 1` → 셀 1개 = 그룹 1개 = anchor 1개

---

## 1. `data.py` — 데이터 파이프라인

### 1.1 스냅샷 필터
```
[data] mature snapshot filter >= 12000: train=1628 val=173
```
`--min_snapshot_step 12000`. 왜 필요한가 — 같은 코드로 GT를 렌더한 상한이 스냅샷 성숙도에 따라 크게 다릅니다.

| 스냅샷 | GT 렌더 PSNR |
|---|---|
| early (<12k) | 16.34 dB |
| late (≥12k) | 20.13 dB |
| 혼합 | 18.29 dB |

즉 미성숙 스냅샷을 섞으면 **모델 성능과 무관하게** 천장이 내려갑니다. 현재 GT 천장은 18.70 dB(eval 8뷰).

### 1.2 점 선택 `_importance` / `_select`
```python
score  = opacity
score += 0.25 * zscore(pruning_scores)      # upstream densify/prune 신호
score += density_importance_weight * (1 - local_density)
```
- `pruning_scores`는 **샘플링에만** 씁니다. 인코더 입력 피처에 넣지 않습니다 (`decode(z_compact)`가 볼 수 없는 정보이므로).
- 중요도 기반 선택이 랜덤보다 같은 K에서 5 dB 이상 좋습니다.
- K를 줄이면 표현 상한 자체가 내려갑니다: **262144 → 21.16 dB / 65536 → 19.62 / 32768 → 17.93**. 그래서 K는 고정입니다.

### 1.3 고정 anchor 배정 `_anchor_assign` — 이게 왜 있는가

**Morton 청킹의 결함(실측).** 청크 경계는 점 *순서*로 정해지므로, densify가 셀 j 앞에 점을 삽입하면 셀 j의 내용이 밀립니다. 인접 스냅샷 쌍에서:

| | 셀 중심 이동량 (그룹 반지름 배수) |
|---|---|
| 점 개수 변한 쌍 (densify 발생) | **13.9×** |
| 점 개수 안 변한 쌍 | 0.1× |

world model이 `z_t → z_{t+1}`을 학습할 수 없는 수준입니다. 고정 anchor로 바꾸면 **0.13×**.

**왜 k-means인가.** FPS는 공간을 균일 덮으므로 밀집 영역에서 anchor가 거대해집니다 — 그룹 반지름 FPS 1.18 vs k-means **0.064**.

**용량 불균형과 스필오버.** anchor는 고정인데 스냅샷 점 개수는 81k~565k로 변합니다. 최근접 anchor만 쓰면:

| 스냅샷 | 최근접만 | 8-근접 스필오버 |
|---|---|---|
| step_005070 | 26.1% 버림 | **15.6%** |
| step_018020 | 15.9% | **6.1%** |
| 희소 스냅샷 | 0.4~5.9% | — |

구현은 **가까운 것 우선 그리디**, 폴백 랭크마다 1회 벡터화 라운드 (`--anchor_spill 8`). 늦게 도착해서 밀리는 게 아니라, 경쟁에서 진 점이 밀립니다. 남는 점은 `anchor_assignment=capacitated`일 때 전역 빈 슬롯으로 한 번 더 복구합니다.

> **한계 명시**: anchor는 점의 **소속**을 정할 뿐 **위치를 구속하지 않습니다**. 디코더는 `centroid + offset`을 내보내고 offset 크기는 그룹 extent 단위로 자유입니다 (§4.2). 실측 `rel_offset_p50 = 1.314` — 점들은 이미 자기 anchor 반지름의 1.3배까지 나가 있습니다.

### 1.4 증강이 anchor를 따라가야 한다 (과거 버그)
```python
if anchors is not None:
    Ra, sa, sh = norm_xf
    anchors = ((anchors @ Ra.T) * sa + sh[None, :])
```
이 3줄이 없던 동안 anchor는 원본 좌표에, 점은 회전/이동된 좌표에 있었고 **슬롯 81%가 비었습니다**. `_augment`가 정규화 공간 변환 `(R, s, shift)`를 4번째 반환값으로 넘기게 고친 것이 이 수정입니다.

또한 `scene_anchors.npy`는 world 좌표(−132..186)로 저장되어 있어 자동 감지 후 정규화합니다:
```
[data] scene_anchors: 4096 anchors, world -> normalised (97.0% inside the unit cube)
```

### 1.5 슬롯 ↔ Gaussian 동일성 `source_index`
NN 조회로 슬롯을 원본 행에 되매핑하면 **0.991만 단사**입니다 — 이 씬들에서 0.16~0.17%의 Gaussian이 *정확히* 겹치고, 추가 ~0.5%가 정규화 왕복의 float32 오차 3e−5보다 가깝습니다. 그래서 슬롯 → npz 행 인덱스를 명시적으로 들고 다닙니다. 속성 감독은 이 동일성이 있어야 성립합니다.

### 1.6 인코더 입력을 크게
`--max_points 262144 --max_input_points 589824` → 그룹당 144점 입력, 64점 출력.

로더가 병목 *앞에서* 버리는 정보가 평균 2.50 dB, 최악 5.32 dB였습니다. 단, 인코더가 디코더보다 *다른* 집합을 보게 하면 world model의 셀에 안정된 Gaussian 동일성이 없어지므로 — **큰 집합은 작은 집합의 상위집합**입니다.

---

## 2. `encoder.py` — 인코더 (1.07M)

### 2.1 `GroupPointPooler` (0.744M) — 학습된 풀링
그룹당 144점(가변) → 고정 patch 256차원.

```python
h = MLP(feats)                       # feats = [그룹로컬 xyz, 표준화된 11속성]
z = CrossAttention(q=learned[16], h) # 학습 쿼리 16개
out = Linear(16*128 → 256)
```

- 인코더가 디코더보다 많이 읽는 순간 **복사할 identity가 없습니다** (identity pack은 xyz 64개를 문자 그대로 복사하는 것이었음). 3DShape2VecSet / Can3Tok / COD-VAE가 모두 쓰는 학습 쿼리 cross-attention 구성.
- *어떤* 점이 살아남는지가 *몇 개*보다 지배적입니다: 예산 초과 스냅샷에서 옛 규칙은 92% 유지 시 5.17 dB 손실, 56% 유지 시 2.92 dB 손실 — 유지율과 손실이 아예 역전.
- 4096 그룹 전체 attention이 메모리에 안 들어가서 `pool_chunk=512`로 그룹축 청킹.

### 2.2 `GroupAttributeEncoder` (0.059M) — 현재 **동결**
pack의 aux 슬롯(과거엔 mask)에 그룹당 64개 숫자로 속성을 압축하던 경로. **pooler가 켜지면 `_build_patch_pooled`가 점에서 직접 patch를 만들므로 이 모듈은 호출되지 않습니다.** 그래서 명시적으로 `requires_grad_(False)` — 체크포인트 호환을 위해 남겨두고, silent-freeze 가드가 계속 경고하지 않도록.

이 모듈이 존재하는 이유(기록용): pack이 xyz+mask뿐이던 동안 scale/rot/opacity/color가 `z_raw`에 **들어간 적이 없었습니다**. 기하만으로 속성 예측 시 held-out R²는 opacity **−0.99**, rotation −0.25, color −0.23 — 데이터셋 평균보다 나쁩니다. 색만 다른 두 씬이 *동일한* latent를 받았습니다.

부가 교훈 2개가 코드 주석에 박혀 있습니다:
- **슬롯 임베딩 없으면 전 슬롯이 같은 함수**를 계산합니다 (전부 permutation-equivariant). 그룹 평균 속성 복원 held-out R²: 임베딩 없음 0.158 → 있음 **0.984**. train R²는 양쪽 1.000 — 즉 64개 숫자는 항상 충분했고, 도달 가능한 코드가 인코딩 대신 암기였던 것.
- **LayerNorm 금지, 채널별 고정 통계 표준화.** 11채널은 서로 비교 가능한 양이 아닙니다 (log_scale 평균 −8.5, opacity 3.71 std, color 1.3 std). 점별 평균을 빼면 그 점의 색/불투명도 오프셋이 함께 날아갑니다 — 실측 R² 0.998 → 0.747.

### 2.3 `pack` (0.066M) — identity 초기화, 이제 **학습 가능**
```python
self.pack = nn.Linear(256, 256); self._init_identity()
if not cfg.pack_trainable: freeze
```
`w_z_raw`가 z_raw를 pack에 맞추는 감독을 하던 동안엔 동결이 옳았습니다 (pack이 *타깃*이므로). 목적이 렌더로 바뀐 뒤엔 동결이 곧 **인코더 기하 경로에 학습 파라미터가 0개**가 되는 것 — 실측 `‖dp‖`가 모든 스텝에서 정확히 0, "인코더"가 Morton 정렬 + Hungarian 배정에 불과했습니다. 현재 `--pack_trainable` ON.

### 2.4 `residual` (0.201M) + `residual_gate` — 제로곱 안장점 사고
```python
z_tok = pack(patch) + tanh(residual_gate) * residual(feat)
```
`residual` 마지막 층과 `residual_gate`를 **둘 다** 0으로 초기화하면 정확한 zero-gradient 안장점입니다:
- `d/d(residual weights) = tanh(0) * (...) = 0`
- `d/d(gate) = sech²(0) * ⟨upstream, residual(feat)⟩ = 0` ∵ `residual(feat)`가 0

**어떤 learning rate에서도 영원히** 움직이지 않습니다. 실측: 4000 스텝 후 gate·weight·bias 모두 정확히 0.0, 같은 기간 attr_encoder는 1.0e−3 이동. 수정은 `gate = 1e-2`(작지만 0 아님) — 분기 출력은 여전히 정확히 0(마지막 층이 0), 그래디언트 경로만 열립니다.

같은 실수를 이 코드베이스에서 **세 번** 잡았습니다: `fold_head`/`shape_xyz`, AttributeDecoder의 cross-attention `out_proj`, 그리고 `GroupAttributeEncoder.out`.

---

## 3. `compressor.py` — `StagedCompressor` (17.04M) → z_compact

### 3.1 순서
```
tokens (B,4096,8,32)
  → [pooled_input이면 스킵] xyz 블록을 그룹 extent로 나눔
  → token_embed: 32 → 448        + token_pos
  → intra SelfAttention × 3      (그룹 내 8토큰, 빈 그룹 마스킹)
  → token_merge: 8*448 → 2*448 → 448   (LayerNorm pre-activation)
  → cell_mix: 448 → 448
  → WindowSelfAttention × 2      (64×64 격자, window 8, 짝수 층 shift)
  → group_head: [group_vec, cell_vec, anchor_pe(centroid), extent, count/64] → 448
  → mid: 448 → 256 → 256         + global_to_mid(log N, fill)
  → budget heads
```

### 3.2 extent 정규화 (rank 붕괴 수정)
그룹 반지름이 한 씬 안에서 **854배** 차이 납니다. 절대 단위로 넣으면 pack 벡터들의 선행 방향이 그냥 "이 그룹이 얼마나 큰가"이고 형상은 4자리 아래로 깔립니다.

| pack 입력 (width 192) | effective rank |
|---|---|
| `xyz - centroid` (모델이 보던 것) | **9.0** |
| `(xyz - centroid) / extent` | **26.9** |

붕괴는 그대로 전파: pack 1.01 → token_embed 1.02 → intra 4.60 → pooling 4.26 → z_compact 5.41. extent는 이미 자기 채널에 있으므로 무손실.

**단, pooler가 켜지면 이 정규화를 끕니다** (`pooled_input`). 학습된 pooler의 patch는 계량적 의미가 없는 코드이고, 임의 슬라이스를 854배 변동하는 양으로 나누면 정규화가 없애려던 스케일 변동을 오히려 주입합니다. 조건은 pooler 생성 조건과 동일.

### 3.3 `token_merge` 단계화
| 지점 | effective rank |
|---|---|
| intra attention 출력 (448) | 34.26 |
| token_merge 출력 (448) | **13.83** |

3584 → 448 압축에서 도달한 방향의 60%를 버립니다. `Linear` 1개 → 2층 MLP → 현재 `3584 → 4d → 2d → d` (`compress_merge_stages`). 대안으로 `--compress_merge_attn`(학습 쿼리 1개의 attention pooling)도 구현돼 있습니다 — COD-VAE(ICCV'25)가 직접 point→latent 맵 대신 쓰는 구성.

### 3.4 budget head — 앵커 반, 자유 반
```python
cen = centroid + tanh(cen_res[:3]) * s * CENTROID_DELTA    # 해석적 centroid에 묶임
cen = cat([cen, u_scale=encode_scale(extent), cen_res[4:]])# ch3 = 그룹 스케일
occ = (2*count/64 - 1) + tanh(occ_res)*0.1                 # 점유율 앵커
shape       = head_shape(mid)     # 19ch 자유
appearance  = head_appear(mid)    #  8ch 자유
```

`mid` 하나에서 shape와 appearance가 갈라집니다 — 손으로 그은 분할이 아니라 **네트워크가** 얼마를 기하에, 얼마를 외관에 쓸지 정합니다. appearance가 마지막 접미사여서 모든 소비자가 prefix slice로 자를 수 있고 `c_app=0`이면 과거 레이아웃과 바이트 동일.

**latent 견고성 실측** — 이 분할이 왜 중요한가:

| 채널 | 노이즈 | PSNR 변화 |
|---|---|---|
| 자유 27ch | 30% | −0.19 dB |
| 앵커 5ch | **1%** | **−0.97 dB** |

앵커 채널은 물리량이라 극히 민감합니다. 그래서 wide 잔차(`to_wide`/`wide_blocks`/`to_compact`)는 **shape 채널에만** 마스크로 적용합니다. 초기 wide 런에서 32채널 전체에 잔차를 걸었더니 pose가 파괴되어 `z_l1`은 떨어지는데 codec eval이 무작위로 망가졌습니다.

### 3.5 staged budget head
단일 `Linear(256→16)`이 인코더 전체의 마지막 단계이자 latent가 쓰이기를 멈추는 지점: **mid 23.63 → z_compact 10.91** (32채널 중). 예산 3분의 2가 아무것도 안 나릅니다. `--compress_head_stages`로 정규화 비선형을 끼워 단계 하강 (DC-AE와 같은 방식).

### 3.6 전역 개수 신호
pad/truncate가 per-group occupancy에서 densify–prune 스케일을 지웁니다. 그래서 씬 레벨 `(log N / log N_max, N/N_max)`를 zero-init MLP로 모든 그룹 `mid`에 더합니다.

---

## 4. `compressor.py` — `StagedDecompressor` (13.58M)

### 4.1 순서
```
z_compact → 셀 → pg (B,4096,32)
  centroid = pg[0:3]          (그대로 읽음)
  scale    = decode_scale(pg[3:4])
  count    = (pg[4]+1)/2*64
  shape    = pg[5:24]         (19ch)
  appear   = pg[24:32]        ( 8ch)
  feat = [pg, anchor_pe(pg[0:3])]
  → group_embed → cell_mix → window attn → intra attn ×3
  → token_out → deep (B,4096,256)
  → folding으로 offsets 생성 → z_raw_hat
```

### 4.2 Folding 디코더 — 위치가 만들어지는 곳
```python
frame = fold_head([shape, ctx_tok])              # 6채널: aniso 3 + axis-angle 3
la    = frame[:3] - mean(frame[:3])
aniso = exp(tanh(la / cap) * cap)                # cap = folding_aniso_log_cap = 1.5
local = fixed_fibonacci_ball(64)                 # 고정 템플릿
local = local + tanh(shape_xyz(h)/1.0)*1.0       # 잔차, cap 1.0
local = local - local.mean(dim=2)                # ★ 평균 제거
local = local * aniso
local = R(axis_angle) @ local
xyz   = centroid + local * group_scale
```

각 요소가 왜 있는가 (전부 실측 근거):

**(a) 잔차 cap.** 무제한이면 192출력 잔차 헤드가 6출력 frame 헤드를 앞질러 비등방성을 자기가 재현합니다 — step 500에 lam2 0.999(완전 구) → **0.243** (GT 0.579), frame 헤드는 여전히 zero init. 봉투는 값싼 경로여야 합니다. cap 비용은 없음: rank-22 oracle chamfer 무제한 0.218 / cap 1.2 0.217 / 0.8 0.221 / 0.4 0.236.

**(b) `folding_res_gain` 램프.** 위와 같은 이유로 frame 헤드가 먼저 학습되도록 잔차를 초기 스텝 동안 0으로 붙잡습니다. 안 하면 intra_chamfer가 *나빠집니다* (0.499 → 0.557).

**(c) `aniso` log-ratio cap = 1.5 (최근 수정).** 기하평균 정규화는 **곱은** 고정하지만 **비율은** 고정하지 않습니다. 한 축이 폭주하는 대신 다른 축이 줄면 통과합니다. J1 체크포인트 실측: aniso가 **0.027 ~ 1411.5** 범위, 이 단계가 offset 크기를 0.620 → **7.976**으로 부풀리면서 effective rank는 5.80 → **4.24**로 *떨어뜨렸습니다*. 바늘 모양 그룹 몇 개가 지배 — 렌더의 거대한 번진 얼룩이 정확히 이것. cap 1.5면 한 축이 여전히 e^1.5 = 4.5배 늘 수 있어 실제 데이터의 비등방성(GT lam2 0.757)을 덮으면서 1411배는 배제합니다.

**(d) 평균 제거.** 템플릿은 중심에 있지만 학습된 잔차는 아니고, 그 평균이 그룹 centroid에 그대로 얹힙니다 — 어떤 활성 손실도 이걸 감독하지 않습니다 (`residual_pack`에서 pack 타깃이 `xyz - centroid`, 두 chamfer 항은 centroid 제거). 실측 centroid rmse가 step 500→2000에 0.00113 → 0.00374 → **0.00719** (제곱오차의 0.7% → 17.7%)로 커져 그룹 내 이득을 잡아먹었습니다.

**Folding 정보 감사** — 실제로 얼마나 쓰이는가:

| 단계 | effective rank |
|---|---|
| shape code (19채널) | 6.58 / 19 |
| → 잔차 (192차원) | 5.33 / 192 |
| → × aniso | **4.24** |

rank-4 코드는 디코더가 무엇을 하든 그룹 내 chamfer를 0.34 근처로 상한 지웁니다. 이 코드베이스 최고 런이 멈춘 지점이 정확히 0.345입니다. 이래서 `w_latent_decorr`(VICReg/Barlow Twins 공분산 항)이 rank를 직접 공격합니다 — `w_latent_std`는 **채널별** 통계라 상관에 눈이 멉니다 (채널 std는 0.25~0.57로 건강한데 rank는 예산을 넓혀도 ~4).

| 런 | shape 채널 | effective rank | 예산 활용 |
|---|---|---|---|
| _fix @10000 | 12 | 4.38 | 36.5% |
| fix2 @2000 | 28 | 4.33 | 15.5% |
| folding @1000 | 28 | 2.61 | 9.3% |

**12 → 28 확장이 아무것도 사지 못했습니다.**

### 4.3 `shortcut_alpha = 0` — 공짜 경로 제거
`_shortcut_tokens`는 비파라미터 사전값입니다. `residual_pack` 모드에서 xyz 사전값은 **0**이고 occupancy만 자유입니다. 절대 pack 모드였다면 centroid를 xyz로 broadcast — 그러면 chamfer의 가장 값싼 답이 "모든 점을 셀 centroid에 두기"였습니다. 현재 `shortcut_alpha=0.0` (eval도 0.0)으로 이 경로를 완전히 닫았습니다.

### 4.4 `ctx` — 아래로 내려가는 것
| 키 | 내용 | 소비자 |
|---|---|---|
| `centroid`, `count`, `scale` | 해석적 앵커 | CodecDecoder, AttributeDecoder |
| `cell_vec`, `group_vec` | 448차원 문맥 | CodecDecoder 이웃 문맥 |
| `appearance` | 8ch | CodecDecoder `_attributes` |
| **`attr_code`** | **shape 19 + appearance 8 = 27ch** | **AttributeDecoder** |
| `shared_code` | 비앵커 전체 27ch | joint decoder (현재 OFF) |
| `res_ratio`, `direct_frac`, `direct_ratio` | 진단 | 로그 |
| `fold_frame`, `fold_local` | 기저 | distillation (현재 OFF) |

`attr_code`가 존재하는 이유는 §5.4에서.

---

## 5. 디코더 2개 — 왜 나눴고, 어떻게 연결되는가

설계 의도: **위치를 먼저 맞춰야 나머지 Gaussian 파라미터를 맞출 수 있다** → `p(A | X, Z)`.

### 5.1 `CodecDecoder` (49.68M) — 기하 담당
```python
patch  = unpack(tok)
coarse = patch[:64*3] + centroid              # residual_pack 복원
q  = slot_query(offset_pe) + slot_embedding + mem_tok.mean() + q_xyz(xyz_pe(coarse))
for sa, ca in zip(self_blocks, blocks):       # 10층
    h = sa(h, key_padding_mask=slot_pad)      # 슬롯 간 self-attn (pad 마스킹)
    h = ca(h, memory)                         # 토큰 + 이웃 셀 cross-attn
xyz      = coarse + tanh(xyz_residual(...)) * group_scale * residual_scale * refine_alpha
presence = (mask_value-0.5)*8 + presence_head(h) + (occ-0.5)*4
```
- 정제 예산이 **그룹 자기 extent의 비율**입니다 (`residual_scale=0.6`, `decoder_refine_alpha` 램프).
- `presence`에 count 사전값이 필요합니다 — 없으면 padded frame에서 "전부 존재"로 붕괴.
- `_attributes`의 속성 헤드들은 **`attr_decoder_layers > 0`이면 죽은 코드**입니다. `attr_pred`가 렌더·파라미터 손실·eval·배포가 읽는 것이므로. (이 때문에 `attr_teacher_prob` teacher forcing이 한동안 아무도 소비하지 않는 헤드에서 돌고 있었습니다 — model.py로 옮겨 수정.)

### 5.2 `AttributeDecoder` (3.63M) — 속성 담당
```python
tok = slot_emb.weight                          # 64개 학습 슬롯 기저
feats = [tok, xyz_pe(p), loc_pe(그룹로컬 정규화 p)]
h = in_proj(cat(feats))
ct = to_code_tok(app_o) + to_shape_tok(shp)    # ★ zero-init 가산 분기
ct = ct + nbr_emb                              # 이웃 순서 표시
h = h + xattn(xnorm(h), ct, ct)
h = SelfAttention × 4
out = [p + tanh(head_nudge(h))*nudge_cap*scale,   # 경계된 위치 nudge
       _bounded_log_scale(head_scale(h), scale),
       normalize(head_rot(h)),
       head_opacity(h), head_color(h)]
```

**(a) `loc_pe` — 그룹 로컬 좌표 인코딩.** 위치는 씬 정규화 좌표로 들어오고 그룹 반지름은 씬의 ~0.5%이므로, `xyz_pe`는 사실상 그룹 *주소*입니다: 그룹 내 변동 / 그룹 간 변동 = 0.27. 그런데 설명해야 할 속성은 반대로 **변동의 52~78%가 그룹 안**에 있습니다. 그룹 중심 재정렬 + 반지름 나눗셈이 이 비율을 **3.52**로 만듭니다 (13배 개선). 이 디코더의 유일한 per-point 입력이므로 다른 어디서도 공급될 수 없습니다. 대체가 아니라 **가산** — 씬 레벨 인코딩은 "씬의 어느 부분인가"를 계속 말해야 외관이 예측 가능합니다.

**(b) `attr_cond = xattn` — FiLM이 아닌 이유.** 8개 스칼라 씬 하나에 같은 네트워크, 이 경로만 바꿔 측정 (`tools/oracle_conditioning.py`):

| 조건화 | nrmse | 그룹 내 nrmse |
|---|---|---|
| none | 0.3021 | 0.3830 |
| **film** | **0.3055** | **0.3876** ← 코드를 무시하는 것보다 나쁨 |
| concat | 0.2983 | 0.3759 |
| **xattn** | **0.2783** | **0.3547** ← film보다 8.9% 좋음 |
| film+cat | 0.2969 | 0.3741 |

FiLM이 `none` 아래로 간 게 결정적이었습니다 — 메커니즘이 약한 게 아니라 코드 값보다 최적화 비용이 더 컸다는 뜻이고, 그러면 8개 appearance 채널이 죽은 무게가 됩니다. FiLM은 `gam`/`bet`을 64슬롯 전체에 broadcast하므로 "슬롯 5는 빨강, 슬롯 6은 파랑"을 말할 수 없습니다.

**(c) `attr_nbr_window` — 하드 분할의 정보 상한.** pack이 씬을 64점 그룹으로 하드 분할하고 각각에 appearance 채널만 주므로, 점의 외관은 그 몇 개 숫자로만 결정됩니다. 그룹 내 패턴은 최대 8차원 공간에 사는데 GT는 64×3 = 192차원.

| 코드 | 색 변동의 그룹 내 비중 |
|---|---|
| 인코더 자기 코드 | 14.9% |
| 같은 크기의 **랜덤** 코드 | 23.3% ← 인코더 코드가 노이즈보다 매끄러움 |
| 분산 16배 코드 | 36.5% |
| **GT** | **74.7%** |

36.5%가 이 하드 분할의 측정된 상한입니다. Can3Tok(이 모델의 원본)은 아예 분할하지 않고 학습 canonical 쿼리가 씬의 모든 Gaussian에 cross-attend합니다 — 262144점에서 O(N²)로 불가능하지만, **로컬 윈도우가 대부분을 회복**합니다. Morton 순서에서 연속 그룹 인덱스는 공간적으로 인접하고, 이건 compressor의 윈도우 attention이 이미 하는 가정과 동일합니다. 현재 M 시리즈 `--attr_nbr_window 1` (양쪽 1칸 = 코드 3개).

**(d) `_bounded_log_scale` — 그룹 상대 비대칭 밴드 (핵심 수정).**

밴드 자체는 있어야 합니다 — 무제한 `exp`에서 Gaussian 하나가 148배까지 커지고, 렌더 손실은 정확히 그 방향으로 밉니다 (구멍을 덮는 가장 값싼 방법이 splat 확대). **틀린 것은 밴드의 폭이 아니라 중심**이었습니다.

중심이 `head_scale.bias` — log-scale이 −18.4 ~ +0.8에 걸친 씬에 대해 전역 스칼라 하나. GT 3.5M개 실측: −7.58 중심 + cap 3이면 **5.94%에 도달 불가** (5.29% 너무 작음, 0.66% 너무 큼). 모델은 실제로 덜 퍼뜨렸습니다 — R8은 GT 퍼짐의 0.739배.

그룹별 재중심이 원인을 고칩니다:
```
median log_scale ≈ 0.787 * log(extent) − 2.559        R² = 0.759   (5 스냅샷 × 4096 그룹)
```
```python
g    = group_scale.clamp(min=1e-6).log()
base = self.scale_base_a * g + self.head_scale.bias    # 둘 다 학습 가능
r    = raw - self.head_scale.bias                      # ★ base가 아니라 bias 기준
r    = where(r < 0, down*tanh(r/down), up*tanh(r/up))
return base + r
```

`r = raw - b`인 이유: `raw = W h + b`이므로 `raw - base = W h - a·log(extent)` — zero init(W≈0)에서 extent < 1인 모든 그룹이 **상한에 포화**하고 모든 Gaussian이 e^1.5배 크게 시작합니다. `raw - b = W h`는 init에서 0이므로 출력이 밴드의 목표인 그룹 상대 중앙값에서 시작합니다.

기준 제거 후 헤드가 내야 하는 잔차는 작습니다 — 그룹 내 p0.1 = −2.54, p99.9 = +2.66, 기준 적합 자체의 잔차 p1 = −1.33, p99 = +1.29. 5.94%를 놓치던 ±3 밴드가 올바르게 중심을 잡으면 **99.93%**에 도달합니다.

비대칭은 데이터 기반이 *아닙니다* — 그룹 상대가 되면 양 꼬리가 거의 같습니다. 상한을 더 좁게 두는 것은 렌더 손실의 거대 splat 지름길에 대한 안전 마진입니다. 두 분기 모두 r=0에서 기울기 1로 통과하므로 C1 연속.

**(e) `to_shape_tok` — function-preserving 가산 분기 (최근 수정).**

문제: 위치 입력이 detach되면 렌더/속성 목적함수가 19개 shape 채널에 **정확히 0.000e+00** 그래디언트를 놓습니다 (실측). 즉 shape는 점공간 항이, appearance는 렌더가 학습 — 모든 셀의 서로소인 두 절반이 서로 다른 목적함수 아래 있습니다. 그런데 채널 연합 실증은 xyz/scale/opacity/color가 **함께** 움직여야만 이득이라고 말합니다.

첫 시도는 `to_code_tok`의 입력을 8 → 27로 **넓히는** 것이었는데, 그러면 `to_code_tok`이 재초기화되어 appearance→슬롯 조건화 경로 전체가 리셋됩니다: held-out PSNR이 15.39 → **9.55** @step100, step250에도 12.11. 런이 "shape 연결됨"과 "appearance 경로 재학습됨"을 섞어서 측정했습니다.

현재는 **분리된 zero-init 가산 분기**:
```python
ct = self.to_code_tok(app_o)                    # 기존 경로 그대로
if self.to_shape_tok is not None:
    ct = ct + self.to_shape_tok(shp)            # weight/bias = 0
```
- step 0에서 모델이 체크포인트와 비트 동일
- 분기 자기 weight는 그래디언트를 받음 (`d out/d W = shape ≠ 0`) → 스스로 열림
- §2.4의 제로곱 안장점이 **아닙니다** — 그건 `tanh(gate) * residual(feat)`로 두 인자가 모두 0이던 경우

검증: `c_app=8 c_shape_read=19 c_code=27 to_shape_tok=True`, detach ON 상태에서 shape 채널 그래디언트 0.000e+00 → **2.9e−06**.

그리고 물리적으로도 맞습니다: Gaussian의 적정 scale은 그 그룹의 **형상**에 달려 있습니다 (얇은 판이면 한 축이 작아야 함) — 이 모듈은 그걸 볼 수 없었습니다.

**(f) `head_nudge` — 경계된 위치 이동 (trust region).**
```python
d_xyz = tanh(head_nudge(h)) * attr_nudge_cap * group_scale
```
그룹 자기 extent 단위. 0.05 (M1) / 0.15 (M3, M4). 0.15는 투영 footprint 하나를 조금 넘습니다 (그룹 반지름 12.0 px에 대해 1.40 px) — 이미지 그래디언트가 아직 옳은 곳을 가리키는 범위.

### 5.3 `model.py` — 두 디코더 연결
```python
pred = decoder(z_raw_hat, cell_vec, scale, centroid, count, group_vec, attr_xyz, appear)

geo = pred[..., 0:3]
if attr_detach_geometry:  geo = geo.detach()          # ← 스케줄로 해제 가능
if attr_xyz is not None and p_tf > 0:                 # teacher forcing
    geo = where(rand < p_tf, true_xyz, geo)           # 점별 혼합

attr_pred = attr_decoder(geo, ctx["attr_code"], ctx["scale"].detach())["pred"]
```

**`appear`는 의도적으로 detach하지 않습니다.** 과거엔 했고, 그 detach 하나가 appearance가 latent에 들어가지 못한 이유였습니다. step-4000 체크포인트 실측: `w_render_attr`(68.0, 전체 목적함수의 ~85%)가 attr_decoder의 3.61M 파라미터에 L2 128.5의 그래디언트를 놓고 `encoder.attr_encoder`·compressor·decompressor에는 **정확히 0**. appearance 코드가 자기 유용성에 대해 동결돼 있었습니다 — 속성 디코더가 어떤 손실도 개선할 수 없는 코드로부터 scale/rot/opacity/color를 예측하라고 요구받은 것. 그 코드에서 그룹 평균 속성을 복원하는 held-out R²가 pack에서 **−0.947**, decompression 후 −0.339 — 디코더의 최선이 데이터셋 평균을 내는 것이었고, 그게 모든 런이 보고한 속성 nRMSE ≈ 1.0입니다.

그래디언트를 통과시키면 compressor/decompressor가 렌더 그래디언트에 노출됩니다. **그게 의도한 결과입니다** — appearance가 z_compact에 도달할 유일한 경로. 기하 디코더는 위의 detach로 계속 보호됩니다.

**`attr_detach_geometry`의 전제와 결론.** 전제는 옳았습니다 — 이미지 그래디언트는 위치 수송에 조건이 나쁩니다 (참 보정과 cosine **0.13**, 중앙값 Gaussian이 1.40 px에 투영되는데 위치 오차가 ~6 px, 24%의 점만 그래디언트를 받음). 결론은 틀렸습니다 — held-out PSNR과 정렬된 유일한 항이 84M 중 3.6M만 학습하게 두고, 기하를 재구성 항에 완전히 넘겼습니다. 실측 그 항들의 이 채널 예산에서의 공동 최적(그룹별 PCA, held-out 사진 8장)이 **9.95 dB**, 모델 자체는 13.4~14.6 dB. **끌어내리고 있습니다.**

조건화 문제는 있어야 할 곳에서 처리합니다 — **coarse-to-fine 래스터화** (§7.1).

### 5.4 정리: 왜 `attr_code`인가
| 경로 | 무엇을 보는가 | 어떤 손실이 학습시키는가 |
|---|---|---|
| CodecDecoder | z_raw_hat (shape 경유), appearance 8ch | 점공간 기하 항 |
| AttributeDecoder (과거) | appearance 8ch만 | 렌더 + 속성 항 |
| **AttributeDecoder (현재)** | **shape 19 + appearance 8** | 렌더 + 속성 항, **shape에도 도달** |

렌더 그래디언트를 **위치 수송**(조건 나쁨)이 아니라 **속성 경로**(조건 좋음)를 통해 shape에 넣는 것입니다.

---

## 6. `losses.py` — 활성 항 16개와 실측 기여도

### 6.1 M1 가중치
| 항 | 가중치 | 대상 |
|---|---|---|
`w_render_attr` | 20.0 | attr_pred (렌더)
`w_shape_sinkhorn` | 12.0 | shape 채널 OT
`w_intra_sinkhorn` | 8.0 | 그룹 내 OT
`w_intra_chamfer` | 6.0 | 그룹 내 chamfer
`w_chamfer` | 5.0 | 다중 스케일 chamfer
`w_splat_area` | 4.0 | splat 면적 힌지
`w_group_centroid` | 4.0 | 그룹 centroid
`w_coverage` | 2.0 | 단측 커버리지
`w_scale` `w_opacity` `w_color` | 1.0 | 속성 L1
`w_latent_std` | 1.0 | 채널 std 밴드
`w_presence` `w_rot` `w_equiv` | 0.5 | 존재/회전/등변성
`w_latent_decorr` | 0.05 | 채널 상관 (rank)

### 6.2 ★ 실측 기여도 — 가장 중요한 발견
R8I에서 각 항의 `weight × value`:

| 항 | 기여 | 비중 |
|---|---|---|
| `group_centroid` | 91.22 | **47.6%** |
| `intra_chamfer` | 54.55 | 28.5% |
| `intra_sinkhorn` | 43.58 | 22.7% |
| **`render_attr`** | **2.31** | **1.2%** |
| `chamfer` | 0.02 | ~0% (실질 OFF) |
| `coverage` | 0.00 | 0% (실질 OFF) |

동시에 polish lr 배율:

| 모듈 | lr 배율 |
|---|---|
| encoder | 0.02 |
| compressor | 0.05 |
| decompressor | 0.05 |
| decoder | 0.10 |
| **attr** | **1.00** |

**목적함수의 98.8%가 거의 동결된 모듈을 밀고, 1.2%가 유일하게 학습되는 모듈에 도달합니다.** 이게 "손실을 과도하게 쓴다"의 정확한 형태입니다 — 항의 *개수*가 아니라 **기여도와 lr의 정렬 실패**.

이 발견이 과거 결론 하나를 무효화합니다. "렌더는 xyz를 움직일 수 없다"는 T3에서 나왔는데, T3의 점공간 항이 ~190, 렌더가 ~2.31 — **같은 파라미터에 대해 80:1로 표결에서 졌습니다**. T1은 이미 학습된 방향의 크기만 조절한 것이었습니다. M3/M4가 기하 항을 ~8배 깎고 렌더를 3배 올려 이걸 처음으로 교란 없이 시험합니다.

### 6.3 개별 항의 존재 이유

**`render_loss` — 유일하게 프록시가 아닌 항.**
> 여기 모든 점공간 지표는 렌더가 나쁜 복원으로 만족시킬 수 있음이 드러났습니다. 가장 명확한 사례가 중복입니다 — 예측 점 2개가 GT 점 1개를 공유하면 대칭 chamfer는 만점을 주는데 표면 절반이 안 덮입니다.

기준은 GT Gaussian의 렌더 + 실제 사진(`--extra_real_views 4`, `--photo_map`). 양쪽을 **같은 카메라, 같은 래스터라이저**로 렌더하므로 차이는 전적으로 Gaussian 때문입니다. 타깃 분기는 `no_grad`.

**presence 게이팅 (최근 추가).** 예전엔 양쪽 다 GT 마스크를 썼으므로 `presence`가 렌더 목적함수에 **아예 들어가지 않았고**, 모델이 추론 시 갖지 못하는 정보로 채점됐습니다:

| 렌더 마스크 | PSNR |
|---|---|
| GT 마스크 | 14.89 dB |
| 예측 presence | 14.76 dB |
| 전 슬롯 존재 | 11.85 dB |

헤드는 실제로 일을 하고 있었고 질문을 받은 적만 없었습니다. 예측 게이팅은 가변 개수 디코더의 전제조건이기도 합니다 — 없으면 빈 영역에 Gaussian을 내는 비용이 0입니다.

**`w_splat_area` 힌지.** 렌더의 거대 splat 지름길에 대한 직접 페널티 (분위 0.80 초과분).

**`covariance3d_loss` (구현됨, M2에서만 사용).**
```python
Cp = R_p diag(s_p²) R_pᵀ ;  Ct = R_t diag(s_t²) R_tᵀ
d  = ‖Cp/‖Ct‖ − Ct/‖Ct‖‖²
```
`scale`과 `rot`을 따로 감독하면 이미지가 관측할 수 없는 인자분해를 고릅니다 — 실측 `attr_rot_nrmse 1.089`, 데이터셋 평균과 다를 바 없음. 단위 테스트: 동일 → 0.0, 랜덤 회전 → 0.5243, scale +0.7 → 9.3342, `q → −q` → **0.0** (쿼터니언 부호 불변, 올바름).

**`sinkhorn_parameter_target` / `responsibility_target`.** 슬롯 순서 L1은 슬롯 i의 예측이 슬롯 i의 GT여야 한다고 요구합니다. 위치가 아직 틀린 동안 그건 잘못된 짝에 대한 감독입니다. Sinkhorn OT로 예측 위치 ↔ GT 점을 부드럽게 짝지어 속성 타깃을 만듭니다.

**`attr_anchor_covered = 0.25`.** 렌더가 이미 말할 수 있는 점에서 앵커를 4배 낮춥니다. 덮인 점에서 둘이 불일치할 때 앵커가 측정된 **10 dB** 차이로 더 나쁩니다 (같은 위치에서 16.74 vs 26.77) — 거기서 앵커가 끌면 점을 그룹 평균으로 되돌리기만 합니다. 덮이지 않은 집합에서는 렌더가 아무것도 주지 않으므로 앵커가 그래디언트의 100%.

**`w_equiv` — 등변성.** 씬을 회전시키고 latent도 같이 회전해야 함. 4스텝마다.

**`w_latent_decorr` vs `w_latent_std`.** §4.2 참조 — std는 rank에 눈이 멀고, decorr가 rank를 직접 공격합니다.

---

## 7. 학습 스케줄 (`schedule.py`)

### 7.1 `render_downscale_at` — coarse-to-fine 래스터화
```python
t = ramp(step, render_start, render_downscale_steps)
v = exp(log(hi) + t * (log(lo) - log(hi)))     # log2 공간 기하 보간
```
M1: 4 → 2 (1500 스텝). M3/M4: 8 → 2 (4000 스텝).

렌더를 기하에서 떼어놨던 이유가 조건화였습니다 (§5.3). **해상도를 낮추면 조건화 자체가 고쳐집니다** — downscale 8에서 픽셀 하나가 그룹 반지름 ~4개를 덮으므로, 전해상도에서 6 px 벗어난 점이 자기 footprint 안에 들어오고 그래디언트가 옳은 방향을 가리킵니다. 그 다음 기하가 수렴하며 해상도를 올립니다. 모든 광도 정렬 기법의 표준 coarse-to-fine.

`t`의 시작점이 `render_start`인 게 중요합니다 — R6은 전역 step 0에서 보간을 시작해서, 렌더가 늦게 켜지면 거의 최종 해상도로 진입했습니다. 기하가 아직 넓은 footprint의 잘 조건화된 그래디언트를 필요로 하는 바로 그때.

### 7.2 `attr_teacher_prob` — teacher forcing → scheduled sampling
```python
ramp(step, attr_start + attr_force_steps, attr_anneal_steps, 1.0 → 0.0)
```
`p(A|X,Z)` 인자분해는 헤드가 **신뢰할 수 있는 위치**에서 A를 배울 때만 의미가 있습니다. 초기엔 예측 위치가 점 간격보다 더 틀리므로, 그러지 않으면 헤드가 기하 오차를 보상하는 법을 배웁니다. 점별 혼합(배치별 아님)이라 모든 스텝이 두 체제를 다 담고, 헤드가 한쪽에 과적합할 깨끗한 구간을 얻지 못합니다.

### 7.3 `attr_detach_geometry` / `attr_detach_release`
```python
attr_detach_geometry AND (release < 0 OR step < release)
```
초기엔 detach가 옳고(조건 나쁨 + 점집합 거침), **영원히는 틀립니다**. M4가 step 6000에 해제합니다.

### 7.4 `folding_res_gain` 램프
§4.2(b).

### 7.5 위상과 lr
```
step < latent_end : latent
step < geo_end    : geo      (geo_weak_scale 0.35 → 1.0 램프)
polish_start > 0 && step >= polish_start : polish
```
M 시리즈는 `latent_end=0, geo_end=0, polish_start=1` — 처음부터 polish. `polish` 위상에서 §6.2의 lr 배율이 적용됩니다.

전역 lr: 선형 warmup(200) → cosine (5e−5 → 5e−6).

`attr`은 모든 cut에서 **의도적으로 빠져 1.0**을 유지합니다 — polish가 encoder/compressor를 동결하는 것은 latent가 디코더 밑에서 움직이는 걸 막기 위함이고, 속성 디코더는 detach된 입력의 하류이므로 무엇을 배워도 latent를 흔들 수 없습니다.

`attr_enc`는 `encoder`와 **별도 그룹**입니다 — train.py가 residual 분기가 꺼져 있으면 `encoder` 그룹 lr을 0으로 만드는데, 그게 latent에 어떤 appearance를 넣을지 배우는 모듈을 동결시켜서는 안 됩니다. 실측으로 500스텝 떨어진 두 체크포인트에서 모든 encoder 텐서가 비트 동일했고 pack의 aux 블록이 초기값 std 0.0163에 그대로 있었습니다.

---

## 8. `train.py` — 학습 루프와 평가

### 8.1 silent-freeze 가드 · `param_groups` orphan 검사
`model.param_groups()`가 학습 가능한데 어떤 옵티마이저 그룹에도 없는 파라미터를 찾으면 **RuntimeError**를 냅니다.
> attr_decoder에게 정확히 그 일이 일어났습니다: 3.325M 파라미터가 런 전체에서 초기값에 앉아 있었고 손실은 학습 중인 것처럼 로깅됐으며, 유일한 증상은 체크포인트 간 eval 지표가 소수 셋째 자리까지 동일한 것이었습니다.

### 8.2 평가 프로토콜
```
--score_metric psnr --eval_view_count 8 --eval_view_seed 1234
--eval_val_indices 0,4,7,31,63,95,127,159
```
`eval_utils.render_eval_metrics()`가 `psnr_photo`, `psnr_gt_photo`, `psnr_teacher`, `ssim_photo`, `psnr_gap`을 냅니다. PSNR은 클수록 좋으니 부호를 뒤집습니다:
```python
sgn = -1.0 if score_metric.startswith("psnr") else 1.0
```

`run_eval`은 `enc_x`/`enc_mask`를 **반드시** 넘겨야 합니다 — 안 넘기던 동안 pooler eval이 학습과 다른 그룹화를 측정했습니다 (rmse **0.675** vs 실제 0.0096).

`group_error_breakdown()`은 그룹으로 reshape하기 *전에* 마스크 압축을 하고 있었는데, 부분 채워진 anchor 그룹에서는 무효입니다. 슬롯 레이아웃 기반 그룹화 + 적응 prefix 폭으로 교체하고 `group_stat_slots`/`group_stat_groups`를 함께 보고합니다.

### 8.3 ⚠ `--init_from`의 step 리셋 함정
`--init_from`은 step 카운터를 0으로 되돌립니다. 그래서 이전 런이 **절대 step**으로 표현한 모든 게이트를 새 런에서 명시적으로 다시 열어야 합니다. 안 열면 속성·렌더 손실이 꺼진 채로 수천 스텝이 돌아 속성 디코더가 완전히 동결됩니다 — **S1/S2가 이렇게 죽었습니다.**

### 8.4 ⚠ `--init_skip` (오늘 발견, 수정 완료)
`scripts/launch_r8i_scale_band.sh:38`이 `--init_skip attr_decoder.head_scale`을 들고 있습니다. R8I에는 옳았습니다 — R8H에서 워밍스타트했고 R8H의 `head_scale`은 **구 전역 bias 밴드** 아래 학습된 것이라 의미가 달랐습니다.

M 시리즈에는 **틀립니다** — R8I 자체에서 출발하고 거기 `head_scale`은 이미 이 밴드로 학습돼 있습니다 (bias −2.51, `scale_base_a` 0.719). 실측 대가: step 100 held-out PSNR **11.03 dB** vs R8I의 16.35 — 세 변종 전부에서 5.3 dB의 워밍스타트를 버렸습니다. `launch_m_series.sh`에서 센티넬로 무효화했습니다 (`EXTRA_ARGS`는 단어 분할 문자열이라 빈 값을 넘길 수 없음). 수정 후 **625/628 텐서 로드**, 새 텐서 3개(`nbr_emb`, `to_shape_tok.weight/bias`)만 초기값 — 설계상 function-preserving.

### 8.5 DDP
`torchrun --nproc_per_node=$NPROC`. **R8I는 3장(유효 배치 3배)이었고 M 시리즈는 1장입니다 — 단일 GPU 20000스텝은 R8I의 20000스텝과 본 샘플 수가 1/3입니다.** 비교할 때 반드시 감안해야 합니다.

---

## 9. 현재 상태

### 9.1 결과 사다리
| 런 | held-out PSNR | 비고 |
|---|---|---|
| Q1 | 12.95 | |
| R3 | 14.92 | 기하 우선 |
| R5 | 14.69 | 기하 우선이 이기지 않음 |
| R8 | 15.39 | anchor + OT + 렌더 |
| R8H | 16.48 | |
| **R8I** | **16.35 @4500** | gap 2.36, 현재 기준선 |
| J1 − J0 | +0.02 | joint refiner 기여 없음 (`jdx 7e−6`) |

R8I 렌더 분해 (`render_metrics_idx63-161.json`):

| | PSNR |
|---|---|
| canon (GT Gaussian 렌더) | 25.70 |
| codec | 18.00 |
| **codec_snap** (xyz를 GT에 스냅) | **19.87 (+1.87)** |
| codec_own | 17.40 |
| nn_unique | 0.533 |

`codec_snap − codec = +1.87 dB`가 **위치 오차가 갚아야 할 빚**입니다. 이게 `attr_nudge_cap 0.15` 안에 들어가는지가 미측정이고, M4가 그 경우를 대비해 detach를 해제합니다.

### 9.2 블러 진단 (R8I step 5000, held-out 5 스냅샷)
| 지표 | 예측 | GT | 비 |
|---|---|---|---|
| linear scale 중앙값 | 1.55e−4 | 1.79e−4 | **0.86×** |
| linear scale 평균 | 2.89e−4 | 5.74e−4 | 0.50× |
| log_scale std | 0.96 | 1.63 | 0.59× |
| p0.1 | −11.03 | −16.62 | **+5.59 (≈270배 큼)** |
| p95 | −7.25 | −6.16 | −1.09 (작음) |

**평균적으로 큰 게 아닙니다. 분포가 자기 평균으로 압축돼 있습니다.** 미세 구조에 GT가 아주 작은 Gaussian을 쓰는 자리에 중간 크기를 놓습니다 → 정확히 블러이고, 국소적으로는 splat이 커 보입니다.

그룹 내 잔차 (그룹 중앙값 제거):

| | 값 |
|---|---|
| std 비 (예측/GT) | **0.444** |
| 예측 p0.1 | −1.47 (GT −10.12) |
| **하한 cap 포화** | **0.00%** ← 클리핑 아님 |
| **GT가 `cap_up 1.5` 초과** | **13.13%** ← 상한은 클리핑 |

하한은 클리핑이 아니라 **모듈이 그룹 안에서 변화할 정보가 없다**는 것입니다 — `slot_emb`가 4096 그룹에서 동일, `attr_nbr_window 0`, appearance 8채널이 64점 × 11속성을 조건화. 상한 13.13%는 p95의 −1.09 부족과 일치합니다.

### 9.3 진행중 실험 (2026-08-18 10:58 재시작, GPU 0/1/2, 20000 스텝)
공통: `--init_from runs/R8I_20260818_035640/ckpt_step00005000.pt`, `cap_up/down 3.0`, `attr_nbr_window 1`, `attr_read_shape 1`, `init_skip` 무효화

| | GPU | out_dir | 차이 |
|---|---|---|---|
| **M1** | 0 | `runs/M1_20260818_105842` | 밴드+정보만. 기하 가중치 원래대로, `w_render_attr 20`, `nudge 0.05`, `downscale 4→2`, `polish_decoder 0.1` |
| **M3** | 1 | `runs/M3_20260818_105902` | 렌더에 위치 소유권. `group_centroid 4→0.5`, `intra_chamfer 6→0.8`, `intra_sinkhorn 8→1.0`, `shape_sinkhorn 12→1.5`, `render_attr 20→60`, `nudge 0.15`, `downscale 8→2`(4000), `polish_decoder 0.5`/`compressor 0.3`/`decompressor 0.3`/`encoder 0.1` |
| **M4** | 2 | `runs/M4_20260818_105922` | M3 + `attr_detach_release 6000` |

(M2 = `w_cov3d 2.0 --w_rot 0.0`는 구현돼 있으나 현재 미실행)

**판정 게이트**
- **M1** → `attr_scale_spread` (0.501)와 `attr_scale_spread_within` (0.410)이 올라가야 함, log_scale std가 GT 1.63 쪽으로
- **M3** → `rel_offset_p50` (1.314)이 내려가야 함
- **M4** → step 6000 근처에서 `rel_offset` 기울기 변화

### 9.4 구현됐으나 꺼져 있는 것
`joint_direct_decoder`, `joint_local_memory`, `structured_local_code`(F3D), `joint_shared_decoder`, `gen_decoder`, `teacher_cycle`, `w_cov3d`. R8I 로그에서 `mem 0.000 cmem 0.000 tr 0.0000 joint_shared_decoder 0`으로 확인.

### 9.5 남은 미측정
1. **손실 기여도 정규화 감사** — `chamfer` 0.02, `coverage` 0.00은 실질 OFF인데 가중치 5.0, 2.0을 들고 있습니다. `group_centroid`가 47.6%.
2. **교정된 latent oracle** — 자유 27채널만, 낮은 lr. 이전 시도는 앵커 채널을 lr 0.02로 최적화해서 무효화됐습니다 (13.34 → 10.18).
3. **F3D 아키텍처 활성화**.
4. **가변 개수 생성 디코더** — presence 게이팅이 전제조건이고 이미 켜져 있습니다.

---

## 부록: 반증된 가설 (7회 이상)

같은 함정을 다시 밟지 않기 위한 기록입니다.

| 가설 | 실측 결과 |
|---|---|
| 재구성 항 전부 제거가 도움 | R1 tmpl 1.7 (붕괴) |
| 기하 우선이 이긴다 | R3 **14.92** > R5 14.69 |
| nudge cap이 병목 | 4배 확대 = **+0.02 dB** |
| scale 밴드가 1순위 | 예측 중앙값 GT의 0.94배, 포화 0.28% |
| cap 4배가 T1의 −0.52를 설명 | 실제 원인은 anchor multiplier 회귀 |
| 잔차가 작아서 템플릿이 지배 | 잔차/ball = 0.679, 원인은 **aniso** (0.027~1411.5) |
| 렌더는 xyz를 움직일 수 없다 | 교란됨 — 점공간 항이 80:1로 표결 승리 |

**채널 연합 사다리** — `{xyz, scale, opacity, color}`가 **함께** 갈 때만 오릅니다:

| 조합 | Δ PSNR |
|---|---|
| 4개 전부 | **+2.24** |
| 모든 진부분집합 | 중립 또는 해로움 |
| xyz + scale (최악) | **−1.58** |

**GT 교체 사다리** — 보상 평형(자기 정합 해)의 증거:

| 교체 | Δ PSNR |
|---|---|
| GT xyz만 | −0.16 |
| GT scale만 | **−1.77** |
| GT 속성 전부 | −0.50 |
| GT 전부 | **+2.77** |

단일 채널을 GT로 바꾸면 **해롭습니다**. 모델이 자기 오차들이 서로 상쇄되는 자기 정합 해에 앉아 있다는 뜻이고, 한 채널만 고치면 그 평형이 깨집니다.
