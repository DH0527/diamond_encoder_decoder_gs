# can3tok 모델 구조

`can3tok_encoder_decoder_new` 기준. 측정된 수치는 전부 출처를 밝혔습니다.
진단과 실행 계획은 [DIAGNOSIS.md](DIAGNOSIS.md), 변경 이력은 [IMPLEMENTED.md](IMPLEMENTED.md).

---

## 0. 무엇을 하는 모델인가

3D Gaussian Splatting 장면 하나(최대 262,144개 가우시안)를 **32×64×64 = 131,072개
숫자**로 압축하고 되돌립니다. 그 latent가 DIAMOND world model의 인터페이스입니다.

**고정 제약 (변경 불가)**

| | 값 | 이유 |
|---|---|---|
| `max_points` | 262,144 | 사용자 요구 |
| `z_compact` | 32 × 64 × 64 | DIAMOND 인터페이스 |
| 파생값 | **0.5 채널/점** | 위 두 개가 정함 |

마지막 줄이 이 문서의 나머지를 지배합니다 — §7 참고.

---

## 1. 전체 흐름

```
npz 프레임: N개 가우시안 (N = 79k ~ 519k), 카메라 1대
    │
    │  (0) data.py — 학습 파라미터 없음
    │      · 장면 정규화 (전역 center/scale)
    │      · drop_outside: |xyz_norm| ≤ 1 밖 제거
    │      · N > 262,144 → 밀도 인지 층화 샘플링으로 262,144개 선택
    │        N ≤ 262,144 → 전부 사용하고 나머지 슬롯은 mask=0 패딩
    │        (복제 없음. `_select`가 n ≤ s면 order를 그대로 반환)
    │      · Morton 정렬 → 32그룹 블록 내 local kd → 4096 그룹 × 64점
    │      · 그룹마다 Hungarian 배정으로 고정 템플릿에 슬롯 정렬
    │      · source_index 기록 (슬롯 → npz 행, 정확한 1:1)
    ▼
4096 그룹 × 64점 × 59채널 타깃
    │
    │  (1) encoder  PatchPackEncoder   0.267M  (0.2%)
    │      identity pack = 사실상 reshape. residual_pack: (xyz − 그룹 centroid) 저장
    │      ⚠ patch = x[..., 0:3] 와 mask 뿐. 59채널 중 **3채널만** 들어갑니다
    │      64×3 xyz + 64 mask = 그룹당 256개 숫자  (patch_dim = g*4, 하드코딩)
    ▼
z_raw : 32ch × 256×128  (32,768 셀 = 4096 그룹 × 8 토큰) = 1,048,576 숫자
    │
    │  (2) compressor  StagedCompressor  17.039M  (15.0%)
    ▼
z_compact : 32 × 64×64 = 131,072 숫자        ◄── diffusion 인터페이스
    │        그룹당 32채널 = centroid 3 + log-extent 1 + shape 28
    │
    ├──► (3) decompressor  13.580M (12.0%)  →  codec decoder  49.253M (43.5%)
    │         교사. z_raw_hat(1,048,576)을 거쳐 디코딩
    │
    └──► (4) gen_decoder  33.099M (29.2%)
              학생. z_compact만 읽음. world model이 쓰는 경로
```

**합계 113.24M** (stage `xyz`) / 113.66M (stage `geometry`, attribute 헤드 +0.42M).

"encoder"는 신경망이 아닙니다 — 0.267M은 step 12000까지 동결되는 선택적 residual
헤드이고, pack 자체는 결정론적입니다. **실제 인코더는 compressor입니다.**

두 디코드 경로 모두 추론 시 **`z_compact`만** 필요합니다. codec은 큰 중간표현
(1,048,576)을 거쳐 정보를 만들고, gen은 world model이 업데이트한 `z_compact`를
직접 소비합니다.

---

## 2. data.py — 학습 파라미터 0

| 단계 | 내용 |
|---|---|
| 정규화 | 전역 `center`/`scale` (데이터셋 통계, `assets/stats_replay_seg.json`) |
| 선택 | `stratified` + 밀도 인지 (희소 영역 가중 0.35) → 정확히 262,144 |
| 공간 정렬 | Morton (`morton_bits`) → 32그룹 블록 안에서 local kd |
| 슬롯 정렬 | `slot_sort=template` — 화이트닝된 고유프레임에서 고정 fibonacci-ball 템플릿에 **Hungarian 배정** |
| augmentation | 회전 ±8°, 스케일 ±3%, 이동 ±1%, crop 35% |
| 카메라 | 16-벡터로 패킹, **augmentation과 함께 이동** (§6.4) |
| `source_index` | 각 슬롯의 npz 행 번호. NN 조회 대신 정확한 대응 |

**local kd를 쓰는 이유** (전역 kd 아님): 전역 kd는 슬롯 순서를 먼저 고치지 않으면
**해칩니다** (오라클 0.00699 → 0.00784). local kd + 템플릿 정렬은 0.00453 (−46.8%)
이면서 prefix 패킹·꽉 찬 그룹·셀→Z-order 격자 국소성을 모두 유지합니다.

**템플릿 정렬 슬롯의 효과**: 평균 잔차 1.010 → 0.408, 도달 가능 chamfer 0.261 → 0.218.

**PCA 슬롯 정렬은 제거했습니다**: 축 단조 순서는 그룹을 직선으로 붕괴시킵니다
(`sqrt(lam2/lam1)` 0.036 vs GT 0.544). rank-k PCA 상한이 그 퇴화를 *보상*하고 있어서
−23.6%로 잘못 읽혔습니다.

---

## 3. compressor — 실제 인코더 (17.0M)

```
z_raw (32ch × 32768 셀)
  │  그룹 extent로 입력 정규화   ← 단일 변경으로 유효 rank 1.01 → 20.48
  │  token_embed: Linear(32 → 448)
  │  token_merge: MLP([8×448 → 896 → 448], hidden_norm)   ← mean이 아니라 concat
  │  + token_pos (학습 파라미터)
  │  intra_blocks × 3   : 그룹 내 8토큰 self-attention
  │  cell_mix           : merge=1이므로 항등에 가까움
  │  window_blocks × 2  : 8×8 윈도우, 홀수 층 shift (Swin 방식)
  │  group_head: MLP([448×2 + anchor_pe + 4 → 448 → 448])
  │  mid: Linear(448→256) LN LeakyReLU(0.1) Linear(256→256) LN LeakyReLU(0.1)
  │  + global_to_mid(전역 N/fill 2개 → 256, zero-init)
  │  heads: centroid(→4) | shape(→28)
  ▼
z_compact 32 × 64 × 64
```

### 왜 이렇게 되어 있는가 (전부 측정 근거)

| 결정 | 근거 |
|---|---|
| **그룹 extent로 입력 정규화** | `residual_pack`이 절대 단위 오프셋을 저장하는데 한 장면 안에서 그룹 반지름이 **854배** 차이납니다. 정규화 후 pack 유효 rank **1.01 → 20.48**. 정보 손실 없음 (extent는 `u_scale` 앵커에 이미 있음) |
| **token_merge를 MLP로** | `Linear(3584→448)` 하나는 init에서 16.23을 통과시키다가 학습이 **5.84**로 몰았습니다. 그동안 `group_head`는 18.64로 회복 — `cell_vec`과 해석적 앵커를 함께 받기 때문. 즉 **점 데이터 경로가 죽고 앵커가 latent를 나릅니다** |
| **mid 32 → 256** | 448 → 32 → 32 → 28에서 32 중 유효 rank 5.77, 유닛의 53%가 LeakyReLU 0.1 기울기 구간. 예산(28)보다 좁은 두 번째 병목이었습니다 |
| **global_to_mid** | pad/truncate가 densify–prune 스케일을 그룹별 occupancy에서 지웁니다. 장면 수준 카운트를 모든 그룹 mid에 주입 |
| **occupancy 0채널** | prefix 패킹 + 꽉 찬 그룹이라 그룹당 카운트가 불필요. 4채널을 shape로 |

---

## 4. z_compact — 채널 예산

```
그룹당 32채널 = centroid 3 + log-extent 1 + shape 28
4096 그룹 × 32 = 131,072
```

* **centroid(3) / log-extent(1)** — 해석적 앵커. 그대로 씁니다(학습된 예측 아님).
  기하량이고 그대로 두어야 합니다.
* **shape(28)** — 학습된 유일한 부분. 현재는 64점의 그룹 내 **배치만** 기술합니다.

`shape`라는 이름이 현재 상태를 정확히 반영하지만, full 3DGS를 목표로 하면
**`content(28)`** 로 재정의되어야 합니다 — local geometry + scale/rotation +
opacity + appearance를 함께 싣고, 기하 디코더와 attribute 디코더가 같은 채널을
**다르게 읽는** 구조.

측정된 배분(§7.8): **24 geometry + 8 appearance**. 8채널이 appearance 곡선의 무릎
(그룹당 8개 → held-out +4.4 dB, 그 이상은 급격히 포화)이고, 기하를 28 → 24로 줄이는
비용은 −0.2 dB 입니다. 이 배분에서 `z_compact` 도 `z_raw` 도 크기가 바뀌지 않습니다.

`12 geo / 12 attr` 처럼 더 균등하게 쪼갤 이유는 측정상 없습니다 — appearance 는
8에서 포화하고 기하는 계속 rank 에 비례해 좋아지므로, 남는 것은 전부 기하로 가는
편이 낫습니다.

측정된 shape 채널 유효 rank: **11.8 / 28** (학습 로그 `lrank`).

`w_latent_decorr`(VICReg/Barlow-Twins 공분산 항)를 0.1 → 0.3으로 올리면 rank
11.8 → 14.0, chamfer 0.245 → 0.243 — **아무것도 아닙니다.** rank 자체가 목표가
아니라는 증거입니다.

---

## 5. 디코더

### 5.1 folding 기반 (codec·gen 공통)

두 경로 모두 그룹을 이렇게 만듭니다:

```python
h        = cat([shape(28), contextualised group token(448)])
frame    = fold_head(h)                    # 3 log-scale + 3 axis-angle
aniso    = softplus(frame[:3]) + 1e-3      # 기하평균 정규화 → 크기는 앵커가 소유
R        = axis_angle_to_matrix(frame[3:6])

local    = fibonacci_ball(64)              # 고정 템플릿, 학습 안 함
local   += tanh(shape_xyz(h)/cap)*cap*gain # 잔차, cap 1.0, step 500부터 ramp
local   -= local.mean(slots)               # 잔차가 centroid를 못 움직이게
local    = local * aniso
offsets  = R @ local
points   = centroid_anchor + scale_anchor * offsets
```

| 설계 | 측정 근거 |
|---|---|
| 자유 MLP 대신 **고정 템플릿** | 기존 `shape_xyz` 출력 유효 rank **4.67** (미학습 랜덤 28→192 선형사상도 26.3). 맨 템플릿이 학습된 사상을 이깁니다: 0.294 vs 0.345 |
| 껍질이 아니라 **채워진 공** | 0.336 vs 0.367 |
| 축정렬이 아니라 **회전 프레임** | 0.301 vs 0.336 |
| 템플릿 스케일 √3 (√5 아님) | 초기 chamfer 0.299 vs 0.331 |
| 잔차 **tanh cap 1.0** | 무제한이면 192출력 잔차가 6출력 frame 헤드를 앞질러 이방성을 자기가 재현 (`aniso_lam2` 0.999 → 0.243) |
| 모든 가산 경로 **cap + ramp** | 잔차와 refine을 끄니 `direct_frac`이 **30.6** — 깊은 경로가 folding 출력의 97%를 빼고 있었습니다 |
| **프레임 전 mean-centre** | 그룹 centroid를 다른 무엇도 감독하지 않음. `cen` 0.00719 → 0.00143 |
| **det = +1 강제** | eigh 기저 + 자유 축부호 뒤집기는 그룹의 **50%**에서 반사행렬이 되는데, `R(axis-angle)·diag(softplus)`는 그걸 절대 못 만듭니다 |
| 헤드가 **shape + 토큰** 둘 다 읽음 | raw 28채널만 읽으면 compressor가 학습 불가: `dL/d(shape)` = 1e-2 vs latent 정규화항 1e+2 → `w_latent_std`가 인코드측 그래디언트의 99.5% 소유, shape 블록 유효 rank ~4/28 |

### 5.2 codec 경로 (decompressor 13.6M + decoder 49.3M)

```
z_compact
  │ group_embed: MLP([32 + anchor_pe → 448 → 448], in_norm, hidden_norm)
  │ window_blocks × 2  →  intra_blocks × 3
  │ token_out: residual_head([896 → 448 → 256]), zero-init   ← 깊은 경로
  │ + folding 경로 (5.1)
  ▼
z_raw_hat (1,048,576)
  │ unpack: Linear(256→256), 항등 초기화
  │ coarse = unpack(token) + centroid
  │ 10 × [ point self-attention → cross-attention(memory = 자기 토큰 + 3×3 이웃 셀) ]
  │ xyz_residual: tanh, 그룹 extent의 0.6배 예산, decoder_refine_alpha로 게이트
  ▼
pred (262144 × target_dim), presence
```

### 5.3 gen 경로 (33.1M) — 배포 경로

```
z_compact
  │ from_compact: MLP([32 → 448 → 448], in_norm, hidden_norm)
  │ window_blocks × 4
  │ group_split → group_blocks × 1
  │ + folding 경로 (5.1, codec과 동일한 입력)
  │ expand_xyz (extent의 0.35배), q_proj(xyz_pe)
  │ 4 × [ point self-attention → cross-attention ]
  │ xyz_residual (extent의 0.6배)
  ▼
gen_pred, gen_presence
```

`gen_detach_latent=True` — gen 손실이 `z_compact`로 역전파되지 않습니다.
(끄면 `hf_ratio`가 0.145 → 0.027로 무너지고 gen은 여전히 나빠졌습니다.)

**교사→학생 전달** (Milestone C에서 추가):
* 입력 대칭화 — gen도 `cat([shape, 문맥 토큰])`을 읽습니다 (이전엔 raw 28채널만)
* **basis distillation** — 최종 점만이 아니라 교사의 `frame`(6개)과 `local`(스케일 전 오프셋)을 직접 distill. `frame_distance()`는 scale은 log 공간, 회전은 행렬 Frobenius (axis-angle L1은 π에서 불연속)
* **anti-inflation** — `w_gen_p2g`(정밀도 단독), `w_gen_radius`(반지름 비). 대칭 chamfer 하나로는 radius 1.83×GT, p→g 1.294 / g→p 0.521을 못 봅니다

### 5.4 attribute 헤드 (stage `geometry`/`full`)

```
attr = attr_mlp(cat([h, xyz_pe(xyz_cond)]))
  ├─ head_scale   → 3   (bias −5.5)
  ├─ head_rot     → 4   (normalize, bias [1,0,0,0])
  ├─ head_opacity → 1   (bias −2.0)
  ├─ head_color   → 3   (SH DC)
  └─ head_sh      → 45  (stage full만)
```

`xyz_cond`는 **geometry-first 조건화**의 커리큘럼 대상입니다:

```
teacher forcing (p=1) → scheduled sampling (1→0) → inference conditioning (p=0)
```

디코더 안에서 **포인트 단위로** 섞고, forward는 1회, **eval은 항상 p=0**입니다.

### 5.5 스테이지

레이아웃이 prefix 순서라 각 스테이지는 단순 절단입니다:

```
xyz 3 | log_scale 3 | quat 4 | logit_opacity 1 | SH DC 3 | SH rest 45
```

| stage | target_dim | sh_dim | 용도 |
|---|---|---|---|
| `xyz` | 3 | 0 | 기하만 |
| **`geometry`** | **14** | 0 | **`sh_degree=0` 래스터라이저가 쓰는 전부** ← attribute 런 |
| `full` | 59 | 45 | + 뷰 의존 SH |

`geometry`는 이전에 `target_dim=59`인데 디코더가 14채널만 내보내 실행 자체가
불가능했습니다. `target_dim − sh_dim`으로 유도하게 고쳤습니다.

---

## 6. 래스터라이저 (`can3tok/render.py`)

`diff_gaussian_rasterization`, `/data/daeho/aabb/gaussian-splatting` 규약.

### 6.1 카메라

npz `state_t['camera']`에 fx, fy, cx, cy, R(3×3), T(3,) → **977×544, FoVx 80.2°**.
`image_path`는 `None`이라 **원본 사진이 없습니다** → 기준은 **GT 가우시안의 렌더**이고,
따라서 목표는 픽셀 진리가 아니라 **뷰 등가성**입니다.

### 6.2 규약 (틀리기 쉬운 곳)

| 필드 | npz 저장 형태 | 렌더에 필요한 형태 |
|---|---|---|
| `color` | **SH DC 계수** (−2.3 ~ 10.1) | `rgb = 0.2821·dc + 0.5`. 변환 없으면 **회색 안개** |
| `opacity` | 이미 활성화 (0~1) | 그대로 |
| `scaling` | 이미 exp 적용 | 그대로 |

### 6.3 손실

`photometric_loss = 0.8·L1 + 0.2·(1 − SSIM)` (3DGS 원 목적함수).
SSIM은 box window (avg_pool2d) — 학습 스텝 안에서 돌아야 하므로.

`amp_ctx()` **바깥**에서 실행합니다. CUDA 래스터라이저는 fp32 전용이고 bf16을 주면
에러가 아니라 **쓰레기 값**으로 통과합니다.

crop 샘플은 양쪽 다 거의 검은 화면이라 손실이 공짜 0이 됩니다 → `min_coverage`
미만이면 스킵하고 **스킵 비율을 로그에 찍습니다** (`rend L1/DSSIM@사용비율`).

### 6.4 augmentation과 카메라

`_augment`가 정규화 공간에서 `x_n → s·R·x_n + shift`. 월드 공간으로는
`x → s·R·x + t_w`, `t_w = center − s·R·center + shift·scale`.

```
R_wc' = R_wc · Rᵀ
T'    = s·T − R_wc·Rᵀ·t_w
```

이면 `x_c' = s·(x_c)`. 투영은 스케일 불변이고 가우시안 scale도 같은 s가 곱해졌으므로
**렌더가 완전히 동일**합니다. 검증: **98.16 dB** (쿼터니언 회전을 빼면 28.45 dB —
검증이 실제로 민감함).

### 6.5 멀티뷰

`perturb_camera_vector()`가 기존 카메라를 자기 축 기준 yaw/pitch 궤도 회전 + 소량
평행이동. 카메라 하나로는 3D 등가성을 증명할 수 없습니다 — 안 보이는 것은 제약이
없고, scale/opacity가 기하 오차를 흡수할 수 있습니다.

---

## 7. 측정된 한계 — 정보 예산

이 절이 나머지 전부를 지배합니다.

> **범위 주의.** 이 절의 모든 수치는 **28채널을 전부 기하에 쓰는 현재 배분**에 대한
> 것입니다. "예산의 최적점에 있다"는 말은 *xyz 재구성*에 대해서만 맞고, **full 3DGS
> latent가 최적이라는 뜻이 아닙니다.** 28채널이 appearance도 실어야 하면 기하 몫이
> 줄어들고 상한도 함께 내려갑니다 — §7.5.

```
채널/점 = 131,072 / 262,144 = 0.5      ← 두 제약이 정함, group size와 무관
```

g=64면 그룹당 32채널로 192개 숫자(64점×3)를, g=16이면 8채널로 48개를 기술합니다.
**비율이 같습니다.** 그룹 크기는 예산의 지렛대가 아닙니다.

### 7.1 rank가 정하는 상한 (`tools/oracle_shape_rank.py`)

템플릿 정렬 타깃 오프셋을 자기 주성분 k개로 투영 = **완벽한 선형 디코더의 상한**:

| rank k | ich | nn_unique | PSNR |
|---|---|---|---|
| 7 | 0.282 | 0.391 | 15.68 |
| 16 | 0.251 | 0.501 | 16.56 |
| **28** | **0.236** | **0.529** | **17.39** |
| 64 | 0.189 | 0.597 | 18.45 |
| 128 | 0.088 | 0.756 | 23.03 |
| 192 | 0.0001 | 0.997 | 81.82 |

**현재 모델: ich 0.251, nn_unique 0.51, template_erank 7.1, PSNR 17.35.**
shape 채널이 28개인데 rank-28 상한과 일치 — **이미 예산의 최적점에 있습니다.**

### 7.2 비선형이 도움이 되는가 (`tools/oracle_nonlinear_shape.py`)

192차원 오프셋에 28차원 병목 MLP AE, train 5장면(17,246 그룹) → held-out 3장면:

| method | nn_unique | ich | set |
|---|---|---|---|
| PCA-28 | 0.548 | 0.2349 | held-out |
| MLP AE-28 | 0.553 | 0.2401 | held-out |

**+0.005.** 벽은 선형 투영도, 손실도, 디코더 구조도 아니라 **예산 자체**입니다.

### 7.3 그런데 렌더는 갇히지 않습니다 (`tools/oracle_attr_compensation.py`)

rank-28 위치를 **고정**하고 attribute만 렌더 손실로 최적화, **held-out 포즈** 평가:

| | fit | held-out (미학습 3뷰) | SSIM |
|---|---|---|---|
| GT attribute | 16.86 | **17.05** | 0.461 |
| 적응, 1뷰 fit | 36.97 | 21.02 | 0.759 |
| 적응, **4뷰 fit** | 33.28 | **26.69** | **0.905** |

canonical baseline(정확한 262k GT 부분집합 + GT attribute)이 **27.8 dB**입니다.

**예산이 제한하는 것은 점집합의 1:1 재현이고, 렌더 품질이 아닙니다.**
빠진 점이 남긴 구멍은 이웃 가우시안이 더 크고 잘 정렬되어 채웁니다.

### 7.4 K를 줄이면 (제약상 하지 않음, 참고)

| K | 채널/점 | 그룹당 rank | 오라클 PSNR |
|---|---|---|---|
| 262,144 (현재) | 0.5 | 28 | 17.4 |
| 65,536 | 2.0 | ~124 | ~23 |
| 32,768 | 4.0 | ~192 | ~lossless |

출판된 Gaussian AE들(GaussianCube, TRELLIS/SLat, L3DG)이 ~32k에서 도는 이유입니다.

### 7.5 그런데 인코더가 attribute를 아예 보지 않습니다 (blocking)

`encoder.build_patch`는 `x[..., 0:3]`과 mask만 가져갑니다. `patch_dim = g*4`가
하드코딩되어 있고, 59채널 중 scale·rot·opacity·color·SH는 **`z_raw`에 들어가지
않으므로 `z_compact`에도 들어가지 않습니다.**

결과: 기하가 같고 색만 다른 두 장면은 **동일한 `z_compact`**를 받습니다. 디코더가
낼 수 있는 attribute는 전부 기하의 함수입니다.

**얼마나 심각한가** (`tools/oracle_attr_from_geometry.py` — 그룹 기하 196차원에서
그룹 attribute를 예측, train 5장면 → held-out 3장면):

| attribute | dim | held-out R² |
|---|---|---|
| log_scale | 3 | **0.227** |
| rot | 4 | **−0.248** |
| opacity | 1 | **−0.993** |
| color_DC | 3 | **−0.234** |
| sh_rest | 45 | **−0.253** |

R² ≤ 0 은 **데이터셋 평균보다도 못하다**는 뜻입니다. 즉 현재 인코더로는 `--stage
geometry`를 돌려도 rotation·opacity·color가 평균 수준으로 수렴하는 것이 상한입니다.
§7.3의 attribute 보상(+9.6 dB)은 **좋은 attribute가 존재한다**를 증명했을 뿐,
**`z_compact`가 그것을 담을 수 있다**를 증명하지 않았습니다. 현재 구조로는 담을 수
없습니다.

### 7.6 attribute **재구성**은 기하보다 비쌉니다 (`tools/oracle_attr_rank.py`)

GT xyz를 정확히 주고 attribute만 rank-k로 제한한 뒤 렌더(4뷰):

| rank k | ch/point | PSNR | scale | rot | opacity | color |
|---|---|---|---|---|---|---|
| 32 | 0.50 | 15.00 | 0.086 | 0.584 | 0.818 | 0.489 |
| 64 | 1.00 | 16.14 | 0.078 | 0.526 | 0.757 | 0.418 |
| 128 | 2.00 | 17.51 | 0.072 | 0.466 | 0.668 | 0.291 |
| 512 | 8.00 | 27.29 | 0.028 | 0.067 | 0.074 | 0.096 |

(오른쪽 4열은 채널별 상대오차.) 같은 0.5 ch/point에서 **기하 17.4 dB vs attribute
15.0 dB** — attribute가 더 압축하기 어렵습니다. within/global std가 0.83–0.93으로
점당 거의 독립이라 그룹화가 도움이 되지 않습니다. `scale`만 예외적으로 잘 압축됩니다
(상대오차 0.078 @ 1 ch/point) — 국소 밀도와 상관되기 때문입니다.

**GT attribute를 재구성하려면 ~8 ch/point가 필요하고, 그건 예산의 16배입니다.**

### 7.7 그런데 **등가**는 훨씬 쌉니다 (`tools/oracle_attr_code_size.py`)

§7.6은 잘못된 질문입니다. 필요한 것은 GT attribute 재현이 아니라 **같은 이미지를
내는 attribute**입니다. 적응된 attribute를 k차원 부분공간(공유 basis + 그룹당 코드
k개 — 채널 k개짜리 디코더가 선형 극한에서 표현 가능한 것)으로 제한하고, 기하는
rank-28(현재 모델)로 고정, **held-out 포즈**로 평가:

| appearance 코드 | ch/point | fit | **held-out** | SSIM |
|---|---|---|---|---|
| GT attribute (적응 없음) | — | 16.86 | **17.05** | 0.461 |
| **8 / group** | **0.12** | 25.56 | **21.43** | 0.771 |
| 16 / group | 0.25 | 28.57 | 22.95 | 0.827 |
| 32 / group | 0.50 | 31.23 | 23.80 | 0.862 |
| 64 / group | 1.00 | 33.38 | 24.33 | 0.875 |

**그룹당 8개 숫자로 +4.4 dB.** 그리고 빠르게 포화합니다 — 0.12 → 1.00 ch/point가
2.9 dB밖에 더 주지 않습니다. fit/held-out 격차는 코드가 클수록 벌어지므로
(k=8: 25.6/21.4, k=64: 33.4/24.3) **작은 코드가 더 강건합니다.**

재구성 1 ch/point = 16.14 dB, 등가 0.12 ch/point = 21.43 dB. **8배 적은 용량으로
5 dB 더 좋습니다.** 이 차이가 이 프로젝트의 목표를 정합니다.

### 7.8 그래서 예산이 닫힙니다 — z_compact 크기를 바꾸지 않고

```
32 ch/group = centroid 3 + log-extent 1 + shape 28
                                          └─ 재배분 대상
```

| 배분 | 기하 | appearance | 예상 held-out |
|---|---|---|---|
| 현재 | 28 → 17.4 dB 상한 | 0 | **17.05 dB** |
| 제안 | 24 → ~17.2 dB | 8 | **~21.4 dB** |

기하 **−0.2 dB**를 내주고 appearance **+4.4 dB**를 얻습니다 (§7.1의 rank→dB 표에서
24는 16과 28 사이 보간). **`z_compact`도 `z_raw`도 크기가 그대로입니다.**

전제 조건은 §7.5입니다 — 인코더가 appearance를 봐야 합니다. 지금은 `x[..., 0:3]`만
pack 하므로 이 8채널에 넣을 정보가 존재하지 않습니다.

### 7.9 이 측정들의 한계

* 오라클은 공유 basis + 그룹당 코드를 렌더에 **직접** 최적화합니다. 실제 디코더는
  그 코드를 인코더로부터 만들어야 하므로 **상한**입니다.
* `k=128`에서 래스터라이저가 CUDA illegal memory access로 죽었습니다. 코드가 커지면
  optimizer가 `scale`을 극단으로 밀어 타일 할당을 깨는 것으로 보입니다. 곡선이 이미
  포화 중이라 결론에는 영향이 없지만, **attribute 헤드에 scale 상한이 필요하다**는
  신호입니다.
* 장면 1개(`step_015830`), 뷰 4 fit / 3 held-out.

---

## 8. 손실

### 8.1 활성 항 (12개, 원래 19개에서)

각 항의 그래디언트 크기와 항간 코사인을 측정해 골랐습니다
(`tools/audit_losses.py`). 19개 항이 4개 독립 방향으로 붕괴했고, 클러스터 내
코사인이 0.75–1.00이었습니다.

| codec | 가중치(pretrain) | 대표하는 것 |
|---|---|---|
| `w_z_residual` | 40 (80) | 7항 ordered 클러스터, 그룹 내 그래디언트의 51.8% |
| `w_z_intra_chamfer` | 25 | 순열 불변, **packed** 복원 위 — 디코더 불필요라 latent 구간을 커버 |
| `w_intra_chamfer` | 14 | 디코더 출력 위 같은 것, 디코더 그래디언트의 99.3% |
| `w_z_raw` | 10 (30) | 4항 latent/mask 클러스터, 항간 코사인 1.00 |
| `w_chamfer` | 20 | 5항 global 클러스터, 디코더 그래디언트의 0.3% |
| `w_latent_std` | 2 | 채널별 std 밴드 [0.5, 3.0] |
| `w_latent_decorr` | 0.1 | shape 채널 상관행렬 비대각 |

gen: `gen_chamfer` 8, `gen_intra_chamfer` 10, `gen_xyz_residual` 24,
`gen_presence` 0.2, `distill` 30, `gen_basis` 20, `gen_p2g` 8, `gen_radius` 12.
그 외 `teacher_cycle` 6, `equiv` 1.0 (4스텝마다).

### 8.2 각 항의 조건수 (`tools/audit_render_direction.py`)

정답이 알려진 상황(타깃 + 알려진 노이즈)에서
`cos(−dL/dxyz, target − pred)`와 그래디언트를 받은 점의 비율:

| 오차 | render(1뷰) | render(4뷰) | chamfer | intra_chamfer | xyz_residual |
|---|---|---|---|---|---|
| 0.6 px | 0.134 @24% | 0.155 @35% | 0.394 @25% | 0.682 @100% | **0.866** @100% |
| 6.0 px | 0.054 @23% | 0.064 @33% | 0.405 @25% | 0.662 @100% | **0.866** @100% |

**렌더 손실은 위치에 ill-conditioned입니다.** 가우시안 화면 반지름 중앙값이 1.40 px인데
모델 오차는 ~6 px — 자기 footprint의 4배라 이미지 그래디언트가 엉뚱한 위치에서
샘플링됩니다(transport 문제). 멀티뷰로 0.02만 오르므로 깊이 blindness가 원인이 아닙니다.

vanilla 3DGS는 이미지 손실만으로 means를 최적화하지만 **adaptive density control
(clone/split/prune)**에 의존합니다. 이 설계는 262144 고정 예산이고 densification이
없습니다.

**그래서 렌더 손실은 기하 단계에서 끕니다.** 켜는 곳:

| 위치 | 이유 |
|---|---|
| attribute | opacity/color/scale은 **이미 차지한 같은 픽셀**을 바꿈 — 수송 불필요 |
| canonicalizer | 결정 변수가 per-Gaussian gate. "보이는가"는 이미지 손실의 질문 |
| 평가 (항상) | `tools/render_compare.py` |

### 8.3 크기 (`tools/audit_render_scale.py`)

가중치를 유추로 정하면 자릿수를 놓칩니다:

| term | weight | \|w·dL/dxyz\| | share |
|---|---|---|---|
| `w_render` | 6.00 | 1.234e+02 | **99.9%** |
| `w_chamfer` | 20.00 | 1.182e-01 | 0.1% |
| `w_intra_chamfer` | 14.00 | 3.515e-02 | 0.0% |
| `w_xyz_residual` | 12.00 | 1.353e-02 | 0.0% |

래스터라이저 그래디언트는 단위 가중치당 chamfer의 **~1000배**입니다. 30% 점유
목표면 `w_render ≈ 0.0035`. **단위가 다른 항을 추가할 때마다 이 측정이 필요합니다.**

### 8.4 제거한 항 (근거 있음)

* `group_cov` — 모멘트 매칭은 노이즈로 만족됩니다 (정밀도 0.238 → 0.439, 커버리지 0.540 → 0.345, 대칭 chamfer 불변)
* `direct_ratio`
* `slot_sort=pca` — §2

### 8.5 절대 게이트로 쓰면 안 되는 지표

`rmse`는 extent가 가장 큰 ~1% 그룹이 지배하고, `rel_offset`은 슬롯 순서 의존입니다.
둘 다 디코더가 출력을 **치환만 해도** 움직입니다: `rel_offset` 0.758 → 1.011,
`rmse` 0.01571 → 0.01453인데 점 *집합*은 그대로였습니다 (0.393 → 0.396).

---

## 9. 학습 스케줄

20,000 스텝, 4-GPU DDP, rank당 batch 1, AdamW lr 2e-4 (warmup 500, cosine → 1e-5),
wd 1e-4, grad clip 1.0, bf16. ~3.9 s/it, rank당 15.6 GB.

| step | 켜지는 것 |
|---|---|
| 0–500 | compressor ↔ decompressor 왕복만. 잔차 gain 0 → 맨 템플릿 × 학습된 frame |
| 500–1000 | folding 잔차 gain 0 → 1 |
| 2500 | **decode 시작** (`geo_weak_scale` 0.15). `w_intra_chamfer` 자체 500스텝 ramp |
| 4000 | `latent_end`. geo 0.15 → 1.0 (10000까지), latent 가중치 감쇠, `decoder_refine_alpha` 0 → 1 |
| 5000 | gen 시작, 1500스텝 ramp |
| 10000 / 12000 / 15000 | geo full / encoder residual 해제 / codec lr ×0.4 |

`attr_teacher_prob`: `attr_start + attr_force_steps`부터 `attr_anneal_steps` 동안 1 → 0.
`w_render`: `render_start`부터 `render_ramp_steps` 동안 0 → 목표값.

---

## 10. 평가

| 지표 | 무엇을 보는가 |
|---|---|
| `intra_chamfer_rel` | **주 점수.** 순열 불변 그룹 내 chamfer / 평균 그룹 반지름. GT 자체 최근접 간격 0.191 = "1.0×" |
| `nn_unique` | 예측점이 도달하는 **서로 다른** GT점의 비율. 중복을 보는 유일한 지표 |
| `template_erank_pred` | 디코더가 만드는 **서로 다른 그룹 shape 수** (스펙트럼 엔트로피의 exp) |
| `thread_pred/gt` | 연속 슬롯 거리 / 반지름. 0에 가까우면 하나의 실로 붕괴 |
| `aniso_lam2` | 그룹당 이방성. 직선 붕괴 탐지 |
| `latent_erank` | shape 채널 유효 rank |
| `hf_ratio` | latent의 절반 Nyquist 위 2D 스펙트럼 에너지 (diffusability 대리) |
| 렌더 | `render_compare.py`: orig / canon / codec / snap 4-way, PSNR·SSIM·nn_unique |

**eval 세트 주의**: `eval_val_indices`에 3DGS 학습 530스텝 이전 프레임 3개가 있습니다
(idx 0은 opacity 최대 0.140, SH 전부 0). 점공간 지표는 영향 없지만(0.6–2.1%, 반지름
정규화라 opacity/SH를 안 봄) **렌더 지표에는 쓸 수 없습니다** — 원본조차 안개라
`codec_snap`이 `codec`보다 낮게 나옵니다(수렴 프레임과 부호 반대).

---

## 11. 아직 없는 것

| | 상태 |
|---|---|
| attribute 단계 실규모(262k) | 배선 완료, smoke만 통과 |
| 멀티뷰 렌더 손실 | 구현 완료, 미사용 |
| render-aware canonicalizer | 미구현. 현재 `canon`은 sampler top-K baseline (27.8 dB) |
| **learned `H_rich` 인코더** | **미구현 — 그리고 attribute 단계를 막고 있습니다 (§7.5).** 현재 pack은 `x[..., 0:3]` + mask 뿐이라 appearance가 latent에 0비트 들어갑니다. 이것이 지금 가장 중요한 구조 변경입니다 |
| 계층적 cell attention, EMPTY 토큰 + 마스크 | 미구현 (빈 셀 42%) |
| full SH 커리큘럼 (deg 0 → low → full) | 미구현 |
| 프레임간 latent permutation 안정성 | **미측정.** Morton 순서가 프레임마다 재계산되므로 셀 *i*가 *t*에는 벽, *t+1*에는 의자일 수 있습니다. autoencoder에는 무해하지만 world model에는 치명적일 수 있습니다 |
| DIAMOND가 만든 latent 디코딩 | 미측정. 현재 validation은 held-out npz의 encode-then-decode뿐 |

---

## 12. 도구

| | 무엇을 답하는가 |
|---|---|
| `tools/render_compare.py` | 체크포인트 → 4-way 렌더 비교 + `nn_unique` |
| `tools/diag_duplication.py` | 중복이 어느 단계(coarse/full/gen)에서 생기는가 |
| `tools/oracle_shape_rank.py` | rank가 정하는 상한. rank → ich/nn_unique/PSNR |
| `tools/oracle_nonlinear_shape.py` | 비선형이 선형 상한을 넘는가 (held-out 분리) |
| `tools/oracle_attr_compensation.py` | attribute가 rank 한계를 보상하는가 (held-out 포즈) |
| `tools/audit_losses.py` | 파라미터 공간 항별 그래디언트 크기 + 항간 코사인 |
| `tools/audit_render_scale.py` | 점 공간 항별 그래디언트 크기 (262k 실규모) |
| `tools/audit_render_direction.py` | 각 항이 **정답 방향**을 가리키는가 |

`audit_losses.py`의 `TERMS`는 **두 번** 쓰입니다 — 측정할 항 선택과 **나머지 전부를
0으로 만들기**. 목록에서 빠진 항은 절대 0이 되지 않고 모든 행을 오염시킵니다.
실제로 `w_latent_decorr`가 빠져서 4개 항이 전부 동일 크기, 코사인 1.00으로 나온 적이
있습니다.
