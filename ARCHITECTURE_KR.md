# can3tok 인코더–디코더 구조 설명

3D Gaussian Splatting 장면을 고정 크기 latent로 압축했다가 Gaussian으로 복원하는
오토인코더. latent는 DIAMOND world model의 입력이므로 **모양과 크기가 고정**이다.

---

## 1. 고정 제약

| 항목 | 값 | 이유 |
|---|---|---|
| `max_points` (디코더 출력) | 262,144 | 요구사항 |
| `z_compact` | 32 × 64 × 64 = **131,072** | world model 인터페이스 |
| Gaussian 하나당 latent | **0.50개** | 위 둘에서 파생 |
| `max_input_points` (인코더 입력) | 589,824 | 새로 분리한 값 (아래 §3) |

원본 3DGS는 점당 59개 파라미터(xyz 3, scale 3, rot 4, opacity 1, SH 48)를 가지므로
262,144 × 59 = 15,466,496개 숫자를 131,072개로 줄인다 — **118배 압축**, 고정 크기,
엔트로피 코딩 없음. 현재 학습은 SH를 DC만 사용해 `target_dim = 14`(xyz 3 + scale 3 +
rot 4 + opacity 1 + SH DC 3)이다.

---

## 2. 전체 데이터 흐름

```
npz (Gaussian 8만~59만 개, 스냅샷마다 다름)
  │
  ├─ 데이터 파이프라인: Morton 정렬 → stratified/importance 선택
  │     ├─ target  : 262,144 슬롯  (손실이 비교하는 대상)
  │     └─ enc_input: 589,824 슬롯 (인코더가 읽는 대상)
  │
  ▼
인코더  PatchPackEncoder                        805,968 params
  │  4096 그룹 × 144점  →  그룹당 patch 256개 숫자
  │  (항등 pack 또는 GroupPointPooler)
  ▼
z_raw   32 × 256 × 128 = 1,048,576
  │
  ▼
compressor  StagedCompressor                  17,038,624 params
  │  token_embed → intra attention → token_merge → cell_mix
  │  → window attention → group_head → mid(256) → budget heads
  ▼
z_compact   32 × 64 × 64 = 131,072            ← world model 인터페이스
  │
  ▼
decompressor  StagedDecompressor              13,573,372 params
  │  folding 으로 pack 재합성 + appearance 슬라이스 통과
  ├──────────────► ctx["appearance"]  (그룹당 16채널)
  ▼
z_raw_hat   32 × 256 × 128
  │
  ├─► decoder  CodecDecoder                   49,684,573 params  → xyz 262,144개
  │      (pack 좌표 직접 읽기 + 유계 잔차)
  │
  └─► attr_decoder  AttributeDecoder           3,611,662 params  → scale/rot/opacity/color
         (cross-attention, 위치는 detach 되어 들어옴)
  │
  ▼
gen_decoder                                   33,524,390 params
  (배포 경로: world model 이 만든 z_compact 에서 직접 Gaussian 생성)

총 118,238,589 params
```

---

## 3. 인코더

### 3.1 그룹 분할

262,144점을 Morton 정렬 후 **4096 그룹 × 64점**으로 나눈다. 4096은 `z_compact`의
셀 수(64×64)와 일치하며, 그룹 하나가 latent 셀 하나에 대응한다.

그룹 내 슬롯 순서는 화이트닝된 공분산 고유좌표계에서 고정 템플릿에 Hungarian 배정으로
정한다(det=+1 강제). 슬롯 *i*가 그룹마다 일관된 의미를 갖게 하기 위한 것이다.

### 3.2 patch 구성

그룹당 **256개 숫자**:

```
[  0:192] 그룹 중심을 뺀 xyz  (64점 × 3)     ← 정확한 좌표
[192:256] aux 64개                          ← GroupAttributeEncoder 출력
```

`patch_dim == token_dim == 256`이고 `pack`은 **얼어붙은 256×256 항등행렬**(bias 0)이라,
xyz는 학습 파라미터 0개를 거쳐 `z_raw`에 그대로 복사된다. 이것이 이 설계의 핵심 성질이며
동시에 한계다(§8 참조).

### 3.3 GroupAttributeEncoder (59,073 params)

그룹의 64점 × 11속성 = **704개 숫자를 64개로 압축**해 aux 블록에 넣는다.

```
채널별 데이터셋 표준화 → MLP(11→64→64) → +슬롯 임베딩 → self-attention → Linear(64→1)
```

슬롯 임베딩이 없으면 모듈 전체가 permutation-equivariant가 되어 **모든 슬롯이 같은
함수를 계산**한다. 격리 측정에서 held-out R²가 0.158 → **0.984**로 바뀐 지점이다.

### 3.4 GroupPointPooler (480,384 params) — 새로 추가

`max_input_points > max_points`일 때만 활성화된다. 항등 pack은 인코더가 디코더와
정확히 같은 수의 점을 읽을 때만 정의되므로, 더 많이 읽으려면 축약을 **학습**해야 한다.

```
그룹의 144점 × (지역 xyz 3 + 속성 11) → MLP(14→128→128)
  → 학습형 query 8개가 cross-attention
  → Linear(8×128 → 256) = patch
```

3DShape2VecSet·Can3Tok·COD-VAE가 점 집합을 고정 latent 집합으로 바꾸는 방식과 같다.
**무엇을 남길지 네트워크가 결정한다** — 기존에는 학습 이전의 고정 규칙이 결정했다.

### 3.5 residual 분기 (200,718 params)

`z_tok + tanh(gate) · residual(feat)` 형태의 보정 경로. `encoder_residual_start`
이후에만 켜진다.

---

## 4. compressor (17,038,624 params)

`z_raw`(32×256×128) → `z_compact`(32×64×64), **8배 압축**.

```
tokens (4096그룹 × 8토큰 × 32채널)
  → 그룹 extent 로 xyz 정규화        (그룹 반지름이 장면 안에서 854배까지 차이남)
  → token_embed  32 → 448
  → intra attention ×3               (셀 안의 토큰끼리)
  → token_merge  8×448=3584 → 448    (MLP, 옵션으로 attention 풀링)
  → cell_mix, window attention ×2    (셀 사이, 창 8)
  → group_head(group_vec, cell_vec, anchor_pe) → 448
  → mid: 448 → 256 → 256             (LayerNorm + LeakyReLU)
  → budget heads
```

### 채널 예산 (그룹당 32채널)

| 블록 | 채널 | 내용 |
|---|---|---|
| centroid | 4 | 그룹 중심 3 + 그룹 스케일 1 (해석적 앵커 + 유계 보정) |
| occupancy | 0 | 미사용 |
| shape | 12 | 그룹 내 형태 |
| **appearance** | **16** | 외형 코드 |

centroid는 해석적 앵커에 `tanh` 보정을 더한 형태라 반쯤 결정돼 있고, 채널 3은 그룹
스케일을 그대로 싣는다.

---

## 5. decompressor (13,573,372 params)

`z_compact` → `z_raw_hat`. 위치를 **folding으로 재합성**한다.

```
direct_xyz  = fold(shape 채널) · scale          ← 템플릿 변형
learned_xyz = (direct + deep_residual) · scale
z_raw_hat   = [learned_xyz | 나머지 채널]
appearance  = z_compact 의 appearance 슬라이스를 그대로 통과
```

appearance가 파라미터를 거치지 않고 지나가므로, render loss의 그래디언트가
decompressor 가중치에는 **정확히 0**으로 도달한다(측정됨).

---

## 6. 두 개의 디코더

의도적으로 분리돼 있고, 구조도 다르다.

| | 기하 디코더 `CodecDecoder` | 외형 디코더 `AttributeDecoder` |
|---|---|---|
| 파라미터 | 49,684,573 | 3,611,662 |
| 층/폭 | 10 layers, dim 448 | 4 layers, dim 256 |
| 방식 | 슬롯 query + memory cross-attention | 슬롯 임베딩 + 코드 cross-attention |
| 출력 | `out["pred"]` (xyz) | `out["attr_pred"]` (scale/rot/opacity/color + 위치 nudge) |

### 6.1 기하 디코더의 위치 생성

```python
patch  = unpack(tokens)
coarse = patch[..., :192] + centroid      # z_raw_hat 의 좌표를 직접 읽음
residual = tanh(xyz_residual(h))
xyz = coarse + residual · scale · residual_scale · refine
```

즉 **직접 읽기 + 유계 잔차**이지 순수 folding이 아니다.

### 6.2 외형 디코더

```python
attr_pred = attr_decoder(pred[..., :3].detach(),   # 기하는 detach
                         ctx["appearance"],         # 그래디언트 통과
                         ctx["scale"].detach())
```

기하 입력이 detach 되어 있어 **render loss가 기하 디코더에 정확히 0의 그래디언트**를
준다(측정 확인). 두 디코더의 분리는 이 한 줄이 강제한다.

헤드 초기값: `head_scale.bias = −7.58`(±3.0 tanh 밴드), `head_opacity.bias = −2.13`,
`head_nudge`는 zero-init에 `tanh` 캡 0.15.

---

## 7. 손실 체계

| 항목 | 가중치 | 대상 |
|---|---|---|
| `w_render_attr` | 68.0 | `attr_pred` 렌더 vs 기준 이미지 |
| `w_z_residual` | 40.0 | `z_raw_hat` 잔차 (xyz 접두부만) |
| `w_z_intra_chamfer` | 25.0 | z공간 그룹 내 chamfer |
| `w_chamfer` | 20.0 | 점 집합 chamfer |
| `w_intra_chamfer` | 14.0 | 그룹 내 chamfer |
| `w_z_raw` | 10.0 | `z_raw_hat` L1 (xyz 접두부만) |
| `w_scale/rot/opacity/color` | 6/3/4/6 | GT 파라미터 앵커 (감쇠) |
| `w_gen_*` | — | student 분기 |

- `geom_pack_dim`이 z공간 손실을 **xyz 접두부로 제한**한다. aux 블록은 학습되는
  코드이므로 비교 대상이 될 수 없다.
- `quat_loss`는 `|⟨q,q̂⟩|`로 부호 모호성을 처리한다(q와 −q는 같은 회전).
- attribute 앵커는 `render_start`부터 `attr_param_decay_steps`에 걸쳐
  `attr_param_floor`까지 감쇠한다.

### render loss의 기준 이미지

두 가지를 지원한다.

1. **GT Gaussian 렌더** (기본) — 타깃 Gaussian을 같은 카메라로 rasterize
2. **실제 사진** (`--photo_map`) — npz 카메라를 COLMAP 포즈로 대조해 매칭

301장의 실제 사진과 카메라가 `assets/view_pool.json`에 있고, `--extra_real_views`로
매 스텝 새 뷰를 뽑는다. npz↔사진 대응은 3000개 전부 포즈 오차 1e-15로 확인했다.

---

## 8. 학습 스케줄

```
latent(0~4000) → geo(~10000) → joint(~15000) → gen(~20000)
attr_start 2500 · render_start 2700 · gen_start 5000
encoder_residual_start 12000 · late_codec_start 15000
```

옵티마이저 그룹은 7개(`encoder`, `attr_enc`, `compressor`, `decompressor`,
`decoder`, `gen`, `attr`)이고 그룹별 학습률 배율이 스케줄에 따라 바뀐다.
`attr_enc`가 `encoder`와 분리돼 있는 이유는 §9-5 참조.

**동결 가드**: 매 로그마다 그룹별 `‖Δp‖`를 찍고, 학습률이 0이 아닌데 3회 연속
움직이지 않으면 `[FROZEN]` 경고를 낸다. 이 프로젝트가 같은 유형에 다섯 번 물렸기 때문이다.

---

## 9. 발견하고 고친 결함

전부 "손실은 정상적으로 로그에 찍히는데 학습만 안 되는" 유형이었다.

1. **permutation equivariance** — `GroupAttributeEncoder`의 모든 슬롯이 같은 함수를
   계산. 슬롯 임베딩 추가로 격리 R² 0.158 → 0.984.
2. **이질 채널 LayerNorm** — 11개 채널에 걸친 정규화가 점의 색·불투명도 편차를 삭제.
   R² 0.998 → 0.747 손실. 채널별 표준화로 교체.
3. **zero-init 출력층** — `out.weight`가 0이라 첫 스텝에 하류 그래디언트 0.
4. **`ap.detach()`** — 목적함수 85%인 render loss가 인코더·compressor에 **정확히 0**을
   전달. appearance만 detach 해제(기하 격리는 유지).
5. **인코더 학습률 0** — `if not flags["encoder_residual"]: scales["encoder"] = 0.0`이
   인코더 **그룹 전체**를 얼림. 두 체크포인트에서 모든 인코더 텐서가 비트 단위로 동일했다.
   `attr_enc`를 독립 그룹으로 분리해 해결.
6. **residual 이중 zero-init 안장점** — `tanh(gate)·residual(feat)`에서 두 인자가 모두
   0이라 양쪽 그래디언트가 정확히 0. 어떤 학습률에서도 탈출 불가. gate를 1e-2로 초기화.
7. **뷰 샘플링 시드 고정** — `persistent_workers=True`라 `set_epoch`가 워커에 전파되지
   않아 각 스냅샷이 평생 같은 뷰만 봄. 매 방문 새 뷰로 변경.

---

## 10. 측정으로 확인한 병목과 반박된 가설

### 확인된 것

| 항목 | 크기 | 근거 |
|---|---|---|
| **인코더–latent–디코더 사상** | **9.4 ~ 12.1 dB** | Representation oracle (아래 §10.1) |
| **셀 의미의 시간 불일치** | **치명 (world model 용도)** | Morton chunk 13.9배 → 고정 anchor 0.13배 (§10.2) |
| **뷰 다양성** | **+4.7 dB** | 총 렌더 800회 고정. 고정 4뷰 +0.02 dB vs 매 스텝 새 뷰(풀 64) +4.67 dB |
| **속성 (동시에 맞아야)** | 3.3 dB | 하나만 GT로 바꾸면 −0.5~+0.2, 하나만 모델로 남기면 −1.5~−2.7 |
| **입력 서브샘플링** | 2.5 dB | 예산 초과 스냅샷 8개 평균, 최대 5.32 dB |
| 커버리지 편향 | 2.0 dB | 모델이 고른 GT점 17.66 vs 무작위 같은 수 19.65 |

### 10.1 Representation oracle — 262k 출력은 충분하다

네트워크를 전부 빼고 262,144개 Gaussian을 `nn.Parameter`로 두고 teacher(스냅샷 전체
Gaussian)의 렌더에 맞췄다. 매 스텝 새 뷰를 뽑고, held-out 뷰 8장으로 평가했다.

| 스냅샷 | teacher | 무작위 서브샘플 시작 | 800스텝 후 held-out |
|---|---|---|---|
| step_005070 | 519,086점 | held-out 21.49 dB | **25.74 dB** |
| step_015830 | 337,946점 | **held-out 28.09 dB** | 25.42 dB |

`step_015830`은 **최적화조차 필요 없다** — GT를 무작위로 262k개 뽑기만 해도 28.09 dB다.
(최적화가 오히려 나빠진 것은 학습률 과다이며, 시작값만으로 결론이 난다.)

같은 기준(teacher 렌더)에서 **현재 모델은 16.02 dB**다. 따라서

> **격차 9.4 ~ 12.1 dB 전부가 인코더–latent–디코더 사상이다.**
> 출력 개수, Gaussian 표현력, SH 차수는 모두 병목이 아니다.

### 10.2 latent 셀의 시간적 일관성 — Morton chunk 의 치명적 결함

`z_t[j]`와 `z_{t+1}[j]`가 같은 공간을 가리켜야 world model 이 `z_t → z_{t+1}`을 배울 수
있다. Morton 순서를 4096×64로 자르면 densification 으로 점이 추가될 때 chunk 경계가
전부 밀린다.

**점 개수가 변하는 인접 스냅샷 쌍**에서 셀 중심 이동 / 그룹 반지름:

| iter | 점수 변화 | 증가율 | 비율 |
|---|---|---|---|
| 680→810 | 116,193 → 132,930 | 14.4% | **14.4배** |
| 890→940 | 132,930 → 142,668 | 7.3% | **12.1배** |
| 1030→1250 | 152,758 → 176,103 | 15.3% | **16.5배** |

**평균 13.9배.** 같은 인덱스의 셀이 완전히 다른 공간을 의미한다.

> 주의: 점 개수가 **동일한** 쌍(114,842 → 114,842)만 고르면 0.1배로 나온다. densification
> 이 없는 구간이라 경계가 안 밀리기 때문이다. 검증 대상을 잘못 고르면 문제가 안 보인다.

**고정 anchor 로 해결된다.** 장면당 4096개 anchor 를 한 번만 만들어 모든 스냅샷에서
동일하게 쓰면:

| 그룹 정의 | 셀 중심 이동 / 그룹 반지름 | 그룹 반지름 |
|---|---|---|
| Morton chunk (기존) | **13.9배** | 0.0022 |
| FPS anchor | 0.05배 | 1.18 ← 너무 넓음 |
| **k-means anchor (채택)** | **0.13배** | **0.064** |

FPS 는 공간을 균일하게 덮으므로 점이 몰린 영역에서 anchor 하나가 지나치게 넓어진다.
k-means 는 점 분포를 따라가므로 **시간 일관성 107배 개선**과 조밀함을 동시에 만족한다
(빈 anchor 0개, 그룹당 26~34점). `assets/scene_anchors.npy` 에 저장돼 있다.

### 10.3 SH 차수 — 병목이 아니다

npz 에는 `color`(3, DC)와 `sh`(45, 고차항)가 **따로** 있고, `render_loss` 는 GT 와 예측
**양쪽 모두** `sh_dc_to_rgb(color)` 만 쓴다. 즉 "target 은 full SH, 예측은 DC" 같은
mismatch 는 없다.

그리고 고차 SH 의 실제 기여가 거의 없다 — `sh(45)` 의 평균 크기가 **0.043**, `dc` 는
**1.116** 으로 26배 차이다. 이 데이터는 최적화 중간 스냅샷이라 고차 SH 가 아직 학습되지
않았다. 1차 SH 를 배열·부호 4가지 조합으로 모두 넣어 봐도 DC-only(23.99 dB)를 넘지
못한다(23.51~23.98). **DC 만 쓰는 현재 선택은 이 데이터에서 옳다.**

### 반박된 것

| 가설 | 반박 근거 |
|---|---|
| latent 용량 부족 | 압축을 완전히 제거해도 13.71 → 11.97 dB (개선 없음) |
| latent 활용률(랭크) | decorr로 34% → 83%로 올려도 PSNR 변화 없음 |
| folding 붕괴 | 예측이 GT보다 **더** 균일 (그룹내 최근접/반지름 0.118~0.193 vs GT 0.000~0.107) |
| 빈 그룹 낭비 | 빈 그룹 0%인 장면도 랭크 9.56 |
| 디코더 재합성 | chamfer 기준 ② 0.207 → ③ 0.214 (거의 무해) |
| 중복/일대일 대응 | 완전 제거해도 +0.03 dB |
| 오차 상위 꼬리 | 모든 점을 GT 위치로 옮겨도 −0.06 dB |

**교훈**: 간접 지표(랭크, 슬롯 오차, 그룹내 분산)로 내린 진단이 연달아 틀렸다.
목표 지표(렌더 PSNR)에 대한 인과를 직접 재야 한다.

---

## 11. 현재 성능과 상한

사진 기준 held-out PSNR:

| | PSNR |
|---|---|
| 현재 모델 | 13.4 ~ 14.6 dB |
| GT Gaussian (입력 자체) | **19.84 dB 평균** (최소 9.83, 최대 22.86) |

**GT 자체가 평균 19.84 dB**다. 이 데이터는 3DGS 최적화 **중간 스냅샷**이라 수렴하지
않았고, iteration 구간별로 18.56 / 19.52 / 20.39 / 21.25 dB다. 따라서 사진 기준
21~28 dB는 이 데이터로 도달할 수 없고, 현실적 목표는 **19~20 dB**다.

---

## 12. 데이터

- **한 장면의 3DGS 최적화 스냅샷 3000개** (여러 장면이 아님). 2700/300 분할도 스냅샷 분할이다.
- 점 개수: 최소 81,140 / 중앙값 236,562 / **최대 565,017**. **40%가 262,144를 초과**한다.
- 실제 사진 301장(977×544), COLMAP 카메라. 증강 시 `transform_camera_vector`가 카메라를
  같은 월드 변환으로 옮겨 대응이 유지된다(증강 켠 상태에서 +11.84 dB 차이로 확인).

---

## 13. 문헌 대비 위치

| 모델 | 점 → latent 할당 | Gaussian당 latent |
|---|---|---|
| [3DShape2VecSet](https://arxiv.org/abs/2301.11445) | 학습형 query + cross-attention | — |
| [COD-VAE](https://arxiv.org/abs/2503.08737) | 2단계 attention 다운샘플링 | — |
| [Can3Tok](https://arxiv.org/html/2508.01464) | 학습형 canonical query | **0.41** |
| [GaussianCube](https://gaussiancube.github.io/) | Optimal Transport 배정 | ~14 |
| [L3DG](https://arxiv.org/abs/2410.13530) | sparse conv + VQ | — |
| **can3tok (이 모델)** | **Morton 하드 분할 → 학습형 풀링** | **0.50** |

예산은 기반 논문 Can3Tok(0.41)과 같은 수준이다. Can3Tok은 L2 파라미터 손실만 쓰고
PSNR을 보고하지 않으며 "high-frequency details are washed out"을 한계로 명시한다.
[HAC](https://arxiv.org/html/2403.14530v2)·FCGS 계열의 75~100× 압축은 **가변 비트율
엔트로피 코딩**이고 렌더 시점에는 anchor feature를 full dimension으로 복원하므로,
고정 크기 dense latent를 쓰는 이 설계와는 다른 문제를 푼다.

FCGS는 geometry 파라미터(opacity·scale·rotation)를 AE에 통과시키면 붕괴한다고 보고하고,
그것들은 AE를 거치지 않고 직접 양자화한다.

---

## 14. 남은 작업

### 1순위 — `data.py` 를 anchor 기반 그룹 배정으로 교체

`assets/scene_anchors.npy` 는 만들었지만 **`data.py` 는 아직 Morton chunk 를 쓴다.**
`_pack_slots` · `patch_anchors` · 슬롯 정렬이 연결돼 있어 신중히 해야 한다.
world model 용도의 **전제 조건**이며, 재구성 PSNR 과 무관하게 반드시 필요하다
(§10.2 — 이 결함은 어떤 재구성 지표에도 나타나지 않는다).

anchor 로 바꾸면 `GroupPointPooler`(§3.4)와 그대로 호환된다. 그룹 정의를
"Morton 순서 chunk" 에서 "고정 anchor 근방" 으로 바꾸기만 하면 된다.

### 2순위 — 9~12 dB 격차의 소재 특정

§10.1 이 격차가 인코더–latent–디코더에 있다는 것까지 확정했다. 그 안에서 어디인지는
아래 두 실험이 가른다. **추측하지 말 것** — §10 의 반박 목록이 그 결과다.

- **Direct decoder oracle**: 인코더를 빼고 스냅샷별 `z_compact` 를 `nn.Parameter` 로
  두고 디코더만 학습. Representation oracle(25~28 dB)에 근접하면 latent+디코더는
  충분하고 **인코더가 병목**. 근접하지 못하면 **디코더 또는 131k latent** 가 병목.
- **Auto-decoder**: 스냅샷 20~50개에 각각 학습형 `z_i` 를 배정하고 디코더는 공유.
  성공하면 인코더/토큰화가 병목이라고 거의 확정된다.

### 3순위 — 구조 단순화

배포 경로(`gen_decoder` 33.5M)와 검증 경로(decompressor + 두 디코더 66.9M)가
이원화돼 있고 실제로 쓰는 것은 전자다. oracle 결과가 나오면 **실제 쓰는 경로 하나만**
남기는 것이 맞다. 118M 파라미터에 중간 표현과 손실이 너무 많아, 하나가 실패해도 다른
손실이 정상적으로 감소해 "학습되는 것처럼" 보인다 — §9 의 결함 7개가 전부 그렇게 숨었다.

### 4순위 — 손실 재설계

GT 1:1 앵커는 폐기해야 한다. 200만~300만 점 장면에서 26만 개와 1:1 매칭은 정의되지
않고 빈 공간을 만든다. 측정상 뷰가 충분하면 render loss 만으로도 속성이 결정되므로
(64뷰에서 +4.2 dB), 대응 없는 감독(render + 약한 Chamfer)으로 가는 것이 맞다.

누적 뷰 수 = `(max_steps / 훈련샘플수) × 스텝당 뷰`. 현재 약 30이고 오라클이 요구한
값은 64다. **스텝당 뷰를 늘리는 것보다 서로 다른 뷰를 많이 보는 것**이 중요하다
(같은 렌더 예산에서 고정 4뷰 +0.02 dB vs 매 스텝 1장 새 뷰 +4.67 dB).

---

## 15. 다음 세션을 위한 요약

**확정된 것**
- 262k 출력, SH 차수, latent 용량·랭크, 점 위치 정확도 — 전부 병목이 **아니다**
- 격차 9~12 dB 는 인코더–latent–디코더 사상에 있다
- Morton chunk 는 world model 용도로 쓸 수 없다 (셀 의미가 매 스냅샷 바뀜)

**적용된 수정**
- 고정 k-means anchor 생성 (`assets/scene_anchors.npy`) — 아직 `data.py` 에 연결 안 됨
- `GroupPointPooler` + `max_input_points` 분리 — 구현 완료, 학습 검증 안 됨
- 매 방문 새 뷰 샘플링 — 구현 완료
- §9 의 결함 7개 — 수정 및 검증 완료

**방법론**
간접 지표(랭크, 슬롯 오차, 그룹내 분산, 유효랭크)로 내린 진단이 이 프로젝트에서 일곱 번
연속으로 틀렸다. 목표 지표(렌더 PSNR)에 대한 인과를 직접 재고, oracle 로 상한을 먼저
확정한 뒤 그 gap 이 어디서 생기는지 사다리를 타고 내려가는 방식만 신뢰할 수 있다.
