# 현재 학습의 gradient 폭증·정체·디테일 손실 분석 — 2026-09-25

## 0. 결론

**현재 문제는 세 가지를 분리해서 고쳐야 한다.**

1. **과거 B1의 실제 발산:** 34k 이후 gradient와 기하 오차가 증가하고 사진 PSNR이 붕괴했다.
2. **첫 B1g의 학습 정체:** 정상적인 해상도·detail 손실 변화까지 고정 gradient 기준으로 차단했다. step은 증가했지만 optimizer가 업데이트되지 않았다.
3. **현재 B1g의 복원 품질:** 재시작 후 gradient는 안정 범위로 돌아왔지만, 큰 splat의 사진 gradient를 모두 끊는 보호장치, 입력/타깃 cell 계약, 속성 손실의 표현 불일치가 남아 있다. 안정적인 loss만으로 난간·floater 문제가 해결됐다고 볼 수 없다.

**우선순위는 디코더를 더 크게 만드는 것이 아니라, gradient 경로와 관측 로그를 바로잡고 → 데이터의 cell/count 의미를 맞추고 → covariance/속성 matching을 일관되게 만드는 것이다.** 그 뒤 같은 latent 예산에서 encoder pooling과 decoder conditioning을 비교한다.

이번 조사에서는 GPU 0·1·2의 학습 프로세스와 학습 코드를 수정하지 않았다. 기존 checkpoint를 읽고, GPU 3에서 optimizer 없는 추론·출력 tensor gradient 진단만 수행했다. 해당 추론의 PyTorch 최대 할당량은 약 1.23 GiB였다.

### 증거와 한계

- 실행 중인 run: `runs/B1g_20260925_053253`, 시작 checkpoint: `B1g_20260924_105249/ckpt_step00042000.pt`.
- 분석용 로그·소스 고정본: [manifest.json](gradient_audit_20260925/manifest.json), 최초 캡처 UTC 06:35:19.
- [history.json](gradient_audit_20260925/history.json), [args_diff.json](gradient_audit_20260925/args_diff.json), [logged_windows.csv](gradient_audit_20260925/logged_windows.csv).
- [frozen_probe.json](gradient_audit_20260925/frozen_probe.json): 기차 028630 / truck 028750, 30k·34k·36k·B1g 42k를 같은 입력으로 검사. **자기 카메라 한 장씩, downscale=2의 진단이며 held-out 성능표가 아니다.** prediction presence를 사용했다.
- [loss_math.json](gradient_audit_20260925/loss_math.json): 실제 loss 함수에 합성 입력을 넣은 수학적 검증.
- 과거 폭증을 일으킨 정확한 minibatch, 카메라 조합, 각 모듈/손실별 parameter gradient는 저장되어 있지 않다. 따라서 **“최초 원인이 특정 CUDA 연산이었다”까지 확정할 수는 없다.** 출력 tensor의 gradient와 전체 모델 parameter gradient를 구분해 아래에 적었다.

## 1. 지금 어떤 실험이 돌아가는가

이 run은 9월 15일의 S/T 또는 초기 speedy 데이터 C1과 다르다.

| 항목 | 현재 값 | 해석 |
|---|---|---|
| 데이터 | vanilla-3dgs train + truck, snapshot ≥12k | 원본이 약 100만 개 이상인 dense Gaussian 장면 |
| 입력 / 출력 capacity | 2,097,152 / 262,144 | cell당 최대 2048개를 읽어 256개를 출력 |
| compact | 1024 cells × `4|1|3|8` = 16,384 scalars | 중심·extent 4, occupancy 1, shape 3, appearance 8 |
| pooler | query 16, dim 128, block 1 | 한 번의 cross-attention으로 cell 입력 요약 |
| geometry decoder | dim 448, 10 layers | 전체 모델 약 101.10M parameters |
| attribute decoder | dim 256, 4 layers | local PE, shape 읽기, 이웃 appearance 읽기 활성화 |
| C1 | `normalize_pooler_xyz=1` | 이전의 pooler 좌표 단위 문제는 이미 수정 |
| 데이터 계약 | holdout=1, fullres extra=1, prev-vis reuse=0 | 과거 누수/이전 샘플 visibility 재사용과 구분 |
| cell/slot | shared-owner=0, fixed-center=0, count-template=0, attr-mask=0 | encoder와 target을 별도로 pack하는 문제는 남음 |
| 사진 loss | `w_render_attr=60`, 5 views | 별도 일반 render loss는 0 |
| geometry 연결 | `attr_detach_release=12000` | 현재 사진·속성 gradient가 geometry 경로로도 흐름 |
| detail phase | 40k 시작, 3k ramp, downscale 2→1 | edge gain 4, VGG .05, Sobel 1로 증가 |
| 안정화 | near-detach .1, splat-area 1, grad mult 3, abs 6000 | 서로 역할·부작용이 다른 장치 |
| optimizer | AdamW, β=(.9,.95), eps=1e-15, clip=1 | bf16 autocast, max_steps=80k |

`B1`과 최초 vanilla C1의 args 차이는 주로 stats의 scene scale 확대다. 이는 배경 잘림뿐 아니라 정규화 좌표, log-scale, cell extent의 숫자까지 바꾼다. “배경 개수만 바꾼 실험”으로 해석하면 안 된다.

## 2. 실제 발산과 학습 정체는 언제 일어났나

### 2.1 B1: 34k 이후 실제 품질 붕괴

| 시점 / 구간 | 기록된 clipping 전 norm | 사진 PSNR: 기차 / truck | 해석 |
|---|---:|---:|---|
| 30–32k | window norm 중앙값 약 625 | 31k: 19.57 / 22.19 | 정상적인 비교 기준 |
| 34k | 808.75 | 18.06 / 21.92 | 이미 기차 성능 저하 |
| 34.1k | 6,461.51 | — | 큰 spike 확인 |
| 35k | 29,918.83 | 13.49 / 15.38 | 두 scene 모두 붕괴 |
| 36k | 35–37k window 중앙값 약 56,940 | 11.98 / 12.25 | 지속적인 불안정 |
| 37.1k | 최대 기록값 306,442.04 | — | 단일 step 최대가 아니라 로그 window 평균 |
| 40k | 수백대로 하락 | 17.03 / 19.39 | 일부 회복했지만 좋은 checkpoint보다 낮음 |

**이 발산은 40k detail phase 이전이다.** Sobel/VGG가 켜져서 34.1k에 발산했다는 설명은 시간 순서가 맞지 않는다. 당시 로그의 detail 가중치는 0이다.

36k에는 Gaussian 이방성도 약 1.7–2 수준으로 낮아졌다. gradient 폭증과 함께 형태·presence·렌더가 함께 손상된 것이다. 반대로 나중에 norm이 작아졌다고 복원이 회복된 것은 아니다.

![학습 이력](gradient_audit_20260925/training_history.png)

그림의 점선은 12k, 34.1k, 40k, 43k이다. B1g의 skip이 있는 구간은 아래의 평균 집계 오류 때문에 절대값 해석에 주의한다.

### 2.2 첫 B1g: 안전장치가 학습을 멈춘 경우

`B1g_20260924_105249`는 좋은 B1 30k에서 재개했다. anchor 503.5를 고정하여 `503.5 × 3 ≈ 1510.5`보다 큰 norm을 버렸다.

- 30,063에서 norm 6523인 update를 건너뛴 기록이 있다. 드문 spike 차단 자체는 유용했다.
- **41,246에서 downscale이 2→1로 바뀐다.** 수식은 부드럽게 변하지만 최종값이 정수로 반올림되므로 실제 렌더 해상도는 여기서 한 번에 바뀐다.
- Sobel/VGG/edge도 동시에 증가한다. 정상 gradient 분포가 바뀌었는데 30k 기준 1510.5를 그대로 사용했다.
- 로그상 42,224의 누적 skip 251회에서 43,324의 1351회까지, **1100 step 동안 skip도 정확히 1100회 증가했다.** 해당 구간에는 optimizer update가 없었다.

이 경우 GPU 사용률과 step 증가만 보면 학습 중으로 보인다. 하지만 forward/backward 계산만 하고 버리므로 가중치는 학습하지 않는다.

### 2.3 현재 B1g: 다시 업데이트되고 있으나 아직 품질 판정 전

현재 재시작은 42k에서 anchor를 새로 수집했고 42,032에 **1198**로 고정했다. 실제 상대 한도는 약 **3594**, 절대 한도는 6000이다. 보통은 더 낮은 상대 한도가 먼저 작동한다.

최초 고정 로그의 42,050–42,750에서 norm은 약 1206–1526, 중앙값 약 1335였으며 GSKIP 기록은 없었다. 이후 읽은 42,850도 약 1430이었다. 이 범위만 보면 과거의 수만~수십만 폭증 상태는 아니다.

**마지막 확인: UTC 06:55:44, step 42,950, norm 1455.78, 재시작 후 GSKIP 기록 0.** 새 run의 43k eval은 아직 로그에 없었다. [live_status.json](gradient_audit_20260925/live_status.json)에 고정했다. 이 구간의 total loss 증가는 detail weight 증가도 포함하므로 서로 다른 step의 total loss만으로 악화 판정을 하지 않는다.

다만 새 코드도 **40k에서 anchor를 한 번 재수집할 뿐, 41,246의 실제 해상도 변화에 맞춰 다시 잡지는 않는다.** 이번에는 이미 full resolution인 42k에서 재시작했기 때문에 문제가 드러나지 않는 것이다. 처음부터 학습하면 같은 문제가 재발할 여지가 있다.

## 3. gradient 폭증을 해석할 때 바로잡아야 할 점

### 3.1 `gn`은 clipping 전 norm이다

[train.py:2353](../can3tok/train.py#L2353)에서 `clip_grad_norm_(..., 1)`의 반환값을 기록한다. `gn=1300`이어도 optimizer에 들어가는 전체 gradient norm은 보통 1 이하이다. PyTorch의 반환값 의미도 동일하다. [PyTorch 문서](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html)

따라서 “1300이 그대로 Adam에 들어간다”는 설명은 틀리다. 그러나 clipping은 **틀린 gradient의 방향, 손실끼리의 충돌, 특정 Gaussian에 집중된 gradient**를 해결하지 못한다. Adam의 parameter update norm을 직접 제한하는 장치도 아니다.

### 3.2 현재 코드의 gradient 연결이 과거 주석과 다르다

[model.py:224](../can3tok/model.py#L224), [schedule.py:207](../can3tok/schedule.py#L207), [attr_decoder.py:328](../can3tok/attr_decoder.py#L328):

```text
사진 loss
 ├─ rendered xyz → geometry decoder
 ├─ scale / rotation / opacity / color
 │   └─ attribute decoder의 xyz positional encoding → geometry decoder
 └─ attribute code → shared encoder / compressor / shape 경로
```

12k 이후 `attr_detach_geometry=False`다. `w_render_attr=60`은 더 이상 “geometry와 격리된 attribute-only loss”가 아니다. 별도 attribute 모듈이 있다는 사실만으로 gradient가 격리되지 않는다.

특히 local PE는 `(p - mean(p)) / mean(||p - mean(p)||)`를 사용한다. **현재는 분모도 예측 위치에 의존하고 미분된다.** 좁거나 붕괴한 cell에서는 나눗셈과 Fourier feature가 위치 gradient를 증폭할 수 있다. `1e-6` clamp는 무한대를 피할 뿐, 충분한 conditioning을 보장하지 않는다.

원근투영도 깊이가 작을 때 위치/크기 변화에 민감하다. 공식 rasterizer backward에는 깊이의 역수·제곱 역수가 들어간다. 다만 현재 자료만으로 “모든 spike가 near-camera Gaussian에서 시작했다”고 확정할 수는 없다. [공식 CUDA backward](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/backward.cu)

**개선:** 사진의 직접 xyz 경로, attribute conditioning의 xyz 경로, shared latent 경로를 따로 제어·측정한다. `attr_detach_release` 하나로 세 경로를 한꺼번에 여는 구조를 바꾼다. 초기 비교는 geometry conditioning을 detach하고 bounded nudge를 유지하는 방식으로 한다. shared encoder까지 격리됐는지는 실제 module gradient로 검증해야 한다.

### 3.3 저장된 checkpoint에서 확인한 gradient

아래는 **가우시안 출력 tensor에 대한 gradient norm**이다. 모델 전체 parameter norm과 직접 비교할 수 없다. 사진 한 뷰, 기본 L1+SSIM, 가중치 60이며 detail 항은 제외했다.

| checkpoint / scene | 보호 없음 | 현재 near/fat detach | 해석 |
|---|---:|---:|---|
| B1 30k 기차 | 435.37 | 335.27 | 일부 민감한 출력 제거 |
| B1 34k 기차 | 479.76 | 332.59 | 같은 방향 |
| B1 36k 기차 | 134.48 | 61.90 | 품질이 무너진 모델의 gradient가 오히려 작을 수 있음 |
| B1g 42k 기차 | 518.33 | 334.21 | 차단 효과는 있으나 여전히 위치 gradient가 대부분 |
| B1g 42k truck | 436.84 | 423.70 | scene/view마다 효과가 다름 |

회전 head의 normalization 전 quaternion norm도 검사했다. 이 표의 체크포인트·두 샘플에서는 최솟값이 약 0.247 이상이었다. **이 샘플들에서는 quaternion이 0에 가까워서 생기는 normalization 폭증이 관측되지 않았다.** 이 경로를 현재 확정 원인으로 지목하면 안 된다.

같은 이유로 이 진단은 과거의 수십만 parameter gradient spike를 재현한 실험이 아니다. 정확한 최초 원인 판정에는 §9의 실패 batch 저장이 필요하다.

## 4. floater를 남기는 현재 보호장치

[losses.py:1540](../can3tok/losses.py#L1540)의 `_detach_near_gaussians`는 다음 중 하나면 Gaussian의 **xyz, scale, rotation, opacity, color를 전부 detach**한다.

- 깊이가 해당 view median의 0.1배보다 작음.
- `f × max(3D scale) / depth`가 전체 median의 4배보다 큼.

### 4.1 화면에서는 남지만 사진으로 교정할 수 없는 Gaussian

현재 시작 checkpoint B1g 42k의 기차 028630을 보면 하늘의 큰 blob이 남아 있다.

![B1g 42k: 사진 / 예측 / packed GT](gradient_audit_20260925/B1g_42000_train_photo_pred_target.png)

같은 입력의 B1 30k:

![B1 30k: 사진 / 예측 / packed GT](gradient_audit_20260925/B1_30000_train_photo_pred_target.png)

자기 카메라 한 장의 PSNR은 30k **21.27**, guarded 42k **14.86**이었다. 이는 held-out 표의 기차 평균 17.00과 다른 측정이다. **guard가 발산을 줄였다고 모든 view 품질이 보존되는 것은 아니다.** 이 비교는 guard·splat loss·resume 이력이 함께 달라져서 어느 한 변경의 단독 인과효과로 해석할 수 없다.

### 4.2 “가린 Gaussian을 삭제하자”도 틀린 해결책

[guard_probe.json](gradient_audit_20260925/guard_probe.json)에서 차단 대상 opacity를 0으로 만드는 진단을 했다.

| scene | visible Gaussian 중 차단 비율 | 삭제 시 RGB가 .05 넘게 바뀐 pixel 비율 | 기본 PSNR → 차단 대상 삭제 |
|---|---:|---:|---:|
| 기차 | 1.57% | 55.83% | 14.86 → 8.67 |
| truck | 0.42% | 31.64% | 21.75 → 13.89 |

**이 집합에는 필요한 배경 Gaussian도 들어 있다.** 점의 개수로는 작지만 담당하는 화면 면적은 크다. 이들을 일괄 삭제하면 심각한 구멍이 생긴다.

같은 출력에서 xyz/scale/rotation만 detach하고 opacity/color는 살리면 차단된 기차 Gaussian의 opacity gradient norm이 **0 → .214**, color는 **0 → .187**로 복구됐다. 이는 출력 수준의 검증이다. 실제 모델에서는 appearance가 xyz PE를 통해 geometry를 움직이지 않도록 §3.2의 경로 분리도 같이 해야 한다.

### 권장 수정

1. `near/fat` 판단을 모든 점의 median 하나로 정하지 않는다. positive depth, 실제 화면 기여, projected covariance를 기준으로 기록한다. `radii>0`도 완전한 visibility/기여도가 아니므로 가능하면 alpha·transmittance 기여를 추가한다.
2. 사진 gradient가 위험한 위치·covariance 경로를 제한하면서 **색·불투명도 교정 경로는 유지**한다.
3. 3D 최대축으로 계산한 구형 반경 대신 화면의 두 covariance 축을 쓴다. 얇고 긴 난간 Gaussian을 큰 구형 blob과 구분한다.
4. camera별 behind point를 전역적으로 제거하지 않는다. 다른 view에서는 정상적인 배경일 수 있다.

## 5. loss와 정규화 항의 구체적인 문제

### 5.1 `covariance3d_loss`는 log-covariance loss가 아니다

[losses.py:1973](../can3tok/losses.py#L1973)의 실제 식은 다음과 같다.

```text
C = R diag(exp(2 log_scale)) Rᵀ
L = || (C_pred - C_target) / max(||C_target||F, floor) ||F²
```

`log_space=True`라는 이름과 “log-Frobenius” 주석이 있지만 행렬 logarithm을 계산하지 않는다. 동일한 구형 target보다 예측 scale이 k배 크면, floor가 작동하지 않는 조건에서 `L=(k²-1)²`다.

| scale 비율 k | loss | 공통 log-scale 방향 미분 |
|---:|---:|---:|
| 2 | 9 | 48 |
| 5 | 576 | 2400 |
| 10 | 9801 | 39600 |
| 30 | 808201 | 3236400 |

실제 함수로 검증했다. 분모의 median floor는 매우 작은 GT를 보호하지만 이 4차 증가 자체를 제거하지 않는다.

**다만 현재 발산의 확정 원인이라고 결론내리지는 않는다.** 현재 설정은 `attr_match_mode=sinkhorn`, `sinkhorn_attr_weight=2`다. position-only matching이 아니다. covariance의 설정값은 2지만 attribute decay floor .5가 적용되어 현재 유효 weight는 **1**이다. B1g 42k 두 샘플의 matched target 기준 weighted covariance loss는 약 **.913 / .787**, 출력 gradient norm은 **.075 / .043**이었다. 반면 순서 그대로의 raw target에 억지로 비교하면 수천 수준으로 커진다. matching을 바꾸면 covariance 항의 규모도 함께 재검증해야 한다.

**권장식:** FP32에서 다음처럼 analytic log-covariance를 구성해 robust loss를 쓴다.

```text
M = log(C) = R diag(2 log_scale) Rᵀ
L_size  = Huber(trace(Mp)/3 - trace(Mt)/3)
L_shape = Huber(dev(Mp) - dev(Mt))  # dev(M) = M - trace(M)I/3
```

이미 R과 log-scale이 있으므로 `eigh`를 통해 log를 다시 구할 필요가 없다. size와 anisotropic shape를 분리하면 큰 scale 오차가 orientation 학습을 압도하는 문제도 다루기 쉽다. 새 식은 기존 설정값 2 또는 유효 weight 1을 그대로 복사하지 말고 실제 weighted gradient로 보정한다.

관련 주석의 “1e40은 fp32에서 finite”, “1e-20은 bf16 최소 normal 1e-38보다 작다”는 서술도 잘못됐다. 실행 수식과 별도로 정정할 대상이다.

### 5.2 splat penalty의 상한이 가장 큰 splat의 복구 gradient를 없앤다

[losses.py:1824](../can3tok/losses.py#L1824):

```text
hinge = relu(r_pred / r_ref - 1).clamp(max=4) ** 2
```

`r_pred/r_ref > 5`이면 크기와 위치 방향의 미분이 0이다. 합성 검사에서도 비율 6, 10, 100에서 모두 0이었다. opacity 미분은 남지만 opacity가 sigmoid saturation 상태면 이 경로도 약해진다. near-detach 조건과 splat loss의 depth-valid 조건은 서로 달라서, 사진·size 두 경로 모두 약해지는 점이 생길 수 있다.

**권장:** loss 값을 잘라 미분을 없애는 대신 `Huber(relu(log(r_pred/r_ref)))` 같은 완만한 증가를 사용한다. 작은 깊이 처리는 별도로 안전하게 정의한다. reference는 scene 전체 q80 하나보다, 같은 영역/깊이의 teacher projected covariance를 사용한다. 긴 축은 유지해야 하는 난간인지, 두 축이 함께 커진 blob인지 구분한다.

### 5.3 covariance는 같아도 attr-set loss는 다르다고 벌한다

global/local attr-set은 log-scale 3개와 quaternion 4개를 표준화해 Euclidean 거리로 비교한다. quaternion의 q/−q 부호는 맞추지만, **축 순열과 그에 따른 회전 변경의 동치성**은 처리하지 않는다.

실제 함수의 합성 검사: x/y scale을 바꾸고 z축으로 90° 회전시켜 같은 covariance를 만들었을 때,

- covariance loss: 약 `5.94e-14`.
- attribute set loss: **4.521**.

즉 렌더가 구분할 수 없는 Gaussian 표현을 현재 attr-set은 크게 다르다고 평가한다. current weight는 global 3 + local 3이다. covariance loss를 추가했다고 기존 속성 metric의 충돌이 사라지는 것은 아니다.

**권장:** matching cost와 attr-set metric 모두에서 scale/quaternion raw 조합을 log-covariance의 대칭 6성분 등으로 교체한다. 색·opacity는 별도 단위로 정규화한다. 고유벡터 부호·축 순열에 민감한 quaternion L2를 주요 matching 기준으로 쓰지 않는다.

### 5.4 Sinkhorn soft matching과 중복 손실

현재 cost에는 attributes도 들어가지만 epsilon .08, 6회 반복의 soft plan이다. finite entropy의 balanced plan은 hard one-to-one assignment와 다르다. quaternion sign-align 후 평균하더라도 서로 다른 Gaussian 방향·크기를 섞을 수 있다.

- plan의 row entropy, 최대 assignment 확률, effective support, row/column marginal 오차를 기록해야 한다.
- matching cost와 실제 회귀 loss가 같은 covariance·좌표 단위를 보도록 만든다.
- 필요하면 geometry가 안정된 뒤 epsilon을 줄인다. 처음부터 Hungarian/hard match로 바꾸는 것은 별도 ablation이다.
- global set + local set + matched scalar + covariance + render를 모두 키우지 않는다. 같은 방향의 중복인지 충돌인지 module별 gradient cosine으로 확인한다.

`codec_attr_local_set`은 이웃 선택에 쓴 xyz를 detach하지만, **attribute output 자체가 geometry PE에 의존하는 경로까지 끊는 것은 아니다.** “local attribute loss는 위치를 움직일 수 없다”는 주석은 현재 12k 이후 연결에서 성립하지 않는다.

### 5.5 좌표 loss의 기준 크기는 예측/입력에 종속되지 않게

현재 `geometry_target_scale=0`, `geometry_center_local=0`, `geometry_centroid_absolute=0`이다. 여러 local geometry loss가 encoder에서 온 decoded scale로 좌표를 나눈다. detach되어 있어 분모를 loss가 직접 속이는 경로는 제한되지만, **입력과 target이 다르게 pack되면 target 오차의 가중치를 다른 점 집합이 결정**한다.

target의 detached extent를 기준으로 위치 오차를 정규화하고, 큰 절대좌표끼리의 거리 계산에는 공통 cell 원점을 먼저 뺀다. shape를 각자 중심화할 경우 translation을 놓치므로 centroid loss를 반드시 별도로 유지한다. 분모의 최소값도 scene·cell 분포에 맞춰 정한다.

### 5.6 optimizer와 precision

- eps=1e-15는 clipping으로 작아진 tail gradient를 살리려는 의도지만, 매우 작은/noisy gradient의 adaptive step도 쉽게 살린다. **현재 폭증의 단독 원인이라는 증거는 없다.** eps=1e-8 또는 1e-6 비교는 별도 실험으로 하고, eps를 바꿔 decoder가 다시 정지하지 않는지 update RMS를 본다. [AdamW 공식 정의](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)
- beta2=.95는 최근 gradient 제곱에 빠르게 반응한다. 갑작스러운 손실/해상도 전환과 함께 검사할 변수다. 다른 수정과 동시에 바꾸지 않는다.
- rasterizer 입력 FP32 변환은 이미 있다. “현재 bf16 tensor를 CUDA rasterizer에 그대로 보낸다”는 문제는 확인되지 않았다.
- covariance, quaternion normalization, local radius, 거리/transport 수식은 FP32 구역에서 계산하는 쪽을 비교한다. 전체 네트워크 bf16을 무조건 끄는 것보다 범위를 특정한다.
- clipping 이전 finite 검사와 DDP 전체 rank의 skip 합의는 항상 필요하다. 현재는 absolute guard를 켰으므로 동작하지만, 둘 다 비활성화하거나 relative anchor 수집 중 abs=0인 설정에서는 nonfinite gradient 보호가 비는 코드 경로가 있다.

### 5.7 이미 올바르게 처리된 부분과 detail 항의 실제 크기

edge-weighted L1은 weight 합으로 나누고, 여러 view loss도 view 수로 평균낸다. edge gain 4를 주었다고 전체 L1이 단순히 5배가 되거나 5-view라서 render weight가 300이 되는 구조는 아니다.

다만 43k에서 완성된 사진 목적함수는 대략 다음과 같다.

```text
60 × mean_views[ .8 × (edge-weighted L1 + Sobel L1)
               + .2 × (1 - SSIM) + .05 × VGG ]
+ 1 × splat_area
```

즉 Sobel의 외부 계수는 48, VGG는 3이다. 수식이 틀렸다는 뜻은 아니지만, 설정의 `w_sobel=1`을 작은 보조항이라고 해석하면 안 된다. term별 gradient를 측정해 Sobel·VGG·해상도 변화를 각각 ramp하고, gate 기준은 실제 해상도 전환과 연동한다.

## 6. encoder/decoder 구조에서 수정할 부분

### 6.1 가장 먼저 cell과 count의 의미를 맞춘다

현재 input과 target은 capacity가 다르고 각각 spill packing된다. 같은 cell 번호가 정확히 같은 source Gaussian 집합을 뜻하지 않는다. 실제 두 샘플에서 encoder 중심과 target 중심 차이는 target extent 대비 중앙값 **기차 .232 / truck .225**, p99는 **1.186 / 1.403**이었다.

[data_contract.json](gradient_audit_20260925/data_contract.json)에서는 실제 14채널 값의 완전일치로 target Gaussian이 encoder 어디에 들어갔는지도 찾았다. 두 샘플의 target은 100% encoder에서 발견됐다. 유일하게 대응되는 비율은 기차 100%, truck 99.9996%였다. 그런데 유일 대응점 중 **기차 9.31%, truck 8.29%가 서로 다른 cell에 속했다.** 정보가 encoder에 존재하는 것과 해당 target cell의 code가 그 정보를 읽는 것은 다르다.

또한 [encoder.py:427](../can3tok/encoder.py#L427)의 count는 **encoder 입력 개수**다. [compressor.py:507](../can3tok/compressor.py#L507)는 이를 출력 `G=256`으로 나누어 occupancy anchor를 만들고, decoder는 다시 256으로 clamp한다. 2048-capacity input count는 target에서 실제로 남은 count와 같지 않다.

이 상태에서 count-aware template만 켜면 잘못된 count prior를 template에도 전달할 수 있다. `count_aware_template=1`은 count 정의를 고치는 기능이 아니다.

| 실측 항목 | 기차 028630 | truck 028750 |
|---|---:|---:|
| encoder cell count 중앙값 | 740.5 | 1393 |
| encoder count가 256 초과인 cell | 93.75% | 99.12% |
| target이 partial인 cell | 25.10% | 29.30% |
| encoder는 256 이상인데 target은 partial인 cell | 19.63% | 28.52% |
| `min(encoder_count,256)`와 target count의 평균 절대 차이 | 19.30 slots/cell | 23.82 slots/cell |
| occupancy anchor 중앙값 | 4.79 | 9.88 |

출력 occupancy anchor가 의도한 기본 범위는 -1~1인데 입력 count 때문에 이 범위를 크게 벗어난다. 작은 `tanh(residual) × .1`로 이 차이를 수정하기도 어렵다. presence head가 이를 뒤에서 보정해야 하는 구조다. 학습 로그의 `n`과 `empty`는 encoder 측 count를 반영하므로 target 슬롯의 유효 개수와 혼동하면 안 된다.

**수정 순서:**

1. 출력 target source 집합과 cell 소속을 먼저 확정한다.
2. 해당 source는 encoder에서도 같은 cell에 넣고, 남는 공간에 context point를 추가한다.
3. 입력 occupancy와 출력 occupancy를 별도 정의한다. decoder count는 실제 출력 예산을 뜻하도록 학습하거나 명시적으로 encode한다.
4. encode/decode가 같은 기준의 centroid·extent·count를 쓴 뒤 count-aware template와 attr slot mask를 적용한다.

기존 `shared_cell_owner=1`을 그대로 켜라는 뜻은 아니다. [16번 보고서](16_experiment_design_review_20260916.md)는 그 구현이 target 선택을 손상시킨 것을 측정했다. target을 보존하면서 소속을 맞추는 구현이 필요하다.

### 6.2 Encoder: C1을 유지하고 작은 구조를 세밀하게 읽게 한다

이미 local xyz normalization이 켜져 있다. 9월 15일의 “xyz를 지워도 PSNR이 안 변함” 결과를 현재 모델에 그대로 적용하면 안 된다.

우선 비교할 구조는 다음과 같다.

- `2048 points → 16 queries` 단일 pooling을 두 단계로 나누고 query끼리 정보를 교환한다. query 수 증가와 layer 추가를 동시에 바꾸지 않는다.
- 실제 3D 이웃을 따라 pooling/context를 구성한다. Morton index ±1은 근접성의 근사이며 항상 가장 가까운 이웃은 아니다.
- thin structure가 소수라는 이유로 평균화되지 않도록 covariance, opacity, local position의 상대관계를 token에 명확히 준다. raw quaternion 대신 covariance 표현을 검토한다.
- 고정 `ATTR_MEAN/STD`는 이전 데이터 통계다. vanilla train split에서 채널 분포와 tail을 다시 측정한다. runtime batch별 whitening 대신 고정 train 통계를 저장하고 checkpoint에 포함한다.
- 모든 code가 같은 평균을 복제하는지 query attention, within-cell reconstruction, input perturbation sensitivity로 검사한다. latent std/rank만 키워서는 입력 디테일 사용을 증명할 수 없다.

### 6.3 Decoder: 더 많은 layer보다 geometry와 appearance의 연결 설계

현재 geometry 10층, attribute 4층이다. 먼저 다음을 바꾼다.

1. **좌표의 역할 분리:** geometry가 예측한 위치는 attribute의 조건으로 사용하되, PE/local-radius 경로의 backward를 따로 제어한다. local radius는 안정된 detached 기준 크기로 정규화하는 실험을 먼저 한다.
2. **작은 위치 수정 유지:** bounded nudge는 살리되, 실제 projected footprint에 비해 얼마나 움직이는지 기록한다. cell extent의 .15가 모든 카메라에서 같은 pixel 이동을 뜻하지는 않는다.
3. **scale의 크기와 모양 분리:** mean log-scale과 trace-free shape를 분리해 구형 평균으로의 도피를 확인한다. rotation은 크기 축과 함께 covariance로 감독한다. raw quaternion을 크게 벌하는 방식으로 난간을 만들지 않는다.
4. **유효 slot만 섞기:** attr self-attention/local PE의 mask는 필요한 개선이다. 현재 두 target도 262144 슬롯 중 약 236–237k만 유효하다. 다만 train에서는 GT mask, deploy에서는 다른 predicted mask를 쓰면 계약이 달라지므로 먼저 count/presence를 맞춘다.
5. **실제 3D 이웃 context:** cell 간 rail이 끊기는 경우 centroid kNN+상대좌표를 쓰는 context를 비교한다.

shape 3 channels는 제약이지만 “shape 3이므로 정확히 절반의 정보만 보존 가능” 같은 수학적 한계는 입증되지 않았다. 이전 PCA 설명분산과 nonlinear decoder의 rate-distortion 한계는 다르다. 이웃 code와 scene별 weight 기억도 영향을 준다.

같은 16,384 scalars에서 budget을 바꾸려면 `4|1|4|7` → `4|1|6|5`처럼 실제 rate를 고정하고 실험한다. appearance를 줄이는 비용도 같이 평가한다. 입력·loss 문제가 남은 상태에서 channel만 재배분하는 것은 우선순위가 낮다.

## 7. 왜 난간과 작은 디테일이 아직 약한가

| 관측 | 가능한 학습 결과 | 확인/개선 |
|---|---|---|
| 100만+ GT에서 약 23만여 출력 point로 선택 | 사진의 빈 coverage를 큰/불투명한 splat으로 메움 | 원본 GT → 필터 GT → packed GT → model render를 분리 평가 |
| input/target cell 소속·중심·count 불일치 | cell code가 설명해야 할 대상이 일관되지 않음 | target 보존 방식의 계약 수정 |
| shape와 appearance를 위치 기반 PE로 강하게 결합 | 색/속성 loss가 위치를 흔듦 | geometry 경로별 gradient gate 및 안정된 local scale |
| scale/quaternion raw metric의 표현 불일치 | 실제 covariance보다 parameter 관례를 맞춤 | covariance-invariant matching/loss |
| low resolution·지연된 detail 감독 | 수 pixel의 난간 신호가 약해짐 | fullres crop과 Sobel을 통제된 단계로 도입 |
| 큰 splat은 사진 gradient 전체 차단 | 이미 잘못된 blob의 색/opacity 수정이 어려움 | geometry만 제한하고 appearance 교정 유지 |

B1g 42k 두 샘플의 이방성 median은 기차 **4.52 vs GT 14.22**, truck **7.45 vs GT 18.02**였다. 현재 training window에서도 대략 5.x vs 12–13이다. 다만 scene 전체 anisotropy를 GT까지 무조건 올리면 배경의 극단적인 needle까지 복제할 수 있다. **난간 ROI에 실제 기여하는 Gaussian의 두 projected 축·위치·opacity**를 함께 봐야 한다.

packed GT의 자기 사진 PSNR도 기차 20.66 / truck 16.80이었다. model truck이 packed GT보다 높다고 원본을 더 정확히 압축했다는 뜻은 아니다. 현재는 사진에 맞춰 선택 손실을 보정하는 목적도 섞여 있다. 원본 전체 Gaussian의 DC render와 photo를 각각 평가하고, codec fidelity용 teacher는 가능하면 **packing 전 원본 render**로 만든다. SH가 없는 14채널 출력과 full-SH teacher의 표현 차이는 별도 ablation으로 분리한다.

## 8. 실험은 이렇게 나눈다

### 8.1 현재 GPU 0·1·2 run의 점검 지점

이번 조사에서 실행 중인 run은 변경하지 않았다. 이 run의 역할은 현재 guard 기준에서 fullres detail phase가 실제로 진행되는지 확인하는 것이다.

- **43k:** detail ramp 완료, 첫 재시작 후 정규 eval/checkpoint. 씬별 held-out PSNR, 같은 난간 crop, sky floater, 실제 update 수를 확인한다.
- **45k:** full detail objective 아래 약 2k 추가 구간의 추세를 본다.
- **48k:** 난간/held-out 성능의 개선 여부를 판정한다. 오래 돌렸다는 이유만으로 80k까지 계속하는 기준을 삼지 않는다.

아래의 진단 실험은 **새 output directory의 독립 run**으로 한다. 현재 run의 loss/guard를 중간에 덮어쓰지 않는다. GPU 0·1·2는 지금 하나의 3-rank DDP 작업이다. 세 GPU에 각각 다른 플래그를 주면 세 ablation이 되는 것이 아니라 잘못된 단일 학습이 된다.

### 8.2 수정 실험 순서

| 단계 | 바꾸는 것 | 출발점 / 최소 관측 구간 | 판단 기준 |
|---|---|---|---|
| D0 | 로그·skip counter·phase별 guard 기준만 수정 | B1 30k, 500–1000 attempted steps | accepted update 수, finite 상태, update RMS, skip 비율 |
| D1 | geometry/PE gradient 제한 + 차단 splat의 color/opacity gradient 보존 | 같은 30k, 먼저 1k, 다음 5k | B1의 34–35k 위험 구간을 통과하고 sky/기차 성능을 보존하는가 |
| D2 | covariance robust화 + 표현이 일치하는 attr matching | 안정화된 공통 출발점, 2–4k | per-term gradient tail, 난간 covariance, matching entropy, 사진 품질 |
| D3 | target 보존 cell/count 계약 | 원칙적으로 from-scratch matched baseline, 8k→16k | geometry/slot 일관성; detail 검증은 기존 40–43k phase까지 별도 필요 |
| D4 | pooling / decoder conditioning 한 변경씩 | 같은 데이터 계약·loss·LR horizon | 같은 rate에서 ROI/held-out 성능, 속도·메모리 비교 |

**D1에서 1k 안정적이었다고 성공 판정하면 안 된다.** 과거 발산은 30k 재개 후 약 4.1k 뒤에 나타났다. detail 관련 안정성은 40k→43k 전환을 통과해야 별도로 검증된다.

여러 보호장치를 동시에 바꾼 이전 B1g와 달리, 가능한 한 한 원인씩 비교한다. D1도 먼저 geometry gradient gate만, 다음에 near/fat appearance 경로 복구를 비교하면 각각의 효과를 분리할 수 있다.

D2는 기존 checkpoint로 loss의 효과를 빠르게 확인하는 짧은 진단이다. 최종 from-scratch baseline을 만들 때는 D3의 데이터 계약을 먼저 확정하고, 선택한 loss와 구조를 그 위에서 다시 비교한다. 두 run의 숫자를 직접 합쳐 하나의 개선율로 보고하지 않는다.

실험 중간 종료용 `stop_at`과 cosine LR의 `max_steps=80000`은 분리한다. `max_steps=1000`으로 바꿔 1k 실험을 만들면 LR 곡선 자체가 달라진다. 동일 checkpoint에서 optimizer state·step을 보존한 resume인지, weight만 가져온 init인지도 명시한다.

### 8.3 판단 기준은 loss 한 개가 아니다

- **안정성:** nonfinite update 0, 실제 accepted update 비율, per-module norm p50/p95/p99, parameter update RMS/weight RMS.
- **복원:** scene별 고정 held-out photo PSNR·SSIM, 원본 render와의 차이, 같은 난간 crop의 edge 오차.
- **형태:** rail ROI에서 projected short/long axis, pixel displacement, opacity; background floater의 화면 기여.
- **latent:** 같은 anchor에서 input local xyz/attribute를 바꾸었을 때 code와 출력이 적절히 바뀌는지; train-view 개선만으로 일반화 판정 금지.

skip이 수백 step 연속되거나, 같은 고정 eval에서 한 scene이 여러 checkpoint 연속 악화하면 그 상태에서 step만 누적하지 않는다. 먼저 gate/데이터/gradient 분포를 확인한다. 예를 들어 held-out 0.3–0.5 dB 이상의 연속 하락은 조사 trigger로 둘 수 있지만, 보편적인 성공/실패 상수는 아니다.

## 9. 다음 코드 수정 때 반드시 함께 고칠 관측 결함

### 9.1 skip과 step을 구분

[train.py:2394](../can3tok/train.py#L2394)는 gradient skip에도 `step += 1` 한다. LR와 detail 가중치는 전진하지만 학습은 진행되지 않을 수 있다. `attempt_step`, `optimizer_step`, `epoch/batch_cursor`를 별도로 저장한다. schedule의 기준을 어느 것으로 할지 명시하고, 기존 run과 비교할 때 동일한 기준을 쓴다.

guard의 phase는 최소한 실제 render downscale과 geometry 연결 상태를 포함해야 한다. 전환 시 absolute ceiling은 유지하면서 별도 정상 분포를 수집한다. 긴 detail ramp 전체를 시작 직후 32개의 norm으로 대표시키지 않는다. 수락/거절된 finite norm 분포를 모두 기록하고, skip이 많다고 threshold를 무조건 올리는 대신 phase 변화인지 실제 발산인지 확인한다. anchor와 phase key도 checkpoint에 저장한다.

### 9.2 로그 평균의 분모 오류

[train.py:2421](../can3tok/train.py#L2421)는 성공한 update의 로그만 `running`에 더하면서 출력은 global step 배수에서 하고, 무조건 `log_every=50`으로 나눈다. skip으로 출력 시점을 놓치면 50개보다 많은 값을 모아 50으로 나누고, skip이 섞인 window에서는 적은 값을 50으로 나눌 수도 있다.

실제 첫 B1g의 41,800 로그에는 **Sobel weight 1.015**가 찍힌다. 스케줄 상한은 1이므로 정상 평균일 수 없다. norm 1790이 고정 허용 한도 1510을 넘는 것으로 보이는 것도 이 집계 문제와 함께 읽어야 한다.

`running_count`로 나누고 skipped batch 통계를 별도 집계한다. log/eval/save trigger도 accepted/attempt 정책에 맞게 분리한다.

### 9.3 resume 비교의 기준점

- parameter movement `_snap`은 checkpoint load **이전**에 생성된다. resume 직후 첫 `d[...]`에는 checkpoint 로딩 변화가 포함된다. 첫 50-step 이동량으로 해석하면 안 된다. 로딩/재초기화 후 snapshot을 생성한다.
- gradient anchor·skip counter·sampler epoch/RNG 상태는 일반 checkpoint에 저장되지 않는다. resume가 중단 지점의 stochastic stream을 그대로 잇는 것은 아니다. 재현 가능한 실패 batch를 별도로 저장한다.
- `args.json`에 없는 플래그를 현재 parser default로 해석해 과거 run에 덧씌우지 않는다. 과거 소스가 완전히 남아 있지 않은 부분은 로그에 확인된 동작으로만 판단한다.

### 9.4 진짜 원인을 찾을 gradient 계측

매 step 전체 loss별 backward를 반복할 필요는 없다. 고정된 소수 diagnostic batch에서 주기적으로 다음을 측정한다.

1. module: pooler / compressor / decompressor / xyz refine / attribute trunk / scale·rotation·opacity·color·nudge head.
2. loss: geometry / covariance / global set / local set / photo L1-SSIM / Sobel / VGG / splat.
3. **weighted parameter gradient norm과 cosine**, output xyz/scale/rotation gradient의 상위 .1% 집중도.
4. camera depth, projected covariance, raw quaternion norm, local PE radius, scale cap hit 비율.
5. spike 발생 시 NPZ ID, scene, camera IDs, RNG seed/state, loss별 값, 모듈 norm, optimizer step 저장. DDP에서는 원인 rank도 기록한다.

기록된 `TOP[render_attr:44% ...]`는 scalar loss 비중이지 gradient 기여율이 아니다. 이 둘을 같은 것으로 읽으면 작은 값의 큰 미분을 놓친다.

## 10. 권장 변경의 요약

**먼저:** skip/평균/phase 측정 수정, geometry gradient 경로 분리, near/fat의 appearance 교정 복구.

**그다음:** target을 보존하는 cell/count 계약, covariance-invariant matching과 robust loss, 안정된 local 좌표 정규화.

**마지막:** query pooling과 실제 3D 이웃 context, 같은 16k 예산의 shape/appearance 배분 비교.

현재 수치로 말할 수 있는 것은 **과거 발산과 첫 guarded run의 정체는 확인됐고, 현재 run은 다시 업데이트되지만 디테일/배경 문제는 남아 있다**는 것이다. 이를 하나의 “gradient clip을 더 세게 걸면 해결될 문제”로 취급하지 않는 것이 핵심이다.

### 재현

저장된 로그 분석은 [audit_history.py](gradient_audit_20260925/audit_history.py), CPU 수식 검사는 [probe_loss_math.py](gradient_audit_20260925/probe_loss_math.py)로 재현할 수 있다. frozen 진단은 [probe_frozen.py](gradient_audit_20260925/probe_frozen.py), 차단 영역 검사는 [probe_guard.py](gradient_audit_20260925/probe_guard.py), 데이터 계약 검사는 [probe_data_contract.py](gradient_audit_20260925/probe_data_contract.py)에 있다. GPU 스크립트는 별도 여유 GPU를 지정해야 하며 실행 중인 학습에 자동 적용되는 수정은 포함하지 않는다.
