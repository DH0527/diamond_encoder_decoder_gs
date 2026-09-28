# 기차 난간·세부 구조 소실의 원인과 인코더–디코더 개선안

작성: 2026-09-15  
대상: T16k 70k checkpoint, 기차 `step_028630.npz`, 977×544의 실제 사진 두 시점  
이전 전체 감사: [12_code_audit_and_improvement_20260915.md](12_code_audit_and_improvement_20260915.md)  
이번 조사 자료: [rail_detail_audit_20260915](rail_detail_audit_20260915/)

## 1. 핵심 결론

**난간은 원본 NPZ와 선택된 복원 타깃에 남아 있지만, 현재 codec은 셀 내부의 정밀한 좌표 배치를 거의 활용하지 않는 상태다.** 복원된 가우시안은 난간 위치에서 몇 픽셀씩 어긋나고, 길쭉한 화면상 모양도 짧고 둥글게 변한다. 그 결과 연속된 선이 점·얼룩으로 바뀌고 차체에 섞인다.

이번에는 일반적인 가능성을 나열하는 대신, 실제 checkpoint에 입력 정보를 선택적으로 제거하고 같은 카메라에서 재렌더했다. 가장 강한 근거는 다음 실험이다.

> 원본 입력으로 계산한 **셀 중심·extent·count는 그대로 유지**하면서, point pooler에 주는 **셀 내부 상대 xyz만 전부 0**으로 만들었다. 최종 렌더 PSNR은 view3에서 **21.483 → 21.480 dB**, view93에서 **15.349 → 15.342 dB**로 거의 변하지 않았다.

반대로 위치와 중심·extent는 유지하고 **셀 내부 attribute만 셀 평균으로 바꾸면**, view3은 **13.074 dB**로 떨어졌다.

따라서 이 checkpoint에서 먼저 해결할 문제는 단순한 출력 점 수 부족보다 **정밀한 기하를 encoder가 읽고 latent로 전달하는 경로가 약하다는 점**이다. 다만 이 개입 실험은 한 snapshot의 frozen model에서 실시했다. 좌표 정규화 하나만 바꾸면 난간이 반드시 복원된다는 학습 결과까지 얻은 것은 아니다.

### 원본부터 복원까지 같은 난간을 비교

![같은 난간 영역의 원본·타깃·모델 비교](rail_detail_audit_20260915/crop_view3_upper_rail.png)

위쪽은 실제 사진, 원본 NPZ SH3, 선택 타깃 SH3다. 아래쪽은 선택 타깃 DC-only, 모델 복원, 실제 extra-view 경로의 축소 후 확대 사진이다. 화면 확대에는 nearest-neighbor 표시를 사용했다. AI로 이미지를 보정하거나 detail을 생성하지 않았다.

**관찰:** 원본과 타깃에는 가로·세로 난간과 통풍구가 남아 있다. 모델에서는 상단 가로 난간이 점선처럼 끊기고 세로 난간은 번진다. 축소된 학습 사진도 원본보다 흐리지만 난간 전체가 사라진 정도는 아니다. **타깃 해상도 문제만으로 모델의 손실 전체를 설명할 수 없다.**

## 2. 실험 조건과 해석 범위

| 항목 | 조건 |
|---|---|
| checkpoint | `runs/T16k_20260913_093332/ckpt_step00070000.pt` |
| NPZ | 기차 `step_028630.npz`, val index 166 |
| 원본 / 타깃 | 원본 81,369개, 범위 필터 후 79,821개, 타깃도 79,821개 |
| view3 | `/data/daeho/train_colmap/images/00004.jpg` |
| view93 | `/data/daeho/train_colmap/images/00094.jpg` |
| 카메라 | 해당 기차 scene의 실제 카메라만 사용 |
| 해상도 | 전체 실험 full resolution, 977×544 |
| 모델 | strict checkpoint load, 70k schedule, eval mode, teacher forcing 없음 |
| 정밀도 | 기본 추론 CUDA BF16, 렌더 FP32 |
| 위치/속성 교체 | GT mask로 점 개수를 고정하고 셀별 위치 기반 Hungarian 대응 |
| encoder 개입 실험 | 모든 조건에서 같은 작은 chunk 사용; 중심·extent·count 입력은 유지 |
| 학습 | 실행하지 않음. weight·optimizer·기존 학습 코드 변경 없음 |

앞선 보고서에서 확인한 held-out 사진 유입이 있으므로 이 결과를 깨끗한 시점 일반화 성능으로 주장하지 않는다. **동일 checkpoint에서 난간 소실이 어디서 일어나는지 분석하는 실험**이다. 특정 ROI의 점수는 난간뿐 아니라 주변 차체 픽셀도 포함한다.

## 3. 난간이 어느 단계에서 사라지는가

### 3.1 상단 난간: 원본과 선택 타깃에는 정보가 있음

사진과 비교한 고정 영역 PSNR이다. `upper rail` 영역은 `(x0,y0,x1,y1)=(395,118,620,345)`로 모든 조건에 동일하게 사용했다.

| 렌더 조건 | view3 전체 | 상단 난간 ROI | 글자 ROI |
|---|---:|---:|---:|
| 원본 전체, SH3 | 26.52 | 26.96 | 27.76 |
| 선택 타깃, SH3 | 21.99 | 26.96 | 27.76 |
| 원본 전체, DC | 20.37 | 23.81 | 23.11 |
| 선택 타깃, DC | 19.21 | 23.81 | 23.11 |
| 모델, predicted presence | 21.43 | 21.57 | 22.02 |

출처: [render_metrics.json](rail_detail_audit_20260915/render_metrics.json).

**상단 난간과 글자 영역에서는 선택 전후의 점수가 사실상 같다.** 이 snapshot은 점 수가 262k보다 작고, 유효한 점들이 target에 모두 들어간다. 앞선 전체 감사의 packing/score 문제는 수정할 필요가 있지만, **이 난간이 안 보이는 직접 원인을 “sampling에서 난간 점을 버렸다”로 돌릴 근거는 없다.**

반면 전체 화면과 앞쪽 계단 난간 영역에는 범위 필터의 손실이 있다. 앞 난간 ROI의 원본 SH3 점수는 26.28인데 선택 타깃 SH3는 17.07이다. 이 ROI는 차체·배경도 포함하며, 특정 Gaussian의 제거가 이미지 합성 전체를 바꿀 수 있다. **모든 난간 부위가 같은 원인으로 손상됐다고 일반화하면 안 된다.**

### 3.2 가까운 난간 시점에서도 모델 손실이 추가됨

view93의 가까운 난간 ROI `(470,350,725,544)`:

| 조건 | 난간 ROI PSNR |
|---|---:|
| 원본 SH3 | 17.44 |
| 선택 타깃 SH3 | 17.44 |
| 선택 타깃 DC | 16.61 |
| 모델 | 12.74 |

![가까운 난간 시점 비교](rail_detail_audit_20260915/crop_view93_near_rail.png)

이 시점은 원본도 완벽하지 않다. 그렇지만 선택 타깃→모델 단계에서 추가 손실이 크다. view3 전체 PSNR만 보면 모델이 DC 타깃보다 높아 보이므로 이런 실패가 가려진다.

### 3.3 SH 생략의 영향은 실제로 큼

현재 renderer는 `sh_degree=0`, 모델 출력도 `sh_dim=0`이다. 원본 NPZ는 15×3개의 추가 SH 계수를 갖는다.

- view3 원본 전체: **SH3 26.52 dB, DC 20.37 dB**.
- 상단 난간: **26.96 → 23.81 dB**.
- 글자: **27.76 → 23.11 dB**.

따라서 코드 주석의 “SH 계수가 작아서 영향이 거의 없다”는 가정은 이 샘플에서 맞지 않는다. 단, DC 타깃에도 난간은 선으로 남아 있으므로 **SH만 추가하면 난간 기하까지 해결된다는 뜻은 아니다.** 기하 전달과 시점 의존 appearance를 별도로 개선해야 한다.

## 4. 인코더의 핵심 부족: 셀 내부 xyz 정보가 약하게 전달됨

### 4.1 코드상 신호 크기가 맞지 않음

**위치:** [encoder.py](../can3tok/encoder.py) L362–401.

현재 pooled 경로는 다음 두 종류의 입력을 바로 합친다.

```text
local xyz = xyz - cell center       # extent로 나누지 않음
attribute = (attribute - mean)/std  # 표준화함
```

난간 band 기여도가 큰 16개 셀에서 local xyz RMS는 대체로 **0.0008–0.002**인데 standardized attribute RMS는 대체로 **0.9–1.3**이었다. 같은 첫 MLP에 들어가지만 좌표 성분의 규모가 약 1,000배 작다.

선형층이 이 차이를 학습으로 보상할 가능성은 있다. 그래서 규모 차이만으로 결론내리지 않고 아래의 실제 제거 실험을 수행했다.

### 4.2 첫 pooling 출력의 변화

16개 셀에서 중심·extent 정보를 바꾸지 않고 각각 한 종류의 내부 정보를 제거했다. 첫 pooler 출력의 `||변경 출력 - 원래 출력|| / ||원래 출력||` 중앙값은:

| 제거한 정보 | pool 출력의 상대 변화 |
|---|---:|
| 셀 내부 xyz → 모두 0 | **0.00795%** |
| 셀 내부 attribute → 셀 평균 | **13.87%** |

이 부분은 작은 차이를 확인하기 위해 **CPU FP32**로 측정했다. BF16에서도 별도 확인했으나 작은 변화가 양자화와 kernel 차이에 민감하므로 FP32와 BF16 수치를 같은 의미로 섞지 않는다. [pool_sensitivity.json](rail_detail_audit_20260915/pool_sensitivity.json).

이는 단순한 feature rank보다 직접적인 근거다. 좌표 분포를 크게 훼손해도 첫 집계 결과가 거의 변하지 않는다.

### 4.3 최종 compact와 렌더도 거의 변하지 않음

이번에는 전체 모델을 다시 실행했다. `GroupPointPooler`의 입력만 바꾸었고, encoder가 원본 점으로 계산하는 analytic center/extent/count는 그대로 두었다. 각 조건은 동일한 checkpoint, BF16, chunk 크기, GT mask를 사용했다.

| 입력 개입 | shape compact 상대 변화 | appearance compact 상대 변화 | view3 PSNR | view93 PSNR |
|---|---:|---:|---:|---:|
| 원래 입력 | — | — | 21.483 | 15.349 |
| local xyz 제거 | **0.209%** | **0.234%** | **21.480** | **15.342** |
| local attribute를 평균으로 | 32.56% | 43.51% | 13.074 | 12.630 |

local xyz를 제거한 이미지와 원래 모델 이미지 사이의 PSNR은 view3 **45.84 dB**, view93 **43.94 dB**로 높다. 이미지가 비슷하다는 의미다. **사진과의 PSNR이 아니다.**

![셀 내부 정보 제거에 대한 최종 렌더 반응](rail_detail_audit_20260915/crop_pool_counterfactual.png)

출처: [pool_to_render.json](rail_detail_audit_20260915/pool_to_render.json).

**판정:** 이 snapshot에서 모델은 정밀한 점 배치보다 **셀의 위치·크기·개수와 속성 패턴**을 바탕으로 형상을 만드는 데 크게 의존한다. 가는 난간의 정확한 위치를 입력에서 전달하기보다, 평균적으로 그럴듯한 구조를 만들 가능성이 크다.

이 실험은 “xyz 전체를 안 쓴다”는 뜻이 아니다. **중심과 extent는 xyz에서 계산되어 계속 사용된다.** 또한 attribute의 Gaussian 회전·scale에도 기하 관련 정보가 있으므로 “기하 정보가 완전히 없다”는 해석도 틀리다. 약한 것은 **셀 안에서 각 Gaussian 중심이 어디에 놓였는지에 대한 직접적인 정보 전달**이다.

### 4.4 attention이 난간 점을 전혀 안 읽는 것은 아님

난간 band에 기여한 점들의 attention mass도 확인했다. 예를 들어 cell 1003은 관련 점이 입력의 78.0%였고 평균 attention mass는 83.6%였다. Cell 235는 21.3% 대 32.2%였다.

따라서 “난간 점이 희소해서 attention이 전부 무시한다”는 설명도 이번 측정에서는 성립하지 않는다. **점을 읽더라도 정확한 좌표 배치를 충분히 반영하지 않고, 주로 속성으로 집계하는 것**이 더 가까운 진단이다. Attention mass만으로 정보 보존을 판정하면 안 된다.

## 5. 디코더 결과: 선의 위치와 모양이 함께 손상됨

### 5.1 위치 오차가 난간 폭에 비해 큼

상단 난간을 따라 7px band를 지정하고, GT 중심이 그 band 안에 있으며 화면상 최대 sigma가 20px 미만인 Gaussian 465개, 40개 셀을 분석했다. 화면 밖·뒤쪽 대응은 제외했다. Band에는 인접 차체가 일부 포함될 수 있으므로 이는 완벽한 semantic 난간 segmentation은 아니다.

| 지표 | 측정 |
|---|---:|
| GT→3D 최근접 예측점의 화면 위치 차이 중앙값 | **2.39 px** |
| 위 화면 위치 차이가 2px보다 큰 비율 | **61.7%** |
| 셀 안 Hungarian 대응의 화면 위치 차이 중앙값 | **5.07 px** |
| GT 화면 단축 sigma 중앙값 | **1.03 px** |

얇은 선은 가우시안 몇 개가 정확히 이어져야 한다. 선의 단축 sigma가 약 1px인데 중심이 2–5px 어긋나면 선이 끊기거나 배경에 섞일 수 있다. 큰 차체에서는 같은 오차가 덜 눈에 띈다.

두 대응 방식은 원본 Gaussian identity를 복원한 정답 대응이 아니다. 최근접 검색은 여러 GT가 한 예측점을 고를 수 있고, Hungarian은 동일 셀 안에서 질량을 맞추면서 더 먼 대응을 만들 수 있다. 그 차이를 함께 공개한 이유다.

### 5.2 길고 얇은 가우시안이 짧고 둥글어짐

같은 focus subset에서 위치 대응한 Gaussian의 투영 covariance를 계산했다. Rasterizer의 저역통과 항과 맞춰 2D covariance에 `0.3 I`를 더한 **sigma**이며 rasterizer가 반환하는 3-sigma 정수 radius와는 다르다.

| 화면상 모양, 중앙값 | GT | 모델 |
|---|---:|---:|
| 단축 sigma | **1.03 px** | **1.57 px** |
| 장축 sigma | **4.80 px** | **2.90 px** |
| 장축/단축 | **3.67** | **1.58** |

단순히 모든 splat이 커진 것만은 아니다. **짧은 축은 두꺼워지고 긴 축은 짧아져**, 난간을 이어 그리는 길쭉한 흔적이 둥근 점처럼 된다. 중심 오차와 합쳐져 상단 난간이 점선·얼룩처럼 보이는 현상과 일치한다. [focused_geometry.json](rail_detail_audit_20260915/focused_geometry.json).

### 5.3 작은 축의 표현 범위도 검사해야 함

현재 attribute scale은 `base(group_extent) ± 3`의 log-scale band에 있다. Focus subset의 **23.2%**는 GT의 가장 작은 scale이 현 checkpoint의 **어느 출력 축으로도 도달할 수 없는 lower bound 아래**였다. 전처리 floor에 잘린 타깃을 제외해도 같은 비율이었다.

이것은 GT의 특정 얇은 covariance를 그대로 복원하는 데 제약이 있다는 증거다. 하지만 가장 작은 3D 축이 시선 방향일 수도 있고, 다른 점 배치로 같은 이미지를 만들 수도 있으므로 **23.2%의 난간 픽셀이 cap 때문에 사라졌다는 뜻은 아니다.**

개선은 모든 cap을 푸는 방식보다, **전체 크기/화면 footprint 제한과 anisotropy 제한을 분리**하여 작은 축을 표현할 여지를 주는 방향이 적절하다.

## 6. 위치·속성 교체가 모두 악화된 이유

“GT scale만 붙이면 살아나는가”를 시험했다. 대응을 임의의 slot index로 정하지 않고, 같은 셀 안 위치 기반 Hungarian으로 맞췄다. 점 개수도 동일하게 고정했다.

| 교체 조건 | view3 전체 PSNR | 상단 난간 ROI |
|---|---:|---:|
| 모델 그대로, 공통 GT mask | 21.48 | 21.57 |
| covariance(scale+rotation)만 GT | 14.19 | 14.81 |
| opacity만 GT | 17.53 | 19.33 |
| color만 GT | 14.27 | 16.11 |
| 전체 attribute를 GT | 13.08 | 13.31 |
| xyz만 GT | 15.07 | 16.54 |

![위치·속성 교체 비교](rail_detail_audit_20260915/crop_ablation.png)

**해석:** 모델의 위치·scale·rotation·opacity·color가 서로 보정하면서 함께 맞춰져 있다. 한 항만 원본 Gaussian의 값으로 바꾸면 그 보정 관계를 깨뜨린다. 또한 위치 기반 대응이 실제 물리적 Gaussian identity를 보장하지도 않는다.

따라서 이 교체 결과를 “GT covariance가 오히려 나쁘다”, “속성은 문제없다”, “GT 속성을 얹은 결과가 기하의 이론적 성능 상한이다”로 해석하면 안 된다. 현재 구조는 **부정확한 기하를 다른 속성으로 감추는 해**에 도달할 수 있으며, 기하·속성의 공동 복원 및 대응을 함께 다뤄야 한다.

## 7. 현재 인코더–디코더 구조에서 개선할 부분

### 7.1 현재 활성 경로와 압축이 일어나는 위치

T16k의 실제 설정은 다음과 같다. `joint_decoder.py`, structured local code 등 저장소에 존재하는 다른 구조를 현재 활성 경로와 혼동하면 안 된다.

```text
셀당 입력 capacity 2048개 × [local xyz 3 + 표준화 attribute 11]
  → point embedding → learned query 16개 × 128
  → flatten + linear: 2048 → 1024
  → pack 1024 → 1024 → 32 tokens × 32
  → token embedding / attention: 32 × 448
  → concat merge: 14336 → 896 → 448
  → 셀 및 주변 context → mid 256
  → compact: [center/extent 4 | count 1 | shape 3 | appearance 8]
  → folding + deep residual
  → trainable unpack → geometry refinement → presence
  → AttributeDecoder: xyz nudge + scale/rotation/opacity/DC
```

총 compact는 **1024셀 × 16 = 16,384 scalar**다. 이번 NPZ의 실제 encoder 유효 점은 79,821개다. **2048은 셀당 capacity이지, 이 실험에서 모든 셀에 실제로 2048점이 있다는 뜻은 아니다.**

큰 차체는 셀 중심과 크기로 대략 배치할 수 있다. 하지만 난간에는 셀 내부에서 선이 지나가는 위치·방향과 다른 표면과의 분리가 필요하다. 현재는 이 정보가 첫 pooler부터 약하게 반영되고, 이후 여러 집계를 거쳐 하나의 셀 표현으로 합쳐진다.

### 7.2 최우선 구조 변경: pooler에 들어가기 전에 상대 좌표를 정규화

**수정 후보:** [encoder.py](../can3tok/encoder.py) L362–401, [compressor.py](../can3tok/compressor.py) L380–410.

첫 비교 실험은 다른 모듈을 늘리지 않고 다음 변경 하나로 시작하는 것이 좋다.

```text
현재: local_xyz = xyz - cell_center
제안: local_xyz = (xyz - cell_center) / max(cell_extent, epsilon)
```

- `cell_extent`는 compact의 extent와 같은 정의·단위를 사용한다. 기존 `encode_scale/decode_scale`의 scalar extent 정의를 재사용하면 해석이 일치한다.
- epsilon, 빈 셀, 점 하나인 셀의 처리를 명시한다. padding에는 mask를 계속 적용한다.
- xyz와 attribute는 별도 embedding을 거친 뒤 합치는 실험도 가능하다. 기하 branch를 추가한다면 gate와 마지막 layer를 동시에 0으로 두어 경로가 닫히지 않도록 한다.
- 여러 주파수의 좌표 encoding이나 상대 거리 bias는 정규화 효과를 확인한 다음 비교한다.

**주의할 구현 위치:** `compressor.py`의 `if not self.pooled_input` 조건을 없애는 방식은 부적절하다. Pooler **이후** 1024개 값은 이미 학습된 feature이며 앞 768개가 xyz라는 보장이 없다. 그 임의 구간을 extent로 나누면 의미 없는 변형이 된다. **metric xyz가 존재하는 pooler 이전을 수정해야 한다.**

입력 분포가 바뀌므로 기존 checkpoint에서 코드 한 줄만 바꾼 렌더를 개선 결과로 취급하지 않는다. 동일 학습 조건의 baseline과 재학습 또는 통제된 fine-tuning으로 비교해야 한다.

통과 기준은 feature rank 증가가 아니다. **셀 중심·크기를 고정한 채 내부 점 배치를 바꾸었을 때, 그 변화가 latent와 복원 기하에 의미 있게 전달되고 실제 난간 위치 오차가 줄어드는가**다. 단순히 입력에 민감한 불안정한 모델도 변화량은 크게 만들 수 있으므로 렌더·위치 정확도와 함께 판단한다.

### 7.3 pooling은 점을 읽는 것과 공간 구조를 보존하는 것을 구분해야 함

**위치:** [encoder.py](../can3tok/encoder.py) L127–199, [compressor.py](../can3tok/compressor.py) L166–193, L412–446.

현재 16개 query는 학습된 content query다. 각각이 셀의 특정 공간 영역이나 선의 특정 구간을 담당한다는 제약은 없다. 한 block만 쓰는 설정에서는 query끼리 별도로 조정하는 단계도 없다. 이후 query를 flatten하고 다시 token으로 나누므로 첫 query의 공간적 역할이 최종 compact까지 보존된다는 보장도 없다.

**부족한 점:** 정확한 좌표 신호가 약한 상태에서는 색·scale·opacity가 비슷한 점들이 비슷하게 집계될 수 있다. 난간 점이 높은 attention을 받아도, 차체와의 상대 위치나 선의 연속성이 보존되지는 않을 수 있다.

정규화 다음의 비교 후보:

1. 정규화된 셀 공간에 query 위치를 두고 거리 bias 또는 국소 이웃을 사용한다. 한 셀에 여러 구조가 있으면 query가 서로 다른 구조를 읽도록 한다.
2. 이미 구현된 `pool_blocks > 1` 경로를 별도 실험한다. block 수 증가만으로 성능 향상이 보장되지는 않는다.
3. 최종 compact로 줄이기 전까지 여러 summary의 구별을 유지하는 집계를 비교한다. 추가 계산량과 학습 시간을 함께 기록한다.

**이번 우선순위는 latent 증설이 아니다.** 고정 `4|1|3|8`에서 입력 기하를 제대로 쓰는지 먼저 확인한다. Shape 3개 값을 MLP로 여러 token으로 펼치는 것은 계산 표현을 바꾸는 것이며, 독립적인 저장 정보가 추가되는 것은 아니다. 반대로 shape 3개만 보고 복원이 수학적으로 불가능하다고 단정할 수도 없다. 현재 decoder는 주변 셀 및 전체 compact context도 읽는다.

### 7.4 부분 셀의 template·count·centroid 정의가 일치해야 함

**위치:** [data.py](../can3tok/data.py)의 anchor packing/template 정렬, [compressor.py](../can3tok/compressor.py)의 folding. 상세 코드 근거는 이전 보고서 §6.3.

이번 난간 focus 셀의 유효 점 수 중앙값은 **88.5개/256개**다. 모든 focus 셀이 부분 셀이다. 그런데 현재 template은 256개 전체를 기준으로 중심을 맞추고, 부분 셀에서는 앞 k개 slot을 유효하게 사용한다. 전체 template의 평균이 0이어도 앞 k개의 평균은 0일 필요가 없다. 타깃의 부분 셀은 full-cell Hungarian 정렬도 건너뛴다.

그 결과 decoder는 “이 셀의 실제 형상”뿐 아니라 “k개만 취했을 때 생기는 template 중심 편향”도 함께 보정해야 한다.

개선 방향:

- 복원 count에 맞는 균형 잡힌 template을 만들거나, 실제 유효 subset을 기준으로 local offset을 mean-center한다.
- deployment에서는 compact에서 복원한 count/presence를 쓴다. GT mask는 초기 학습·진단 용도로만 사용하고 실제 저장 latent 복원에 추가 입력으로 요구하지 않는다.
- count가 바뀔 때 slot 순서가 크게 뒤집히지 않도록 한다. NPZ snapshot 간 시간적 일관성도 확인한다.
- encoder와 target의 셀 소속 및 중심·count 정의를 공유한다. 독립 capacity packing의 문제는 이전 보고서 §5.1에 있다. 다만 이번 후기 snapshot의 소속 차이는 약 0.08%였으므로 상단 난간 실패의 주원인으로 과장하지 않는다.

### 7.5 좌표를 여러 단계가 수정함: 역할을 명시하되 기존 unpack을 바로 지우지 말 것

**위치:** [compressor.py](../can3tok/compressor.py)의 folding/deep residual, [decoder.py](../can3tok/decoder.py) L54, L223–226, [attr_decoder.py](../can3tok/attr_decoder.py) L344–350.

현재 좌표는 folding, deep residual, trainable unpack, refinement, attribute nudge를 거친다. 특히 `unpack`은 identity로 초기화한 **학습 가능한 전체 선형층**이다. xyz와 aux 구간이 계속 독립이라는 보장이 없다. 새 구조에서 “folding이 frame을 책임지고 나머지는 작은 국소 잔차만 만든다”는 의도를 갖는다면 실제 코드도 그 역할을 제한해야 한다.

하지만 **학습된 unpack을 지금 identity로 돌리는 것은 권하지 않는다.** 이번 checkpoint의 유효 셀 centroid 오차 중앙값을 decoded extent로 나누면 다음과 같다.

| 단계 | 유효점 centroid 오차 / extent |
|---|---:|
| folding | 0.349 |
| + deep residual | 0.283 |
| unpack 이후 | **0.124** |
| refinement 이후 | **0.108** |
| attribute nudge 이후 | 0.165 |

현재 unpack은 앞 단계의 오차를 실제로 보정하고 있다. Nudge가 centroid 오차를 늘려도 사진 손실에는 도움이 될 수 있으므로 이 표만으로 nudge를 오류라고 판정하지 않는다. [decoder_stages.json](rail_detail_audit_20260915/decoder_stages.json).

새 학습에서는 **셀 중심 이동, extent/방향, 중심을 보존하는 세부 residual**의 역할을 명시하는 구조를 비교한다. xyz/aux block 분리, bounded residual, 유효점 기준 zero-mean 등을 후보로 두되 학습된 baseline과 비교해야 한다. 단계별 좌표 오차와 최종 사진 품질을 함께 기록한다.

### 7.6 확정된 불필요한 의존성: 빈 슬롯이 유효 가우시안의 속성을 바꿈

**위치:** [attr_decoder.py](../can3tok/attr_decoder.py) L312–315, L331–332, L352–354.

AttributeDecoder의 local 좌표 평균·반경은 전체 256개로 계산하고, self-attention에도 유효 slot mask를 전달하지 않는다. 따라서 렌더에서 버리는 slot의 위치가 실제 사용하는 slot의 속성 계산에 참여한다.

이를 직접 검증했다. Compact와 유효 coarse xyz는 고정한 채 **GT-inactive slot의 x만 셀 extent의 0.5배 이동**시키고 AttributeDecoder만 다시 실행했다.

| 조건 | view3 전체 PSNR | 상단 난간 ROI |
|---|---:|---:|
| 원래 decoder 입력 | 21.48 | 21.57 |
| inactive xyz만 이동 | **21.12** | **21.09** |

원래 입력으로 AttributeDecoder를 재실행한 값은 기존 출력과 최대 차이가 **0**이었다. 따라서 비교 경로 불일치로 생긴 차이가 아니다. 유효 Gaussian의 attribute도 달라졌다. [setup.json](rail_detail_audit_20260915/setup.json), [render_metrics.json](rail_detail_audit_20260915/render_metrics.json).

**판정:** 실제로 그려지지 않는 점의 좌표에 유효 출력이 의존한다. 이동한 빈 점이 화면에 직접 그려졌다는 뜻은 아니다.

개선은 유효 count/presence로 weighted mean·radius를 계산하고, self-attention key mask 또는 동등한 가중 방식을 전달하는 것이다. 완전히 빈 셀에서는 all-masked attention의 NaN 처리도 필요하다. GT mask 사용 시의 결과와 predicted presence 사용 시의 결과를 모두 검증한다.

### 7.7 covariance·이웃 context·시점 의존 색상

**Covariance:** 단순 xyz Chamfer만으로는 가는 선의 방향·두께까지 복원되지 않는다. 현재 이미 covariance 관련 감독이 있으므로 손실을 새로 추가한다고 해결되는 문제가 아니다. 같은 대응에서 위치와 covariance가 함께 맞는지, 화면 단축/장축이 어떻게 바뀌는지 추적해야 한다. Scale은 전체 크기와 anisotropy를 분리하고, 큰 footprint를 억제하는 제약이 작은 축까지 막지 않는 표현을 비교한다. Quaternion 성분 L2보다 축 순서·회전 표현의 동치성을 고려한 실제 covariance 비교가 적합하다.

**이웃:** `attr_nbr_window=1`은 배열상 앞뒤 셀이다. [attr_decoder.py](../can3tok/attr_decoder.py) L368–377. 3D 거리상 최근접 셀이라는 보장은 없다. Decode된 compact 중심으로 실제 공간 이웃을 구하는 대안을 비교할 수 있다. 추가 GT geometry 없이 계산할 수 있어야 하고, 이웃 검색 비용·snapshot 간 변화도 평가해야 한다.

**시점 의존 색상:** SH3→DC 차이가 측정되었으므로 appearance 8개에서 SH 계수나 view-conditioned color를 생성하는 대안을 검토할 근거는 충분하다. Decoder의 출력 채널을 늘리는 것과 저장 latent를 늘리는 것은 다르다. 다만 같은 8개 값에서 더 많은 함수를 학습하므로 학습 난도·복원 한계는 별도로 검증해야 한다. 현재 pipeline이 로드한 원본 SH 45개도 학습 입력/타깃까지 일관되게 사용하도록 바꿔야 한다.

3DGS 원 논문은 anisotropic covariance 최적화를 핵심으로 설명한다. 이번 진단의 수치 근거는 논문이 아니라 위 checkpoint 실험이다. [저자 공식 프로젝트](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). 투영 sigma 계산에 참고한 저역통과 항은 [공식 rasterizer 구현](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu)의 `computeCov2D`에도 있다.

## 8. 손실·평가 코드에서 먼저 바로잡아야 할 부분

구조 변경의 효과를 판단하려면 다음 오류가 통제된 baseline이 필요하다. 상세 재현은 이전 보고서 `12`에 있다.

| 항목 | 현재 문제 | 개선 및 판단 기준 |
|---|---|---|
| 최종 decode | `decode_compact()`가 최종 AttributeDecoder/nudge를 빠뜨림; latent 정규화도 encode 기본값과 불일치 | 저장→복원 결과가 학습 forward의 최종 Gaussian·presence·렌더와 일치해야 함 |
| 카메라/사진 | scene이 다른 view가 섞인 평가와 held-out 사진 유입 | scene별 평가 및 사진 분리; clean holdout을 별도로 구성 |
| 상세 사진 감독 | extra view를 절반 크기로 읽은 뒤 확대 | detail 단계에서 원본 해상도 타깃 사용; 실제 loss target crop 저장 |
| edge/visibility | pointer 기반 edge cache, 이전 샘플 visibility 재사용 | 현재 scene/view/source identity로 계산; cache 제거부터 비교 |
| 공간 감독 | 정렬 후 stride/prefix 선택이 공간 일부를 누락 | 전체 범위가 포함되는 선택 및 유효 mask; 샘플 분포 시각 확인 |
| Gaussian 대응 | geometry와 attribute 손실의 대응이 다름; soft attribute 혼합 가능 | 공동 대응의 위치/covariance/opacity/color 오차 및 실제 Sinkhorn 수렴 확인 |

이번 추가 실험은 올바른 최종 `attr_pred`와 실제 scene 카메라로 렌더했다. 따라서 위 public decode 오류가 **이번 forward 이미지의 난간 소실까지 설명하는 것은 아니다.** 저장 latent로 별도 복원한 이미지라면 이 오류가 추가로 작용할 수 있다.

Edge 가중치, p2g, Sinkhorn 등은 이미 코드에 있다. 이를 새 아이디어처럼 나열하거나 가중치부터 올리기보다 **올바른 점·사진에 손실이 적용되는지**를 먼저 수정해야 한다. `attr_detach_geometry`도 현재 70k에서는 해제된 schedule을 사용했다. 과거 주석만 보고 render gradient가 영구적으로 차단됐다고 해석하면 안 된다.

## 9. 권장 실험 순서와 성공 기준

모든 구조 실험은 우선 **같은 16,384 scalar와 `4|1|3|8` 배분**을 유지한다. 난간만 좋아지고 다른 셀·시점이 나빠지는 변경을 걸러내도록 같은 카메라·ROI·학습 횟수·seed를 사용한다.

| 순서 | 변경 | 확인할 것 |
|---|---|---|
| A | §8의 평가·추론·타깃 오류를 바로잡은 baseline | 최종 forward/decode 일치, 실제 full-resolution 사진, scene별 결과 |
| B1 | **pooler 전 local xyz의 extent 정규화만 변경** | 내부 좌표 개입에 대한 의미 있는 기하 반응, 난간 2D 위치 오차 감소, 고정 ROI 개선 |
| B2 | B1을 기준으로 별도 geometry embedding 또는 공간 query 중 하나 | 추가 branch가 실제 gradient와 정보를 전달하는지; 학습 비용 대비 개선 |
| C1 | count에 맞는 template/중심 처리 | 부분 셀 중심 오차, count 변화와 snapshot 간 안정성 |
| C2 | attribute 통계와 attention에 valid mask 반영 | inactive slot 이동 시 유효 출력의 불필요한 변화 제거 |
| D1 | 위치·covariance 공동 대응 및 작은 축 표현 개선 | 단축/장축 sigma, 위치 오차, 연결된 난간, 큰 splat/floater 악화 여부 |
| D2 | 같은 appearance 예산에서 SH 또는 시점 의존 색상 출력 | DC 대비 여러 시점의 선명도·색상 개선과 geometry 유지 |

B1/B2/C1/C2를 처음부터 한꺼번에 바꾸면 무엇이 효과가 있었는지 알기 어렵다. 작은 단일 snapshot fitting으로 기본 전달 경로를 확인한 다음, 동일 예산의 여러 snapshot·scene 학습으로 확장한다. 수정안의 성공을 주장하려면 다음을 함께 제시해야 한다.

- 원본 NPZ → 필터 후 타깃 → 선택 타깃 → 모델의 동일 시점 crop. 상단 난간과 앞 계단 난간은 별도로 본다.
- 전체 PSNR/SSIM뿐 아니라 난간·글자 ROI의 gradient 오차와 선의 연결 상태. ROI 점수는 인접 차체 때문에 좋아질 수 있으므로 이미지도 확인한다.
- 난간 기여 Gaussian의 화면 위치 오차 및 투영 covariance. 대응 방식과 subset 선택 조건을 고정한다.
- GT mask와 predicted presence의 차이, 셀 count 오차, 주변 시점에서 이동해 보이는 floater.
- 셀 내부 xyz 개입/attribute 개입에 대한 latent와 렌더 반응. 입력 변화량과 학습 성능을 함께 기록한다.
- 압축 예산·실제 저장 bytes·학습 시간·peak memory. Decoder가 원본 NPZ나 GT slot mask에 의존하지 않는 독립 decode.

이번에 손으로 지정한 난간 band는 **원인 분석용**이다. 일반 학습에 쓸 가중치라면 여러 사진의 edge와 3D 가시성 등 재현 가능한 기준으로 구성하고, 평가용 사진에서 정보를 가져오지 않아야 한다.

## 10. 재현 자료와 한계

### 10.1 저장한 자료

| 자료 | 용도 |
|---|---|
| [probe_rail_detail.py](rail_detail_audit_20260915/probe_rail_detail.py) | 실제 데이터 경로의 추론, raw/selected/model 렌더, 위치·속성 교체, inactive slot 개입 |
| [render_selected_sh.py](rail_detail_audit_20260915/render_selected_sh.py) | 원본 source index를 유지한 선택 타깃 SH3 렌더 |
| [analyze_focus_and_figures.py](rail_detail_audit_20260915/analyze_focus_and_figures.py) | CPU focus 기하 분석, 첫 pooler 입력 개입, 비교 crop 생성 |
| [probe_pool_to_render.py](rail_detail_audit_20260915/probe_pool_to_render.py) | analytic anchor를 보존한 전체 encoder 입력 개입 |
| [setup.json](rail_detail_audit_20260915/setup.json) | checkpoint/NPZ/ROI/교체 조건 |
| [render_metrics.json](rail_detail_audit_20260915/render_metrics.json) | 각 렌더의 사진·타깃 비교와 고정 ROI 점수 |
| [focused_geometry.json](rail_detail_audit_20260915/focused_geometry.json) | 465개 focus Gaussian의 위치·투영 모양·scale band 통계 |
| [pool_sensitivity.json](rail_detail_audit_20260915/pool_sensitivity.json), [pool_to_render.json](rail_detail_audit_20260915/pool_to_render.json) | local xyz/attribute 개입 결과 |
| [decoder_stages.json](rail_detail_audit_20260915/decoder_stages.json) | 단계별 유효점 centroid 오차 |
| [gaussian_bundle.npz](rail_detail_audit_20260915/gaussian_bundle.npz), [rail_contributions.npz](rail_detail_audit_20260915/rail_contributions.npz) | 원본 index·대응·좌표·속성·band 합성 기여도 |
| [verification.json](rail_detail_audit_20260915/verification.json) | 기존 소스 107개 해시 일치, 진단 스크립트 4개 문법 및 수치 JSON·문서 링크 점검 |

`rail_geometry.json`은 band와 겹치는 모든 Gaussian을 먼저 집계한 탐색 자료다. 화면 밖 중심의 큰 splat도 포함하므로 그 파일의 극단적인 투영 오차를 난간 위치 오차로 인용하면 안 된다. 본문은 이후 화면 중심·깊이·크기 조건을 고정한 **focused_geometry.json**을 사용했다.

### 10.2 재현 명령

프로젝트 루트에서 사용한 환경은 `/home/super/anaconda3/envs/can3tok/bin/python`이다. 앞 단계가 만드는 파일을 다음 단계가 읽으므로 순서대로 실행한다. 첫 명령은 `--device`를 받지만, 나머지 CUDA 스크립트는 GPU 3을 지정한다. 다른 GPU에서 재현하려면 해당 스크립트의 device 설정도 맞춘다. CPU 분석은 checkpoint와 앞선 렌더 자료만 읽는다.

```bash
cd /data/daeho/aacd_proj/can3tok_encoder_decoder_new_fix_7

/home/super/anaconda3/envs/can3tok/bin/python mdmd/rail_detail_audit_20260915/probe_rail_detail.py --device cuda:3

/home/super/anaconda3/envs/can3tok/bin/python mdmd/rail_detail_audit_20260915/render_selected_sh.py

/home/super/anaconda3/envs/can3tok/bin/python mdmd/rail_detail_audit_20260915/probe_pool_to_render.py

/home/super/anaconda3/envs/can3tok/bin/python mdmd/rail_detail_audit_20260915/analyze_focus_and_figures.py
```

SH 계산은 로컬 원본 구현 `/data/daeho/aabb/gaussian-splatting/utils/sh_utils.py`를 사용한다. NPZ의 DC와 15×3 SH 계수를 합쳐 Gaussian→camera 상대 방향에 따라 degree 3 색을 계산하고 같은 renderer에서 비교했다.

§3/§6의 기본 추론과 §4.3의 전체 입력 개입은 chunk 크기가 다르다. BF16 연산 순서에 따른 작은 차이로 상단 ROI baseline이 각각 21.57/21.58 dB다. **각 표 내부에서는 모든 비교 조건의 chunk·정밀도·mask가 같다.**

### 10.3 이번에 확정하지 않은 것

- 기차 NPZ 하나, T70k checkpoint 하나, 사진 두 시점의 진단이다. 모든 scene·학습 단계에 같은 기여율을 적용할 수 없다.
- 위치 정규화가 약한 좌표 전달의 유력한 원인이지만, 그 변경을 재학습하여 난간 복원 개선까지 입증한 것은 아니다. 가중치의 학습 이력과 objective도 함께 작용할 수 있다.
- SH·geometry·covariance·사진 해상도의 영향은 비선형으로 결합한다. 각 dB 차이를 더해서 원인별 퍼센트로 분해할 수 없다.
- Gaussian identity에 대한 정답 대응은 없다. Hungarian/최근접 및 hand-drawn band의 한계를 본문에 명시했다.
- 원본 NPZ 저장 이전의 FP32 상태는 이번 실험에 없다. 다만 저장된 NPZ에도 보이는 난간을 모델이 잃는 현상은 직접 확인했다.
- 특정 floater 하나의 생성 경로를 모든 시점에서 역추적한 실험은 아니다. 빈 slot 의존성·부분 셀 중심·위치/속성 보상은 관련 구조 문제이며, 각 floater의 단독 원인으로 단정하지 않는다.

## 최종 제안

**가장 먼저 할 구조 실험은 pooler 이전의 셀 내부 xyz 정규화다.** 현재 모델이 실제 점 배치를 거의 무시해도 비슷한 이미지를 만든다는 직접 근거가 있기 때문이다. 그다음 부분 셀의 좌표 정의와 AttributeDecoder의 mask 처리를 맞추고, 얇은 covariance를 표현·감독할 수 있도록 개선하는 순서가 적절하다.

난간 소실은 출력 점 수나 loss 가중치 하나의 문제가 아니다. **정밀 좌표의 입력 전달 → 셀 안의 올바른 점 배치 → 길고 얇은 covariance → 올바른 사진 감독**이 함께 맞아야 연속된 선이 복원된다.
