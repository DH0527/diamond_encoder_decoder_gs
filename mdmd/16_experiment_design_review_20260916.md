# B-only 제안 재검토와 GPU 0·1·2 실험 설계

작성: 2026-09-16  
검토 대상: `14_contract_baseline_20260916.md`, `15_contract_verify_20260916.md`, 현재 코드·S/T/E1 args·런처  
추가 검증: 데이터 8개 snapshot CPU 비교, 4개 snapshot × 2개 카메라 타깃 렌더, 기존 계약 테스트 9개  
**이번 작업에서 새 학습은 실행하지 않았다.**

## 1. 결론: B-only의 취지는 맞지만 제시된 플래그를 그대로 켜면 안 됨

평가·추론을 바로잡은 상태에서 새로운 기준선을 만들고, 그다음 구조를 하나씩 바꾸는 순서는 타당하다. 다만 현재 코드를 실제 데이터에 적용하니 다음 문제가 확인되었다.

1. **`shared_cell_owner=1`은 소속만 고치는 변경이 아니다.** 현재는 anchor에 가까운 256개를 타깃으로 취하면서 복원할 점 집합도 크게 바꾼다. 초기 truck snapshot의 타깃 PSNR이 **18.16 → 11.31 dB**로 악화됐다. 학습 결과가 아니라 모델에 주기 전 타깃의 변화다.
2. 기존 런처의 **`max_steps=16000`은 S의 64k cosine 학습률도 16k로 줄인다.** S와 같은 스케줄이라고 부르면 안 된다.
3. S는 **12k에 geometry detach 해제**, **40k에 detail 단계 시작**, **43k에 detail ramp 완료**다. 8k에서 난간 복원의 최종 성공·실패를 결정할 수 없다.
4. Raw latent의 decode 일치는 확인됐다. 그러나 정규화한 latent의 왕복 오차까지 전부 0인 것은 아니다.

**추천:** 당장 비교할 두 조건은 `B0`와 `C1`이다. B0는 사진 누수·해상도·이전 visibility 문제를 정리하되, 현재 타깃을 크게 바꾸는 shared-owner 구현은 끈다. C1은 B0에서 **pooler xyz 정규화 하나만** 켠다. 공유 소속은 기존 타깃을 보존하는 방식으로 따로 수정·검증한 뒤 별도 실험한다.

이것은 기존 독립 packing을 최종 설계로 인정한다는 뜻이 아니다. **아직 문제가 있는 새 선택 정책을 첫 기준선에 섞지 않기 위한 비교 조건**이다.

## 2. 제시된 설명에서 맞는 것과 보완할 것

| 설명 | 확인 결과 |
|---|---|
| raw compact decode가 forward와 같음 | 세 checkpoint·두 snapshot에서 xyz/최종 attribute 차이 0. 현재 공통 `_decode_gaussians()` 경로와도 일치 |
| 자기 씬 PSNR이 이전 감사와 일치 | `verify.json`과 표 일치. 18.33/21.51은 해당 snapshot·9장·평가 해상도 조건의 값 |
| 혼합 14.x를 주 지표에서 제외 | 맞음. 다른 scene 카메라의 사진과 비교한 값이 섞여 있음 |
| `holdout_own_photo=1` 필요 | 맞음. 101은 train split의 누수 항목 수, 238은 photo map 전체 수로 모집단이 다름 |
| E1을 첫 기준선으로 삼지 않음 | 합리적인 실험 선택. 단, 16k E1과 62k/70k S/T만으로 E1 구조의 최종 열세를 입증하지는 못함 |
| T를 resume하지 않고 새로 학습 | 새 데이터 조건의 비교에는 적절. “centroid가 무너져 복구 불가능”까지 입증된 것은 아님 |
| fingerprint가 바뀌면 해결됨 | `best_score`만 초기화. 모델·optimizer·학습 이력을 초기화하지 않으며 B/C 설정 전체를 추적하는 fingerprint도 아님 |
| B는 데이터 정의만 고침 | 범주상 맞지만 실제 타깃 선택까지 바뀜. 코드의 unconditional sampler/projection/edge 변경도 이전 S와 다름 |
| 8k를 최종 난간 gate로 사용 | 부적절. 초기 상태 점검과 detail 학습 후 평가를 구분해야 함 |

### 2.1 정규화한 latent의 왕복은 추가 확인 필요

[verify.json](verify_20260916/verify.json)의 `normalized_xyz_max_vs_forward`는 다음과 같다.

| 모델 | 기차 | truck |
|---|---:|---:|
| T70k | 0.04833 | 0.000746 |
| S62k | 0.002128 | 0.04436 |
| E1 16k | 0.000691 | 0.01412 |

Raw `forward()['z_compact']`를 decode한 경로는 정확히 같다. 위 값은 별도 `encode_compact(normalized=True)` 호출 후 되돌린 경로다. BF16 연산·나눗셈/곱셈·별도 encode의 차이를 분리하지 않은 상태이므로 **원인을 수치 정밀도로 단정하지 않는다.**

최대값은 전체 슬롯의 normalized coarse xyz 기준이며 유효점과 padding을 분리하지 않았다. 배포 검증에서는 같은 raw z를 FP32로 normalize→denormalize하는 경우와 저장 dtype 왕복을 따로 비교하고, **최종 attr_pred의 유효 xyz·속성·presence·렌더** 차이를 측정해야 한다. 이 확인은 raw latent로 비교하는 B0/C1 학습 설계와 별개의 항목이다.

### 2.2 “진짜 점수”의 범위

현재 검증과 `run_eval()`은 `args.render_downscale=2`를 사용한다. 따라서 18.33/21.51은 full-resolution 난간 품질 점수가 아니다. 보고서 13의 977px 렌더와 직접 숫자를 비교하면 안 된다.

사진 누수를 막아도 그 의미는 우선 **codec의 직접 사진 감독에서 제외한 시점**이다. 원본 3DGS의 train/test 분리도 별도다. 저장된 원본 `cfg_args`는 `eval=True`, 현재 원본 loader는 정렬 후 매 8번째 사진을 test로 사용한다. 현재 codec held-out 인덱스 `[3,26,42,65,93,229,238,239,243]`에는 8의 배수가 없다. 전체 사진을 정렬하는 asset 생성 코드와 대조하면 이 9장은 원본 3DGS의 train 시점에 해당하는 것으로 해석된다. 완전히 새로운 scene 일반화나 원본 3DGS도 보지 않은 시점의 성능으로 표현하지 않는다.

## 3. 새로 확인한 shared-owner 타깃 손상

### 3.1 실제 구현

[data.py](../can3tok/data.py)의 `_pack_shared()`는 다음 순서다.

```text
encoder capacity 2048로 source owner 배정
  → 각 cell의 live index에서 앞 256개 선택
  → 그 점들을 target으로 사용
```

`_anchor_assign()`의 셀 내부 순서는 anchor 거리다. 따라서 이 prefix는 셀 전체를 균형 있게 대표하는 선택이 아니라 **중심에 가까운 점을 우선하는 선택**이다. 기존 `_select(... max_points)`로 고른 타깃을 유지하지도 않는다.

### 3.2 점 집합 및 렌더 검증

기존 평가에 사용하던 8개 snapshot에서 검사한 결과 **같은 source의 encoder/target owner 불일치는 0**이었다. 그 계약은 만족한다. 하지만 타깃 집합 보존은 별개의 문제다.

| scene / snapshot | 기존 타깃 개수 | shared 타깃 개수 | view3 기존 PSNR | shared PSNR | view93 기존 → shared |
|---|---:|---:|---:|---:|---:|
| 기차 013250 | 227,640 | 212,236 | 17.68 | **15.83** | 17.08 → 15.15 |
| 기차 028630 | 79,821 | 79,756 | 19.21 | 19.20 | 19.96 → 19.94 |
| truck 013250 | 227,052 | 242,433 | 18.16 | **11.31** | 19.80 → **12.38** |
| truck 028750 | 158,577 | 150,623 | 21.72 | **21.07** | 20.76 → **19.64** |

조건: **모델 미사용**, 각 snapshot의 선택 타깃 DC-only, 같은 사진·카메라, 원본 해상도. 정렬만 바꾸는 Hungarian 순열은 생략했으며 CPU 원래 경로의 타깃 개수와 일치함을 검사했다. 수치 출처: [target_render.json](experiment_design_20260916/target_render.json), [preflight_data.json](experiment_design_20260916/preflight_data.json).

초기 truck은 점 수가 더 많아졌는데도 나빠졌다. 단순 keep ratio로 선택 품질을 판정할 수 없다. 기차 013250의 상단 난간 ROI도 **22.69 → 21.66 dB**로 떨어졌다. 이전 확인에 쓴 후기 기차 028630만 보면 이 문제가 거의 보이지 않는다.

![왼쪽 사진, 가운데 기존 타깃, 오른쪽 shared-owner 타깃](experiment_design_20260916/targets_val229_view3.png)

이 표가 최종 학습 모델도 정확히 같은 dB만큼 나빠질 것이라는 뜻은 아니다. 모델은 사진 손실로 타깃 오류를 보정할 수 있다. 그러나 “정보는 유지하면서 소속만 고친 기준선”이라는 전제는 성립하지 않는다.

### 3.3 공유 소속을 고칠 방향

권장하는 첫 구현은 **기존 타깃을 먼저 확정하고 encoder에 그 소속을 일치시키는 방식**이다.

1. 현재 target selection·packing으로 `(source_id, target_cell)`을 확정한다.
2. Encoder 입력에는 선택된 target source를 반드시 포함하고, 그 source는 같은 cell에 넣는다.
3. 남는 encoder capacity에 추가 점을 넣는다. 타깃 source를 다른 cell로 옮기거나 빼지 않는다.
4. Cell overflow의 정책을 명시하고, target/encoder source mapping과 중복을 검사한다.
5. 기존 target과 새 target의 source 집합 및 렌더가 같음을 먼저 확인한다.

최종적으로 같은 소속과 타깃 품질을 모두 확보해야 한다. 단순히 prefix를 균등 sampling으로 바꾸는 것도 별도 선택 정책이므로 렌더 검증이 필요하다.

또한 owner를 맞췄다고 중심·count까지 자동으로 같아지는 것은 아니다. Encoder가 읽는 점은 여전히 최대 2048개, 타깃은 최대 256개다. 공통 좌표 원점을 무엇으로 정의하고 어떤 중심 오차를 감독할지 별도로 정해야 한다. 이를 해결하려고 `use_fixed_anchor_center`를 이번 비교에서 같이 바꾸지는 않는다.

## 4. 학습률과 단계: 16k latent와 16k 학습 step을 구분

`S16k`의 16k는 **저장 latent 16,384 scalar**를 뜻하고, S 학습의 `max_steps`는 **64,000**이다.

[schedule.py](../can3tok/schedule.py)의 `global_lr()`는 `args.max_steps`를 cosine 분모로 쓴다.

| step | S와 같은 64k horizon | 현재 계약 런처의 16k horizon |
|---|---:|---:|
| 8k | 1.9353e-4 | 1.0981e-4 |
| 12k | 1.8503e-4 | 3.9548e-5 |
| 16k | 1.7341e-4 | **1.0000e-5** |

16k 중간 점검을 하고 싶다는 이유로 `max_steps=16000`을 넣으면, 모델이 그 전에 받는 업데이트 크기까지 바뀐다. 학습을 잠시 멈추는 시점과 LR horizon은 분리해야 한다.

이번 런처는 **64k horizon을 유지**한다. 8k/16k에 자동으로 멈추는 기능은 없다. 자동 중간 종료가 필요하다면 `stop_after_steps`를 LR horizon과 독립된 실행 옵션으로 추가하고 모든 DDP rank가 같은 step에서 저장·종료하도록 해야 한다. 현재 코드에는 그 기능을 추가하지 않았다.

| S schedule의 시점 | 의미 | 판단 용도 |
|---|---|---|
| 2k | 렌더 및 attribute 학습 초기, teacher forcing 중 | NaN·입출력·mask·학습 경로 확인 |
| 8k | teacher forcing 종료, render downscale 약 2, geometry detach는 유지 | 초기 기하와 정보 전달 진단 |
| 12k | geometry detach 해제 | 사진 gradient가 geometry에 들어가기 시작 |
| 16k / 24k | 해제 후 학습 | B0/C1 중간 비교, 붕괴·floater·난간 추세 |
| 40k | detail 단계 시작 | full-resolution 및 edge/perceptual ramp 시작 |
| 43k | detail ramp 완료 | 높은 해상도 감독의 효과 확인 가능 |
| 48k / 64k | detail 학습 누적 | 난간 디테일의 주 비교 지점 |

따라서 S schedule을 유지하면서 8k에 난간이 안 살아났다는 이유만으로 C1을 폐기하지 않는다. 반대로 명확한 발산·NaN·완전한 기하 붕괴라면 긴 실행을 지속할 이유는 없다.

## 5. 추천하는 실험 행렬

A의 scene-tagged eval과 공통 decode, 이미 반영된 edge/sampler/projection 수정은 **모든 새 조건에 공통**이다. S/T와의 차이를 B의 단일 인과 효과로 해석하지 않는다.

| 조건 | B0: 첫 기준선 | C1: 첫 구조 실험 | C2a: 후속 | C2b: 후속 |
|---|---:|---:|---:|---:|
| holdout_own_photo | 1 | 1 | 1 | 1 |
| keep_extra_fullres | 1 | 1 | 1 | 1 |
| reuse_prev_vis | 0 | 0 | 0 | 0 |
| shared_cell_owner, 현재 구현 | **0** | **0** | **0** | **0** |
| normalize_pooler_xyz | 0 | **1** | 1 | 1 |
| attr_slot_mask | 0 | 0 | **1** | 1 |
| count_aware_template | 0 | 0 | 0 | **1** |
| eval_pred_mask | 1 | 1 | 1 | 1 |
| latent / 배분 | 16,384 / `4\|1\|3\|8` | 동일 | 동일 | 동일 |
| LR·단계 schedule | S의 64k | 동일 | 동일 | 동일 |
| 시작 weight | fresh | fresh | fresh | fresh |

C2a/C2b는 C1의 결과를 본 뒤 진행한다. Mask와 count template을 함께 바꾸는 것보다 따로 비교하면 원인을 구분하기 쉽다. 효과가 없는 변경을 자동으로 누적하지 않는다.

**핵심 비교는 B0 ↔ C1이다.** 둘의 데이터·타깃·코드·LR 조건을 같게 하고, 첫 pooler의 좌표 정규화만 다르게 한다. 보고서 13에서 직접 확인한 약한 좌표 전달이 개선되는지 가장 명확하게 검증할 수 있다.

공유 소속을 먼저 고치기로 한다면, §3.3의 target-preserving 구현을 검증한 뒤 **두 조건 모두** 같은 구현을 켠다. 한 조건만 타깃 집합을 바꾸고 C1 효과를 비교하면 안 된다.

### 5.1 Resume와 from-scratch의 구분

- B0를 시작할 때 S/T/E1 weight·optimizer를 가져오지 않는다.
- C1도 독립적으로 같은 초기화 조건에서 시작한다. B0 checkpoint에 normalize flag만 켜서 이어 학습한 것은 **별도의 fine-tuning 실험**이다.
- 동일한 B0 실험이 중단되어 그 조건 그대로 재개되는 것은 가능하다. 그것과 S/T를 새 기준선으로 재사용하는 것은 다르다.
- 한 비교 쌍을 수행하는 동안 데이터 정책·loss 구현·schedule을 고정하고 소스 해시를 남긴다.

### 5.2 같은 seed만으로 사진 스트림까지 같아지지는 않음

현재 [data.py](../can3tok/data.py)는 own-photo 대체와 extra-view 선택에 seed 없는 `np.random.default_rng()`를 사용한다. `seed=42`를 같게 해도 사진 순서가 정확히 맞지 않는다. NumPy는 seed가 없으면 OS에서 새 entropy를 가져온다고 명시한다. [NumPy 공식 문서](https://numpy.org/doc/stable/reference/random/generator.html).

엄격한 paired comparison에는 다음 공통 준비가 필요하다.

- rank/worker별로 seed를 명시한 **지속되는 view RNG**를 만든다. 호출마다 같은 seed로 새 RNG를 만들면 매번 같은 사진만 고르는 또 다른 문제가 생긴다.
- worker persistent 상태와 재개 시 RNG 상태를 다룬다. 기존 `dataset.set_epoch()`만으로 worker 사본까지 갱신된다고 가정하지 않는다.
- 선택한 own/extra image ID, scene, camera를 기록하고 held-out 교집합이 0인지 검사한다.

이 RNG 변경은 이번에 적용하지 않았다. 현재 런처로도 pilot은 가능하지만 **사진까지 완전히 같은 결정론적 A/B**라고 부르면 안 된다. 효과가 작으면 seed 반복과 stream 차이를 함께 평가한다.

## 6. GPU 0·1·2 운영

확인 당시 세 GPU는 각각 RTX 5000 Ada 32GB이며 idle이었다. GPU 3에는 별도 작업이 있었다.

**첫 권장 배치는 GPU 0·1·2를 함께 사용하는 3-rank DDP 한 실험**이다. B0를 실행하고 동일 GPU 구성으로 C1을 순차 실행한다.

```text
GPU 0: rank 0 ─┐
GPU 1: rank 1 ─┼─ 한 모델, gradient 동기화
GPU 2: rank 2 ─┘
```

S의 per-rank batch size 1을 유지하면 한 optimizer step에 3개 snapshot을 처리한다. DDP는 rank별 replica의 gradient를 동기화하며 입력 분할은 sampler가 맡는다. 현재 코드도 `DistributedSampler`를 사용한다. [PyTorch 공식 DDP 문서](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html).

GPU마다 다른 실험 하나씩을 띄우는 방법도 있지만, 현재 코드에 gradient accumulation이 없으므로 batch 1×1 GPU와 batch 1×3 GPU는 동일 학습 조건이 아니다. 별도 single-GPU 탐색으로 설계하는 경우에만 사용하고 기존 S/DDP 조건과 같은 step 숫자를 직접 비교하지 않는다.

장시간 실행 전에는 기본 실행뿐 아니라 **12k gradient 경로와 43k full-resolution 조건의 backward 메모리**도 확인해야 한다. 초반 짧은 smoke test만으로 후기 메모리를 검증할 수 없다. 필요하면 양 조건에 동일한 chunk 조정을 적용하고 기록한다.

기존 S 로그는 구간에 따라 약 4.3–5.0초/step이었다. 단순 환산하면 16k는 약 19–22시간, 64k는 약 76–89시간이며 평가·I/O·full-resolution 변경 비용은 별도다. 이는 예약 시간 보장이 아니다. 새 조건의 실측 처리량으로 다시 산정한다.

## 7. 기록할 지표와 의사결정

### 항상 고정할 평가 집합

- 기존 val index 8개: `[25,76,127,166,229,280,331,382]`.
- 기차·truck 각각 후기 snapshot뿐 아니라 초기/중기 snapshot도 포함한다.
- 기존 9개 camera는 **codec validation**으로 유지하고 scene별로 분리한다.
- 기존 half-resolution PSNR은 비교용으로 유지한다. **full-resolution PSNR/SSIM·난간 crop은 별도 evaluator 설정으로 기록**한다.
- `run_eval()`에서 `args.render_downscale`를 읽는 부분이 있으므로, full-resolution 평가를 위해 학습의 `render_downscale` 자체를 1로 바꾸지 않는다. 평가 전용 downscale을 따로 전달하거나 오프라인 evaluator를 사용한다.
- 동일한 step의 checkpoint끼리 우선 비교한다. 서로 다른 시점의 best checkpoint만 비교하면 학습량 차이가 생긴다.

### 표와 이미지에 함께 남길 것

| 영역 | 지표 |
|---|---|
| 최종 렌더 | scene별 사진 PSNR/SSIM, predicted presence 기준 |
| 선택 타깃 | 원본/선택 DC/선택 SH/모델을 구분. 선택 DC 점수를 이론적 천장이라고 부르지 않음 |
| 난간·글자 | 동일 ROI crop, gradient 오차, 선의 연결 상태. 인접 차체 점수에 가려지지 않게 이미지 확인 |
| 기하 | 유효 Gaussian의 투영 중심 오차, 단축·장축 sigma와 이방성 |
| 정보 전달 | anchor 고정 + local xyz 제거 실험의 latent/최종 렌더 반응 |
| mask/count | GT mask와 predicted presence 차이, 빈 slot 개입에 대한 유효 출력 변화 |
| 안정성 | NaN/gradient skip, bbox/centroid 이상, 여러 시점의 floater, 처리량·peak memory |

C1의 성공은 feature rank나 `relq` 하나의 상승이 아니다. **기하 입력이 의미 있게 전달되면서 난간 위치·연속성·여러 시점 렌더가 함께 개선되는 것**이다. 평균 PSNR이 좋아져도 truck/기차 한쪽이나 난간 ROI가 악화되면 trade-off를 그대로 기록한다. 작은 차이는 반복 seed로 확인한다.

## 8. 준비한 런처와 사용법

[scripts/launch_detail_ablation.py](../scripts/launch_detail_ablation.py)를 추가했다. 기존 S args를 읽고 다음을 명시적으로 고정한다.

- fresh initialization: `resume=''`, `init_from=''`.
- `max_steps=64000`, S의 LR·render·detach·detail schedule 유지.
- C 플래그의 parser 기본값에 의존하지 않음.
- B0/C1 모두 현재 `shared_cell_owner=0`.
- `CUDA_VISIBLE_DEVICES=0,1,2`, DDP 3 rank.
- 실행할 때 최종 인자와 핵심 소스 해시를 `experiment_manifest.json`에 보관.

**기본 동작은 dry-run이며 이번에는 dry-run만 실행했다.** 생성된 설정:
[plan_B0.json](experiment_design_20260916/plan_B0.json), [plan_C1.json](experiment_design_20260916/plan_C1.json).

```bash
cd /data/daeho/aacd_proj/can3tok_encoder_decoder_new_fix_7

# 실행하지 않고 조건 확인
/home/super/anaconda3/envs/can3tok/bin/python scripts/launch_detail_ablation.py --variant B0
/home/super/anaconda3/envs/can3tok/bin/python scripts/launch_detail_ablation.py --variant C1

# 준비·메모리 점검 후 첫 실험을 시작하는 명령
/home/super/anaconda3/envs/can3tok/bin/python scripts/launch_detail_ablation.py --variant B0 --execute --detached

# B0와 같은 자원을 동시에 점유하지 않도록 순차 실행
/home/super/anaconda3/envs/can3tok/bin/python scripts/launch_detail_ablation.py --variant C1 --execute --detached
```

위 실행 명령은 64k까지 진행하는 설정이며 8k/16k에서 자동 정지하지 않는다. C1은 B0 뒤에 수동으로 실행하는 별도 명령이다. 두 명령을 연달아 실행하면 같은 GPU에 두 학습이 겹치므로 동시에 실행하지 않는다.

원래 `launch_contract_baseline.sh`는 여전히 A+B+C와 16k horizon을 켠다. 이번 두 조건의 실행에는 새 런처를 사용한다.

## 9. 이번 검증 산출물과 남은 작업

| 자료 | 내용 |
|---|---|
| [preflight_data.py](experiment_design_20260916/preflight_data.py) / [결과](experiment_design_20260916/preflight_data.json) | 8개 snapshot의 source 집합·소속·count·중심 및 LR schedule |
| [render_data_contract.py](experiment_design_20260916/render_data_contract.py) / [결과](experiment_design_20260916/target_render.json) | 4개 snapshot × 2시점, 기존/shared 타깃의 사진 대비 렌더 |
| [초기 truck 비교](experiment_design_20260916/targets_val229_view3.png) | 사진 / 기존 타깃 / shared 타깃 |
| [초기 기차 난간](experiment_design_20260916/upper_rail_val25.png) | 같은 난간 ROI의 타깃 변화 |
| [plan_B0.json](experiment_design_20260916/plan_B0.json) / [plan_C1.json](experiment_design_20260916/plan_C1.json) | dry-run으로 검증한 최종 인자와 소스 해시 |

- 기존 `tests/test_audit_contracts.py` 9개가 통과했다. 하지만 이 테스트의 shared-owner 검사는 소속 불일치만 검사하며 타깃 렌더 품질은 검사하지 않는다. 이번 실데이터 검증이 추가로 필요했던 이유다.
- 새 런처는 인자 parse와 B0/C1 간 차이를 검사했다. 실제 3-GPU 학습·후기 backward smoke는 아직 실행하지 않았다.
- Production 데이터·모델·loss·schedule 코드는 이번 작업에서 바꾸지 않았다. 추가한 것은 진단 스크립트·실험 런처·문서다.

**최종 추천:** 현재 shared-owner prefix 정책을 첫 기준선에서 제외한 B0를 만들고, 같은 조건의 C1로 정밀 좌표 전달 개선을 확인한다. 그 사이 공유 소속을 타깃 보존 방식으로 정리한다. 8k는 초기 점검, 난간 디테일의 주 판단은 같은 schedule의 43k 이후로 구분한다.
