# Gaussian NPZ 인코더–디코더 복원 품질 감사와 개선안

작성일: 2026-09-15 (UTC)  
대상: `/data/daeho/aacd_proj/can3tok_encoder_decoder_new_fix_7`  
검증 자료: [audit_20260915](audit_20260915/)  
범위: 코드·설정·실험 기록 검토, 실제 NPZ 검사, 기존 체크포인트 추론. **이번 조사에서 학습 코드는 수정하지 않았고 새 학습도 실행하지 않았다.**

후속 조사: [13 — 기차 난간·세부 구조 소실과 구조 개선안](13_handrail_detail_and_architecture_20260915.md). 실제 난간 crop과 입력 개입으로 local xyz 전달의 약함을 확인했고, 학습된 unpack이 현재 오차를 보정한다는 추가 근거를 포함했다.

## 1. 결론

현재 문제를 **“16k latent가 작아서”, “shape 3채널이라서”, “렌더 손실이 약해서” 중 하나로 단정할 근거는 부족하다.** 그 전에 고쳐야 할 평가 오류와 데이터·학습·추론 경로의 불일치가 실제로 존재한다.

특히 다음 순서로 접근하는 것이 타당하다.

1. **평가와 배포 경로를 바로잡는다.** 두 장면의 카메라가 섞인 평가, held-out 사진의 학습 유입, 학습된 속성 디코더를 빠뜨리는 `decode_compact()`를 수정한다.
2. **같은 셀이 같은 가우시안 집합을 뜻하게 만든다.** 입력과 타깃을 독립적으로 capacity packing하여 생기는 소속·중심·개수 차이를 제거한다.
3. **렌더링되는 집합에 맞춰 기하 감독을 정리한다.** 공간 샘플링 편향, projection 손실의 padding 참여, 부분 셀 template, 자유롭게 학습되는 `unpack`의 우회 경로를 처리한다.
4. **세부 정보가 손실에 제대로 도달하게 한다.** 축소된 사진의 재확대, 이전 샘플 visibility 재사용, 잘못된 edge cache, 속성 대응의 혼합을 수정한다.
5. 이후 **16,384개 scalar와 출력 262,144개 슬롯을 유지한 상태에서** encoder 정규화·pooling·decoder 표현력을 한 요인씩 비교한다.

확인된 오류를 고치면 품질이 반드시 얼마만큼 오른다는 뜻은 아니다. 아래에서는 **확정된 동작**, **실제로 측정한 현상**, **아직 검증해야 할 인과관계**를 구분한다.

### 가장 중요한 발견

| 우선순위 | 발견 | 확인 방법 | 의미 |
|---|---|---|---|
| P0 | 각 NPZ를 두 장면의 카메라·사진 전체로 평가 | 코드 추적 + S/T 체크포인트 재렌더 | 기존 multi-scene PSNR과 실험 순위를 다시 확인해야 함 |
| P0 | held-out 사진이 own-photo 경로로 학습에 유입 | 현재 train split·photo map 전수 대조: **101개 NPZ** | 카메라 일반화 평가가 오염됨 |
| P0 | `decode_compact()`가 `AttributeDecoder`를 호출하지 않음 | 실제 T70k forward와 비교 | 저장 latent로 복원할 때 학습 때와 다른 scale/rotation/opacity/color 사용 |
| P1 | 입력/타깃의 같은 원본 점이 다른 셀에 배치 | 실제 4개 NPZ 추적: 최대 **10.31%** | 셀 중심·개수·기하 감독이 충돌 |
| P1 | 공간 sampler가 정렬된 집합의 뒤쪽을 누락 | 100→60 샘플링에서 인덱스 0–59만 선택 | 특정 공간 영역이 반복적으로 덜 감독됨 |
| P1 | 부분 셀은 template 정렬 제외, template의 prefix는 중심이 치우침 | 실제 부분 셀 비율 + template 계산 | count가 작은 셀에서 기하 prior가 잘못 시작 |
| P1 | `unpack`은 identity 초기화된 **학습 가능한 Linear** | E1/S/T weight 검사 | frame/refine gate만으로는 기하 우회 경로가 닫히지 않음 |
| P1 | 추가 사진 4장이 half-resolution 상태로 들어옴 | dataset resize→render resize 추적 | full-resolution detail 단계의 감독 자체가 흐림 |
| P1 | 이전 배치 visibility와 pointer 기반 edge map 재사용 | 코드 추적 + cache 재현 | 다른 샘플·사진의 가중치로 현재 손실 계산 가능 |
| P2 | pooled encoder의 local xyz에 extent 정규화가 없음 | 활성 경로 검사 | 작은 구조의 좌표 신호가 약해질 수 있음; A/B 필요 |

P0는 **평가·배포의 정확성**, P1은 **학습 데이터와 목적함수의 일관성**, P2는 **표현력과 품질의 추가 개선**을 뜻한다.

## 2. 무엇을 조사했는가

### 2.1 정적 검토 범위

- `mdmd/README.md`, `01`–`11` 전체를 읽고 이전 시도와 제약을 확인했다.
- `can3tok/`의 데이터, encoder, compressor/decompressor, codec/attribute decoder, loss, renderer, schedule, train/eval 연결을 추적했다.
- `config`, `layers`, `template`, `morton`, `stats`, 비활성 `gen_decoder`·`joint_decoder`의 역할과 활성 조건을 확인했다.
- `tools/`, `tests/`, `scripts/`를 목록화하고 현재 체크포인트 평가·진단·런 재개와 관련된 구현을 확인했다. 과거 모든 oracle 실험을 재실행한 것은 아니다.
- Python 파일 76개, 약 20,276줄의 구조 목록을 [inventory.json](audit_20260915/inventory.json)에 저장했다. 조사 당시 소스·스크립트·기록의 해시는 [source_manifest.json](audit_20260915/source_manifest.json)에 있다.

아래 `L숫자`는 조사 당시 파일의 줄 번호이다. 런 이름에 붙은 `train`은 기차 장면 이름이며, 데이터 split의 train과 구별한다.

### 2.2 실제 실행한 검증

| 검증 | 범위 |
|---|---|
| CPU 재현 | spatial sampler, padding/projection, attribute set self 비교, edge cache, template prefix, schedule |
| 사진 split 감사 | 현재 train NPZ 2,867개와 held-out 이미지 경로 전수 대조 |
| NPZ/packing 추적 | val 25, 166, 229, 382: 두 장면의 초기·후기 snapshot |
| 모델 추론 | S16k 62,000 및 T16k 70,000, 각각 val 166·382 |
| 카메라 교차 평가 | 샘플마다 자기 장면 9개 + 다른 장면 9개, target/GT-mask pred/predicted-mask pred |
| encoder 측정 | pooler, pack, pre-merge, token-merge, mid의 셀 간 분산 스펙트럼 |
| weight 검사 | E1 2,000, S16k 62,000, T16k 70,000의 unpack·latent scale·best score |

모델은 현재 코드에 **strict checkpoint load 성공**을 확인하고, checkpoint step에 맞는 residual/refine/detach schedule을 적용했다. 추론은 BF16 autocast, 렌더는 FP32, 사진 비교는 `downscale=2`로 통일했다. 사용 중인 GPU 0–2의 학습 작업은 건드리지 않고 GPU 3에서 검증했다.

S16k 40k는 launcher가 가리키지만 조사 당시 해당 checkpoint 파일은 없어 **실제 존재하는 62k를 측정**했다. E1은 진행 중이므로 초기 성능으로 최종 성공·실패를 판정하지 않았다.

## 3. 현재 활성 구조: 문서와 실제 실행을 구분해야 한다

T16k의 [args.json](../runs/T16k_20260913_093332/args.json)을 기준으로 한다.

```text
NPZ: xyz, linear scale, quaternion, opacity, DC, SH, camera, ids, pruning_scores
  ↓ load_npz_state: ids/pruning_scores 누락
  ↓ scene별 정규화 / 샘플 선택 / anchor capacity packing
  ├─ target: 1024 cells × 256 slots = 262,144 slots
  └─ encoder: 1024 cells × 2048 slots = 2,097,152 input capacity
       ↓ cell별 point pooler: 16 queries × 128 → 1024
       ↓ pack: 1024 → 1024, cell당 32 tokens × 32 channels
       ↓ token_embed: 32 × 448
       ↓ concat merge: 14,336 → 896 → 448
       ↓ group/window processing → mid 256
       ↓ compact: 1024 × [center/extent 4 | count 1 | shape 3 | appearance 8]
                  = 16 × 32 × 32 = 16,384 scalar
       ↓ decompressor: compact + 주변 context → folding + deep residual
       ↓ CodecDecoder: trainable unpack + 10-layer refine → pred / presence
       ↓ AttributeDecoder: code + xyz → nudged xyz / scale / rot / opacity / DC
       ↓ render: attr_pred와 predicted presence
```

- 이번에 읽은 NPZ는 SH 45개를 포함하여 **target 59채널**, loader의 input은 부가 4채널을 포함하여 **63채널**이다. `stage=geometry`가 모델 target을 14채널로 자르고 SH 출력은 끈다. Pooler는 local xyz와 속성 11개를 읽는다. “원본부터 14채널이고 input 15채널”로 계산하면 실제 메모리·I/O를 잘못 추정한다.
- 고정 anchor가 **packing의 기준**으로 쓰이는 것과, 그 anchor를 **decoder의 고정 원점**으로 쓰는 것은 다르다. 현재 `use_fixed_anchor_center=0`이며 `model.py:L167–168`에서 입력 `group_anchor`를 버리고 입력 점들의 평균 중심을 사용한다.
- 현재 `no_gen=true`, joint/structured 경로는 꺼져 있다. 이 설정에서 student/gen branch를 주원인으로 볼 근거는 없다.
- latent의 `shape 3`과 `appearance 8`은 명시적 채널 배분이다. 그러나 decompressor context는 전체 compact를 읽고 `attr_read_shape`도 켜져 있어 **기하·속성 함수가 완전히 분리된 3차원/8차원 함수는 아니다.**
- latent의 scalar 개수와 실제 압축 bit 수는 다르다. 16,384개 값 자체는 FP16 약 32 KiB, FP32 약 64 KiB이며 모델 weight·scene 정규화·포맷 metadata 등의 비용은 별도다.

## 4. P0: 평가와 배포 경로부터 수정

### 4.1 다른 장면의 카메라와 사진이 섞이는 평가 — 확정

**위치:** [train.py](../can3tok/train.py) L1225–1282, L1395 부근 / [eval_utils.py](../can3tok/eval_utils.py) L161–207 / [eval_per_scene.py](../tools/eval_per_scene.py).

`load_eval_views()`는 여러 scene pool을 순회하면서 `(cam, photo)`만 하나의 리스트에 넣는다. scene 식별자를 보존하지 않는다. `run_eval()`은 모든 NPZ에 이 리스트 전체를 전달하고, `render_eval_metrics()`는 전부 렌더링한다.

따라서 기차 가우시안을 truck 카메라로 렌더해 truck 사진과 비교하는 항목이 지표에 포함된다. 이후 결과를 scene별로 평균내도 이미 개별 결과에 섞인 잘못된 카메라 항목은 없어지지 않는다. `tools/eval_per_scene.py`도 같은 함수를 같은 방식으로 호출한다.

**재측정: 자기 scene 9개 카메라만 사용할 때와 기존 18개 혼합을 비교.** 아래 pred는 학습된 `attr_pred`와 predicted presence를 사용했다.

| checkpoint / snapshot | 자기 scene pred PSNR | 다른 scene pred PSNR | 기존 혼합 pred PSNR | 자기 scene target PSNR |
|---|---:|---:|---:|---:|
| S62k / train 028630 | 18.482 | 9.662 | 14.072 | 20.075 |
| T70k / train 028630 | 18.324 | 9.727 | 14.025 | 20.075 |
| S62k / truck 028750 | 21.397 | 9.482 | 15.440 | 20.604 |
| T70k / truck 028750 | 21.510 | 9.311 | 15.410 | 20.604 |

출처: [T model_probes.json](audit_20260915/model_probes.json), [S model_probes.json](audit_20260915/S16k_62000/model_probes.json).

**해석:** 전체 런 평균 14.x dB를 그대로 복원 성능으로 해석하면 안 된다. 특히 truck 예시는 자기 카메라에서는 T가 S보다 높지만 혼합에서는 반대다. 다만 이것은 두 snapshot의 결과이고, 전체 validation 순위를 확정한 재평가는 아니다. 블러가 사라졌다는 뜻도 아니다.

**개선:**

- `EvalView(scene_id, image_id, camera, photo)` 또는 `dict[scene_id, list[view]]`를 사용한다.
- `item['scene']`에 해당하는 카메라만 평가하고, scene–camera 불일치를 assertion으로 막는다.
- scene별 동일한 카메라·snapshot 목록을 고정하고 전체 S/T checkpoint를 다시 평가한다.
- 잘못된 조합의 점수는 학습 성능과 섞지 말고 평가 버전이 다른 과거 기록으로 남긴다.

**수정 검증:** 두 가짜 scene의 evaluator 호출을 추적하여 다른 scene 카메라가 0회 사용되는지 검사. 각 scene의 사용 view ID를 결과 JSON에 반드시 저장.

### 4.2 held-out 사진이 own-photo 경로로 들어옴 — 확정

**위치:** [data.py](../can3tok/data.py) L915–946, L992–1000.

`view_exclude`는 extra real views에만 적용된다. `photo_map`에서 현재 NPZ의 own photo를 읽는 경로에는 제외 조건이 없다. 현재 split에서 held-out 사진을 own photo로 쓰는 **train NPZ 101개: 기차 49개, truck 52개**를 찾았다.

현재 held indices는 `[3, 26, 42, 65, 93, 229, 238, 239, 243]`이며 전체 목록은 [view_audit.json](audit_20260915/view_audit.json)에 있다. 이는 “고유 held-out 이미지가 101장”이라는 뜻이 아니라, 그 이미지들과 연결된 **NPZ 레코드가 101개**라는 뜻이다.

**개선:**

- scene + 정규화된 image path/image ID 기준으로 own·extra 두 경로에 같은 제외 정책을 적용한다.
- own photo가 held-out이면 train용 다른 카메라·사진 쌍을 선택하거나 그 own-photo 항목을 제외한다. 카메라를 두고 사진만 바꾸면 안 된다.
- extra pool이 전부 제외되었을 때 전체 pool로 되돌아가는 fallback도 제거한다.
- NPZ snapshot holdout과 camera holdout을 따로 기록한다. 전자는 학습 과정의 상태 복원, 후자는 시점 일반화를 측정한다.

현재 checkpoint는 이미 사진을 보았으므로 evaluator만 고쳐 **깨끗한 camera generalization**으로 되돌릴 수 없다. 기존 weight 재평가는 진단용으로 사용하고, 공정한 일반화 비교에는 정리된 split의 새 학습이 필요하다. 원본 3DGS가 이미 관측했던 카메라와 codec이 직접 loss로 본 카메라의 구분도 기록해야 한다.

### 4.3 `decode_compact()`가 학습된 최종 Gaussian을 반환하지 않음 — 확정

**위치:** [model.py](../can3tok/model.py) L123–152와 L286–318 비교.

`forward()`는 별도의 `AttributeDecoder`에서 학습한 속성과 bounded xyz nudge를 `attr_pred`에 넣는다. 학습 render는 이것을 사용한다. 그러나 `decode_compact()`는 decompressor와 구형 `CodecDecoder`까지만 실행한다. 구형 decoder의 속성 채널은 현재 목적함수에서 최종 속성으로 학습되지 않는다.

실제 T70k의 raw `z_compact`를 바로 decode한 결과:

- `decode_compact(z)['pred'].xyz`와 `forward()['pred'].xyz`의 최대 차이는 **0**이었다.
- 하지만 `attr_pred` 키가 없고, 학습된 속성과 반환 속성의 혼합 채널 MAE는 기차 **1.751**, truck **1.534**였다. 단위가 다른 속성을 섞은 숫자이므로 품질 지표로 쓰지는 않는다. 같은 결과가 아니라는 재현 증거다.
- raw latent decode에서 xyz만 일치하므로 기존 `test_decode_compact_matches_training_codec_path`는 이 문제를 잡지 못한다.

이 버그는 **latent만 저장하고 public decode API로 복원하는 경우 직접적인 실패 원인**이다. `forward()['attr_pred']`로 만든 기존 학습 렌더의 블러까지 이 버그 하나로 설명할 수는 없다.

또한 `encode_compact(..., normalized=True)`가 기본값인데 반환값은 `z / latent_scale`이고, `decode_compact()`는 이를 원복하지 않는다. T70k의 scale은 **0.912512**다. encode→decode를 기본 인자만으로 연결하면 center, extent, count를 포함한 모든 채널을 다른 값으로 해석한다.

**개선:**

1. `_decode_gaussians(raw_z, ...)` 공통 함수를 만들고 `forward()`와 public decode가 같은 decompressor→codec→attribute→nudge→presence 경로를 사용하게 한다.
2. raw/normalized 여부를 명시적 인자 또는 artifact metadata로 저장하고, normalized 입력은 decode 전에 정확히 역변환한다.
3. final xyz/log-scale/quaternion/logit-opacity/DC/presence와 scene center/scale의 계약을 명시한다. 좌표를 외부 타깃에서 복사하는 방식으로 맞추지 않는다.
4. teacher forcing을 끈 eval 조건에서 **전체 Gaussian 속성, presence, 여러 카메라의 렌더**까지 round-trip 비교한다.

### 4.4 resume 시 best-score의 의미가 바뀌었는데 이전 숫자를 그대로 사용 — 확정

**위치:** [train.py](../can3tok/train.py) L1605–1613, L2244–2279, L2351.

T는 S의 `score_metric=psnr`에서 `psnr_gap_worst`로 바꾸어 resume한다. 그런데 `best_score`는 과거의 `-PSNR` 값 그대로 복사한다. T70k checkpoint에는 **`best_score=-14.2814228634`**, metric에는 `psnr_gap_worst`가 저장되어 있고, T 런 디렉터리에는 `ckpt_best.pt`가 없다.

**개선:** checkpoint selection에는 metric 이름·평가 버전·split fingerprint를 함께 저장한다. 하나라도 바뀌면 이전 best를 새 metric으로 재평가하거나 best-score 기록만 초기화한다. optimizer와 학습 step을 이유 없이 초기화할 필요는 없다.

### 4.5 보조 평가 도구와 출력 파일도 통일 필요

- [render_eval_figs.py](../tools/render_eval_figs.py) L63–69는 큰 `enc_input`/`enc_mask`를 모델에 전달하지 않는다. 현재 2M 입력 모델에서는 262k 타깃 텐서를 다른 group 크기로 재해석하므로 학습 경로와 다르다.
- [render_compare.py](../tools/render_compare.py) `load_run()`은 refine을 1로 강제하고 일부 step별 flag를 복원하지 않는다. 후기 S/T에서는 관련 gate가 이미 열렸지만, E1 초기 checkpoint 진단에는 잘못된 조건이 된다.
- [eval_utils.py](../can3tok/eval_utils.py) L488–519는 GT mask로 pred를 저장한다. 이 PLY는 [io_utils.py](../can3tok/io_utils.py) `write_ply()`가 쓰는 **xyz 중심의 point cloud**이며, 전체 3DGS 속성 PLY가 아니다.
- `run_eval()`의 파일명은 NPZ basename 기반이다. 두 scene의 `step_013250.npz` 등이 충돌하여 결과 파일이 덮어써질 수 있다.

**개선:** checkpoint load·schedule·encoder input·최종 Gaussian 선택을 공통 helper로 통합하고 `scene_id/snapshot_id`를 파일 경로에 넣는다. 실제 배포 결과는 전체 속성 NPZ/3DGS-compatible PLY와 predicted mask로 따로 저장한다.

## 5. P1: 입력 NPZ와 복원 타깃의 정의

### 5.1 입력/타깃을 독립적으로 packing하면 같은 cell ID가 같은 집합이 아님 — 확정

**위치:** [data.py](../can3tok/data.py) L782, L846–877 / [encoder.py](../can3tok/encoder.py) L404–418 / [compressor.py](../can3tok/compressor.py) L499–508, L768–771.

현재 타깃은 capacity 256, 입력은 capacity 2048로 `_anchor_assign()`을 **두 번 독립 실행**한다. anchor와 spill 수가 같아도 capacity가 다르면 overflow 시 이동하는 셀이 다르다.

**실제 NPZ 추적 결과:** `valid`는 scene 정규화 범위 필터 후 개수다. 비율은 양쪽에 존재하는 원본 점의 source identity를 추적해 계산했다.

| scene / snapshot | 원본 | valid | target live | encoder live | 같은 점의 셀 소속 변경 | 부분 셀 비율¹ |
|---|---:|---:|---:|---:|---:|---:|
| train / 013250 | 293,663 | 284,597 | 227,643 | 284,597 | 9.71% | 28.06% |
| train / 028630 | 81,369 | 79,821 | 79,821 | 79,821 | 0.081% | 99.70% |
| truck / 013250 | 688,870 | 654,114 | 227,061 | 654,114 | 10.31% | 24.75% |
| truck / 028750 | 177,833 | 171,023 | 158,577 | 171,023 | 5.02% | 70.17% |

¹ 비어 있지 않은 셀 중 1–255개만 채워진 셀의 비율. 출처: [data_probes.json](audit_20260915/data_probes.json).

네 샘플 모두 **target 점 자체가 encoder 전체 집합에서 빠진 경우는 0**이었다. 따라서 여기서 확인된 것은 “입력이 타깃을 전혀 포함하지 않는다”가 아니라 **같은 점의 cell identity가 달라진다**는 문제다. 데이터가 2M보다 커지는 경우까지 superset이 보장된 것은 아니다.

#### 왜 중심과 count에도 문제가 생기는가

Compressor는 **입력 셀의 평균·extent·count**를 analytic anchor로 저장한다. 그런데 감독은 타깃 셀에 붙는다.

- 초기 snapshot의 `||input mean - target mean|| / input extent` 중앙값은 기차 **0.201**, truck **0.211**이다. 90백분위는 각각 **0.886**, **0.990**이다.
- centroid head가 이동할 수 있는 범위는 축당 **0.05 × extent**다. 3축 거리로 보아도 최대 약 0.0866 extent이므로 위 차이의 상당 부분을 이 채널만으로 정정할 수 없다.
- count는 `2 * input_count / 256 - 1`에 작은 residual을 더한 뒤 decoder에서 0–256으로 clamp한다. input count가 256을 훨씬 넘으면 타깃의 빈자리 구조를 직접 나타내지 못한다.
- `min(input_count,256)`과 target count의 셀당 평균 절대차는 기차 초기 **17.48개**, truck 초기 **19.22개**, truck 후기 **7.77개**다.

결과적으로 좌표 residual·unpack·presence가 analytic anchor의 불일치를 우회해서 해결해야 한다. 작은 구조를 복원하기 전에 소속·중심·개수 오차를 보정하는 데 용량과 gradient가 사용된다.

**개선:**

1. 원본 Gaussian ID에 대해 **한 번 계산한 cell ownership**을 입력과 타깃이 공유한다.
2. 각 cell 내부에서 256개 이하의 복원 대표점을 선택하고, 추가 encoder 점들도 같은 cell owner를 유지한다.
3. overflow를 재배치해야 한다면 입력/타깃에 공유되는 계획으로 수행한다. 전역 target 선택 후 서로 다른 capacity로 다시 spill시키지 않는다.
4. centroid/extent/count가 “전체 input”인지 “복원 대표 집합”인지 명시한다. 복원 대표 집합 기준이라면 **encoder-side preprocessing에서 계산하여 compact 안에 넣는다.** decoder가 원본 타깃을 추가로 받아서는 안 된다.
5. 셀 소속, 선택 ID, overflow count를 diagnostic 출력으로 남긴다. temporal 안정성도 동일 source ID의 owner 변화율로 확인한다.

고정 anchor를 decoder 원점으로 쓰는 별도 설계는 가능하지만 현재 사본의 기본 동작을 조용히 바꾸는 수정과 섞지 않는다. 먼저 같은 packing 정책 안에서 identity 계약을 맞추는 것이 우선이다.

**수정 검증:** `target_source_id ⊆ encoder_source_id`, 공유 source ID의 owner 일치율 100%, 비어 있는 셀·256 경계·overflow 사례에서 count/center round-trip 확인. 중복 xyz가 있으므로 좌표 최근접 검색 대신 ID를 사용한다.

### 5.2 pruning score가 파일에 있는데 loader에서 사라짐 — 확정

**위치:** [io_utils.py](../can3tok/io_utils.py) L45–135 / [data.py](../can3tok/data.py) L306–323, L376–426.

검사한 NPZ 네 개 모두 최상위 `pruning_scores`가 있다. upstream 저장 형식은 단순 배열이 아니라 `scores`와 metadata를 포함한 dict다. 하지만 `load_npz_state()`는 Gaussian attribute와 camera만 반환한다. `_importance()`가 기대하는 `gs['pruning_scores']`는 실제 NPZ 경로에서 전달되지 않는다.

따라서 현 sampling은 pruning score를 결합했다는 주석과 달리 opacity logit와 설정된 density 항에 의존한다. 직접 dict를 만들어 넘기는 기존 sampling test로는 실제 loader 누락을 찾을 수 없다.

**개선:**

- replay/flat NPZ 두 경로 모두 score schema를 읽고, Gaussian `ids` 또는 검증된 저장 순서로 score를 정렬한다.
- 길이만 맞는 다른 시점의 score를 그대로 결합하지 않는다. ID·step·finite 값 검사를 추가한다.
- score 결합 후에도 anchor capacity 단계에서 어떤 중요한 점이 탈락하는지 확인한다.
- `original → selected target` 렌더 손실과 edge 영역 보존율을 측정하여 실제 개선 여부를 판단한다. score를 읽는 것만으로 detail 보존이 자동 보장되지는 않는다.

이번 개선안은 기존 의도대로 score를 sampling에 쓰는 범위를 우선한다. “decoder가 score를 직접 보지 않으므로 encoder에도 원리상 넣을 수 없다”는 주석의 논리는 일반적으로 성립하지 않는다. Encoder 전용 입력과 decoder side-channel은 다른 문제다.

### 5.3 원본·선택·모델의 손실을 구분해야 함

원본 NPZ 자체, 범위 필터, 262k 제한, local capacity packing, 모델 압축을 한꺼번에 비교하면 어느 단계에서 난간이 사라졌는지 알 수 없다. 예를 들어 truck 초기에는 원본 688,870개가 타깃 227,061개가 된다.

다음 사다리를 **같은 scene/camera**에서 저장해야 한다.

| 단계 | 내용 | 비교로 알 수 있는 것 |
|---|---|---|
| A | 가능하면 원본 FP32 3DGS + full SH | 원본 표현의 기준 |
| B | 저장된 NPZ 전체 + full SH | NPZ 저장 정밀도의 손실 |
| C | 저장된 NPZ 전체 + DC only | SH 생략 손실 |
| D | 필터·sampling·packing한 target + DC | 선택에서 사라진 구조 |
| E | 모델 final Gaussian + predicted presence | codec의 추가 손실 |
| F | 각 단계와 실제 사진 비교 | 원본 3DGS의 오차와 codec 오차 구별 |

이 사다리 전체는 이번 조사에서 실행하지 않았다. 현재 직접 측정한 teacher는 **D**다. D보다 이미 앞 단계에서 사라진 detail을 decoder만으로 복원하겠다는 목표인지도 별도로 정의해야 한다.

## 6. P1: 기하 손실과 decoder의 실제 좌표 생성

### 6.1 공간 샘플링의 뒤쪽 누락과 multiscale prefix 편향 — 확정

**위치:** [losses.py](../can3tok/losses.py) L89–103, L194–211.

`_sample_spatial_indices()`는 voxel key로 정렬한 뒤 다음을 수행한다.

```python
stride = N // K
picked = sorted_indices[::stride][:K]
```

N이 K의 배수가 아니면 정렬된 공간의 뒤쪽이 체계적으로 누락된다. CPU에서 x축으로 정렬한 100점을 60개로 줄이면 **0–59만 선택되고 60–99는 전부 빠진다.** 단순한 확률적 sampling noise가 아니다.

여기에 `multiscale_chamfer()`는 최대 샘플 수로 만든 정렬 리스트에서 작은 scale마다 다시 `[:n]`을 취한다. 최대 scale이 장면 전체를 덮어도 작은 scale은 한쪽 공간에 집중한다.

**왜 detail/floater와 관계가 있는가:** 경계·배경·난간 등 공간 위치에 따라 GT→pred와 pred→GT 압력의 빈도가 달라진다. 가중치를 높이면 같은 편향이 강해질 수 있다.

**개선:** 전체 정렬 범위를 K개 구간으로 나눈 뒤 각 구간에서 대표를 뽑거나 voxel별 할당을 명시한다. 작은 scale도 매번 전체 공간을 덮게 별도 stratification 또는 재현 가능한 분산 순서를 사용한다. 수정 후 공간 histogram과 선택 범위를 기록한다.

**수정 검증:** N/K 경계값, N이 K보다 약간 큰 경우, 여러 분리된 cluster에서 전체 범위가 포함되는지 검사한다. 큰 샘플 리스트를 그대로 prefix로 자르는 작은 scale까지 함께 검증해야 한다.

### 6.2 어떤 slot을 감독하고 어떤 slot을 렌더하는지 다름 — 일부는 이미 수정, 일부는 잔존

**위치:** [losses.py](../can3tok/losses.py) L307–348, L1293 부근, L1719–1767 / [eval_utils.py](../can3tok/eval_utils.py) L161–179.

현재 코드를 정확히 구분하면:

| 경로 | pred 쪽 사용하는 집합 |
|---|---|
| branch의 multiscale Chamfer / coverage / voxel occupancy | **GT mask** — 여기는 padding 제외가 이미 적용됨 |
| intra-group Chamfer / Sinkhorn / attr set | 대체로 양쪽에 GT mask |
| projected Chamfer / projected histogram | pred 전체 슬롯; padding 참여 문제 잔존 |
| 학습 render | `presence > 0`; 64개 미만이면 GT mask로 fallback |
| 현재 render eval / point PLY 저장 | GT mask |
| 실제 latent-only 배포 | 예측 presence를 사용해야 함 |

따라서 **“모든 Chamfer에 padding이 들어간다”는 진단도 틀리다.** 문제는 projection 항과 배포 집합의 불일치다.

CPU 재현에서 유효한 8개의 점은 그대로 두고 비활성 8개 좌표만 옮겼는데, projected histogram loss가 **87.114 → 100.000**으로 바뀌었다. 렌더되지 않을 좌표가 손실을 바꾼다.

또한 GT mask가 0인 위치에 예측 presence가 1이면 해당 floater는 기존 GT-mask 기반 geometry/eval에서 빠질 수 있다. hard threshold는 그 자체로 presence에 photometric gradient를 전달하지 않는다.

**개선:**

- projection 함수에 pred mask/weight를 명시적으로 받게 하고, 비활성 슬롯의 영향을 제거한다.
- count를 GT 정보 없이 복원하는 배포 결과를 **주 평가**로 사용한다. GT-mask 평가는 원인 분리용으로만 병기한다.
- 학습 초반 GT mask를 쓰는 warmup은 명시하고, 후반에는 predicted active set의 false-positive에도 pred→GT 또는 visibility-aware penalty를 준다.
- soft presence × opacity를 렌더에 쓰는 방안은 가능하나 count collapse를 막는 count/BCE 항과 같이 실험한다. hard/soft 정책을 train/eval에서 명확히 구분한다.
- 64개 미만 fallback은 실패를 가리지 않게 별도 로그·검증 실패로 처리한다.

**현재 영향의 크기:** T70k의 늦은 두 snapshot에서 GT-mask와 predicted-mask PSNR 차이는 약 0.004 dB 이하로 작았다. false-positive는 각각 397개·162개였다. 따라서 이 두 후기 샘플의 블러를 presence 오류가 지배한다고 주장할 수는 없다. 구조적으로 고칠 필요와 현재 지배적인 원인인지는 구별해야 한다.

### 6.3 부분 셀과 Fibonacci template의 불일치 — 확정된 prior 문제

**위치:** [data.py](../can3tok/data.py) L641–655, L676–683 / [template.py](../can3tok/template.py) L62–70 / [compressor.py](../can3tok/compressor.py) L718–729.

타깃은 256개가 꽉 찬 셀만 Hungarian template 정렬한다. 부분 셀은 앞 k개를 live로 유지한다. 반면 decoder는 항상 256개 template을 만들며 전체 256개를 기준으로 mean-center한다.

현재 Fibonacci 구현은 index와 방향·반경이 연결되어 있다. **전체 집합의 중심이 0이어도 prefix의 중심은 0이 아니다.**

| 유효 prefix k | 원 template에서 prefix 평균의 norm |
|---:|---:|
| 32 | 0.743 |
| 64 | 0.770 |
| 80 | 0.752 |
| 128 | 0.628 |
| 192 | 0.354 |
| 256 | 약 0 |

위 값은 template 자체 단위이며 최종 world 이동량은 frame·extent에 따라 달라진다. 기차 후기의 nonempty 셀 **99.70%가 부분 셀**이므로 드문 예외가 아니다.

**개선:** count별로 고르게 분포하는 deterministic subset/template을 정의하고, valid subset 기준의 중심을 보장한다. 부분 셀 정렬도 같은 기준을 사용한다. 시간에 따라 count가 바뀔 때 slot이 대거 뒤바뀌지 않도록 nested ordering·ID 안정성을 함께 검사한다.

count를 decoder가 예측한다는 점을 고려하여 active subset centering과 count supervision을 설계해야 한다. GT mask를 decode 인자로 추가해 문제를 숨기면 latent-only 복원이 성립하지 않는다.

### 6.4 `identity-unpack`은 identity가 아니다: E1의 frame-first에도 열린 좌표 경로 — 확정

**위치:** [decoder.py](../can3tok/decoder.py) L54, L89–94, L117–131, L223–230 / [compressor.py](../can3tok/compressor.py) L802–808.

`self.unpack = nn.Linear(1024,1024)`는 identity로 **초기화**될 뿐, 현재 codec 경로에서 identity로 고정되어 있지 않다. 앞 768개 xyz와 뒤 256개 auxiliary 값 사이의 교차 weight도 학습된다. `set_refine_trainable()` 대상에도 unpack은 없다.

| checkpoint | `||W-I||F / ||I||F` | xyz←aux block norm | unpack bias norm |
|---|---:|---:|---:|
| E1 2k | 0.586 | 2.058 | 0.724 |
| S16k 62k | 4.396 | 4.824 | 7.087 |
| T16k 70k | 4.547 | 4.869 | 7.434 |

출처: [checkpoint_metadata.json](audit_20260915/checkpoint_metadata.json).

E1은 folding residual과 decoder refine을 8k까지 닫지만, **2k에서 이미 unpack이 identity를 벗어났다.** 그러므로 “그 기간에는 6-DoF frame만 기하를 학습한다”는 해석은 성립하지 않는다. 이 층의 사용 자체가 무조건 나쁘다는 뜻은 아니며, 실험이 의도한 격리가 이루어지지 않았다는 뜻이다.

현재 좌표에는 다음 경로가 겹친다.

1. folding template + shape residual: 전체 슬롯 기준 중심 제거.
2. `deep_xyz`: 별도 residual이며 mean-zero 보장은 없음.
3. trainable unpack: 좌표 변환·bias·aux→xyz 우회.
4. decoder refine: 별도 xyz residual.
5. attribute decoder nudge: 렌더에 쓰이는 최종 xyz 이동.

**개선:**

- frame-first 실험에서는 unpack을 실제 identity 연산으로 고정하거나 xyz/aux 분리 block으로 제한한다. 그 뒤 잔차를 단계적으로 연다.
- 임의 Linear로 기존 학습을 보정해온 checkpoint에 갑자기 identity를 덮어쓰면 출력이 깨진다. 새 실험 또는 명시적인 변환·재적응 실험으로 다룬다.
- 기하 계약을 `xyz = encoded_center + extent × centered_local + bounded_final_correction`처럼 명확히 하고, 모든 우회 경로에서 계약을 검사한다.
- center loss를 encoded center와 출력 valid centroid에 각각 측정한다. nudge 이후 최종 Gaussian의 기하 지표도 기록한다.
- 단순히 residual gate를 더 늦추기 전에, gate가 닫힌 상태에서 어떤 파라미터가 xyz를 바꿀 수 있는지 gradient/perturbation 검사로 확인한다.

### 6.5 floater와 두께는 point count만으로 진단할 수 없음

T70k의 **predicted-active 최종 xyz**를 target에 최근접 대응하면 다음과 같다.

| snapshot | pred→GT 거리 p50, world 단위 | p99 | GT의 nonzero NN spacing 중앙값 |
|---|---:|---:|---:|
| train 028630 | 0.0243 | 2.014 | 0.0183 |
| truck 028750 | 0.0707 | 2.992 | 0.0411 |

이 값은 scene 좌표 단위이며 미터라고 가정하지 않았다. 먼 배경·희소 영역도 섞여 있으므로 p99 전체를 floater 비율이라고 부르면 안 된다. 다만 오차의 긴 꼬리가 남아 있음을 보여준다.

기존 `w_p2g=3`, intra Sinkhorn, radius, spacing 항은 이미 존재한다. 따라서 “pred→GT 정밀도 손실이 없어서 floater가 생긴다”는 단순 설명은 맞지 않는다. 실제 active set, 국소 표면, camera visibility, Gaussian footprint를 기준으로 **어디서 이 항들이 효력을 잃는지** 확인해야 한다.

추천 추가 지표는 visible Gaussian의 projected distance, screen-space radius, alpha contribution, 국소 GT spacing으로 나눈 pred→GT 거리, 얇은 구조의 normal 방향 두께다. 위치를 그대로 두고 opacity만 줄인 결과도 구별해야 한다.

## 7. P1: 속성 복원과 렌더 감독

### 7.1 detail 단계에서도 추가 사진 4장은 이미 흐린 타깃 — 확정

**위치:** [data.py](../can3tok/data.py) L1007–1009 / [losses.py](../can3tok/losses.py) L1380–1385 / [schedule.py](../can3tok/schedule.py) `render_downscale_at`, `detail_gains`.

T의 `extra_real_views=4`, `view_downscale=2`다. Dataset은 추가 사진을 PIL bilinear로 절반 크기로 줄인다. 이후 detail 단계에서 `render_downscale=1`이 되어도 원본을 다시 읽지 않고 그 작은 사진을 확대한다.

즉 정상적인 own+extra 구성에서는 **5개 중 4개 시점의 고주파 정보가 먼저 사라진다.** 원본 977px 폭 이미지가 488px로 내려간 뒤 977px로 확대되는 셈이다. VGG와 edge 가중치를 올려도 이미 없어진 가는 선을 타깃에서 되살릴 수 없다.

**개선:** full-resolution 원본을 cache하고 현재 render 해상도에 맞는 pyramid level을 선택한다. Dataset worker의 설정을 중간에 바꿀 경우 persistent worker에 전달되지 않는 문제까지 고려한다. 간단한 첫 수정은 full-res를 보관하고 runtime에서 필요한 크기로만 내려 쓰는 것이다.

**수정 검증:** detail 단계에서 참조 사진의 provenance와 원본 해상도를 기록한다. 난간·글자·차륜 영역에서 원본과 실제 loss target crop을 나란히 확인한다. 원본 손실 없이 고해상도 target을 쓰는 조건을 맞춘 뒤 edge/VGG의 추가 효과를 측정한다.

### 7.2 이전 batch의 visibility로 현재 속성 loss를 가중 — 확정

**위치:** [train.py](../can3tok/train.py) L1742–1747, L1827–1828, L1963 / [losses.py](../can3tok/losses.py) L2161–2175.

직전 render에서 나온 `visible`을 `prev_vis`에 저장하고, 다음 batch와 shape가 같으면 `out['render_visible']`에 넣는다. 현재는 shuffle된 snapshot·scene을 처리하므로 배열 shape가 같아도 **동일 Gaussian·동일 camera의 visibility가 아니다.**

`attr_anchor_covered=0.25`라서 이 잘못된 boolean이 현재 속성 anchor 강도를 1 또는 0.25로 바꾼다. “바로 전 step이라 비슷하다”는 가정은 샘플 identity가 바뀌는 DataLoader에 적용할 수 없다.

**개선:** 현재 batch·현재 predicted set·현재 카메라의 visibility를 계산한 뒤 속성 loss에 사용한다. 또는 정확한 identity key를 가진 cache로 바꾼다. 단기 수정으로 해당 재사용을 끄고 모든 점에 동일한 anchor 가중을 주는 비교도 가능하다.

검증은 shape 비교가 아니라 `scene/snapshot/source IDs/view IDs` 일치로 해야 한다. 다른 scene으로 바뀌는 두 batch의 synthetic 검사로 재발을 막을 수 있다.

### 7.3 edge map cache의 key가 이미지 identity를 보장하지 않음 — 확정

**위치:** [render.py](../can3tok/render.py) L248–276.

현재 key는 `(ref_img.data_ptr(), ref_img.shape, gain)`이다. 메모리 주소는 이미지 내용이나 scene/view의 ID가 아니다. 임시 tensor가 해제되고 allocator가 주소를 재사용하면 다른 이미지에 이전 edge 가중치가 적용될 수 있다.

CPU에서 같은 tensor의 내용을 검정 이미지에서 경계가 있는 이미지로 바꿨을 때, cache는 이전 가중치를 그대로 반환했고 새로 계산한 값과 최대 **16.0** 차이가 났다. 이 실험은 key의 부정확성을 입증한다. 실제 학습에서 주소 재사용이 몇 회 발생했는지는 측정하지 않았다.

**개선:** 우선 edge 계산의 cache를 제거한다. 필요하다면 immutable image ID + resize/crop/augmentation + device/dtype + gain으로 cache한다. 원본 reference를 확실히 유지하는 방식도 가능하다. 캐시 최적화 전에 정답 일치를 검사한다.

### 7.4 geometry와 attribute의 대응 관계가 하나로 공유되지 않음

**위치:** [losses.py](../can3tok/losses.py) L447–529, L772–866, L2178–2195.

현재 다음 세 목적이 동시에 작동한다.

| 항 | 대응 기준 | 남는 문제 |
|---|---|---|
| geometry Sinkhorn | 위치 거리 | 위치가 비슷한 다른 속성의 Gaussian을 구별하지 못함 |
| attribute parameter target | nudged 위치 + standardized attribute, `attr_weight=2` | geometry와 다른 plan; 예측 속성에 따라 바뀌는 detached target |
| attribute set distance | scale/quaternion/opacity/DC 공간의 양방향 NN | 공간 대응·개별 Gaussian의 multiplicity까지 보장하지 않음 |

따라서 “같은 one-to-one plan으로 xyz와 속성을 감독한다”는 설명은 현재 구현과 다르다. attribute set 항은 **양방향 Chamfer형 NN 손실**이며 balanced optimal transport 자체는 아니다.

#### 혼합된 속성 타깃

`sinkhorn_parameter_target()`는 plan의 각 row로 log-scale·opacity logit·DC 등을 가중 평균한다. Quaternion은 부호를 맞춘 뒤 평균하고 normalize한다. 이것은 hard한 한 Gaussian 선택과 다르다.

한 위치에 수직으로 얇은 Gaussian과 수평으로 얇은 Gaussian이 있으면, 두 개를 분리해서 유지해야 할 때도 중간 속성을 목표로 만들 수 있다. 다만 현재 attribute-aware cost가 이를 얼마나 줄이는지는 실제 plan을 확인해야 하며, 모든 블러를 평균 때문이라고 단정하지 않는다.

#### Sinkhorn annealing의 수렴 확인이 없음

T는 epsilon을 0.08에서 0.005까지 낮추지만 iteration은 6이다. Log-domain이라는 이유만으로 6회 후 행·열 질량이 충분히 맞는 것은 아니다. geometry plan에는 row-min shift가 있지만 parameter-target plan에는 같은 보정이 없다.

**개선:**

1. plan의 행·열 marginal error, entropy, 최대 match mass, 중복 대응을 로그로 남긴다.
2. 필요한 경우 tolerance 기준 반복 또는 더 충분한 반복을 사용한다. epsilon을 내리기 전에 수렴을 검증한다.
3. 좌표는 양쪽에 같은 local center를 뺀 뒤 extent로 나눠 거리를 계산한다. 절대 좌표를 작은 extent로 나누어 큰 수끼리 `cdist`하는 수치 오차를 줄인다.
4. “먼저 attribute 평균을 만든 뒤 loss”와 “각 GT Gaussian에 대한 covariance/attribute cost의 plan-weighted 기대값”을 비교한다. 후자는 평균 타깃 자체를 만들지 않는다.
5. geometry/attribute에 shared plan을 쓰는 실험은 대응 가정을 명시해서 별도 비교한다. 모든 항을 한꺼번에 바꾸거나 기존 anchor를 전부 끄지 않는다.

이번 조사에서는 실제 checkpoint의 Sinkhorn marginal residual을 새로 측정하지 않았다. 따라서 **epsilon 0.005/6회가 실패했다는 확정 판정이 아니라, 수렴을 검사하지 않는 위험한 가정**으로 분류한다.

### 7.5 covariance 감독은 있으나 속성 표현·범위·padding 영향이 남음

**위치:** [attr_decoder.py](../can3tok/attr_decoder.py) L270–351, L355–400 / [losses.py](../can3tok/losses.py) L1596–1636.

현재 `covariance3d_loss`와 group-relative scale band는 이미 있다. `w_rot=0`이라고 해서 Gaussian 회전에 모든 감독이 없는 것은 아니다. 렌더와 covariance를 통해 회전이 영향을 받는다.

#### 실제 anisotropy 감소

최종 predicted-active Gaussian의 `max(linear scale)/min(linear scale)` 중앙값:

| T70k snapshot | target | prediction |
|---|---:|---:|
| train 028630 | 16.92 | 3.74 |
| truck 028750 | 17.02 | 5.55 |

최근접 대응한 GT 중 anisotropy 상위 20%를 참조하는 prediction만 보아도 중앙값은 약 6.69·11.04였다. 이는 방향성 있는 얇은 타깃들이 더 둥근 Gaussian으로 표현되는 현상과 일치한다. 다만 최근접 대응은 진정한 one-to-one 대응이 아니고, 분포 차이만으로 원인을 확정하지 않는다.

상위 극단값에는 원본 FP16 scale의 0/underflow와 전처리 floor 영향이 섞인다. 특히 target 최대 anisotropy를 그대로 따라 하도록 cap을 풀어서는 안 된다.

#### 살펴볼 세 가지

- **표현 비식별성:** 축을 순열하고 회전을 함께 바꾸면 같은 covariance를 만들 수 있다. raw quaternion·3축 scale 벡터 거리는 같은 Gaussian에도 비용을 부과할 수 있다. 현재 covariance 항 외의 raw attr set/transport cost도 함께 검토한다.
- **scale band 포화:** 축별 `base + bounded residual`의 residual 범위는 ±3이다. 가는 축이 lower cap에 막히는지, 큰 splat 억제를 위해 필요한 upper cap과 구별해 검사한다. 전체 cap-hit 한 숫자 대신 각 축·GT anisotropy·화면 radius별로 본다.
- **부분 셀 padding:** `AttributeDecoder`는 mask를 받지 않는다. local PE의 중심·평균 반경을 모든 256개 좌표로 계산하고 self-attention도 전체 슬롯을 읽는다. 실제 live가 70개인 셀에서 나머지 186개 비활성 좌표가 live 속성에 영향을 줄 수 있다. 이는 코드상 의존성이며 checkpoint에서의 민감도 크기는 추가 측정이 필요하다.

**개선:** live/count-aware local normalization과 attention mask를 설계한다. 배포 시 얻는 presence/count를 기준으로 하고 warmup 동안의 GT 사용은 명시한다. covariance 기반 대응 비용과 작은 축 복원 능력을 같이 비교한다. 무조건 scale cap을 넓히는 수정은 피한다.

### 7.6 현재 attr-set의 self-loss가 약 1이라는 가정은 맞지 않음 — 재현으로 정정

현재 `intra_group_attr_set_loss()`는 셀의 256×256 대응을 계산한다. 별도 부분표본 둘을 비교하는 방식이 아니다.

- 합성 256-slot `GT↔GT`: **0.000556**.
- 실제 train 028630 `GT↔GT`: **0.001676**.
- 실제 truck 028750 `GT↔GT`: **0.001534**.

0에 가까운 작은 잔차이며 FP32 `cdist`·표현 정규화의 수치 영향 등을 포함한다. 과거 부분표본 oracle에서 얻은 self floor를 현재 전체 셀 학습 loss의 바닥으로 가져오면 안 된다. 다만 self-loss가 낮다는 것만으로 손실의 목표가 적절하다는 뜻은 아니다. 공간 대응과 multiplicity 문제는 따로 남는다.

## 8. P2: encoder 표현력과 원본 포맷

### 8.1 pooler 전에 local xyz를 extent로 정규화하지 않음

**위치:** [encoder.py](../can3tok/encoder.py) L362–401 / [compressor.py](../can3tok/compressor.py) L387–412.

활성 pooled 경로는 `loc = xyz - centroid`를 입력하고, 속성 11개는 dataset 통계로 표준화한다. 작은 셀의 좌표 extent는 0.002 정도인데 속성은 대략 표준화 단위다. 좌표 신호가 상대적으로 작다.

Compressor의 `if not self.pooled_input` 정규화가 꺼지는 것은 맞다. **이미 학습된 feature vector의 앞부분을 xyz라고 간주해 나누면 안 되기 때문이다.** 그러나 그 대신 pooler **앞**에서 해야 할 metric xyz 정규화가 빠져 있다.

**개선 후보:**

```text
cell owner 고정
    → input/target과 합의한 center, extent 계산
    → local_xyz = (xyz - center) / max(extent, floor)
    → [local_xyz, standardized attributes]를 pooler에 입력
    → center/extent는 기존 compact 4채널에 유지
```

첫 비교는 이 한 변경만 적용한다. normalization floor, 빈 셀, 매우 평평한 셀의 처리를 검사한다. 기존 checkpoint에 적용하면 입력 분포가 달라지므로 baseline과 같은 학습 시작점 또는 명시적 재적응 조건으로 비교해야 한다.

### 8.2 현재 16k checkpoint에서도 중간 feature의 분산이 몇 방향에 집중 — 측정

과거 131k의 rank 측정을 그대로 적용하지 않고 S62k/T70k에서 다시 측정했다. 아래는 **encoder nonempty 셀만 사용**, feature별 평균 제거 후 covariance eigenvalue의 entropy rank다.

정의: `λᵢ`를 centered feature matrix의 squared singular value라고 할 때, `pᵢ=λᵢ/Σλ`, `rank=exp(-Σ pᵢ log pᵢ)`. 이는 대수적 rank나 채널별 표준화 후 correlation rank와 다르다.

| checkpoint / scene | pooler out, 1024 | pack, 1024 | pre-merge, 14336 | merge out, 448 | mid, 256 |
|---|---:|---:|---:|---:|---:|
| S62k / train | 1.715 | 1.648 | 1.759 | 2.268 | 3.217 |
| T70k / train | 1.721 | 1.631 | 1.766 | 2.320 | 3.193 |
| S62k / truck | 1.640 | 1.578 | 1.691 | 2.242 | 3.156 |
| T70k / truck | 1.642 | 1.558 | 1.695 | 2.292 | 3.126 |

출처: [T rank_probes.json](audit_20260915/rank_probes.json), [S rank_probes.json](audit_20260915/S16k_62000/rank_probes.json).

**알 수 있는 것:** 넓은 feature 폭을 그대로 독립 정보량으로 해석할 수 없다. 두 샘플에서 pooler/pack 단계부터 분산이 소수 방향에 집중하며 S→T 변경이 이 현상을 크게 바꾸지 않았다.

**알 수 없는 것:** 이 숫자가 곧 “shape 정보가 2개만 남았다”, “latent가 원리상 복원 불가능하다”, “merge가 모든 문제의 원인이다”를 뜻하지 않는다. 작은 분산 방향에 중요한 정보가 있을 수 있고, nonlinear decoder는 이를 사용할 수 있다. 넓은 행렬의 FP32 작은 eigenvalue에는 수치 한계도 있다.

**후속 측정:**

- geometry를 미세하게 바꾸고 attribute를 고정했을 때 compact/출력이 얼마나 반응하는지 확인한다.
- 반대로 attribute만 바꾸어 두 신호의 민감도를 비교한다.
- 좌표 정규화 전후의 covariance rank와 channel-standardized rank를 모두 기록한다.
- 셀별 얇은 구조·다중 표면·count를 설명하는 probe를 **별도 holdout 셀**에서 평가한다.
- rank가 오르더라도 최종 렌더·기하가 좋아지지 않으면 성공으로 처리하지 않는다.

그 다음에야 `pool_blocks=2`, query 간 interaction, 단계적 token merge 등을 **같은 16k 예산**으로 각각 비교한다. 재구성한 여러 token을 독립 저장 정보가 늘어난 것처럼 세지 않는다.

### 8.3 주변 셀 context가 실제 3D 이웃과 일치하는지 확인 필요

**위치:** [attr_decoder.py](../can3tok/attr_decoder.py) L369–383 / [morton.py](../can3tok/morton.py) / [compressor.py](../can3tok/compressor.py) window 처리.

Attribute decoder의 `nbr=1`은 cell index 앞뒤 한 개씩이다. 실제 3D kNN 세 개가 아니다. Morton 순서·2D grid의 인접성은 일부 공간 국소성을 유지하지만 모든 실제 이웃을 보장하지 않는다.

현재 기본 packing을 뒤집기보다, decoded centroid로 deterministic 이웃을 구하는 옵션과 비교할 수 있다. 이때 decoder가 알 수 있는 compact center만 사용하고, scene anchor/GT 좌표를 외부로 추가 전달하지 않는다. Thin structure가 cell 경계를 넘을 때의 seam 지표로 판단한다.

### 8.4 full SH 생략은 별도 품질 한계

**위치:** [io_utils.py](../can3tok/io_utils.py) L119–123 / [train.py](../can3tok/train.py) `build_config` / [render.py](../can3tok/render.py) L144–169.

이번 NPZ는 Gaussian당 `features_rest=(15,3)`을 갖는다. 현재 renderer는 `sh_degree=0`, `colors_precomp=DC×C0+0.5`만 사용한다. 모델도 `sh_dim=0`이다.

시점별 색 변화가 있는 부분은 full-SH 원본과 동일하게 표현하기 어렵다. 공식 3DGS renderer는 활성 SH 차수와 시선 방향을 사용하는 경로를 제공한다. [공식 구현](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py).

“SH 계수가 DC보다 작다”는 비교만으로 렌더 영향이 작다고 결론내릴 수 없다. **전체 원본 full SH vs DC-only**를 같은 카메라에서 먼저 비교한다. SH가 큰 잔차를 설명할 때만 제한된 latent 예산에서 SH/시선 조건부 appearance를 어떻게 표현할지 실험한다. SH를 켜는 것 자체가 복원률 개선의 보장은 아니다.

### 8.5 NPZ FP16 저장에서 이미 작은 구조가 손실될 수 있음

네 NPZ 모두 Gaussian xyz, scaling, rotation, opacity, color, SH가 FP16 저장이었다. Loader에서 FP32로 바꿔도 저장 전에 잃은 정보는 돌아오지 않는다.

- 절대 좌표의 크기가 커질수록 FP16의 인접 표현 간격도 커진다. 작은 Gaussian의 center 차이가 저장 시 사라질 수 있다.
- raw linear scaling의 0 비율은 약 **0.0044%–0.0204%**였다. 실제 0이 upstream에서 생겼는지, FP16 저장으로 underflow했는지는 FP32 원본 대조가 필요하다.
- 선형 scale의 0은 이후 log floor와 clamp를 거쳐 아주 얇은 Gaussian으로 해석될 수 있다.

**개선:** 우선 xyz FP32 저장 또는 정규화된 상대좌표 저장을 비교한다. Scale은 FP32 또는 범위가 관리되는 log-scale 저장을 검토한다. 원본 PLY/checkpoint와 직접 렌더 비교하여 저장 format의 손실을 분리한다. 이번에 FP32 원본까지 되돌아가 정량 비교하지는 않았다.

카메라의 R/T, scale/log-scale, opacity/logit, DC/RGB 변환은 활성 경로를 확인했으며, 현재 자료에서 모든 품질 문제를 설명하는 단순한 좌표계 반전은 찾지 못했다. `Camera`의 `W=2cx, H=2cy` 가정은 일반 데이터에는 부적절하지만 검사한 두 scene의 사진 크기와는 맞았다. 실제 width/height를 metadata로 보존하는 guard는 추가할 가치가 있다.

## 9. 기존 기록에서 정정하거나 보류해야 하는 해석

`mdmd`는 이전 실패를 반복하지 않도록 해 주는 중요한 기록이다. 다만 당시 코드·실험 조건과 현재 실행을 구분해야 한다.

| 기존 설명 또는 성급한 해석 | 현재 확인 결과 |
|---|---|
| geometry는 항상 detach되어 render gradient를 받지 못함 | T의 `attr_detach_release=12000`; 12k 이후 `attr_detach_geometry=False`. `attr_read_shape=1` 경로도 존재 |
| 기하와 속성 latent는 완전히 분리됨 | decompressor context가 전체 compact를 읽고 attribute decoder도 shape를 읽음 |
| input 15 / target 14 | 실제 loader는 input 63 / target 59, 활성 모델 출력 14 |
| anchor·spill 수가 같으므로 입력과 타깃 셀이 일치 | capacity 차이로 같은 점의 owner가 최대 10.31% 다름 |
| 모든 group이 template에 맞게 정렬됨 | 256개 full group만 정렬; 부분 셀은 제외 |
| identity-unpack은 고정된 역연산 | 학습 가능한 1024×1024 Linear; 실제 checkpoint에서 크게 변경 |
| E1은 8k까지 frame만 기하를 결정 | unpack 경로는 열려 있음 |
| 모든 geometry loss가 padding을 포함 | Chamfer/coverage/voxel은 GT mask 적용; projection 항에 문제 잔존 |
| floater를 당기는 p2g 항이 없음 | 현재 p2g·radius·Sinkhorn이 존재; 적용 집합·sampling·gradient를 조사해야 함 |
| attr set의 GT self-floor가 약 1 | 현재 실제 full-group self 비교는 약 0.0015–0.0017 |
| 모든 xyz/attribute가 같은 transport plan 사용 | 위치-only 기하 plan과 위치+속성 parameter plan이 별도 |
| 현재 held-out 카메라는 학습에 안 들어감 | own-photo 경로에 101개 train NPZ 유입 |
| scene별 PSNR이면 장면 섞임이 해결됨 | 개별 NPZ에 이미 다른 scene 카메라가 섞임 |
| PSNR이 target보다 높으면 반드시 blur로 속인 것 | smoothing일 수도, 실제 사진 적합도가 좋아진 것일 수도 있음. crop/SSIM/geometry로 구분 |
| relq가 √2 근처면 복원 실패가 수학적으로 확정 | 순서가 다른 동일 점 집합도 slot별 residual error가 클 수 있음. set metric·render와 함께 해석 |
| 높은/낮은 feature rank 하나로 정보량과 성공 판정 가능 | rank 정의·표준화·빈 셀·작은 분산 신호에 의존; 민감도와 복원 성능 확인 필요 |

특히 `relq`, raw xyz RMSE, raw quaternion NRMSE는 **slot 대응 또는 표현의 선택에 민감한 지표**다. 동일 점 집합을 다른 순서로 출력해도 slot별 오차는 크다. 서로 상관이 약한 동일 분산 offset을 index끼리 비교하면 상대 RMS가 √2에 가까워질 수 있으므로, 그 숫자만으로 정보가 사라졌다고 단정할 수 없다.

또한 loss scalar의 비중은 parameter gradient의 비중이 아니다. 단위·감도·gradient clipping 때문에 작은 loss도 큰 업데이트를 만들 수 있다. “총 loss의 0.5%이므로 영향이 없다” 같은 결론 대신 공통 parameter에서 각 loss의 gradient norm과 gradient 간 cosine을 소수 batch에서 측정해야 한다.

S/T의 과거 모든 순위가 반드시 뒤집힌다는 뜻은 아니다. **평가 bug와 split 오염을 고려하면, 그 순위를 근거로 구조적 가능성을 영구히 닫을 수 없다는 뜻이다.** N1/P1의 131k 성과도 16k의 재현 성공을 대신하지 않는다.

## 10. 구체적인 수정 순서와 통과 기준

### 단계 A — 재학습 없이 평가·추론의 계약부터 맞춤

| 수정 파일 | 구체적인 변경 | 통과 기준 |
|---|---|---|
| `train.py`, `eval_utils.py`, 평가 tools | scene-tagged view, scene별 필터 | foreign scene camera 사용 0회 |
| `model.py` | final Gaussian decode 공통화; raw/normalized 구분 | 같은 raw z에서 forward/decode 최종 속성·presence 일치 |
| `train.py` checkpoint 처리 | metric/evaluator/split 변경 감지 | best checkpoint 갱신 기준이 현재 metric과 일치 |
| `tools/render_compare.py`, `render_eval_figs.py` | 공통 loader와 step flags, enc input 전달 | 같은 checkpoint/input/precision에서 train evaluator와 동일 결과 |
| eval output | scene prefix, full Gaussian 저장, predicted mask | 파일 충돌 0건; latent-only export로 재렌더 가능 |

이 단계에서 고정 snapshot 8개와 scene별 9개 카메라로 S62k/T70k를 재평가한다. GT-mask와 predicted-mask, target↔photo와 pred↔target을 모두 남긴다. 기존 weight의 사진 유입 사실을 결과에 표시한다.

**필수 round-trip 검사:** raw z, normalized z 두 경로 모두 전체 Gaussian을 비교한다. FP32에서는 같은 계산 순서라면 일치해야 하며, 별도 실행/혼합 정밀도에서 허용 오차가 필요하면 출력뿐 아니라 이미지 차이 기준까지 먼저 정한다. xyz만 비교하는 기존 테스트를 통과했다고 배포 검증을 완료하면 안 된다.

### 단계 B — 타깃 생성과 손실 구현을 수정

권장 패치 단위:

1. own/extra photo holdout을 통합하고 재현 가능한 view RNG를 사용한다.
2. pruning score/ID loader와 공통 cell ownership을 구현한다. 셀별 target count/center/extent 계약을 맞춘다.
3. spatial sampler의 전체 범위 보존과 multiscale 분포를 수정한다.
4. projection padding, 최종 active set 평가, 현재 batch visibility를 수정한다.
5. edge cache를 정정하고 full-resolution target을 확보한다.

각 패치는 작은 재현 검사로 확인하고, **모두 합친 데이터·loss 정정 baseline**을 새로운 깨끗한 split에서 학습한다. 이 baseline은 이후 구조 실험의 공통 출발점이다. 기존 S/T에 파이프라인을 부분 적용한 재평가와 새 baseline 학습 결과를 같은 이름으로 섞지 않는다.

Packing을 바꾸면 target의 slot 의미도 바뀐다. 과거 optimizer 상태를 그대로 이어서 다른 target으로 학습한 결과를 “한 하이퍼파라미터만 바꾼 실험”이라고 기록하면 안 된다.

### 단계 C — 같은 16k 예산에서 구조를 한 요인씩 비교

| 실험 | baseline 대비 변경 | 검증할 가설 |
|---|---|---|
| C1 | pooler 전 local xyz/extent 정규화만 | 작은 셀의 기하 신호가 더 잘 보존되는가 |
| C2 | 부분 셀 template/active centering 계약 | 낮은 count 셀의 centroid·표면 두께가 좋아지는가 |
| C3 | frame-first에서 unpack 고정 또는 xyz/aux 분리 | frame이 기하 envelope를 실제로 학습하는가 |
| C4 | C1 이후 pool blocks 또는 staged merge 중 하나 | 같은 latent 예산에서 local detail 전달이 나아지는가 |
| C5 | covariance-aware transport/타깃 평균 방식 변경 | 얇은 Gaussian의 분리가 유지되는가 |
| C6 | 실제 3D 이웃 기반 attribute context | cell 경계의 seam과 가는 선 연결이 좋아지는가 |

각 실험은 먼저 단일 snapshot의 작은 고정 view 집합에서 충분히 fitting되는지 확인하고, 그 다음 두 scene 전체로 넓힌다. 단일 snapshot 실패는 구현/최적화/표현력의 한계를 분리하는 데 도움이 되고, 성공만으로 일반화를 주장할 수는 없다.

구조 실험의 기본 조건은 **262,144 output slots, 16,384 scalar, `4|1|3|8`**이다. appearance 8을 줄이거나 shape 3을 나누어 “독립 token이 늘었다”고 해석하는 변경은 첫 순서가 아니다. 외부 xyz/centroid side-channel로 렌더를 맞추는 방법도 latent-only 복원 목표와 구분한다.

### 단계 D — 예산/표현 형식의 품질 한계 확인

단계 B/C 후에도 특정 detail이 남으면 §5.3의 A–F 렌더 사다리를 완성한다.

- D 단계부터 이미 난간이 없으면 sampling/원본 표현 문제를 먼저 해결한다.
- D에는 있고 E에서 사라지면 codec 정보 전달·decoder·목적함수 문제다.
- full SH에서만 복원되면 시점 의존 appearance의 예산을 검토한다.
- raw Gaussian 자체가 FP16 저장으로 손실되었다면 NPZ 생성 경로를 바꾼다.
- 충분히 최적화한 same-budget oracle도 실패할 때 비로소 고정 16k 예산의 한계를 논의한다. 선형 PCA oracle 하나만으로 nonlinear codec의 불가능성을 증명할 수는 없다.

## 11. 다음 실험의 대시보드와 회귀 검사

### 11.1 PSNR 한 숫자로 고르지 않기

| 영역 | 최소 기록 항목 |
|---|---|
| identity | source ID 포함율, encoder/target owner 일치율, cell count MAE, overflow |
| 화면 | scene/view별 PSNR·SSIM, pred↔target 지표, 같은 ROI의 full-res crop |
| detail | 난간·글자·차륜·트럭 외곽 등 고정 ROI의 edge/고주파 오차 |
| active set | GT/예측 count, false-positive/negative, fallback 횟수 |
| geometry | 최종 nudged xyz의 p2g/g2p, 국소 spacing 정규화 거리, active centroid/radius, 표면 두께 |
| splat | 화면 radius·alpha contribution·visible outlier, covariance anisotropy, 축별 scale cap-hit |
| transport | 행/열 marginal error, entropy, 최대 match mass |
| latent | shape/app 민감도, 정의를 명시한 rank, noise/quantization 후 변화 |
| 최적화 | 대표 parameter군별 loss gradient norm, clipping 전후 norm, nonfinite gradient 횟수 |

ROI 좌표와 카메라 목록은 실험 전에 고정한다. 높은 PSNR 한 장만 고르거나 서로 다른 snapshot/해상도의 crop을 비교하지 않는다. Scene별 평균과 함께 worst-view/하위 분위도 저장한다.

### 11.2 추가하면 가치가 큰 회귀 검사

1. 실제 저장 schema의 작은 NPZ fixture에서 score/ID가 loader→sampling까지 전달되는지.
2. 서로 다른 capacity에서도 공유 source ID의 cell owner가 일치하는지.
3. 여러 분리된 cluster와 N/K 경계값에서 sampler가 전역 범위를 덮는지.
4. 비활성 slot의 좌표를 바꿔도 해당 slot을 제외하는 loss가 변하지 않는지.
5. count 1/32/64/128/255/256에서 valid template 중심과 coverage가 계약대로인지.
6. frame-only gate가 닫혔을 때 허용하지 않은 parameter가 xyz를 바꾸지 못하는지.
7. 동일 주소의 다른 reference에 edge cache가 잘못 적중하지 않는지.
8. 서로 다른 scene의 사진/카메라가 train/eval에 섞이지 않는지.
9. raw/normalized compact decode가 전체 속성과 렌더까지 일치하는지.
10. epsilon·cost 범위별 Sinkhorn marginal error가 사전에 정한 tolerance 안에 드는지.

이번 작업에서는 위 수정용 테스트들을 학습 코드에 추가하지 않았다. [reproduce_audit.py](audit_20260915/reproduce_audit.py)는 **현재 문제를 재현하고 수치를 저장하는 진단 도구**다.

## 12. 재현 자료와 남은 한계

### 12.1 산출물

| 파일 | 내용 |
|---|---|
| [reproduce_audit.py](audit_20260915/reproduce_audit.py) | CPU/데이터/체크포인트/렌더/rank 재현 |
| [cpu_probes.json](audit_20260915/cpu_probes.json) | 작은 반례·self 비교·schedule |
| [view_audit.json](audit_20260915/view_audit.json) | held view 목록과 101개 train NPZ 유입 목록 |
| [data_probes.json](audit_20260915/data_probes.json) | 원본 dtype, 점 개수, source owner, count/center/extent 차이 |
| [model_probes.json](audit_20260915/model_probes.json) | T70k, 두 snapshot의 18개 카메라별 점수·속성·거리 |
| [S model_probes.json](audit_20260915/S16k_62000/model_probes.json) | S62k의 동일 검사 |
| [rank_probes.json](audit_20260915/rank_probes.json) / [S rank](audit_20260915/S16k_62000/rank_probes.json) | 빈 셀 제외를 포함한 rank 후속 측정 |
| [checkpoint_metadata.json](audit_20260915/checkpoint_metadata.json) | E1/S/T unpack 및 selection state |
| [inventory.json](audit_20260915/inventory.json) / [source_manifest.json](audit_20260915/source_manifest.json) | 검토 파일 구조와 소스 해시 |

기존 이미지의 육안 확인 예: [T70k 기차 view3](../runs/T16k_20260913_093332/render_70000/step70000_scene0_view3_model.png). 난간·차륜 주변의 뭉침과 배경의 흐림을 확인했다. 이 기존 PNG 자체가 위 재측정 점수의 입력은 아니며, 점수표는 본 감사에서 별도로 렌더한 결과다.

### 12.2 재현 명령

저장소 루트에서 실행한다. GPU 번호는 사용 가능한 장치에 맞춘다. 진단 파일 출력만 쓰며 checkpoint·dataset·학습 코드를 변경하지 않는다.

```bash
# CPU 반례 + 데이터/held-view 감사
/home/super/anaconda3/envs/can3tok/bin/python \
  mdmd/audit_20260915/reproduce_audit.py

# T70k 모델/렌더 검사만
/home/super/anaconda3/envs/can3tok/bin/python \
  mdmd/audit_20260915/reproduce_audit.py --model-only --device cuda:3

# S62k 모델/렌더 검사만
/home/super/anaconda3/envs/can3tok/bin/python \
  mdmd/audit_20260915/reproduce_audit.py --model-only --device cuda:3 \
  --args runs/S16k_20260910_012456/args.json \
  --checkpoint runs/S16k_20260910_012456/ckpt_step00062000.pt \
  --out-dir mdmd/audit_20260915/S16k_62000

# 렌더 생략, nonempty 셀 rank만 다시 측정
/home/super/anaconda3/envs/can3tok/bin/python \
  mdmd/audit_20260915/reproduce_audit.py --model-only --ranks-only --device cuda:3

# E1/S/T unpack weight와 checkpoint selection state 검사, CPU만 사용
/home/super/anaconda3/envs/can3tok/bin/python \
  mdmd/audit_20260915/reproduce_audit.py --metadata-only
```

첫 GPU 진단은 current CUDA device를 명시하지 않아 rasterizer에서 illegal-memory-access로 종료됐다. 진단 스크립트에 `torch.cuda.set_device(device)`를 적용한 새 프로세스에서 S/T 렌더 검사가 정상 종료됐다. 실패 로그는 `run.log`, 성공한 모델 검사는 `model_run.log`·`S16k_run.log`에 남겼다. **이 진단 장치 선택 문제를 학습 중인 모델의 오류 증거로 사용하지 않았다.**

### 12.3 이번 조사로 아직 확정하지 않은 것

- 각 수정이 최종 PSNR/SSIM을 얼마나 높일지: 재학습 A/B는 실행하지 않았다.
- 전체 validation·모든 과거 checkpoint의 정정된 순위: 이번 모델 측정은 두 snapshot씩이다.
- 실제 Sinkhorn plan의 불균형 정도, scale cap 포화 위치, padding에 대한 attribute 민감도의 크기.
- 원본 FP32→NPZ FP16, full SH→DC, 전체 원본→선택 target의 개별 렌더 손실.
- 16k latent가 이 두 scene에서 도달 가능한 최상의 상세 복원 품질.

원본 Gaussian의 covariance·시점 의존 표현을 확인하는 외부 기준은 [3D Gaussian Splatting 공식 프로젝트](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)와 [공식 renderer](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py)를 참고했다. 이 보고서의 버그 판정과 수치는 외부 사례를 유추한 것이 아니라 **현재 저장소와 실제 데이터·체크포인트 검사**에 근거한다.

---

## 최종 제안

다음 대규모 학습을 시작하기 전에 **scene-correct 평가 → full Gaussian decode → 공통 cell ownership → sampling/mask/photo/visibility 정정 baseline**을 먼저 완성하는 것이 좋다.

그 다음 같은 `4|1|3|8` 예산에서 **pooler 앞 좌표 정규화**와 **부분 셀·unpack의 기하 계약**을 우선 비교한다. 현재 확인된 문제들은 단순히 latent 폭, 가우시안 개수, loss 가중치 하나를 바꾸는 것으로 해결되는 종류가 아니다.
