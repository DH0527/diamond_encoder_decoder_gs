# 2026-09-16 계약 재측정

작성: 2026-09-16  
스크립트: `tools/verify_contracts.py`  
결과: [`verify_20260916/verify.json`](verify_20260916/verify.json), 그림 [`verify_20260916/figs/`](verify_20260916/figs/)  
장치: `cuda:0`. 학습 없음. 옛 `args.json`이라 B/C 플래그는 모두 꺼져 있다.

같은 val 인덱스 **166 = 기차 `step_028630`**, **382 = truck `step_028750`**.  
held-out 9장/씬, 인덱스 `[3, 26, 42, 65, 93, 229, 238, 239, 243]`.

## 1. `decode_compact` = 학습 decode

09-15 감사에서 T70k raw decode는 xyz만 같고 속성 MAE가 기차 1.751 / truck 1.534였다. `attr_pred` 키도 없었다.

| checkpoint | snapshot | attr_pred 키 | xyz max | attr MAE | encode를 정규화 없이 decode |
|---|---|---|---:|---:|---:|
| T70k | 028630 | 있음 | 0 | **0** | 0.968 |
| T70k | 028750 | 있음 | 0 | **0** | 2.435 |
| S62k | 두 장 | 있음 | 0 | **0** | — |
| E1 16k | 두 장 | 있음 | 0 | **0** | 2.461 / — |

T `latent_scale=0.9125`. 정규화를 안 되돌리면 좌표가 깨진다.

## 2. 씬을 가린 PSNR (자기 / 다른 / 혼합)

숫자는 pred + GT mask. 괄호는 09-15 감사의 pred PSNR.

| checkpoint | snapshot | 자기 씬 9장 | 다른 씬 9장 | 혼합 18장 |
|---|---|---:|---:|---:|
| S62k | 기차 028630 | **18.475** (18.482) | 9.712 (9.662) | 14.094 (14.072) |
| T70k | 기차 028630 | **18.327** (18.324) | 9.739 (9.727) | 14.033 (14.025) |
| S62k | truck 028750 | **21.399** (21.397) | 9.482 (9.482) | 15.440 (15.440) |
| T70k | truck 028750 | **21.508** (21.510) | 9.311 (9.311) | 15.409 (15.410) |
| E1 16k | 기차 028630 | 15.209 | 9.171 | 12.190 |
| E1 16k | truck 028750 | 17.305 | 9.133 | 13.219 |

자기 씬과 감사가 0.01 dB 안에서 맞는다. 혼합 14.x는 다른 씬 카메라가 끌어내린 값이다.  
GT 천장: 기차 20.075, truck 20.604. E1 16k는 residual/refine이 막 열린 직후라 S/T보다 낮다.

GT mask와 predicted presence의 자기 씬 차이는 T 기차 18.327 vs 18.324, truck 21.508 vs 21.511.

## 3. 그림 (왼쪽 사진 / 오른쪽 예측)

각 파일은 그 씬의 held-out **첫 카메라** 한 장이다. 9장 평균이 아니다.

- 기차 자기: `T16k_70k_step_028630_own_photo_pred.png` (이 한 장 21.52 dB)
- 기차×truck 카메라: `T16k_70k_step_028630_foreign_photo_pred.png` (7.73 dB)
- truck 자기: `T16k_70k_step_028750_own_photo_pred.png`
- S, E1도 같은 이름 규칙

## 4. 옛 args는 구조 플래그 off

세 체크포인트 모두:

```
normalize_pooler_xyz=0 count_aware_template=0 attr_slot_mask=0
shared_cell_owner=0 holdout_own_photo=0 keep_extra_fullres=0
use_fixed_anchor_center=0
```

`eval_fingerprint`가 예전 ckpt에 없어서 resume하면 `best_score`는 초기화된다. T의 저장값 `-14.28`은 예전 혼합 PSNR 부호.

## 5. holdout own-photo

플래그가 꺼진 현재 로더 기준으로, photo_map 전체에서 held-out 이미지와 연결된 항목은 **238**개다. 09-15의 101은 train split만 센 수다. 새 런에서 `--holdout_own_photo 1`이면 이 경로가 막힌다.

## 다시 돌리기

```
CUDA_VISIBLE_DEVICES=0 python -u tools/verify_contracts.py \
  --device cuda:0 --out mdmd/verify_20260916
```
