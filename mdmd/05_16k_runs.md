# 16k 런 상세: L16k → E1

작성: 2026-09-15  
잠재 **16×32×32 = 16,384**. 출력 262,144. 씬 train + truck.
레이아웃 기본: 1024칸 × 256점, compact 16ch = cen 4 | occ 1 | shape 3 | app 8.
예외: L17k만 shape 7 / app 4.

숫자는 각 `eval/step*/metrics.json`에서 이 문서 작성 시 재확인.

---

## 0. 왜 16k로 갔는가

월드모델(DIAMOND) 입력 크기를 줄이려는 목표. 성공 줄(N1/P1)은 131k·한 씬.
`M_speedy_both` 등은 이미 씬 2개였지만 **여전히 group 64, 잠재 32×64×64**.

L16k는 한 번에:

- 잠재 8배 축소 (131k → 16k)
- 칸 64 → 256
- shape 19 → 3
- 씬 2개

HISTORY가 금지한 다변수 실험. 이후 모든 16k 해석은 이 교란을 깔고 간다.

centroid를 compact에 **남긴 채** 16k를 지키면 64×4096은 불가
(앵커 5ch × 4096 = 20,480 > 16,384). `FIX7_NOTES.md` §4.

---

## 1. L16k — `fix_6/runs/L16k_20260827_063453`

**왜:** 위 다변수. 당시 문서화는 “작은 잠재로 두 씬”.

**결과:**

| 시점 | relq | uniq | photo | xyz rmse | cen rmse |
|---|---:|---:|---:|---:|---:|
| 8k | 1.376 | 0.419 | 11.85 | 0.0332 | (중심은 학습 중) |
| 60k | **1.416** | 0.484 | 14.42 (GT 가우시안 13.99) | 0.0328 | **0.0083** |

xyz rmse는 52k 스텝 동안 1.2%. 모델은 **칸 중심 + 반지름 + 평균 색**.
트럭은 GT floater를 뭉개 PSNR이 유리, 기관차는 글자·난간이 빔.
`latent_erank` 11.0/11: decorr가 appearance까지 먹음.

렌더: `../can3tok_encoder_decoder_new_fix_6/runs/L16k_20260827_063453/render_58000/`

**다음 근거:** 코드 버그 4건 + 가드 부재 (`FIX7_NOTES.md`). 레이아웃은 그대로 두고
fix_7에서 재학습. 동시에 “shape가 3이면 256점을 못 푼다”는 가드가 생김
(`allow_low_shape_budget` 없이는 거절).

---

## 2. fix_7 코드 수정 (학습 아님)

대상은 L16k 로그/metrics. 자세한 식은 `FIX7_NOTES.md`.

| # | 무엇이 틀렸는가 | 증거 | 수정 |
|---|---|---|---|
| 1.1 | `shape_channel_mask`가 앵커 이후 11ch를 shape로 봄 | `latent_erank` 11/11, decorr가 색↔기하 | `c_shape=3` 상한 |
| 1.2 | 퇴화 그룹에서 radius 비율 2.9억 배 | 로그 loss 23.6 → 600073 | 퇴화 그룹 리덕션 제외 |
| 1.3 | 로그 `loss` ≠ backprop total | render/splat 등이 로그 뒤에 더해짐 | 마지막에 logs["total"] 덮기 |
| 1.4 | DDP `latent_scale` 랭크 분란 | `broadcast_buffers=False` | 50스텝 all-reduce |
| 2.1 | shape/점 0.012 레이아웃이 통과 | 3/256 | 하드 0.02, 경고 0.05, 이스케이프 플래그 |
| 2.2 | 스파이크가 AdamW 2차 모멘트를 오염 | 50스텝 평균의 9%가 100× | `loss_spike_mult`, DDP MAX 합의 |
| 3.x | 씬 평균이 기관차 패배를 숨김 | truck +dB, loco −2.2, 보고 +0.45 | 씬별 eval, `psnr_gap_worst` |

**다음:** 이 코드 위에서 L17k(재분배)와 Q16k(shape 3 유지).

---

## 3. L17k — `runs/L17k_20260830_072543`

**왜:** 같은 16384에서 P2를 “제대로” 연장하려면 남은 11ch를 shape에 몰아야 한다.
약한 버전: shape 3→7, app 8→4. app를 깎은 근거: L16k에서 속성 NRMSE가 전부 ~1.0이라
8채널이 아무것도 안 샀다는 측정. `w_latent_decorr` 5→1 (버그 1.1).
`score_metric=psnr_gap_worst`, `loss_spike_mult=25`.

group_size 64를 안 쓴 이유: 이 예산에서 앵커 5ch×4096=125%. 실행 불가.

**결과 @53k** (`eval/step00053000/metrics.json`):

| | 값 |
|---|---:|
| photo | **12.98** |
| relq | **0.973** (s0 1.049 / s1 0.897) |
| uniq | **0.308** (s0 0.280 / s1 0.336) |
| cen rmse | 0.0114 |
| scene0 / scene1 photo | 12.73 / 13.22 |

`relq`를 √2에서 내린 **유일한 16k 런**. 동시에 uniq·PSNR 악화. opacity NRMSE 1.62
(문서 기록). 죽은 extra shape 채널.

**다음:** 같은 256점 칸에서 app를 깎아 shape만 올리는 복제는 닫힘.
단, “relq를 깨는 임계가 shape/점 ≈0.03 부근”일 수 있다는 측정은 남김.
10.B의 64점·shape 3은 0.047로 그 위. **그림이 안 산 것과 relq가 움직인 것을
한 문장으로 지우면 안 됨.**

---

## 4. Q16k — `runs/Q16k_20260904_023431`

**왜:** 다시 shape 3 / app 8. `allow_low_shape_budget`. 부가 항:
dispersion, plane chamfer, proj hist. 버그 수정 코드 위 from-scratch.

**결과 @30k:** photo 14.05, relq 1.406, uniq 0.454.
scene0 13.43, scene1 14.67.

Q16k README (`runs/Q16k_.../renders/README.md`)가 남긴 것:

- 트럭 held-out은 GT 절단본을 앞섬 (구멍을 메움).
- 기관차 028630이 약점. 가우시안 81,369 = 예산의 0.31배.
- **자기 카메라 PSNR이 눈을 이김:** GT 가우시안 19.21, 모델 20.38이지만
  난간·WESTERN PACIFIC은 뭉개짐. L2가 매끄러운 쪽을 보상.
- 홀드아웃은 스냅샷 축과 사진 축 **둘 다**. `00004.jpg`는 학습 스냅샷과 공유.

**다음:** VGG·edge로 디테일을 살릴 수 있는지 → Q16kD.

---

## 5. Q16kD — `runs/Q16kD_20260907_054951`

**왜:** VGG·edge @50k. PSNR vs 눈.

**결과 @64k:** photo 14.72, relq 1.419, uniq 0.471.
scene0 14.15, scene1 15.30.

`relq`/`uniq`는 Q16k와 같음. PSNR은 올랐고 글자는 README 패턴과 같음.

**다음:** 홀드아웃 누수를 더 막고 (S16k split), 디코더 폭만 키우는 대조 (R16k).

---

## 6. R16k — `R16k_20260909_034005`

**왜:** “작은 코드에서는 똑똑한 자유 디코더가 불리” (`FIX2` rank 4.67).
현재 레이아웃에서 **디코더 `shape_xyz` 폭만** 1024→2048. 파라미터 101.10M → 105.49M.
Q16k와 split·시드 동일. 한 변수.

**결과** (step 8000–17000, 10개 eval 평균, `ANALYSIS_ADDENDUM_KR.md` B):

| 지표 | R16k − Q16k | 부호 |
|---|---|---|
| template_erank | **+1.0** | 10/10 앞섬 |
| photo PSNR | **−0.20 dB** | 10/10 뒤짐 |
| SSIM | −0.015 | 10/10 |
| relq | 불변 | |

**다음으로 닫힌 것:** 이 레이아웃에서 디코더를 먼저 키우기.
병목은 MLP 폭이 아니라 shape 3이 가리키는 물체(256점 혼합).

---

## 7. S16k — `runs/S16k_20260910_012456`

**왜:** holdout 엄격. `assets/split_speedy_both_blockB.json`:
`--eval_view_force 3`, 해당 학습 스냅샷 7개 제거. 이전 분할은 val의 98%가
train과 10 3DGS-스텝. 새 분할 최소 210스텝. `detail_start=40000`.
from-scratch 16k 기본선.

**결과:**

| | 40k | 62k |
|---|---:|---:|
| photo | 14.28 | 14.80 |
| uniq | 0.459 | 0.467 |
| relq | 1.406 | 1.412 |
| cen rmse | (문서 0.034) | **0.035** |
| scene0 (기관차, GT 14.81) | 13.68 | 14.14 |
| scene1 (트럭, GT 14.09) | 14.88 | 15.45 |

VGG(`detail_start=40000`) 이후 PSNR은 오르고 `relq`/`uniq`는 안 움직임.
centroid는 L16k @60k의 0.008보다 이미 나쁨.

**다음:** VGG가 켜진 뒤 floater가 커 보임. 깨끗한 40k에서 splat+spacing을 먼저
켜고 detail을 늦춤 → T16k. shape 3은 안 바꿈 (바꾸려면 from-scratch).

---

## 8. T16k — `runs/T16k_20260913_093332`

재개: S16k `ckpt_step00040000.pt`. 런처 `scripts/launch_t16k_resume_40k.sh`.

켠 것:

- `w_splat_area` 0→1.0
- `w_intra_spacing` 0→2.0 (평균 NN 힌지)
- sinkhorn ε 0.08→0.005 (40k부터 12k)
- `detail_start` 40k→44k
- `score_metric=psnr_gap_worst`
- `max_steps` 72k

**결과 @70k** (`eval/step00070000/metrics.json`):

| | S16k 40k | S16k 62k | T16k 70k |
|---|---:|---:|---:|
| photo | 14.28 | 14.80 | **14.80** |
| uniq | 0.459 | 0.467 | 0.471 |
| relq | 1.406 | 1.412 | 1.415 |
| cen rmse | 0.034 | 0.035 | **0.038** |
| scene0 | 13.68 | 14.14 | 14.19 (gap **+0.63**) |
| scene1 | 14.88 | 15.45 | 15.42 (gap **−1.33**) |

L16k @60k 대비: 사진 14.42→14.80, centroid 0.008→0.038, offset rmse 0.0298 vs 0.0299.
**칸 안은 원래 안 맞았고, T16k가 깨뜨린 것은 중심.**

로그 `[joint] step 70000`: loss 35.33. TOP은 proj_hist 27%, attr_set 11%,
coverage 8%, Sinkhorn 6%. splat ×w=1 ≈ 총손실 **0.5%**. spacing은 TOP 밖.
움직인 쪽은 렌더 L1/VGG.

view3 모델 렌더 (대화 측정): 이 뷰 PSNR은 44k→70k에서 오르는데
`745` / `WESTERN PACIFIC`·난간은 뭉개짐.

**다음으로 닫힌 것:** 같은 16k·shape 3에서 splat/VGG/스텝만 더하기.

---

## 9. E1 — `runs/E1_20260915_041813`

**왜:** T16k는 `folding_res_start=0`, refine_start=400. folding 주석:
잔차가 프레임을 앞지르면 공이 찌그러짐. E1은 레이아웃 불변,
`folding_res_start=8000`, `decoder_refine_start=8000`, from-scratch.
게이트는 eval **8k**에서 글자·이방성. VGG는 `detail_start=20000`으로 이 확인 밖.
**N1 재현이 아님** (N1도 residual 0부터). 스케줄만의 대조.

`args.json` 확인: `budget_shape=3`, `folding_res_start=8000`,
`decoder_refine_start=8000`.

**결과 (이 문서 작성 시 eval은 3k까지).** 8k 게이트 미도달.

| | E1 @3k |
|---|---:|
| photo | 9.16 (GT 14.45, gap 5.29) |
| relq | 1.367 |
| uniq | 0.412 |
| cen rmse | 0.029 |
| tmpl_erank pred/GT | 5.53 / 36.3 |
| scene0 / scene1 photo | 8.13 / 10.19 |

3k는 S16k @1k와 비교할 자리이지, T16k @70k와 비교할 자리가 아님.
잔차·refine가 아직 꺼져 있으므로 tmpl_erank 5.5는 **의도된 템플릿 시작**에 가깝다.
8k에서 글자가 안 움직이면 “스케줄만으로는 256점·shape 3이 안 된다”는 쪽으로
읽는다. 그 전에는 E1을 실패로 닫지 말 것.

---

## 10. 10.B 오라클 (학습 아님, `PLAN_10B_ANCHOR_KR.md`)

같은 16,384 / 같은 k-means 파티션에서 칸 크기만 바꾼 기하 오라클
(`tools/oracle_layout_tradeoff.py`, 2026-09-15):

| 레이아웃 | abs_chamfer |
|---|---:|
| 3ch × 1024 (현재) | 0.00341 |
| 7ch × 1024 (L17k) | 0.00330 |
| 3ch × 4096 (10.B, occ 유지) | **0.00247** |
| 4ch × 4096 (occ 제거) | **0.00236** |
| 19ch × 4096 (N1 예산) | 0.00250 |

64점 칸에서 기하는 4채널이면 포화. 19ch가 4ch보다 나쁨.
`ich`는 큰 칸에 유리해서 교차 레이아웃 비교 금지.
사용자는 같은 날 **latent packing을 유지**하라고 해서 10.B는 이 사본의 기본이 아님.
appearance 0 설계는 사용자가 거절한 축.

---

## 11. 한 표로 16k 학습 런

| 런 | 한 일 | photo | relq | uniq | 판정 |
|---|---|---:|---:|---:|---|
| L16k @60k | 다변수 16k | 14.42 | 1.416 | 0.484 | 칸 안 정지, 중심은 맞음 |
| L17k @53k | shape 3→7, app 8→4 | 12.98 | **0.973** | **0.308** | relq만 이김, 그림 죽음 |
| Q16k @30k | shape 3 + 부가 항 | 14.05 | 1.406 | 0.454 | 기본선 |
| Q16kD @64k | VGG·edge | 14.72 | 1.419 | 0.471 | 사진↑ 기하 불변 |
| R16k | 디코더 폭만 | Q16k−0.20 | 불변 | — | 폭 확대 닫힘 |
| S16k @40k | 홀드아웃 수정 | 14.28 | 1.406 | 0.459 | 16k 기본선 |
| S16k @62k | +VGG | 14.80 | 1.412 | 0.467 | 사진↑ 기하 불변 |
| T16k @70k | splat/spacing/sinkhorn | 14.80 | 1.415 | 0.471 | 사진= S16k62k, **cen 악화** |
| E1 @3k | residual/refine 8000 | 9.16 | 1.367 | 0.412 | 8k 게이트 전 |
