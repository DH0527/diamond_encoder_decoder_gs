# 실험 연대기: 설계 → 결과 → 다음 근거

작성: 2026-09-15  
범위: 2026-08-01 ~ 09-15, 출력 262,144 고정.  
잠재는 8/17까지 131,072 (32×64×64), 8/27 L16k부터 16,384 (16×32×32).

각 절은 세 덩어리다.

1. **왜 이렇게 설계했는가**
2. **결과가 무엇인가** (숫자 + 출처)
3. **그다음을 무슨 근거로 바꿨는가**

더 긴 표와 오라클은 `ANALYSIS_CODEC_WM_KR.md`, `HISTORY_KR.md`.
8/1–8/17을 버그·오라클까지 풀어 쓴 것은 `08_aug01_aug17_detail.md`.

---

## A. 토크나이저 이전 ~ can3tok 재작성 (8/1–8/6)

### 왜

투영만 맞으면 3D가 거짓이어도 넘어가는 실패를 피하려고, voxel/앵커로 기하를 먼저
두었다 (`GSStateTokenizer`, `CoordFirstGSAE` → `can3tok`). 주 지표는
`codec_xyz_rmse`. 렌더 손실은 아직 없다.

재작성 이유 (`can3tok_encoder_decoder_new/README.md`): `z_raw` 절반이 항등 패딩 0.
compact가 상수를 압축. 262k에서 점당 유효 채널 0.25. 그래서 `validate_layout`이
생겼다.

### 결과 / 수정 (fix A–F)

| 실패 | 증거 | 수정 |
|---|---|---|
| 점을 prefix에만 팩 | `empty_frac≈0.42` | 슬롯 재분배 + kd |
| 산 그룹은 항상 꽉 참 | presence 붕괴 | occupancy 채널 |
| equiv가 shape 불변 | 오라클 rmse 악화 | shape에 equiv 0 |
| dispersion margin = GT 반경 5.7배 | 상시 blur | margin 축소 |
| gen이 공유 슬롯 템플릿만 | 간격 0.4× | `shape_xyz` 직결 |

점 수 13만 축소는 여기서 거절되었다.

### 다음

`shape_xyz` MLP가 데이터를 못 따라간다 → folding.

---

## B. Folding (`_fix_2`, 8/7–8/8)

### 왜

`shape_xyz` MLP 최고 chamfer 0.345 = 데이터 해상도의 1.81배. 학습된 맵 유효 rank
**4.67**. 랜덤 28→192 선형맵은 26.3. 학습이 해로운 구간이라는 측정 (`FIX2_NOTES.md`).

설계: `R · diag(s) · (고정 Fibonacci 공 + 잔차)`. 프레임이 싸고, 잔차는 캡.

### 결과

- 학습 0인 템플릿 chamfer **0.294** (MLP 0.345보다 좋음)
- 템플릿 정렬 + rank-22 오라클 **0.218**
- `slot_sort=pca` → 전 그룹 줄무늬, 폐기
- `group_cov` → 노이즈로 만족, coverage 악화, 폐기

### 다음

rmse / 슬롯 `rel_offset`으로 게이트 금지 (순열만 바꿔도 움직임). 기하는 이 상한에
붙었다고 기록하고 속성·렌더로 넘어감.

---

## C. 속성·렌더, 목표 뒤집힘 (8/8–8/15)

### 왜

기하 1:1이 rank-28 상한에 붙음 (ich 0.25, uniq 0.53, PSNR 17.35). “기하는 끝”으로
적었다.

### 결과 (목표가 바뀐 측정)

위치 rank-28 고정 + 속성만 렌더 → held-out **26.7 dB**. 중복 제거 +0.03 dB.
GT 속성 **재구성**은 ~8 ch/point. **같은 그림(등가)** 은 그룹당 8채널로 21.43 dB
(`MODEL.md` §7.6–7.7). 재구성 1 ch/point = 16.14 dB. 등가 0.12 ch/point = 21.43 dB.

속성 경로 버그 7개 (`HISTORY_KR.md` §4.3): 슬롯 동변, 이질 LayerNorm, zero-init
출력, `ap.detach()`로 렌더→인코더 0, 인코더 lr 0, residual 안장점, DataLoader
워커 시드. 전부 **손실은 찍히는데 compact에 정보가 안 흐름**.

실사진·뷰 다양성 **+4.7 dB** (이 프로젝트에서 인과로 잰 가장 큰 레버). T4(뷰 5→10)는
R8 15.39를 못 넘음 — 구성이 바뀌면 재현되지 않음.

Morton → k-means 앵커: 시간 일관성. **사진 PSNR 이득 0.00 dB.** 앵커 5ch에 1%
노이즈 → **−0.97 dB**. 자유 채널 30% → −0.19 dB.

간접 지표 일곱 번 연속 오진: 랭크 34%→83%여도 PSNR 불변 등 (`HISTORY_KR.md` §5.2).

### 다음

등가 8ch appearance를 compact에 넣고, 렌더가 기하를 이기지 않게 스케줄을 짠다.
GT-swap: {xyz, scale, opacity, color}를 **같이** 바꿀 때만 그림이 좋아짐.
shape에 대한 렌더 그래디언트는 xyz detach 아래 **정확히 0**.

---

## D. R 계열과 F3 (8/16–8/17)

레이아웃은 아직 **4096×64, 잠재 131k**. 한 변수씩.

| 런 | 왜 | 결과 | 다음 |
|---|---|---|---|
| Q1 | ich를 score | photo 12.95. ich만 개선, rot/opacity 발산 | 렌더 PSNR을 eval에 넣을 것 |
| R1 | `w_z_*` 거의 0 | tmpl_erank 1.7 | 집합거리를 끄면 안 됨 |
| R3 | 집합거리 복원 | 14.92 @10k | 유지 |
| T1 nudge×4 | 헤드 여유 | +0.02 dB | 닫힘 |
| T3 detach 0부터 | 렌더로 xyz | 14.86→13.84 | 렌더로 위치 금지 (이미 반복 실패) |
| R5 geometry-first | detach 늦게 | 14.69 < R3 | M4에서 가중으로 재시도 |
| R8 | 앵커+pooler+Sinkhorn, shape 19/app 8 | **15.39** @10k 당시 최고 | polish |
| J0/J1 | R8 위 zero-init joint 잔차 | J1−J0 → 0 | 붕괴한 기하 위 잔차 금지 |
| F3/B/C/D | shared-27를 직접 디코드 | ~9.5 dB. centroid 0.04→0.006. tmpl_erank~3 | 27ch를 **한 벡터로 만들어 64 쿼리에 방송**. 로컬 토큰 무시. **지금 16k와 같은 계약** (점만 64→256) |

compressor 숙제 (131k에서 측정, `STRUCTURE_KR.md`): shape 19 중 **6.58**만 사용.
`token_merge` effective rank intra **34.26 → 13.83**. mid 23.6 → compact 10.9.
이 숫자는 **shape 19 / 그룹 64 계열**에서 잰 것이다. 16k·shape 3에서 다시 안 쟀다.

---

## E. 그림이 움직인 줄: R8I → M4 → N1 → P1 (8/17–8/20)

**한 변수씩.** 레이아웃 끝까지 4096×64, shape 19, app 8, 잠재 131k, **씬 하나**.

| 런 | 왜 (한 가지) | photo PSNR | 그다음 |
|---|---|---|---|
| R8H | R8 polish | 16.57 @9500 | 거대 splat |
| R8I | 그룹 상대 scale 밴드 + splat 힌지 | 16.35 @5k (추가 5k는 손해) | 기하 가중 |
| M4 | 기하 가중 ↓, `w_render_attr 60`, detach 해제 | **17.75** @16k | 속성 집합 |
| N1 | **`w_attr_set 3`** | **18.67** @15k, GT 18.70, gap 0.04. relq=**1.278**, cen=0.006 | 인코더가 칸을 버림 |
| P1 | 인코더 칸당 입력 144→512, overflow 23%→0.2% | **19.20** @14k (로더 크래시). relq=1.282 | 16k·멀티씬으로 넘어감 |

N1 설정 재확인 (`fix_4/runs/N1_20260819_111156/args.json`): group 64, occupancy 1,
shape 19, app 8, `folding_res_start=0`, `decoder_refine_start=400`.
**지연 잔차로 이긴 런이 아니다.**

실패한 곁가지: M1 `init_skip head_scale` → 11 dB. M5 decorr만 → 랭크만 오름.
N4 같은 레시피 from-scratch, 렌더 늦음 → 15.80. coverage 가중 2.0은 목적함수의
0.02% (단위 불일치). T3는 렌더:기하 ≈ 1:190.

N1에서 남은 것: uniq ≈ 0.49, 구멍을 큰 splat으로 메움, 웜스타트 체인.
**relq≈1.28인데도 글자·난간이 읽혔다.**

P1을 **학습 안 한 트럭**에 넣으면 9.59 dB, 학습한 씬 14.79 (`ANALYSIS_ADDENDUM_KR.md`
A). 디코더가 씬을 외운다. 16k 멀티씬으로 넘어가기 전에 이미 일반화는 실패.

참고 렌더:

- N1: `../can3tok_encoder_decoder_new_fix_4/runs/N1_20260819_111156/render/step00014000_val_018400_028630/step_028630.png`
- P1: `../can3tok_encoder_decoder_new_fix_5/runs/P1_20260820_081200/render_step12000/step_028630.png`

---

## F. 그룹 32→64 (P2) vs 나중에 64→256

### 왜 P2가 그룹을 키웠는가

이긴 숫자는 사진이 아니라 **그룹 오프셋 PCA 천장** (`launch_fix2_262k_ddp.sh` 헤더).

| | 이전 | P2 |
|---|---|---|
| group_size | 32 | **64** |
| 그룹 수 | 8192 | 4096 |
| merge | 2 | **1** |
| occupancy | ~0 | **0** |
| shape | 12 | **28** |
| 오라클 offset rmse | (P1 후 0.00651) | **0.00502 (−41%)** |

centroid 4채널은 그룹마다 나간다. 8192그룹이면 131k의 12.5%가 중심에 묶임.
4096으로 줄이면 그 세금이 shape로 이동. 당시 점 MSE의 99.3%는 중심이 아니라
칸 안 오프셋. 결론: **그룹을 키워서가 아니라, 같은 잠재에서 중심 세금을 덜 내고
shape에 넣고, 셀을 쪼개지 마라 (merge=1).**

N1 학습 계약은 P2 오라클과 다르다: occ 1, shape 19, app 8, shape/점 = 0.297.
칸 64와 merge=1만 P2를 따른다. “N1 = shape/점 0.44”는 오라클 P2를 N1에 붙인
오류다.

### 왜 16k에서 256이 되었는가

centroid를 compact에 **남긴 채** 16k를 지키면 64×4096은 불가.

| group_size | 그룹 수 | 앵커 5ch (cen4+occ1) | 16384 대비 |
|---|---|---|---|
| 64 | 4096 | 20480 | **125% 실행 불가** |
| 256 | 1024 | 5120 | 31% |

`FIX7_NOTES.md` §4. L16k `launch_cmd.txt`는 앞에 `--group_size 64`가 있고 뒤에서
256으로 덮는다. 64가 후보이었고 이 예산에서 막힌 흔적. **16k에서 64 vs 256 렌더
A/B는 문서에 없다.**

P2의 이득(세금을 shape로)을 L16k는 따라가지 못했다. 그룹을 키워 세금은 31%로
줄였지만 남은 채널을 appearance 8에 주고 shape는 3.

---

## G. 16k 줄 (8/24–9/15)

### G.1 L16k (fix_6, 8/27–)

**왜:** 월드모델 크기를 16k로. 한 번에 잠재 8배 축소, 칸 64→256, shape 19→3,
씬 2개. HISTORY가 금지한 다변수 실험.

**결과** (`L16k_20260827_063453`):

| 시점 | relq | uniq | photo | xyz rmse | cen rmse |
|---|---|---|---|---|---|
| 8k | 1.376 | 0.419 | 11.85 | 0.0332 | (중심은 학습 중) |
| 60k | **1.416** | 0.484 | 14.42 (GT 가우시안 13.99) | 0.0328 | **0.0083** |

xyz rmse는 52k 스텝 동안 1.2%. 모델은 칸 중심 + 반지름 + 평균 색.
트럭은 GT floater를 뭉개 PSNR이 유리, 기관차는 글자·난간이 빔.
렌더: `../can3tok_encoder_decoder_new_fix_6/runs/L16k_.../render_58000/`

`latent_erank` 11.0/11: decorr가 appearance까지 먹음 (버그).

**다음:** fix_7에서 버그·가드·씬별 eval. 레이아웃은 그대로 두고 재학습.

### G.2 fix_7 버그 수정 (코드, 학습 아님)

`FIX7_NOTES.md`. L16k 로그/metrics에서 측정한 것.

1. `shape_channel_mask`가 앵커 이후 전부(11ch)를 shape로 봄 → decorr가 색과 기하를
   무상관으로 밈. 수정: 상한 `c_shape=3`.
2. `codec_radius`가 퇴화 그룹에서 2.9억 배. 로그 `loss`와 backprop total 불일치.
3. DDP `latent_scale` 분란.
4. shape/점 0.02 미만 하드 플로어 + 0.05 경고. L16k 3/256=0.012는
   `allow_low_shape_budget` 없이는 거절.
5. 씬별 eval, `psnr_gap_worst`.

이 수정 **위에서** 16k 학습을 다시 돌린 것이 Q16k → S16k → T16k.

### G.3 L17k

**왜:** 같은 16384에서 P2를 “제대로” 연장하려면 남은 11ch를 shape에 몰아야 한다.
약한 버전: shape 3→7, app 8→4.

**결과:** relq **0.973** @53k — √2를 깬 **유일한 16k 런**. 동시에 uniq **0.308**,
PSNR **12.98**, opacity NRMSE 1.62. 죽은 extra shape 채널.

**다음:** 같은 256점 칸에서 app를 깎아 shape만 올리는 복제는 닫힘. relq 실패검출을
깨는 임계가 shape/점 ≈0.03 부근일 수 있다는 측정은 남김 (10.B의 64점·shape 3은
0.047).

### G.4 Q16k / Q16kD / S16k / R16k

다시 shape 3 / app 8.

| 런 | 왜 | relq / uniq | 비고 |
|---|---|---|---|
| Q16k | dispersion, plane chamfer, proj hist | 1.41 / 0.45 | `allow_low_shape_budget` |
| Q16kD | VGG·edge @50k | 1.42 / 0.47 | PSNR vs 눈: 글자는 안 오름 |
| S16k | holdout 엄격 (자기 사진 누수 차단), detail@40k | 1.41 / 0.47 | from-scratch 16k 기본선 |
| R16k | **디코더 `shape_xyz` 폭만** 1024→2048 | relq 불변 | tmpl_erank +1.0 (10/10), photo **−0.20 dB** (10/10) vs Q16k |

R16k가 닫은 것: 이 레이아웃에서 디코더를 먼저 키우기. 병목은 MLP 폭이 아니라
shape 3이 가리키는 물체(256점 혼합).

S16k @40k / @62k (이 기록 작성 시 `metrics.json`에서 재확인):

| | S16k 40k | S16k 62k |
|---|---:|---:|
| photo | 14.28 | 14.80 |
| nn_unique | 0.459 | 0.467 |
| relq | 1.406 | 1.412 |
| centroid rmse | 0.034 | 0.035 |
| scene0 (기관차, GT 14.81) | 13.68 | 14.14 |
| scene1 (트럭, GT 14.09) | 14.88 | 15.45 |

VGG(`detail_start=40000`) 이후 PSNR은 오르고, `relq`/`uniq`는 안 움직인다.

### G.5 T16k (9/13–, S16k 40k 재개)

**왜:** S16k에서 VGG가 켜진 뒤 PSNR은 오르고 floater가 커 보임. 깨끗한 40k에서
splat+spacing을 먼저 켜고 detail을 4k 늦춤. shape 3은 안 바꿈 (런처 주석:
바꾸려면 from-scratch).

켠 것: `w_splat_area` 0→1.0, `w_intra_spacing` 0→2.0 (평균 NN 힌지),
sinkhorn ε 0.08→0.005 (40k부터 12k), `detail_start` 40k→44k,
`score_metric=psnr_gap_worst`, max_steps 72k.

**결과 @70k** (`eval/step00070000/metrics.json`, 로그 step 70000):

| | S16k 40k (재개점) | S16k 62k | T16k 70k |
|---|---:|---:|---:|
| photo | 14.28 | 14.80 | **14.80** |
| nn_unique | 0.459 | 0.467 | 0.471 |
| relq | 1.406 | 1.412 | 1.415 |
| centroid rmse | 0.034 | 0.035 | **0.038** |
| scene0 | 13.68 | 14.14 | 14.19 |
| scene1 | 14.88 | 15.45 | 15.42 |

L16k @60k와 비교하면 사진은 14.42→14.80, centroid 0.008→0.038, offset rmse는
0.0298 vs 0.0299. **칸 안은 원래 안 맞았고, T16k가 깨뜨린 것은 중심.**

로그 `[joint] step 70000`: loss 35.33. TOP은 proj_hist 27%, attr_set 11%,
coverage 8%, Sinkhorn 6%. splat ×w=1 ≈ 총손실 **0.5%**. spacing은 TOP 밖.
움직인 쪽은 렌더 L1/VGG.

view3 모델 렌더 (대화 측정, 자기 카메라 혼입 가능): 기관차/트럭 이 뷰 PSNR은
44k→70k에서 오르는데 `745` / `WESTERN PACIFIC`·난간은 뭉개짐. L2는 매끄러운
쪽을 보상.

**다음으로 닫힌 것:** 같은 16k·shape 3에서 splat/VGG/스텝만 더하기. `relq`/`uniq`
불변, centroid 악화.

### G.6 E1 (9/15, 스케줄만)

**왜:** T16k는 `folding_res_start=0`, refine_start=400. 잔차가 프레임을 앞지른다는
이전 측정(folding 주석, step 500에 공이 찌그러짐). E1은 레이아웃 불변,
`folding_res_start=8000`, `decoder_refine_start=8000`, from-scratch.
게이트는 eval **8k**에서 글자·이방성이지 PSNR이 아님. VGG는 `detail_start=20000`으로
이 확인 런 밖. N1도 residual을 0부터 열었으므로 **N1 재현이 아님.**

`args.json` 확인: `budget_shape=3`, 두 start 모두 8000.
경로: `runs/E1_20260915_041813`.

**결과 (작성 시 eval은 3k까지, 8k 게이트 미도달):**

| | E1 @3000 |
|---|---:|
| photo | 9.16 (GT 14.45) |
| relq | 1.367 |
| uniq | 0.412 |
| cen rmse | 0.029 |
| tmpl_erank pred/GT | 5.53 / 36.3 |
| scene0 / scene1 | 8.13 / 10.19 |

3k를 T16k @70k와 비교하지 말 것. 잔차·refine가 아직 꺼져 있어 tmpl_erank 5.5는
템플릿 시작으로 읽는 편이 맞다. 8k에서 글자가 안 움직이면 “스케줄만으로는
256점·shape 3이 안 된다”. 그 전에는 실패로 닫지 말 것.

상세 표: `05_16k_runs.md`.

---

## H. 16k 셀이 실제로 담는 것 (T16k/S16k 계약)

칸당 16숫자:

```
cen 0–2 : 그룹 xyz 원점. use_fixed_anchor_center=0 이면 인코더 표본 평균
          + tanh nudge × 0.05×extent. k-means npy가 아님.
cen 3   : log-extent (folding 길이 단위). npy에 없음.
occ 1   : live count
shape 3 : 칸 안 기하. fold_head(6-DoF)와 shape_xyz(256×3 잔차)가 셋 다 읽음
app 8   : appearance, 256 슬롯에 방송
```

`decode_compact(z)`는 앵커를 받지 않음. `partition_mode=morton`은
`scene_anchors`가 있으면 죽은 플래그. 그룹핑은 k-means + spill 8.

디코더 식:

```
점 = xyz + 반지름 × 프레임(shape) × (템플릿 + 잔차(shape))
겉모양 = appearance 8 방송
```

CodecDecoder 10층 refine는 접힌 벡터에서 T 토큰을 다시 만든다. F3
`structured_local` + joint 디코더는 이 런처에서 꺼져 있다.

네 겹의 평균 (코드 경로, 16k에서 다시 안 잰 것도 포함):

1. 256점 혼합 k-means 셀 — 한 코드가 획이 아니라 이웃
2. pooler: 최대 2048점 → 16 쿼리, `pool_blocks=1`. N1은 144→16
3. `token_merge`: T×448 → 1 벡터. 랭크 34.26→13.83은 **shape 19 모델**
4. shape 3이 256점을 풀고 app 8은 방송

---

## I. 닫힌 레버 vs 열린 레버 (9/15 분석 문서 기준)

닫힘 (`ANALYSIS_CODEC_WM_KR.md` §9) — 이 축만 다시 돌리지 말 것. 레이아웃이
바뀌면 같은 이름이 다시 열릴 수 있다.

- 칸 256 유지 + shape만 3→7 (L17k)
- 같은 레이아웃에서 스텝만
- splat/VGG/edge만 (T16k)
- 디코더 폭 (R16k)
- `w_attr_set` 가중만
- joint 전면 재작성 (F3)
- 렌더로 xyz
- 출력 K 축소, 잠재 131k 복귀 (현재 목표와 반대)

열림, 한 변수씩 (§10):

- **10.A** 16k + centroid를 compact에 유지, 256칸. appearance 팔레트/등가, 프레임
  먼저, spacing을 가까운 점 비율, 전역 pred→표면, VGG는 기하가 움직인 뒤.
  한 번에 다 바꾸지 말 것.
- **10.B** centroid를 compact 밖으로, 4096×64 재개방. 당시
  `budget_centroid < 4`는 ValueError. appearance 0 오라클과는 별개.
- **10.C** 멀티 데이터셋은 셀 의미가 생긴 다음. 세 번째 씬 생성은 병행 가능.

사용자 후속 제약 (같은 날 대화, `03_session_20260915_fix8.md`): 지금은
**latent 담는 방식을 고정**하고, 인코더–디코더 구조만 손보려 함. 10.B로 바로
가지 않음.

10.B 기하 오라클과 E1 @3k 숫자는 `05_16k_runs.md`. 난간/floater는 `06_visual_failure_and_wm.md`.
닫힌/열린 레버는 `07_closed_open_methodology.md`. N1 줄은 `04_n1_p1_success_line.md`.
