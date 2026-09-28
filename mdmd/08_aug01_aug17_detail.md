# 8/1–8/17 상세 (HISTORY 재구성 + 8/16 이후 1차 측정)

작성: 2026-09-15  
`HISTORY_KR.md`가 8/17에서 끝난다. 8/1–8/15 숫자는 **재구성**(프로젝트가 스스로 기록,
이 폴더 작성자가 재실행하지 않음). 8/16–8/17은 **1차 측정**.
요약 연대기는 `02_experiment_chronology.md`.

고정: `max_points=262,144`, `z_compact=32×64×64=131,072`. 점 수 축소는 사용자가 거부.

측정 프로토콜 (8/16 이후): held-out 실사진 8장, val 스냅샷 8개, downscale 2.
`psnr_gap` = 그 스냅샷의 GT 사진 PSNR − 예측.

---

## 1단계 (8/1–8/4) — CoordFirstGSAE, hybrid merge

런: `CoordFirstGSAE/runs/hybrid_merge1_gen_v5_c32_*`

- 8/1 `..._c32` — 32채널 hybrid merge
- 8/1–8/2 `..._c32_262k` — 262k 확장
- 8/3–8/4 `..._c32_262k_deeptoken` — 토큰 경로 심화

주 지표 `codec_xyz_rmse_norm`. 렌더 손실 없음.
투영만 맞으면 3D가 거짓이어도 넘어가는 실패를 피하려고 voxel/앵커로 기하를 먼저
(`GSStateTokenizer`, `CoordFirstGSAE` → `can3tok`).

---

## 2단계 (8/5–8/6) — can3tok 이식과 fix A–F

재작성 이유: `z_raw` 절반이 항등 패딩 0. compact가 상수를 압축. 262k에서 점당
유효 채널 0.25. `validate_layout`이 생긴 이유.

| | 원인 | 증거 | 대응 |
|---|---|---|---|
| A | 점을 prefix에만 팩 → 빈 그룹이 compact 24–70% 소비 | `empty_frac≈0.42`, 재분배 시 PCA 한계 −12~−59% | `slot_redistribute` + kd |
| B | `alive*G` count 가정 | 부분 채움에서 presence 붕괴 | `budget_occupancy=1`, shape=11 |
| C | equivariance가 shape 불변 | 오라클 rmse 0.009→0.013, `lat_std` floor 고착 | `equiv_shape_weight=0` |
| D | `dispersion_margin=0.01` | GT 그룹 반경 p50의 5.7배 → 상시 blur | margin 0.001 + 0.35×GT std |
| E | gen이 공유 슬롯 템플릿만 | gen spread p50 0.48, nn spacing 0.4× | `shape_xyz` 직결 |
| F | set chamfer / coverage / density-aware | 이전 minimal fix | 유지 |

의도적으로 안 한 것: max_points→131k, soft-clamp / adaptive_shape / mid↑ /
map_blocks↑ / residual_scale↑ (이전 붕괴), 이웃 centroid 프레임 정규화 (효과 ≈0).
기대: codec rmse 0.0082 → ~0.0047.

**다음:** `shape_xyz` MLP가 데이터를 못 따라감 → folding.

---

## 3단계 (8/7–8/8) — folding (`_fix_2`)

런 14개, 대부분 짧게 중단. `slot_sort=pca` 중단, `minloss`로 손실 항 19→12,
`rankfix`, `genfix`.

지표: 순열 불변 그룹 내 chamfer, 그룹 평균 반지름으로 정규화 (`intra_chamfer_rel`).
GT 자기 최근접 간격 0.191 = 데이터 해상도 1.0배.

| 디코더 | chamfer | × 간격 |
|---|---:|---:|
| `_fix` @6000, 순수 `shape_xyz` MLP | 0.345 | 1.81 |
| 첫 folding @2000 | 0.382 | 2.00 |
| folding envelope 단독, **학습 0** | 0.294 | 1.54 |
| folding + rank-22 residual, 템플릿 정렬 | **0.218** | **1.14** |

용량/가중치 문제가 아닌 세 측정:

1. 타깃이 거의 비압축적. 19,131개 GT 그룹 반지름 정규화 PCA: rank-28 잔차 0.593,
   rank-64도 0.454, `lam1/lam28` 16.9. 스펙트럼이 평평.
2. L2 최적해가 조건부 평균. 학습된 맵을 랜덤 코드로 탐침하면 유효랭크 **4.67**.
   학습하지 않은 랜덤 28→192 선형맵은 26.3. 예측 반지름 GT의 0.706.
   `template_erank` 5.1 vs GT 59.2.
3. 저랭크 보정이 안 하는 것보다 나쁨. 고정 템플릿 + rank-k 오라클: k=0 → 0.524,
   k=8 → 0.332, **k=22에서야** 0.263으로 순수 템플릿 0.302를 이김.
   기존 디코더는 rank 4.67 = **학습이 해로운 구간**.

도입식: `offsets_g = R(a_g) diag(s_g) (template + residual_g)`
- template: 고정 Fibonacci **채운 공**, 전 그룹 공유, 학습 안 함
- R diag(s): fold_head, axis-angle 3 + 축별 스케일 3, zero-init
- residual: shape_xyz, 마지막 층 zero-init → 순수 템플릿에서 시작

템플릿 오라클: 껍질×축정렬 0.367, 껍질×회전 0.320, 채운 공×축정렬 0.336,
채운 공×회전 **0.301**.

`--slot_sort template`: 중심화 → 공분산 고유프레임 화이트닝 → 같은 템플릿에 64×64 배정.

| 슬롯 순서 | 평균\|residual\| | rank-22 상한 | chamfer |
|---|---:|---:|---:|
| Morton | 1.010 | 1.176 | 0.261 |
| 최적 배정 | 0.408 | 0.679 | **0.218** |

이게 없으면 residual이 가용 22차원을 순열 되돌리기에 씀 — 첫 folding이
자기 envelope 오라클보다 나빴던 이유.

부호 규약: eigh + 자유 부호 반전은 50% 그룹에서 반사. 디코더 R은 항상 det>0.
skew가 0에 가까운 축을 뒤집어 det=+1 강제.

제거 (측정 있음):

- `group_cov`: 노이즈로 만족. 반지름 0.706→0.965, precision↑, coverage 0.540→**0.345**,
  chamfer 불변. `thread` 0.050→1.038 (GT 0.576) — 슬롯 순서 뒤섞음
- `w_direct_ratio`: folding이 direct/deep 분리를 없앰
- `slot_sort=pca`: 전 그룹 줄무늬 (`sqrt(lam2/lam1)` 0.036 vs 0.544)

기각:

- 셀 완전 활용: 오라클 −17~−20%지만 학습이 평평. 추정(미검증): occupancy 마스크가
  고엔트로피가 되어 shape 예산을 먹음
- 정규 재배열로 스펙트럼 평탄화: rank-28 0.593→0.555 (−6%). **순서는 평평한
  스펙트럼을 고칠 수 없다**
- 교차 그룹 문맥: 이웃 4/8/16이 자기 그룹보다 나쁨

결론: rmse / 슬롯 `rel_offset`으로 게이트 금지. `group_cov` 런에서 집합은 불변인데
rel_offset·rmse만 움직임. `--score_metric intra_chamfer`가 기본이 됨.
(나중에 Q1이 이 기본값으로 렌더를 안 봐서 실패.)

---

## 4단계 (8/8–8/12) — attribute 도입 (`new/`)

런: mC, mBC, FAILED mE (ckpt/OOM), mE_attr ×5, mF_slotemb, mG/mH full/fast.

### 진단이 세 번 바뀜

1차: 중복 — 증상은 맞았고 원인은 아님.
2차: 정보 예산 (`oracle_shape_rank.py`). 당시 모델 ich 0.251 / uniq 0.51 / PSNR 17.35
= **rank-28 상한과 일치**. 기하 종료로 기록.
3차: 위치 rank-28 고정 + 속성만 렌더, 4뷰 맞춤 held-out **26.69 dB**.
예산이 제한하는 것은 점집합 1:1이지 렌더 품질이 아님.

철회한 진술 4개:

| 앞서 한 말 | 실제 |
|---|---|
| 중복이 17.2 dB 격차의 지배 원인 | 아님. 속성 GT 고정 탓. 적응하면 26.7 |
| `nn_unique>0.85`가 attribute 진입 조건 | **반대.** 이 게이트가 해결책을 막음 |
| `w_xyz_residual`이 가장 필요 | 0.51→0.53, ~0 dB |
| geometry 먼저, attribute 나중 | 슬롯별 재구성 가정. 렌더 등가면 겹친 점은 문제 아님 |

속성 경로 버그 7개 (손실은 찍히는데 compact에 정보가 안 흐름):

1. permutation equivariance — 슬롯 임베딩으로 격리 R² 0.158→**0.984**
2. 이질 채널 LayerNorm — 색·불투명도 편차 삭제. R² 0.998→0.747. 채널별 표준화
3. zero-init 출력층 — 첫 스텝 하류 그래디언트 0
4. `ap.detach()` — 목적함수 85%인 render가 인코더에 **정확히 0**
5. 인코더 학습률 0 — 두 체크포인트에서 인코더 텐서 비트 단위 동일. `attr_enc` 분리
6. residual 이중 zero-init 안장점 — gate를 1e-2로 초기화
7. DataLoader `persistent_workers` — `set_epoch`가 워커에 안 감. 평생 같은 뷰

등가 vs 재구성 표는 `06_visual_failure_and_wm.md` §7. **이 표가 프로젝트 목표를 정함.**

---

## 5단계 (8/12–8/15) — 렌더 실규모, 실사진

런: mJ_capacity/dssim, mK_nbr/perc, **mL_photo**, mM_photo_anchor/perc, N1_mv, N2_mv_stage.

| 항목 | 크기 | 근거 |
|---|---|---|
| **뷰 다양성** | **+4.7 dB** | 총 렌더 800회 고정. 고정 4뷰 +0.02 vs 매 스텝 새 뷰 **+4.67** |
| 속성 (동시에 맞아야) | 3.3 dB | 하나만 GT −0.5~+0.2, 하나만 모델 −1.5~−2.7 |
| 입력 서브샘플링 | 2.5 dB | 예산 초과 스냅샷, 최대 5.32 |
| 커버리지 편향 | 2.0 dB | 모델이 고른 GT 17.66 vs 무작위 같은 수 19.65 |

실사진 당시: 모델 14.55, 타깃 Gaussian 21.75, 모델 vs 타깃 렌더 16.02.

**월드모델 전제 — 셀 시간 일관성.** Morton 청크는 densify 때 셀 중심이 그룹 반지름의
평균 **13.9배** 이동. k-means 앵커 **0.13배**, 반지름 0.064. FPS는 0.05배이나
반지름 1.18로 너무 넓음. `scene_anchors.npy`. 점 개수가 *동일한* 쌍만 고르면
0.1배로 나와 문제가 안 보임.

반박된 가설 (`ARCHITECTURE_KR.md` §10): 압축 제거해도 13.71→11.97 (개선 없음),
decorr로 랭크 34%→83%여도 PSNR 불변, 빈 그룹 0%인 장면도 랭크 9.56,
디코더 재합성 chamfer 거의 무해, 중복 완전 제거 +0.03 dB, 모든 점을 GT 위치로
옮겨도 −0.06 dB. **간접 지표 일곱 번 연속 오진.**

---

## 6단계 (8/15–8/16) — P/Q, 선택 기준만 개선

P1_decorr, P2_mergeattn, Q1/Q2/Q3 뷰 개수 A/B.

Q 증상 step 2000→6000: xyz rmse 악화, rel_offset 악화, rot NRMSE 5→**153**,
opacity 발산, color≈평균, **intra_chamfer만 개선**. `run_eval`은 렌더 PSNR 없음.
total loss 25.6→703 (gen이 삼킴). 배포 경로 `gen_decoder`는 detach된 latent로
발산 (gen ich 0.705 vs codec 0.254)인데 체크포인트는 codec만 봄.

여기서 8/16 세션이 시작 (`SESSION_LOG_KR.md`).

---

## 7단계 (8/16–8/17) — 1차 측정

### 오라클: 재구성 항이 끌어내림

기하 12ch + appearance 16ch (당시 배분) 공동 최적해 **9.95 dB**. 모델은 이미
13.4–14.6. 가중치 상위가 거의 전부 재구성 (`w_z_residual` 40 등).

K 자유 파라미터 피팅: K=262,144는 무작위 선택만으로 19.78, 최적화 불필요.
저K는 ~15 dB에서 멈춤 (densify 없는 Adam 바닥). 이 실험으로 K 축소는 판정되지 않음.
사용자는 이후에도 K 축소를 거부.

Q1 @6000 새 프로토콜: photo **12.95**, gap 5.49.

### 버그 11개 (에러 없이 조용히 잘못)

앵커 world 좌표, augmentation이 앵커를 안 따라가 슬롯 81% 빔, eval이 enc_input을
안 넘겨 rmse 0.675 vs 실제 0.0096, pooler가 attr_encoder 미호출, group_error
가짜 그룹, spill 비대칭, teacher forcing이 죽은 헤드, `--init_from`이 step을 0으로
리셋해 attr/render 게이트 닫힘, render가 GT 마스크로 선택, aniso 비율 미제한
(0.027–1411.5). 3·4·9·11은 이 세션의 실수.

### 런 (한 줄씩, 상세는 `02` D절·`04`)

R1 ❌ tmpl 1.7 — 순열 불변 집합거리까지 끊음.
R3 ✅ 14.92 @10k — 집합거리 복원.
R5 ❌ 14.69 — geometry-first가 R3를 못 이김.
R2/R4/R6 폐기 (버그 3, 4, 사망).
앵커 가둠 **0.00 dB**, 넘침 −0.22, 시간 일관성 13.9→0.13.
T1 nudge×4 = +0.02. T2 scale band 3변수 교란 중단. T3 detach 0부터 −1.0 dB.
T4 뷰 5→10, R8 15.39를 못 넘음 — +4.7 dB가 이 구성에서 재현 안 됨.
T5 appearance 폭 8→27 리셋, 설계 실수. 올바른 형태는 zero-init 가산 (J1).
R8 ✅ 15.39 @10k. J0/J1: J1−J0→0. F3 ~9.5 dB.

GT-swap·노이즈·K 곡선·folding 감사(shape 19 중 6.58, aniso 1411배, 렌더→shape
그래디언트 0)는 `HISTORY_KR.md` §7.5, `07_closed_open_methodology.md`.

스냅샷 나이: 초기(<12k) GT 사진 평균 16.34, 후기 20.13, 당시 eval 혼합 18.29.
모델은 “미숙한 파라미터를 복원하라”와 “그보다 좋은 최종 사진을 렌더하라”를 동시에 받음.

---

## 그 다음

8/17 이후는 HISTORY에 없다. N1/P1은 `04_n1_p1_success_line.md`,
16k는 `05_16k_runs.md`.
