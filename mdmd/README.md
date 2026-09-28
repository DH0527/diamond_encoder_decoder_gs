# mdmd — 실험 기록

작성: 2026-09-15  
위치: `can3tok_encoder_decoder_new_fix_7/mdmd/`

이 폴더는 **왜 그렇게 설계했는지, 결과가 무엇이었는지, 그다음을 무슨 근거로 바꿨는지**를
시간 순으로 남긴다. 분석 본문(`ANALYSIS_CODEC_WM_KR.md`)과 숫자가 겹치지만,
여기는 해석 에세이가 아니라 **결정의 연쇄**를 적는 곳이다.

**현재 코드가 무엇을 하고 어떻게 학습하는가**는 `09`–`11`이다. 루트 README는 131k 시절
설명이라 지금 16k 런과 다르다.

읽는 순서:

- 실험 역사: `01` → `08` → `02` → (`04`·`05`) → `06`·`07` → `03`
- 현재 구조·학습: `09` → `10` → `11`
- 코드·실데이터 감사 및 개선안: [`12_code_audit_and_improvement_20260915.md`](12_code_audit_and_improvement_20260915.md) — 기존 설명의 정정과 재현 자료 포함
- 난간·세부 구조 집중 분석: [`13_handrail_detail_and_architecture_20260915.md`](13_handrail_detail_and_architecture_20260915.md) — 실제 난간 crop, 입력 좌표 제거 실험, 위치·covariance 측정, 구조 개선 순서
- 계약 baseline 구현: [`14_contract_baseline_20260916.md`](14_contract_baseline_20260916.md) — A/B/C 코드. 학습은 아직 없음
- 계약 재측정: [`15_contract_verify_20260916.md`](15_contract_verify_20260916.md) — S/T/E1에서 씬별 PSNR·decode 일치. 그림은 `verify_20260916/figs/`
- 다음 실험 설계 재검토: [`16_experiment_design_review_20260916.md`](16_experiment_design_review_20260916.md) — shared-owner 타깃 손상 실측, 64k LR horizon, GPU 0·1·2의 B0/C1 비교
- 현재 vanilla/B1g 학습 감사: [`18_gradient_training_audit_20260925.md`](18_gradient_training_audit_20260925.md) — 34k 발산, 42k skip 정체, floater gradient 차단, cell/count 불일치와 encoder·decoder·loss 개선안. 9월 25일 실행 상태는 이 문서 기준

| 파일 | 내용 |
|---|---|
| `01_goal_and_constraints.md` | 최종 목표, 고정 제약, 숫자를 어떻게 읽는지 |
| `02_experiment_chronology.md` | 2026-08-01 ~ 09-15. 런 단위로 설계 → 결과 → 다음 근거 |
| `03_session_20260915_fix8.md` | 같은 날 encoder–decoder 개선 대화와 `fix_8`이 세 번 뒤집힌 기록 |
| `04_n1_p1_success_line.md` | 그림이 움직인 줄 R8I→M4→N1→P1. 한 변수씩 |
| `05_16k_runs.md` | L16k부터 T16k·E1까지 16k 런 표와 근거 |
| `06_visual_failure_and_wm.md` | 난간·floater·흐림, 월드모델이 요구하는 것 |
| `07_closed_open_methodology.md` | 닫힌 레버, 열린 레버, 세 번 나온 방법론 |
| `08_aug01_aug17_detail.md` | HISTORY §1–9를 설계→결과→다음으로 풀어 쓴 8/1–8/17 |
| `09_current_repo_and_layout.md` | 트리, 16k 산술, 켜진/꺼진 모듈, README가 틀린 점 |
| `10_current_data_and_forward.md` | npz→풀러→compact→folding→attr→렌더 |
| `11_current_training.md` | 스텝, 커리큘럼, 손실 가중, eval, S16k/T16k/E1 차이 |
| `12_code_audit_and_improvement_20260915.md` | 실제 NPZ·S/T 체크포인트 기반 오류 조사, 우선순위별 개선안, 검증 기준 |
| `13_handrail_detail_and_architecture_20260915.md` | 난간 소실의 직접 개입 실험: 약한 local xyz 전달, 둥글어진 splat, 빈 슬롯 영향과 구조 개선안 |
| `14_contract_baseline_20260916.md` | 감사 A/B/C 계약 구현. 16k·`4\|1\|3\|8` 유지. 학습 미실행 |
| `15_contract_verify_20260916.md` | S62k/T70k/E1 16k 재측정. decode MAE 0, 자기 씬 PSNR이 09-15 감사와 일치 |
| `16_experiment_design_review_20260916.md` | 제안된 B-only의 추가 검증과 수정 실험안. 현재 shared-owner prefix의 렌더 악화 및 B0/C1 런처 |
| `18_gradient_training_audit_20260925.md` | 현재 GPU 0·1·2의 B1g 진단. 과거 로그·동결 checkpoint·실제 loss gradient·데이터 계약 실측과 수정 실험 순서 |

숫자 출처는 각 절에 경로를 붙였다.

- **측정**: `metrics.json` / 로그 / 오라클 스크립트에 파일이 있음
- **재구성**: `HISTORY_KR.md`, `FIX*_NOTES.md` 등 프로젝트 자체 기록
- **대화 측정**: 이 대화에서 쟀으나 재현 스크립트가 트리에 없음. 게이트로 쓰지 말 것
- **미실행**: `fix_8` 학습은 한 번도 돌리지 않음. 이 문서를 쓰는 시점에 `can3tok_encoder_decoder_new_fix_8` **디렉터리 자체가 없음** (대화에서 복사·구현했다고 적었으나 디스크에 남지 않음)

관련 기존 문서 (이 폴더가 대체하지 않음):

- `HISTORY_KR.md` — 8/1–8/17 1차 기록
- `SESSION_LOG_KR.md` — 8/16–8/17 세션
- `ANALYSIS_CODEC_WM_KR.md` — 16k·월드모델 분석 (재검토 포함)
- `ANALYSIS_ADDENDUM_KR.md` — 원자료
- `FIX7_NOTES.md` — L16k 버그 4건
- `PLAN_10B_ANCHOR_KR.md` — centroid를 compact 밖으로 (사용자: 이 사본의 기본 아님)
- `STRUCTURE_KR.md`, `MODEL.md`, `FIX2_NOTES.md`, `FIX3_DESIGN_KR.md`
