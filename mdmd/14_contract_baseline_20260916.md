# 2026-09-16 계약 baseline 구현

작성: 2026-09-16  
근거: `12_code_audit_and_improvement_20260915.md`, `13_handrail_detail_and_architecture_20260915.md`  
학습은 **돌리지 않았다.** E1 진행 중 프로세스는 건드리지 않았다.

고정: 출력 262,144, 잠재 16,384, 배분 `4|1|3|8`.  
하지 않은 것: identity-unpack 덮어쓰기, `use_fixed_anchor_center` 기본값 변경, 장시간 학습.

## 무엇을 넣었는가

### A — 항상 켜짐 (평가·배포 계약)

- `EvalView(scene_id, image_id, camera, photo)`로 held-out 사진을 씬 태깅.
- `run_eval` / `eval_per_scene`는 `item['scene_key']`와 맞는 카메라만 렌더. 불일치는 예외.
- `decode_compact` = 학습 decode (`_decode_gaussians`: decompressor → codec → AttributeDecoder → nudge).
- `encode_compact(normalized=True)`는 decode 전에 `latent_scale`을 역변환.
- resume 시 `eval_fingerprint`(metric + split + contract)가 바뀌면 `best_score`만 초기화.
- eval 파일명에 scene prefix. 배포 PLY는 predicted presence를 쓸 수 있음.
- `apply_eval_schedule`: 도구가 refine=1을 강제하지 않음. `render_eval_figs`는 `enc_input`을 전달.

### B — 새 런 기본 1, 옛 `args.json`은 0

parser default=1, `getattr(..., 0)` / dataset 속성 default=0.

| 플래그 | 내용 |
|---|---|
| `shared_cell_owner` | 인코더 용량으로 한 번 소속. 타깃은 셀 prefix |
| `holdout_own_photo` | own-photo에도 held-out 제외. 카메라도 같이 교체 |
| `keep_extra_fullres` | extra 4장을 원해상도로 보관 |
| extra fallback 제거 | allowed가 비면 전체 pool로 돌아가지 않음 (항상) |
| `reuse_prev_vis` | 기본 0. 이전 batch visibility 재사용 금지 |
| edge cache | pointer key 캐시 삭제 (이미 반영) |
| 공간 샘플 | 정렬 구간을 K개로 나눠 뒤쪽 누락 제거 (이미 반영) |
| projection mask | 호출부가 GT mask를 pred_mask로 전달 (이미 반영) |

### C — 한 번에 켜되, 옛 체크포인트 재평가와 분리

같은 default 규칙. 레이아웃은 그대로.

| 플래그 | 내용 |
|---|---|
| `normalize_pooler_xyz` | pooler **앞**에서 `(xyz-cen)/extent` |
| `count_aware_template` | 예측 count의 live prefix로 템플릿 중심. 부분 셀 Hungarian도 같은 prefix |
| `attr_slot_mask` | AttributeDecoder PE / self-attn이 빈 슬롯을 무시. 마스크는 예측 presence/count (GT 아님). 그래서 `decode_compact`와 `forward`가 같다. |

`use_fixed_anchor_center`는 그대로 기본 0.

## 옛 런을 재평가할 때

S/T/E1 `args.json`에는 위 키가 없다. `build_config` / dataset getattr 기본 0이라 C·B 구조 플래그는 꺼진 채로 로드된다.  
씬 태깅 eval은 항상 켜지므로 **혼합 카메라 14.x dB와 같은 숫자가 아니다.** `best_score` fingerprint가 이를 기록한다.

E1 런처는 재시작해도 구조 플래그가 켜지지 않게 0으로 고정했다. 실행 중인 E1은 이 코드 변경을 보지 않는다.

## 다음 학습 (아직 안 함)

새 from-scratch 런은 parser 기본값으로 A+B+C가 켜진다.

```
bash scripts/launch_contract_baseline.sh --detached
```

한 변수씩 보려면 C 플래그를 하나씩 끄고 비교하면 된다. 16k / `4|1|3|8`은 유지.

## 테스트

`tests/test_audit_contracts.py`: 씬 필터, fingerprint, 샘플러 꼬리, projection mask, prefix 0-mean, 공유 소속, holdout fallback 없음, 옛 args 플래그 off, `decode_compact`↔`forward` + scale invert.
