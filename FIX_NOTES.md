# can3tok_fix — 재작성 노트 (2026-08-06, v2)

원본: `/data/daeho/aacd_proj/can3tok_encoder_decoer`  
제약: **`max_points=262144` 유지**, `z_compact=32×64×64` 유지.

## 진단 요약

| 원인 | 증거 | 대응 |
|---|---|---|
| A. 정보 예산 낭비: 실제 점을 prefix에만 팩 → 빈 그룹이 compact 24–70% 소비 | `empty_frac`≈0.42, PCA 한계 −12~−59% when redistributed | **`slot_redistribute` + kd partition** |
| B. `alive*G` count 가정 | 부분 채움 그룹에서 presence 붕괴 | **`budget_occupancy=1`, shape=11** |
| C. equivariance shape 불변 강제 | 오라클 rmse 0.009→0.013, lat_std floor에 고착 | **`equiv_shape_weight=0`** |
| D. dispersion_margin=0.01 | GT 그룹 반경 p50의 5.7배 → 상시 blur | **margin=0.001 + 0.35×GT std** |
| E. gen이 공유 slot 템플릿만 사용 | gen spread p50=0.48, nn spacing 0.4× | **`shape_xyz` 직결 경로** |
| F. (유지) set chamfer / coverage / density-aware | 이전 minimal fix | 유지 |

## 의도적으로 안 한 것

- `max_points` → 131k (가우시안 34% 손실 — 사용자 거부)
- soft-clamp / adaptive_shape / mid↑ / map_blocks↑ / residual_scale↑ (이전 붕괴 원인)
- 이웃 centroid 프레임 정규화 (측정 효과 ≈0)

## 기대

점 손실 0으로 PCA 한계 ~−34% (재분배+kd) + equivariance 여유 회수 → codec rmse 대략 0.0082→~0.0047 방향.
