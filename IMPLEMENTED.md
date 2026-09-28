# 구현 내역 전체

> **개정 2 (리뷰 반영).** 외부 리뷰 27개 항목 중 7개가 코드 변경으로 반영됨:
> 원본 인덱스 보존(§6), chamfer 감도 표현 정정(§7), frame 손실 표현 변경(§15),
> teacher decoder 동결(§17), 멀티뷰 렌더(§10), audit 도구에 신규 항 등록(§21),
> 게이트 이원화(§22·§23). 상세는 §14 개정 이력.

`can3tok_encoder_decoder_new` = `can3tok_encoder_decoer_fix_2` 전체 복사 + 아래 변경.
`runs/` 는 복사하지 않았습니다.

베이스 대비 변경된 파일: **11개 모듈, 2개 문서, 4개 신규 파일.**

```
can3tok/render.py              신규   래스터라이저 래퍼 + 측광 손실
can3tok/io_utils.py            +46    카메라 패킹 / 역패킹 / augmentation 변환
can3tok/data.py                +19    카메라를 배치에 실어 보냄, augment 시 카메라 동반 이동
can3tok/losses.py             +192    render_loss, intra chamfer 분해, gen 4개 항
can3tok/train.py               +82    렌더 블록, attribute 커리큘럼, 14개 인자, 로그
can3tok/schedule.py            +24    w_render 램프, attr_teacher_prob, gen 항 램프
can3tok/decoder.py             +11    scheduled sampling (attribute 조건화)
can3tok/config.py               +5    attr_teacher_prob, teacher_cycle
can3tok/eval_utils.py          +19    nn_unique 지표
can3tok/model.py               +22    fold basis 노출, pred_from_true_pack
can3tok/gen_decoder.py         +41    컨텍스트 입력 대칭화, basis 반환
can3tok/compressor.py           +8    teacher basis 를 ctx 로 발행

tools/render_compare.py        신규   체크포인트 → 4-way 렌더 비교
tools/diag_duplication.py      신규   중복이 어느 단계에서 생기는지
scripts/launch_coverage_262k_ddp.sh  신규
ASSESSMENT.md / PLAN.md / ARCHITECTURE.md  갱신
```

---

## 1. 래스터라이저 — `can3tok/render.py` (신규)

`/data/daeho/aabb/gaussian-splatting` 의 `utils/graphics_utils.py` 규약을 그대로 따릅니다.

### 1.1 카메라

npz 의 `state_t['camera']` 에 fx, fy, cx, cy, R(3×3), T(3,) 가 전부 있습니다.
`image_path` 는 `None` 이라 **원본 사진은 없습니다.**

```python
class Camera:                       # npz camera dict → 래스터라이저 입력
    w, h  = round(2*cx), round(2*cy)        # 977 x 544
    FoVx  = 2*atan(w / (2*fx))              # 80.2 deg
    world_view_transform = getWorld2View2(R, T).T     # Rt[:3,:3] = R.T
    full_proj_transform  = w2v @ projection(znear, zfar, FoVx, FoVy)
    camera_center        = w2v.inverse()[3,:3]
```

`downscale` 인자로 977×544 → 488×272 등으로 줄일 수 있습니다 (학습용).

### 1.2 색상 규약 — 반드시 필요했던 수정

```python
C0 = 0.28209479177387814
def sh_dc_to_rgb(dc):  return dc * C0 + 0.5
def rgb_to_sh_dc(rgb): return (rgb - 0.5) / C0
```

npz 의 `color` 필드는 **RGB 가 아니라 SH DC 계수**입니다. 원시 배열 범위가
−2.3 ~ 10.1 로 3DGS 체크포인트의 `features_dc` 범위입니다. 변환 없이 그대로
`colors_precomp` 로 넣으면 **평평한 회색 안개**로 렌더됩니다 (처음에 그렇게 나왔습니다).

반면 `opacity` 와 `scaling` 은 npz 에 **이미 활성화된 값**입니다 (sigmoid/exp 적용 후).
`gaussian_target_channels` 는 여기에 다시 logit / log 를 씌워 타깃을 만들므로,
렌더할 때는 되돌려야 합니다 — 그게 `gaussians_from_target`.

### 1.3 렌더 함수

```python
render_gaussians(xyz, scaling, rotation, opacity, colors, cam) -> (3,H,W)
```

모든 가우시안 인자에 대해 미분 가능. `sh_degree=0` + `colors_precomp` 로
뷰 의존 경로를 뺐습니다 — rest-band SH 는 이 데이터셋에서 ±0.9 vs DC 의 ±10 이고
초기 프레임에서는 정확히 0이라, 지금 단계에서 넣을 이유가 없습니다.

### 1.4 측정 함수

```python
ssim_value(a, b, window=11)   # 미분 가능. box window (avg_pool2d 4회)
psnr(a, b)                    # no_grad
ssim(a, b)                    # no_grad wrapper
photometric_loss(pred, ref, lam_dssim=0.2) -> (loss, l1, dssim)
                              # 3DGS 원 목적함수: (1-λ)·L1 + λ·(1-SSIM)
```

box window 를 쓴 이유: 학습 스텝 안에서 977×544 에 대해 돌아야 해서
가우시안 window 의 4번 separable conv 대신 avg_pool2d 1회씩으로 끝냅니다.

---

## 2. 카메라를 학습 루프까지 배달 — `io_utils.py` + `data.py`

### 2.1 패킹 (`io_utils.py`)

`load_npz_state` 가 카메라를 **16-벡터**로 평탄화해서 반환합니다.
dict-of-ragged-arrays 는 DataLoader 의 기본 collate 를 통과하지 못합니다.

```
[0:4]  fx, fy, cx, cy
[4:13] R (3x3)
[13:16] T
```

`camera_from_vector(v)` 가 역변환. 카메라가 없는 프레임은 0 벡터 → 렌더 손실이 건너뜁니다.
(실측: 데이터셋 3000개 중 샘플 82개 전부 카메라 보유, fx≤0 없음.)

### 2.2 augmentation 과 카메라 동반 이동 — `transform_camera_vector`

**이걸 안 하면 렌더 손실이 학습 샘플 100% 에서 틀립니다.**

`_augment` 는 정규화 공간에서 `x_n → s·R·x_n + shift`. 월드 공간으로 옮기면

```
x → s·R·x + t_w ,   t_w = center − s·R·center + shift·scale
```

이 similarity 를 카메라에 그대로 적용합니다. world-to-camera 를 `x_c = R_wc·x + T`
(= `getWorld2View2` 가 만드는 것, 저장된 R 은 `R_wc^T`) 라 쓰면

```
R_wc' = R_wc · Rᵀ
T'    = s·T − R_wc·Rᵀ·t_w
```

이면 `x_c' = s·(R_wc·x + T)` 가 됩니다. 카메라 공간 좌표가 **s배 균일 스케일**되는데,
투영은 스케일 불변이고 (`x/z` 불변), 가우시안 자체 scale 도 같은 s가 곱해졌으므로
투영된 크기도 그대로입니다. 즉 렌더 결과가 **완전히 동일**합니다.

**검증:** augment(xyz + scale + 쿼터니언 회전)한 클라우드를 변환된 카메라로 렌더한 결과가
원본 렌더와 **98.16 dB** 일치. (쿼터니언 회전을 빼면 28.45 dB — 즉 이 검증은 실제로
민감합니다.)

`_augment` 는 이제 `(xyz_new, target, (R, s, t_w))` 를 반환하고, `__getitem__` 이
`(R, s, t_w)` 로 카메라를 옮긴 뒤 `batch["camera"]` 로 실어 보냅니다.

---

## 3. 렌더 손실 — `losses.render_loss`

```python
render_loss(pred, target, mask, camera_vec, center, scale, layout,
            lam_dssim=0.2, downscale=2, max_points=0, min_coverage=0.25)
 -> {"render", "render_l1", "render_dssim", "render_used"}
```

목적함수에서 **task space 에 가장 가까운 항**입니다 (단일 뷰 + GT 가우시안 렌더를
reference 로 쓰므로 "proxy 가 아니다"라고까지 말하면 과합니다 — §14.6).

* 기준(reference)은 사진이 아니라 **타깃 가우시안의 렌더**입니다. npz 에 사진이 없기 때문.
  따라서 이 항이 요구하는 것은 픽셀 진리가 아니라 **뷰 등가성**입니다 — 이게 attribute
  예산이 닫히는 이유입니다. 저장하면 compact latent 의 8~112배가 필요하지만,
  group code 로부터 외형만 맞추는 건 추가 비용이 0입니다.
* 양쪽을 **같은 카메라, 같은 래스터라이저**로 렌더하므로 차이는 전적으로 가우시안 탓입니다.
* 타깃 브랜치는 `no_grad`, 그래디언트는 `pred` 로만 갑니다.
* `target_dim == 3` (xyz 단계)일 때는 **타깃의 attribute 를 양쪽에 똑같이** 넣고 detach 합니다.
  → 이미지 차이가 순수하게 기하 오차만 반영합니다.

### 3.1 crop 샘플 스킵 — `min_coverage`

랜덤 crop 은 공간 부분영역만 남기므로 카메라는 여전히 전체 장면을 프레이밍하는데
그 안이 거의 비어 있습니다. 양쪽 다 거의 검은 화면 → **L1 → 0, SSIM → 1** 이 되어
아무것도 배우지 않은 채 손실이 0으로 떨어집니다.

smoke 설정에서 실측: 6샘플 중 2개가 프레임의 11%, 26% 만 채움.

그래서 기준 렌더의 non-background 픽셀 비율이 `min_coverage` 미만이면 건너뛰고,
**건너뛴 비율을 반드시 로그에 찍습니다** (`rend 0.0857/0.237@0.65`).
조용히 1/3을 버리면 "렌더 손실이 수렴했다"로 읽히기 때문입니다.

### 3.2 학습 루프 배치 — `train.py`

`amp_ctx()` **바깥**에 둡니다. CUDA 래스터라이저는 fp32 전용이고, bf16 텐서를 주면
에러가 아니라 **쓰레기 값**으로 통과합니다. 그래서 명시적 `.float()` 캐스트 + 블록 분리.

```python
w_render = weights["w_render"]                       # schedule 이 램프
if w_render > 0 and flags["run_decode"] and step >= args.render_start:
    for br, key in (("", "pred"), ("gen_", "gen_pred")):
        ...
        r = render_loss(out[key].float(), target_full, mask,
                        batch["camera"], batch["center"], batch["scale"], layout, ...)
        loss = loss + wb * r["render"]
```

`target_full` 은 `target_dim==3` 슬라이스 **이전**의 59채널 텐서 — 렌더는
xyz 단계에서도 scale/rot/opacity/color 가 필요합니다.

`schedule.py` 에서 `render_start` 부터 `render_ramp_steps` 동안 0 → 1 로 램프.
시작 시점의 점군은 이미지 그래디언트가 대부분 노이즈일 만큼 거칠어서,
full weight 로 들어오면 안내가 아니라 파괴가 됩니다.

---

## 4. Attribute 디코딩 — geometry-first 조건화

디코더에는 이미 attribute 헤드가 있었습니다 (`attr_mlp` + scale/rot/opacity/color/SH
개별 헤드) 그리고 `attr_xyz` 훅도 배선되어 있었습니다. **그런데 아무도 그 훅을 몰지
않았습니다** — `--stage full` 을 켜면 헤드가 자기 자신의 틀린 위치 위에서 학습됩니다.

### 4.1 무엇을 연결했나

`p(X, A | Z) = p(X | Z) · p(A | X, Z)` 의 조건화 위치를 커리큘럼으로 넘깁니다:

```
teacher forcing  →  scheduled sampling  →  inference conditioning
   (p = 1)              (1 → 0)                 (p = 0)
```

`decoder._attributes` 안에서 **포인트 단위로** 섞습니다:

```python
p = cfg.attr_teacher_prob if self.training else 0.0
if attr_xyz is not None and p > 0.0:
    xyz_cond = attr_xyz[...]
    if p < 1.0:
        keep = torch.rand(b, chunk, gsz, 1) < p
        xyz_cond = torch.where(keep, xyz_cond, xyz)   # xyz = 디코더 자신의 예측
else:
    xyz_cond = xyz
```

* **forward 1회**로 끝납니다 (두 번 디코드하지 않음).
* 배치 단위가 아니라 포인트 단위로 섞으므로, 매 스텝에 두 regime 이 모두 들어갑니다 —
  한 쪽만 연속으로 보고 거기에 overfit 하는 일이 없습니다.
* **eval 은 항상 `p = 0`** (`self.training` 체크). 보고되는 숫자가 teacher-forced 인 경우는
  없습니다.
* 하드 스위치가 아니라 anneal 인 이유: 헤드 입력 분포가 한 스텝에 point spacing 이상
  움직입니다.

`schedule.py`:
```
attr_teacher_prob = ramp(step, attr_start + attr_force_steps, attr_anneal_steps, 1.0, 0.0)
```

---

## 5. `nn_unique` — 새 평가 지표

`eval_utils.group_error_breakdown` 에 추가. 매 eval 마다 기록됩니다.

```python
def _uniq(pred_offsets, gt_offsets):      # 그룹 단위
    nn  = cdist(p, g).argmin(dim=2)       # 각 예측점의 최근접 GT
    hit = zeros_like(nn, bool).scatter_(1, nn, True)
    return hit.sum() / (num_groups * group_size)
```

**여기 있는 다른 지표들은 중복에 대한 감도가 너무 약합니다.** 같은 GT점 위에 예측점 두 개가
앉으면 p→g 는 만점이고, symmetric chamfer 의 g→p 절반은 비어버린 GT점 하나만
그룹 전체에 희석해서 물립니다.

---

## 6. Milestone C (gen 브랜치) — 앞선 대화에서 만든 것, 이번에 완성

### 6.1 학생이 교사와 같은 것을 읽게 함 (`gen_decoder.py`)

```python
# 이전: fold_head = MLP([shape_in, 64, fold_frame]),  frame = fold_head(shp)
# 지금: fold_head = MLP([shape_in + d_ctx, 128, fold_frame])
        hin   = torch.cat([shp, gt], dim=-1)     # gt = contextualised group token
        frame = fold_head(hin)
        res   = shape_xyz(hin)
```

codec 의 folding 헤드는 `cat([shape, contextualised group token])` 을 읽는데
gen 은 raw 28채널만 읽고 있었습니다 — 두 디코드 경로 사이에 남아 있던 마지막 비대칭.

### 6.2 basis distillation

두 디코더 모두 `R(a)·diag(s)·(fixed_template + residual)` 로 그룹을 만듭니다.
지금까지는 **최종 점만** distill 했기 때문에(`w_distill`), 학생이 다른 basis 로
비슷한 점집합에 도달할 자유가 있었고, 실제로 찾아낸 basis 가 팽창합니다.

이제 교사의 `frame`(6 파라미터)과 `local`(pre-scale 오프셋, 그 크기가 곧 그룹 반지름)을
양쪽 경로에서 노출시켜 직접 distill 합니다:

```
w_gen_basis * ( |gen_frame - sg(teacher_frame)| + |gen_local - sg(teacher_local)| )
```

* `compressor.py`: `self._last_basis = {"frame": frame, "local": local}` → `ctx` 로 발행
* `model.py`: `teacher_fold_frame/local`, `gen_fold_frame/local` 를 out 에 실음
* `gen_decoder.py`: `_decode_range` 가 4개 텐서 반환, forward 가 `bfs`/`bls` 수집

### 6.3 anti-inflation 항

`intra_group_chamfer(..., return_parts=True)` 가 분해값을 반환하도록 바꿨습니다:

```python
{"p2g": 정밀도, "g2p": 커버리지, "radius": |r_pred/r_gt - 1|}
```

symmetric chamfer 하나만으로는 gen 이 radius 1.83×GT, p→g 1.294 / g→p 0.521 에
도달할 수 있었고, **대칭 항 하나로는 두 방향을 분리해서 볼 수 없습니다.**

* `--w_gen_p2g` — 정밀도만 따로
* `--w_gen_radius` — 반지름 비 페널티

### 6.4 teacher cycle

`--w_teacher_cycle`: 같은 디코더를 **참 pack** 위에서 한 번 더 돌려
(`no_grad`, `cfg.teacher_cycle` 로 게이트) `D(D_T(H_hat), D_T(H))` 를 만듭니다.
z_compact 가 "pack 이 담고 있는 것"이 아니라 "**디코더가 필요로 하는 것**"을 보존하는지
확인하는 항입니다. `model.py` 의 `pred_from_true_pack`.

---

## 7. 새 CLI 인자 전체

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--w_render` | 0.0 | 측광 손실 가중치. 0이면 래스터라이저를 아예 부르지 않음 |
| `--render_start` | 6000 | 이 전에는 점군이 너무 거칠어 이미지 그래디언트가 도움이 안 됨 |
| `--render_ramp_steps` | 1000 | 0 → w_render 램프 |
| `--render_lam_dssim` | 0.2 | 3DGS 자신의 L1 / D-SSIM 배합 |
| `--render_downscale` | 2 | 977×544 → 488×272. 래스터 시간과 backward 메모리 절반 |
| `--render_max_points` | 0 | 0 = 전체. 캡을 주면 pred/target 을 **동일하게** 서브샘플 |
| `--render_min_coverage` | 0.25 | 기준 렌더가 프레임의 이 비율 미만이면 스킵 (crop 대응) |
| `--render_gen_scale` | 1.0 | gen 브랜치의 w_render 배수 |
| `--attr_force_steps` | 2000 | 순수 teacher forcing 구간 |
| `--attr_anneal_steps` | 4000 | scheduled sampling 구간 |
| `--w_gen_p2g` | 0.0 | gen 정밀도(p→g) 단독 항 |
| `--w_gen_radius` | 0.0 | gen 그룹 반지름 비 페널티 |
| `--w_gen_basis` | 0.0 | 교사 folding basis distillation |
| `--w_teacher_cycle` | 0.0 | 참 pack 디코드와의 일치 |

로그 라인에 추가된 필드:
```
rend <L1>/<DSSIM>@<사용비율>     w_render > 0 일 때
tf <teacher_prob>                with_attrs 일 때
```

---

## 8. 새 도구

### `tools/render_compare.py`

체크포인트를 로드해 held-out 장면을 **4가지로 렌더**합니다. 실패를 단계에 귀속시키기 위함:

| | 내용 |
|---|---|
| A `orig` | npz 의 모든 가우시안 |
| B `canon` | 샘플러가 고른 ≤max_points 만, GT attribute |
| C `codec`/`gen` | 예측 xyz + 최근접 GT 의 attribute |
| D `*_snap` | 각 예측점을 최근접 GT점으로 **치환** — 위치 오차를 전부 제거한 천장 |

B vs A = 선택/절단 비용. C vs B = autoencoder 기하 단독 비용 (attribute 를 공짜로
주므로 **하한**). D 는 "점을 어디에 두느냐만 고쳐서 도달 가능한 상한".

출력: PSNR / SSIM / `nn_unique` / side-by-side PNG / `render_metrics.json`.

`canon_nn_unique` 는 **정합성 검사**입니다 — 선택된 점들은 GT 의 부분집합이므로
NN 매핑이 단사여야 하고, 1.0 이 아니면 전송 자체가 손실적이라 예측 쪽 숫자를
비교할 수 없습니다. 실측 0.991.

### `tools/diag_duplication.py`

중복이 **어느 단계**에서 생기는지: coarse(refine off) / full(refine on) / gen /
target(대조군) 각각의 `nn_unique` 와 `intra_chamfer`.

---

## 9. 측정 결과

### 9.1 렌더 — 최고 체크포인트 (`intra_chamfer` 0.249), held-out 7 프레임

| | PSNR | SSIM |
|---|---|---|
| 원본 → canonical (262k 선택) | **30.3** | 0.961 |
| 원본 → 복원 | **17.2** | 0.518 |
| 원본 → 복원, 최근접 GT 로 snap | **19.2** | |

복원 이미지는 **알아볼 수 있습니다.** 장면 배치, 구조, 색이 살아 있고,
무너지는 것은 미세 디테일과 어둡고 텍스처 없는 영역입니다.

장면 크기가 클수록 canonical 손실이 큽니다 — 413k 점 장면은 262k 절단으로 28.2 dB.
262144 라는 예산 자체가 이미 실질 비용을 치르고 있다는 뜻이고, Milestone D 의
K 결정에 들어가는 숫자입니다.

### 9.2 중복 — 이번에 나온 가장 중요한 숫자

모든 예측점을 최근접 GT 로 snap 하면 위치 오차가 **전부** 사라지는데도 2 dB 만 오릅니다.

```
서로 다른 GT점 / 예측점 수 = 0.540   모델
                             0.991   선택된 GT 부분집합 (대조군)
```

**262k 예산의 46% 가 이미 다른 예측점이 차지한 자리에 떨어집니다.**
symmetric chamfer 는 중복에 대한 감도가 너무 약하므로 (중복점 자신은 p→g 만점,
비어버린 GT점은 g→p 에서 한 번만, 그것도 그룹 전체에 평균되어 물림) 1.23× 간격과
17.9 dB 가 공존합니다.

### 9.3 캘리브레이션 — GT 에 등방성 노이즈

| σ / 반지름 | intra_chamfer | nn_unique |
|---|---|---|
| 0.00 | 0.000 | 0.999 |
| 0.05 | 0.070 | 0.828 |
| 0.10 | 0.123 | 0.728 |
| 0.20 | 0.208 | 0.636 |
| 0.30 | 0.280 | 0.587 |
| 0.45 | 0.383 | 0.538 |
| **모델** | **0.246** | **0.475** |

모델의 오차는 **자기보다 2배 큰 랜덤 노이즈보다도 더 뭉쳐 있습니다.**
점은 올바른 표면 위에 있는데(chamfer 는 대조군보다 훨씬 좋음) 배치가 틀렸습니다.
`template_erank_pred` 7.2 (GT 22.5) 이 반대편에서 같은 얘기를 합니다 — 서로 다른
그룹 shape 이 ~7개.

### 9.4 어느 단계에서 생기나

`tools/diag_duplication.py`, 3 장면 평균:

| 단계 | nn_unique | intra_chamfer |
|---|---|---|
| coarse (refine off) | 0.499 | 0.253 |
| full (refine on) | 0.499 | 0.253 |
| target (대조군) | 0.999 | 0.000 |

**refine 스택 탓이 아닙니다.** coarse 출력에 이미 있습니다.
(주: 해당 체크포인트는 `a_rf 0.00` 구간이라 refine 헤드가 아직 zero-init 이므로
두 행이 같은 것은 이 부분에서는 정보가 없습니다. coarse 자체가 0.499 라는 것이 결론.)

### 9.5 원인

디코더 출력에 **1:1 supervision 이 하나도 없습니다.**
`w_xyz`, `w_xyz_mse`, `w_xyz_hard`, `w_xyz_residual` 전부 0.
남은 것은 `w_chamfer`(20) 와 `w_intra_chamfer`(14) 뿐이고 **둘 다 순열 불변**이라
구조상 중복을 볼 수 없습니다.

`w_z_residual`(40)은 1:1 이지만 **packed 중간표현**에만 걸립니다. unpack 이후는
전부 자유롭고, `rel_offset_p50` = **0.499** (그룹 반지름의 절반).

45개 eval 에 걸친 상관:

| | vs intra_chamfer |
|---|---|
| `rel_offset_p50` | r = **+0.729** |
| `template_erank_pred` | r = **−0.606** |
| `aniso_lam2_pred` | r = +0.424 |
| `thread_pred/gt` | r = +0.260 |

둘 다 같은 방향을 가리킵니다.

### 9.6 렌더 손실 감도

xyz 를 point spacing 배수로 흔들었을 때 (등방 노이즈, 구조적 오차보다 비관적):

| 흔든 양 | PSNR | SSIM |
|---|---|---|
| 0.25× | 19.14 | 0.571 |
| 1.00× | 15.30 | 0.304 |
| 2.00× | 14.09 | 0.245 |

이미지는 점 위치에 매우 민감합니다 — 렌더 손실이 강한 신호라는 뜻이자,
점이 표면 위에 **정합적으로** 놓이지 않으면 chamfer ~1× 만으로는 높은 PSNR 이
나오지 않는다는 뜻입니다.

---

## 10. 검증한 것

| | 방법 | 결과 |
|---|---|---|
| 래스터라이저 정확성 | 실제 장면 렌더 | 선명한 정상 이미지 |
| 색상 규약 | SH DC 변환 전/후 비교 | 변환 없으면 회색 안개 |
| 카메라 augmentation | augment 클라우드 + 변환 카메라 vs 원본 | **98.16 dB** |
| 렌더 손실 그래디언트 | `pred.grad` | norm 0.505, 40000점 중 25541 (나머지는 절두체 밖 — 정상) |
| crop 스킵 | `render_used` 카운터 | smoke 에서 6샘플 중 2개가 11%/26% 커버 |
| end-to-end | `--stage full` + 커리큘럼 + 렌더, 12스텝 | 통과. render L1 0.334→0.086, tf 1.00→0.00 |
| 전체 컴파일 | `py_compile can3tok/*.py tools/*.py` | OK |
| `nn_unique` 지표 | GT 부분집합 대조군 | 0.991 (단사여야 하는 값) |

## 11. 검증하지 **않은** 것

* `--stage full` 을 **실규모(262k)** 로 돌린 적 없음. smoke(8192점)만 통과.
* 렌더 손실을 실규모 학습에서 돌린 적 없음. 비용 추정치(샘플당 ~0.35 s)는 추정입니다.
* `--w_xyz_residual` 을 codec 에 켠 효과 — 다음 런의 가설입니다.
* LPIPS 없음 (SSIM 은 box window 라 논문용 숫자가 아니라 순위용).
* npz 는 프레임당 카메라가 **1개**뿐이라 멀티뷰 평가 불가.

---

## 12. 다음 런

`scripts/launch_coverage_262k_ddp.sh`

두 개의 레버, 둘 다 `nn_unique` 를 겨냥합니다:

1. `--w_render 6`, `--render_start 4000` — proxy 가 아닌 유일한 항이자 중복을
   직접 보는 유일한 항 (겹친 점은 한 곳의 opacity 를 낭비하고 다른 곳에 구멍을 남기며,
   둘 다 이미지 오차입니다).
2. `--w_xyz_residual 12` — 디코더 **출력**에 대한 extent-normalised 1:1 항.

### 사전 확약 게이트

| step | 요구 |
|---|---|
| 4000 | codec `nn_unique` > 0.60 (0.499 → ; 이 chamfer 에서의 노이즈 대조군이 0.61) |
| 8000 | codec `nn_unique` > 0.70 **그리고** `intra_chamfer` < 0.25 **그리고** 렌더 PSNR > 20 dB |

렌더 손실은 떨어지는데 `nn_unique` 가 안 움직이면, 렌더 항이 기하가 아니라
**opacity/scale 보정**으로 만족되고 있다는 뜻이고, 다음 테스트는 attribute 를
GT 로 고정한 렌더 손실입니다.

### 현재 진행 중

`runs/mBC_262k_g64_20260808_100309` — Milestone C (gen 수정) 검증. step ~2900/20000,
게이트는 6500 에서 gen `ich` < 0.45, gen radius/GT < 1.2. 이게 끝나야 다음 런을
겹치지 않게 시작합니다.

---

## 13. 순서에 대한 반성

제안서도 저도 렌더링을 뒤에 뒀습니다 (제안서 76개 중 43번, 제 계획에서는 G).
**처음에 했어야 했습니다.** 파일 하나였고, 재학습이 필요 없었고, 문제를
"디테일 정확도"에서 "커버리지"로 재분류했습니다.

제안서의 #36 — 평가 지표와 학습 지표를 분리하라 — 를 둘 다 충분히 멀리 가져가지
못한 것이 원인입니다. 평가 지표는 **시스템이 존재하는 그 공간**에 있어야 합니다.

---

## 14. 개정 이력 — 외부 리뷰 27개 항목 반영

리뷰의 7개 항목이 코드 변경, 5개가 문서/계획 변경으로 들어갔습니다.
나머지는 이미 반영되어 있었거나(§12·§14·§16·§24), 다음 단계에 배치했습니다(§11·§13·§19·§20·§25).

### 14.1 코드 변경

**§6 — `canon_nn_unique = 0.991` 을 그냥 넘기지 말라.** 원인을 측정했습니다:
GT 자체에 **정확히 겹치는 가우시안이 0.16–0.17%** 있고(NN 거리 = 0), 추가로
float32 정규화 왕복 오차(3e-5)보다 가까운 쌍들이 있습니다. 버그는 아니지만
지적이 맞습니다 — training target association 을 NN 으로 복구할 이유가 없습니다.

`data.py` 가 이제 `source_index` (max_points, int64, 빈 슬롯 −1)를 반환합니다.
각 슬롯이 npz 의 몇 번째 행에서 왔는지 정확히 기록합니다.
`render_compare.py` 는 NN 조회를 버리고 이 인덱스를 씁니다:

```
canon_src_unique  0.991 → 1.000
canon_xyz_err            0.000     (인덱스 매핑 검증용, augment off 이므로 0이어야 함)
```

이제 canonical 은 **증명 가능하게** GT 부분집합이고, 27.8 dB 는 전부 절단 손실이며
조회 artefact 성분이 0입니다. Milestone D 의 canonicalizer 가 이겨야 할 baseline 이
확정되었습니다.

**§7 — chamfer 표현 정정.** "symmetric chamfer 는 중복을 보지 못한다"는 틀렸습니다.
g→p 항이 비어버린 GT점을 실제로 물립니다. 정확한 진술은:

> 중복점 자신은 p→g 만점, 비어버린 GT점은 g→p 에서 **한 번만**, 그것도 64 슬롯 전체에
> 평균되어 물림 → **many-to-one correspondence 에 대한 감도가 너무 약하다.**

`eval_utils.py`, `ASSESSMENT.md`, `IMPLEMENTED.md`, `diag_duplication.py`,
launch script 전부 수정했습니다.

**§15 — frame 손실을 표현 공간에 맞게 분리.** raw 6채널 L1 은 양쪽 모두 틀렸습니다.
`losses.frame_distance()` 신규:

```python
# scale: softplus + 기하평균 정규화를 디코더와 동일하게 적용한 뒤 log 공간에서 L1
#        → 손실이 scale *비율* 에 비례하고 전체 크기에 불변
d_scale = (a_p.log() - a_t.log()).abs().mean(-1)
# rotation: axis-angle → 행렬 → Frobenius
d_rot   = (R_p - R_t).flatten(-2).norm(dim=-1)
```

axis-angle 에 L1 을 주면 `v` 와 `v(1 − 2π/|v|)` 가 **같은 회전인데** 멀리 떨어지고,
`|v| = π` 에서 불연속입니다 — 교사의 회전을 이미 갖고 있는 학생이 벌을 받습니다.
geodesic angle(`arccos`) 대신 Frobenius 를 쓴 이유: geodesic 에 단조이고, 유계이며,
0에서 gradient 가 유한합니다(`arccos` 는 발산).

`logs["gen_basis_frame"]` 로 별도 로깅.

**§17 — teacher decoder 동결.** 지적이 맞았습니다. `pred_from_true_pack` 은
`no_grad` 였지만 손실이 `out["pred"]` 로 역전파되므로, 디코더가 "입력에 둔감해지는"
방향으로 손실을 줄일 수 있었습니다.

`model.py` 가 이제 **양쪽 모두** 디코더 파라미터를 detach 한 상태로 디코드합니다
(`torch.func.functional_call`):

```python
frozen = ({k: v.detach() for k,v in decoder.named_parameters()},
          {k: v.detach() for k,v in decoder.named_buffers()})
out["pred_cycle"]          = functional_call(decoder, frozen, (z_raw_hat, ...))  # grad → 입력
out["pred_from_true_pack"] = functional_call(decoder, frozen, (z_raw, ...))      # no_grad
```

**검증** (cycle 손실만으로 backward):

```
decoder      grad sum = 0.000000e+00   <- 0이어야 함
compressor   grad sum = 4.288499e-03   <- >0 이어야 함
```

`--no_teacher_cycle_freeze` 로 A/B 가능(기본 동결).
**비용:** grad-carrying 디코드가 하나 늘어납니다. 262k 에서 이것 때문에 audit 이
OOM 났으므로(학습이 4 GPU 점유 중이었음) 실규모 메모리는 다음 런에서 확인해야 합니다.

**§10 — 멀티뷰 렌더.** 카메라 하나로는 3D 등가성을 증명할 수 없다는 지적이 정확합니다.
카메라가 못 보는 것은 제약이 없고, scale/opacity 가 기하 오차를 일부 흡수할 수 있습니다.

사진은 없지만 **렌더는 임의 포즈에서 만들 수 있으므로**:

```
L = (1/M) · Σ_m  D( R(pred, C_m), R(target, C_m) )
```

`render.perturb_camera_vector()` 가 기존 카메라를 자기 축 기준 yaw/pitch 로 궤도
회전 + 소량 평행이동시킵니다(같은 내용이 프레임에 남도록). `--render_views M`,
`--render_view_jitter_deg`. `views=1` 이 기본이고, 추가 뷰마다 reference 렌더 1회 +
gradient 렌더 1회가 듭니다. 커버리지 미달 뷰는 개별로 스킵합니다.

이건 canonicalizer 에서 더 중요합니다 — 뒤에 가려진 가우시안을 지우는 게이트는
그 한 뷰에서 비용이 0이고 다른 뷰에서 구멍으로 나타납니다. `PLAN.md` Milestone D 에
"canonicalizer 는 더 큰 M 과 더 넓은 궤도를 써야 한다"로 기록했습니다.

**§21 — audit 도구에 신규 항 등록.** `TERMS` 는 **두 번** 쓰입니다: 측정할 항을 고를 때,
그리고 **나머지 전부를 0으로 만들 때**. 목록에 없는 항은 절대 0이 되지 않고 모든 행을
오염시킵니다 — 실제로 `w_latent_decorr` 가 빠져서 4개 항이 전부 동일 크기, cosine 1.00
으로 나온 적이 있습니다. `w_gen_basis`, `w_gen_p2g`, `w_gen_radius`, `w_teacher_cycle`
추가. 그리고 `w_render` 는 `total_loss` 항이 아니라(학습 루프에서 autocast 밖에서
더해짐) 별도 측정 경로를 넣었습니다.

### 14.2 게이트 이원화 (§22, §23)

같은 숫자로 두 가지 질문에 답하고 있었습니다. 분리했습니다:

| 질문 | 기준 |
|---|---|
| **A. 이 레버가 작동하는가?** (이번 런) | `nn_unique` > 0.60 @4000, > 0.70 @8000, `intra_chamfer` < 0.25, 렌더 PSNR > 20 dB |
| **B. attribute 를 붙일 만큼 기하가 좋은가?** (`--stage full` 진입) | **`nn_unique` > 0.85** |

0.70 이면 출력의 30% 가 여전히 다른 예측점과 최근접 GT 를 공유합니다. attribute 는
슬롯 단위이므로, 그 위에서 학습하면 attribute 디코더가 **존재하지 않는 correspondence**
를 배웁니다. 0.70 은 이번 런을 통과시키지만 다음 런을 통과시키지 않습니다.

gen 게이트도 갱신: 기존 `ich < 0.45` + `radius < 1.2` 는 중복을 전혀 보지 못하므로
**`gen_nn_unique` > 0.65** 를 추가했습니다. `gen_nn_unique` 는 이미 매 eval 기록됩니다.

### 14.3 다음 후보를 repulsion 이 아니라 OT 로 명시 (§8)

`w_xyz_residual` 로도 `nn_unique` 가 안 오르면 다음은 **local one-to-one assignment**
입니다 — 그룹별 64×64 비용행렬 `C_ij = |x_i − x_j|²` 에 Sinkhorn/Hungarian,
`min Σ P_ij C_ij`. 262k 전역 매칭은 불가능하지만 64×64 는 현실적입니다.

**repulsion 은 명시적으로 배제**했습니다: GT 밀도와 무관하게 점을 밀어내므로
복원 대상인 분포 자체를 왜곡합니다. `PLAN.md` 와 launch script 양쪽에 기록.

### 14.4 순서 재배치 (§19, §25)

Milestone 문자는 더 이상 순서가 아닙니다. `PLAN.md` 에 측정된 순서를 기록:

```
1. Milestone C (진행 중)  — 단, gen_nn_unique 를 같이 읽음
2. Geometry coverage      — 목표는 chamfer 가 아니라 nn_unique
3. Render-aware canonicalizer  — --stage full 보다 먼저
4. Full Gaussian attributes @262k  — 진입 조건 nn_unique > 0.85
5. full-SH renderer → DIAMOND latent
```

canonicalizer 를 attribute 앞에 놓는 이유(§19): 현재 `canon` 은 이미 25–30 dB 를
잃은 sampler baseline 이고, 그 위에서 attribute 디코더를 학습하면 **손상된 표현이
ground truth 가 됩니다.**

### 14.5 다음 단계로 넘긴 항목 — `PLAN.md` Milestone E/F 에 기록

* **§11 SH curriculum.** 현재 `sh_degree=0` 은 기하 진단에는 맞고 최종 모델에는
  틀립니다 — degree-0 렌더 손실은 뷰 의존 외형을 검사할 수 없습니다.
  geometry(deg 0) → appearance(DC + low SH) → full SH.
* **§13 scheduled sampling 입도.** 지금 attribute 헤드가 per-point MLP 라서
  point-wise mixing 이 맞습니다. 그룹 내 attention 을 쓰게 되면 GT/predicted 가
  임의로 섞인 그룹은 inference 에 존재하지 않는 분포이므로, group-wise / sample-wise
  를 A/B 해야 합니다.
* **§24 attribute compensation 대조군.** predicted attribute 렌더 vs GT-frozen
  attribute 렌더를 둘 다 돌려 기하 개선과 attribute 보상을 분리합니다.
  xyz stage 에서는 이미 차단되어 있으므로 `--stage full` 부터 필요합니다.
* **§16 `w_gen_p2g` / `w_gen_radius` 는 failure-specific regularizer.**
  학생이 안정된 뒤 ablate 해야 합니다. 그대로 두면 자기 지표에 hand-tune 한 것으로
  보입니다.

### 14.6 반영하지 않은 것과 이유

**§9 — 렌더 손실이 geometry 손실을 대체하면 안 된다.** 동의하며, 대체하지 않습니다.
`launch_coverage_262k_ddp.sh` 는 `w_chamfer 20` + `w_intra_chamfer 14` +
`w_xyz_residual 12` + `w_render 6` 을 **함께** 씁니다. 다만 IMPLEMENTED.md §3 의
"proxy 가 아닌 유일한 항"이라는 표현은 과했습니다 — 단일 뷰이고 reference 가 GT
가우시안 렌더이므로, 정확히는 **task space 에 가장 가까운 항**입니다.

---

## 15. §21 gradient audit 결과 — `w_render 6` 은 1000배 틀렸습니다

리뷰의 §21("raw weight 만 보고 balance 를 결정하지 말고 gradient magnitude 를 다시
측정해라")이 **정확했습니다.** 제 추측값 `w_render 6` 은 자릿수가 세 개 틀렸습니다.

### 15.1 측정 방법을 바꿔야 했다

`tools/audit_losses.py` 는 파라미터 공간에서 측정하는데, 학습이 4 GPU 를 점유한 상태로는
262k 에서 OOM 났습니다 (teacher-cycle 동결이 grad-carrying 디코드를 하나 늘린 것도 일부
기여). 축소 규모로 내려도 문제가 남습니다: **래스터라이저의 포인트당 그래디언트는 픽셀당
가우시안 수에 의존하므로 점 개수를 줄인 측정은 전이되지 않습니다.**

그래서 `tools/audit_render_scale.py` 를 새로 만들었습니다 — **예측점에 대한**
`|dL/dxyz|` 를 262k 실규모에서, 모델 없이 측정합니다. 래스터라이저와 chamfer 만 필요하니
학습 옆에서 돌아가고, 스케일이 정확합니다.

### 15.2 결과

`step_015830.npz`, n=262144, 그룹 반지름의 0.5배 등방 노이즈, 나머지 항 전부 0:

| term | weight | \|w·dL/dxyz\| | per-point | share |
|---|---|---|---|---|
| **w_render** | 6.00 | **1.2340e+02** | 2.833e-02 | **99.9%** |
| w_chamfer | 20.00 | 1.1823e-01 | 6.480e-05 | 0.1% |
| w_intra_chamfer | 14.00 | 3.5151e-02 | 4.688e-05 | 0.0% |
| w_xyz_residual | 12.00 | 1.3532e-02 | 2.643e-05 | 0.0% |

`w_render 6` 이면 스텝의 **99.9%** 를 렌더 항이 차지합니다. 기하 항 셋은 사실상 꺼진
것과 같고, 이 런은 "chamfer + slot + render 를 함께 쓴다"가 아니라 "render 만 쓴다"가
됩니다 — 리뷰 §9 가 하지 말라고 한 바로 그것입니다.

### 15.3 아티팩트가 아님

| 조건 | \|render\| / \|chamfer\| |
|---|---|
| scene 160, σ = 0.5 | 1044× |
| scene 160, σ = 0.1 | 999× |
| scene 240, σ = 0.5 | 505× |
| scene 80, σ = 0.5 | 1027× |

**오차 크기에 거의 무관**(σ 0.1 ↔ 0.5 에서 999 ↔ 1044)하고 장면이 바뀌어도 유지됩니다.
즉 오차 구조의 성질이 아니라 **래스터라이저의 스케일 성질**입니다. 이미지는 점 위치에
극도로 민감하다는 앞선 측정(0.25× spacing → 19 dB)과 일치합니다.

### 15.4 새 가중치

결합 그래디언트에서 렌더가 ~30% 를 차지하도록 풀면:

```
point-space 합계 = 0.1182 + 0.0352 + 0.0135 = 0.1669
w_render = 6 × (0.1669 × 0.3/0.7) / 123.4 ≈ 0.0035
```

`W_RENDER` 기본값을 **6 → 0.004** 로 내렸습니다. `--w_render` 의 help 에도
"point-space 가중치와 비교 가능한 값이 아니다, 유용한 범위는 ~1e-3" 를 명시했습니다.

### 15.5 이 항목이 알려주는 것

`w_render` 뿐 아니라 **단위가 다른 손실을 추가할 때마다** 이 측정을 해야 합니다.
비교 가능한 척도가 없는 두 항의 가중치를 유추로 정하면 자릿수를 놓칩니다.
`tools/audit_render_scale.py` 를 재사용 가능하게 만들어 뒀습니다
(`--w_render`, `--w_xyz_residual` 오버라이드, `--views`, `--sigma`, `--scene`).

---

## 16. 렌더 손실을 geometry 단계에서 빼기로 결정 — 측정 근거

질문: *"렌더링 로스는 geometry를 학습하는 부분에서는 사용하지 않는게 좋지 않을까?"*
**측정 결과 맞습니다.** `w_render` 를 geometry 런에서 **0으로 내렸습니다.**

§15 는 *크기* 만 봤습니다 (래스터라이저 그래디언트가 chamfer 의 ~1000배).
크기가 크다고 유용한 건 아닙니다 — 정답 방향을 가리키지 않으면 크기는 해롭습니다.

### 16.1 측정 방법 — `tools/audit_render_direction.py`

여기서는 **정답을 정확히 알고 있습니다.** 클라우드가 타깃 + 알려진 노이즈이므로
`target − pred` 가 이상적인 하강 방향입니다. 그래서 각 손실에 대해

```
cos( −dL/dxyz , target − pred )
```

를 포인트 단위로, 오차 크기의 함수로 측정합니다. 그래디언트를 **받은** 점만
집계하고, 받은 비율도 같이 보고합니다.

### 16.2 결과 (262k, `step_015830`, 488×272)

```
projected Gaussian radius: median 1.40 px
one group radius on screen: ~12.0 px  (0.5-radius 오차 = ~6.0 px)
```

| 오차 | render (1 view) | render (4 views) | chamfer | intra_chamfer | xyz_residual |
|---|---|---|---|---|---|
| 0.6 px | **0.134** @24% | 0.155 @35% | 0.394 @25% | 0.682 @100% | **0.866** @100% |
| 1.2 px | 0.135 @24% | — | 0.424 @25% | 0.692 @100% | 0.866 @100% |
| 2.4 px | 0.114 @23% | 0.129 @34% | 0.449 @25% | 0.670 @100% | 0.866 @100% |
| 4.2 px | 0.078 @23% | — | 0.434 @25% | 0.655 @100% | 0.866 @100% |
| 6.0 px | **0.054** @23% | 0.064 @33% | 0.405 @25% | 0.662 @100% | 0.866 @100% |
| 9.0 px | 0.036 @22% | — | 0.380 @25% | 0.691 @100% | 0.866 @100% |

### 16.3 세 가지 문제가 한 열에 다 있습니다

**(1) 방향이 거의 직교합니다.** 서브픽셀 오차에서도 cos 0.134 — 정답 방향과 **약 82°**
어긋납니다. 그리고 오차가 커지면 **더 나빠집니다** (0.134 → 0.036).
다른 항들은 오차 크기에 무관하게 평평합니다.

**(2) 점의 24% 만 그래디언트를 받습니다.** 절두체 밖 + 가려짐. 262k 중 4분의 3이
이 항으로부터 아무 감독도 받지 않습니다. `xyz_residual`/`intra_chamfer` 는 100%,
`chamfer` 는 25%(서브샘플 `4096,16384,65536` 때문).

**(3) 원인은 transport 입니다.** 가우시안의 화면 반지름 중앙값이 **1.40 px** 인데
모델의 실제 오차(`rel_offset_p50` 0.499)는 **~6 px** — 자기 footprint 의 4배입니다.
그 거리를 넘으면 이미지 그래디언트는 **엉뚱한 위치**에서 샘플링되고 점이 어디로
가야 하는지 알 수 없습니다.

**멀티뷰가 원인이 아니라는 확인.** 등방 노이즈는 에너지의 ~1/3 이 시선 방향이라
단일 투영으로는 원리적으로 안 보입니다 → 단일 뷰 코사인 상한은 √(2/3) ≈ 0.816.
4뷰로 올려도 0.134 → **0.155** (커버리지 24% → 35%) 로 0.02 만 오릅니다.
0.816 에 한참 못 미치므로 **깊이 blindness 가 원인이 아니고 transport 가 원인**입니다.

### 16.4 앞서 한 주장 하나를 철회합니다

launch script 에 "렌더 손실이 중복을 직접 보는 유일한 항"이라고 썼습니다.
**detect 와 fix 를 혼동했습니다.** 렌더 손실은 중복을 *감지*합니다(겹친 점은 한 곳의
opacity 를 낭비하고, 구멍은 배경으로 읽힘). 하지만 *고칠* 수 없습니다 — 고치려면
점을 자기 footprint 보다 훨씬 먼 간격을 건너 **수송**해야 하고, 그런 그래디언트 경로가
없습니다.

### 16.5 3DGS 와의 차이 — 왜 vanilla 는 되는데 우리는 안 되는가

vanilla 3DGS 는 실제로 이미지 손실만으로 means 를 최적화합니다. 단,
**adaptive density control (clone / split / prune)** 에 의존해서 gradient descent 가
닿을 수 없는 커버리지 오류를 고칩니다. 이 설계는 **262144 고정 예산이고 densification
이 없습니다.** 즉 3DGS 기계장치의 절반만 가져다 쓰는 셈이고, 그 절반은 나머지 절반에
의존합니다.

### 16.6 그래서 어디에 쓰는가

렌더 손실은 **지웠지 않고** 켜는 위치만 바꿨습니다. 결정 변수가 **수송을 요구하지 않는**
곳에서는 조건수가 좋습니다:

| 위치 | 이유 |
|---|---|
| **attribute (`--stage full`)** | opacity / color / scale 은 **이미 차지한 같은 픽셀**을 바꿉니다. 수송 없음 → 조건수 양호 |
| **canonicalizer** | 결정 변수가 per-Gaussian gate 이고, "이 가우시안이 보이는가 / 지우면 이미지가 바뀌는가"는 이미지 손실이 답하기에 정확히 맞는 질문 |
| **평가 (항상)** | `tools/render_compare.py` |

### 16.7 코드 변경

* `scripts/launch_render_262k_ddp.sh` → **`scripts/launch_coverage_262k_ddp.sh`**
  로 이름 변경. `W_RENDER` 기본값 **0.004 → 0**.
* 이제 이 런의 레버는 **하나** 입니다: `--w_xyz_residual 12`.
  측정된 조건수도 가장 좋습니다 (cos 0.866, 100% 커버리지, 오차 크기 무관).
* 측정 표와 철회 내용을 script 헤더에 기록. attribute 단계에서 켤 때의 값(0.004)도
  같이 남겼습니다.
* `tools/audit_render_direction.py` 신규 (`--views`, `--sigmas`, `--scene`).

### 16.8 부수적으로 얻은 것 — 실험 위생

원래 이 런은 레버 2개(`w_render` + `w_xyz_residual`)를 동시에 켤 예정이었습니다.
지금은 1개입니다. 이전 세션에서 7번 재시작하며 변경을 묶어 attribution 을 잃은
문제를 이번엔 반복하지 않습니다.

---

## 17. eval 세트의 퇴화 프레임 — 측정 결과 점공간 지표에는 영향 없음

§16 직전에 "eval 8장 중 3장(idx 0, 4, 7)이 3DGS 학습 530스텝 이전이라 지금까지의
`ich` / `nn_unique` 절대값이 희석됐다"고 했습니다. **점공간 지표에 대해서는 그 진술이
과했습니다.** 체크포인트 step 2000 에서 장면별로 재봤습니다:

| idx | file | N | opac_max | ich | nn_unique | tmpl_erank |
|---|---|---|---|---|---|---|
| 0 | step_000010 | 112,840 | **0.140** | 0.2378 | 0.5104 | 7.01 |
| 4 | step_000450 | 112,840 | 0.998 | 0.2527 | 0.5178 | 7.18 |
| 7 | step_000530 | 112,842 | 0.999 | 0.2534 | 0.5181 | 7.19 |
| 63 | step_005070 | 262,144 | 1.000 | 0.2612 | 0.5118 | 7.43 |
| 74 | step_006130 | 107,434 | 1.000 | 0.2520 | 0.5013 | 7.14 |
| 138 | step_012730 | 209,929 | 1.000 | 0.2505 | 0.4604 | 7.08 |
| 183 | step_018020 | 226,926 | 1.000 | 0.2483 | 0.4750 | 7.13 |
| 288 | step_028630 | 79,557 | 1.000 | 0.2497 | 0.4943 | 7.05 |

| | 8장 전체 | 초기 3장 | 수렴 5장 | 보고값 오차 |
|---|---|---|---|---|
| `intra_chamfer_rel` | 0.2507 | 0.2480 | 0.2523 | **−0.0016** |
| `nn_unique` | 0.4986 | 0.5154 | 0.4886 | **+0.0101** |
| `template_erank_pred` | 7.152 | 7.128 | 7.166 | −0.015 |

0.6% / 2.1% 로, 장면 간 산포(`ich` 0.238–0.261) 안에 들어갑니다.

**이유:** `ich` 와 `nn_unique` 는 **그룹 반지름으로 정규화**되고 opacity 나 SH 를 아예
보지 않습니다. 초기 프레임도 xyz 분포 자체는 정상적인 점군(SfM 초기화 + 114k 점)이라
기하 구조가 비교 가능합니다. 즉 지금까지의 점공간 수치는 유효합니다.

**단, 렌더 지표에는 쓸 수 없습니다.** idx 0 은 `opac_max` 0.140 으로 원본조차 안개이고,
실제로 이 프레임에서는 `codec_snap`(15.50)이 `codec`(16.03)보다 **낮게** 나옵니다 —
수렴 프레임과 부호가 반대입니다. `render_compare` 에는 수렴 프레임만 넘겨야 합니다
(지금까지 보고한 렌더 수치는 idx 40/80/120/160/200/240/280 을 썼으므로 해당 없음).

**부수 소득:** `nn_unique` 가 8장 전부에서 0.46–0.52, `tmpl_erank` 가 7.0–7.4 로
**극히 균일**합니다. 중복 문제는 특정 장면의 성질이 아니라 **모델의 성질**입니다.
이건 §16 의 진단을 강화합니다.

`eval_val_indices` 는 **바꾸지 않았습니다.** 바꾸면 이전 모든 런과의 비교가 깨지고,
위 수치가 바꿀 이유가 없다고 말하고 있습니다.
