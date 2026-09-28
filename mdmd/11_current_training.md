# 현재 학습 방법

작성: 2026-09-15  
루프: `can3tok/train.py`. 가중 램프: `schedule.py`. 손실: `losses.py`.  
기본 숫자: `runs/E1_20260915_041813/args.json`. S16k/T16k와의 차이는 §7.

실행:

```
torchrun --standalone --nproc_per_node=3 train.py  <args.json에서 복원한 인자>
# 또는
bash scripts/launch_e1_frame_first.sh --detached
```

파이썬: `/home/super/anaconda3/envs/can3tok/bin/python`. DDP 3 GPU가 최근 16k 런의 기본.

---

## 1. 한 스텝이 하는 일

`main()` 요지:

1. `build_parser` → `resolve_shape_args` → DDP init, seed
2. `make_datasets` (train/val, 씬별 앵커·stats)
3. `load_eval_views`: 홀드아웃 사진을 풀에서 **제거** (`view_exclude`)
4. `build_config` + `describe_layout` 로그
5. `build_model`, `pack_trainable`/residual 설정
6. AdamW, 모듈별 param group (`core.param_groups`)
7. 루프:
   - `schedule_flags(step)` → `run_decode`, `run_gen`, `with_attrs`, downscale
   - `folding_res_gain`, `decoder_refine_alpha`, `shortcut_alpha`, `attr_teacher_prob`를 cfg에 씀
   - `effective_weights(step)` → `total_loss`
   - amp bf16, `grad_clip=1.0`
   - `loss_spike_mult=25`: 중앙값 대비 스파이크면 배치 skip. DDP는 MAX로 합의한 뒤
     전원 skip (한 랭크만 빠지면 all-reduce 데드락)
   - 로그 `G[...]` 기하, `U[...]` (rad, c3d, aset, p2g, …), 클리핑 전 `gn`
   - `logs["total"]`은 **모든 항을 더한 뒤**의 값 (fix_7: 예전엔 렌더 추가 전 값)
8. eval / save / `score_metric`으로 `ckpt_best`

`enc_x`/`enc_mask`를 넘긴다. eval도 동일. 안 넘기면 풀러가 다른 그룹핑을 봄
(측정: eval rmse 0.675 vs 실제 0.0096).

---

## 2. 커리큘럼 (`schedule.py`)

원래 네 단계: `latent` → `geo` → `joint`/`gen` → 옵션 `polish`.
전이는 전부 `_ramp`. 한 스텝에 가중을 깎으면 latent 다양성이 죽어서.

**E1 (및 S16k/T16k 계열)은 이 표를 거의 비활성화한다.**

| 인자 | E1 | 효과 |
|---|---|---|
| `latent_end` | **0** | 처음부터 decode. latent-only 프리트레인 없음 |
| `late_decode_steps` | 0 | |
| `geo_end` | 800 | 800스텝만 geo 약한 창. 이후 m_geo=1 |
| `geo_weak_scale` | 0.35 | geo 창에서 절대단위 기하를 약하게 |
| `gen_start` / `gen_end` | 10^9 | gen 안 켬 (`no_gen_branch`와 함께) |
| `polish_start` | 0 | polish 꺼짐 |
| `folding_res_start` | **8000** | 그 전 `folding_res_gain=0` (S16k/T16k는 **0**) |
| `decoder_refine_start` | **8000** | 그 전 refine α=0 (S16k/T16k는 **400**) |
| `attr_start` | 1200 | 속성 손실 램프 |
| `render_start` | 1200 | 렌더 램프 |
| `attr_detach_release` | 12000 | 그 전 렌더→xyz detach. E1 max 16k라 끝에서만 풀림 |
| `detail_start` | **20000** | max_steps=16000보다 큼 → **VGG/edge 없음** |
| `equiv_start` | 4000 | 매 4스텝 equivariance |
| `encoder_residual_start` | 0 | residual 처음부터 |

`phase_of`: `latent_end=0`이면 즉시 `geo` 또는 `joint`. gen_end가 거대 센티널이면
geo_end 이후는 `joint`이지만 `run_gen`은 플래그가 막음.

`m_detail`: decode가 켜진 뒤 `detail_ramp_steps=100`으로 intra chamfer / p2g /
radius / attr_set / sinkhorn이 **geo_weak와 별개로** 빨리 올라감. 절대단위 항만
geo 창에서 약하다.

Sinkhorn ε: E1은 `sinkhorn_epsilon=0.08` 고정 (anneal_steps=1). T16k는 40k부터
0.005로 12k 동안 내림.

---

## 3. 지금 켜진 손실 (E1 기본 가중)

`total_loss`는 가중 0인 항은 더하지 않는다. 아래는 **베이스 가중 ≠ 0**인 것.
실제 스텝 값은 `effective_weights`가 램프를 곱한 값.

슬롯 순서 L2 (`w_xyz`, `w_z_raw`, `w_z_residual`, …)는 **전부 0**.
순열 불변 집합거리만 기하를 민다 (R1이 둘을 같이 끄고 붕괴한 이유).

### 3.1 기하 (점 집합)

| 키 | w | 무엇 |
|---|---:|---|
| `w_chamfer` | 5 | 멀티스케일 대칭 chamfer (4096/16384/65536) |
| `w_coverage` | 1200 | GT→pred. 빈 표면. 단위가 작아 가중을 키움 |
| `w_p2g` | 3 | pred→GT. 뜬 점 |
| `w_plane_chamfer` | 2.5 | 평면 투영 |
| `w_proj_hist` | 2.5 | 2D 소프트 히스토그램. T16k에서 TOP ~27% |
| `w_dispersion` | 3 | 패치가 한 점에 안 뭉치게. margin 0.001 |
| `w_intra_chamfer` | 3 | 칸 안 순열 불변 chamfer / 반경 |
| `w_intra_sinkhorn` | 4 | 칸 안 OT. `sinkhorn_attr_weight=2`면 속성이 비용에 들어감 |
| `w_group_centroid` | 2 | 칸 중심 |
| `w_radius` | 4 | 칸 반경 비. 퇴화 GT 그룹은 리덕션에서 제외 (fix_7) |
| `w_presence` | 0.5 | 살아 있는 슬롯 수 |

E1에서 **0**: `w_intra_spacing`, `w_splat_area`, `w_voxel_occ`, `w_xyz*`.

T16k만: `w_splat_area=1`, `w_intra_spacing=2` (평균 NN 힌지).

### 3.2 속성

| 키 | w | 무엇 |
|---|---:|---|
| `w_attr_set` | 3 | 칸 안 속성 집합 거리. 가로/세로가 원으로 평균되지 않게 |
| `w_cov3d` | 2 | 3×3 공분산. `w_rot=0` |
| `w_scale` / `w_opacity` / `w_color` | 1 | 책임 매칭 후 파라미터 |
| `w_rot` | 0 | cov3d가 대체 |

`attr_match_mode=sinkhorn`, `attr_responsibility=1`.

256점 칸에서 aset 바닥 ≈1.0 (완벽해도). T16k aset≈1.33. 가중만 올리는 길은 닫힘.

### 3.3 잠재 / 정규화

| 키 | w | 무엇 |
|---|---:|---|
| `w_latent_decorr` | 5 | shape 채널만 (`c_shape` 상한, fix_7). appearance를 먹으면 안 됨 |
| `w_latent_std` | 1 | 채널 std 밴드 |
| `w_shape_sinkhorn` | 1.5 | shape 코드 쪽 OT |
| `w_equiv` | 0.5 | 회전 후 compact 동변. `equiv_shape_weight=0` (shape 불변 금지) |
| `w_kl` | 0 | VAE 아님 |

### 3.4 렌더

`w_render=0`. 실제 렌더는 `w_render_attr=60`이 `render_loss`에 들어감
(attr_start/render_start 이후 램프).

`w_render_perc=0.05`, `edge_gain=4`는 `detail_gains`가 `detail_start` 전에 0으로 둠.
E1에서는 사실상 꺼짐. S16k는 40k, T16k는 44k.

뷰: `render_views=5`, `extra_real_views=4`, `view_pool`, 홀드아웃 제외.
다운스케일 8→2 (`render_downscale_start`→`render_downscale`), 스텝 8000.

---

## 4. 옵티마이저·안정화

- lr 2e-4, min 1e-5, warmup 500, AdamW wd 1e-4
- `grad_clip=1.0`. 그래디언트 노름이 크면 항상 클립 → AdamW eps를 기본보다
  낮게 (train.py 주석: 클립 후 좌표 RMS가 1e-8 eps에 먹힘)
- amp **bf16**
- `loss_spike_mult=25`, 히스토리 32의 **중앙값**. 웜업 동안 가드 꺼짐
- 비유한 loss는 항상 skip
- `latent_scale_sync_every=50`
- batch_size 1 × 3 GPU
- `CHAMFER_PAIR_CHUNK=4096`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

모듈별 lr (`lr_scales`): encoder / compressor / decoder / attr. polish가 켜지면
encoder·compressor를 0에 가깝게. E1은 polish 없음.
`late_codec_start`가 거대 센티널이라 late 0.4×도 안 탐.

---

## 5. Eval과 체크포인트

`run_eval`:

- `eval_val_indices=25,76,127,166,229,280,331,382` (기관차 4 + 트럭 4)
- refine α는 **그 스텝 스케줄과 같게** (처음부터 1로 열면 초기 eval이 부풀음)
- `enc_input` 전달
- 기하 지표는 `pred` xyz, 속성은 `attr_pred`
- 사진: 홀드아웃 뷰, `eval_view_force=3` 등
- `metrics.json`에 풀 평균 + `scene0/` `scene1/`
- 산출: eval PLY, 비교 png

`score_metric=psnr_gap_worst` (E1, T16k): 씬별 (GT사진 − pred)의 **최악**을 최소화.
풀 `psnr`은 트럭이 기관차를 가림. S16k는 `psnr`이었다.

`ckpt_best.pt` / `ckpt_step*.pt` / `save_every=2000`.
`init_from`은 step을 0으로 리셋하면 attr/render 게이트가 다시 닫힘 (버그 9).
이어갈 때는 `resume`.

---

## 6. 로그에서 볼 것

`[joint] step t`: `loss`, TOP 항 이름과 %. splat/spacing이 켜져 있어도 0.5%면
목적함수가 그걸 안 본다는 뜻 (T16k @70k).

`relq` = `codec_rel_offset_p50`. √2 ≈ 1.414면 칸 안 오프셋 무상관. 성공 게이트로
쓰지 말 것.

`U[rad …]`: radius 항이 예전에 로그에 없어서 스파이크를 못 쫓음.

씬별 `s0`/`s1` photo. 부호가 반대면 풀 평균을 믿지 말 것.

---

## 7. S16k / T16k / E1 — 같은 모델, 다른 학습

레이아웃·데이터·모듈은 같다. 학습 순서와 부가 항만 다름.

| | S16k | T16k | E1 |
|---|---|---|---|
| 시작 | from-scratch | S16k 40k resume | from-scratch |
| max_steps | 64000 | 72000 | 16000 |
| folding_res_start | 0 | 0 | **8000** |
| decoder_refine_start | 400 | 400 | **8000** |
| detail_start | 40000 | 44000 | 20000 (런 밖) |
| w_splat_area | 0 | **1.0** | 0 |
| w_intra_spacing | 0 | **2.0** | 0 |
| sinkhorn ε | 0.08 | 40k→0.005 | 0.08 |
| score_metric | psnr | psnr_gap_worst | psnr_gap_worst |
| 게이트 | — | 70k 사진= S16k62k, cen 악화 | **8k 글자·이방성** |

T16k가 켠 splat/spacing은 총손실 ~0.5%/TOP 밖. `relq`/`uniq` 불변.

E1 @3k는 아직 잔차·refine·속성 램프 중. 70k와 비교하지 말 것.

런처:

- `scripts/launch_e1_frame_first.sh` — S16k args.json을 읽어 스케줄만 덮음
- `scripts/launch_t16k_resume_40k.sh`
- `scripts/launch_l17k_shape_rebalance.sh` — shape 7/app 4. **다른 레이아웃**

`scripts/argv_from_argsjson.py`가 json → CLI.

---

## 8. 데이터·뷰가 학습에 들어가는 방식

- 실사진: `photo_map`이 npz→이미지. 스냅샷 자기 사진은 `view_exclude`를 안 거침.
  일반화는 held-out 스냅샷 × held-out 사진.
- `view_downscale=2`
- `cache_slots=true`이지만 `augment=true`면 캐시 히트 없음 (증강마다 재배정)

---

## 9. 코드에 있는 “학습 방법”이 지금 아닌 것

README §4 curriculum (latent 2k / geo 3k / gen 5k)은 **기본값 주석**이지 E1이 아님.

`w_gen_*`가 args에 남아 있어도 `run_gen=false`면 `total_loss`가 안 더함.

`joint_*` 디코더를 켜면 CodecDecoder를 freeze하고 다른 식을 쓴다. 현재 런처는 0.

새 항을 넣을 때: 한 변수, held-out 사진+글자, 컨트롤 런. 방법론은
`07_closed_open_methodology.md`.
