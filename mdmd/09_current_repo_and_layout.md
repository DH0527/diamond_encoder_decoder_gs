# 현재 트리와 16k 레이아웃

작성: 2026-09-15  
대상: `/data/daeho/aacd_proj/can3tok_encoder_decoder_new_fix_7`  
기준 런: `runs/E1_20260915_041813/args.json` (S16k/T16k와 **같은 레이아웃**, 스케줄만 다름).

루트 `README.md`, `ARCHITECTURE_KR.md`, `MODEL.md`, `STRUCTURE_KR.md`는 대부분
**32×64×64 = 131,072** 시절을 적는다. 지금 학습이 쓰는 계약은 그 문서가 아니라
아래와 `args.json`이다.

---

## 1. 디렉터리

```
can3tok_encoder_decoder_new_fix_7/
  train.py                 torchrun 진입점 → can3tok.train.main
  can3tok/                 모델·데이터·손실·학습 루프
  scripts/                 런처 (현재 16k: launch_e1_frame_first.sh, launch_t16k_resume_40k.sh, …)
  assets/                  앵커 npy, split json, stats, photo_map, view_pool
  tests/
  tools/                   오라클·에셋 생성
  runs/                    학습 산출 (이 문서가 설명하는 코드가 아님)
  mdmd/                    실험 기록 + 이 구조 문서
```

`can3tok/` 모듈 (학습이 import하는 것):

| 파일 | 역할 |
|---|---|
| `config.py` | `Can3TokConfig`, `patch_layout`, `channel_budget`, `validate_layout` |
| `model.py` | `Can3TokAE`: encode → compress → decompress → decode → attr |
| `encoder.py` | `PatchPackEncoder` + `GroupPointPooler` |
| `compressor.py` | `StagedCompressor` / `StagedDecompressor` + folding |
| `decoder.py` | `CodecDecoder` (xyz refine, presence) |
| `attr_decoder.py` | `AttributeDecoder` (scale/rot/opacity/color + nudge) |
| `gen_decoder.py` | 생성 분기. **지금 `no_gen_branch=true`라 안 탐** |
| `joint_decoder.py` | F3 계열. **지금 꺼짐** |
| `data.py` | `ReplayGaussianDataset` |
| `io_utils.py` | npz 로드, 59→타깃 채널 팩 |
| `losses.py` | `total_loss` |
| `schedule.py` | phase, 가중 램프, refine/residual 게이트 |
| `train.py` | DDP 루프, eval, 체크포인트 |
| `render.py` | 가우시안 래스터, photometric, VGG |
| `eval_utils.py` | codec 지표, PLY, 비교 그림 |
| `template.py` | Fibonacci 채운 공 |
| `morton.py` | Morton, kd 그룹, 격자 z-order |
| `layers.py` | MLP, self/cross/window attention |
| `stats.py` | 씬 center/scale |

진입은 루트 `train.py` 한 줄:

```python
from can3tok.train import main
```

---

## 2. “현재”가 의미하는 학습 계약

E1 / S16k / T16k가 공유하는 것 (`args.json`):

| 항목 | 값 | 산술 |
|---|---|---|
| 디코더 출력 | `max_points=262144` | 고정 |
| 인코더 입력 cap | `max_input_points=2097152` | 칸당 최대 2048점 |
| 칸 | `group_size=256`, 그룹 수 **1024** | 262144/256 |
| `z_compact` | **16 × 32 × 32 = 16,384** | WM 인터페이스 |
| `z_raw` | 32 × 256 × 128 = 1,048,576 | 학습 중간. WM은 안 봄 |
| 토큰/그룹 | `local_tokens_per_group=32` | token_dim = 32×32 = 1024 = patch_dim |
| 채널 예산/칸 | cen **4** \| occ **1** \| shape **3** \| app **8** | 합 16 |
| merge | **1** | compact 셀 수 = 그룹 수 |
| 씬 | train + truck | 앵커·stats 각각 |
| gen | `no_gen_branch=true` | `decode_compact`가 배포 경로 |
| joint / structured_local | 0 | F3 꺼짐 |
| `compress_merge_attn` | false | concat MLP merge |
| `compress_merge_stages` | 0 | 2층 merge (`T*d → 2d → d`) |
| `pool_blocks` | 1 | 한 번에 2048→16 쿼리 |
| folding | on | 프레임 6-DoF + 잔차 |
| `allow_low_shape_budget` | **true** | 3/256=0.0117을 통과시킴 |
| `use_fixed_anchor_center` | 0 | compact의 xyz는 npy가 아님 |
| `attr_decoder_layers` | 4 | 기하와 분리된 속성 디코더 |
| `attr_pack_dim` | 11 | 풀러가 속성도 봄 |
| `attr_read_shape` | 1 | 속성이 shape 3을 가산 분기로 읽음 |

shape/점 = 3/256 = **0.0117**. `validate_layout`은 원래 0.02 미만을 ValueError로
막는다. 16k 런은 이스케이프를 켠다.

---

## 3. `patch_layout` / `channel_budget`이 지금 내는 숫자

`config.py`. E1 값으로 계산하면:

```
num_groups     = ceil(262144 / 256) = 1024
raw_cells_used = 1024 * 32 = 32768
latent_hw      = 256×128 = 32768  → 패딩 셀 0
compact_cells  = 32×32 = 1024
merge          = ceil(1024 / 1024) = 1
points/cell    = 256
per_group      = 16 / 1 = 16
token_dim      = 32 * 32 = 1024
patch_dim      = 256 * 4 = 1024   ← identity pack 폭 (xyz 768 + aux 256)
```

`describe_layout`이 매 런 시작에 찍는다. `effective` = 16/256 = **0.0625 compact
채널/점**. 옛 131k 런은 0.50.

`budget_centroid < 4`는 ValueError. xyz+log-extent가 칸 안 길이 단위이기 때문.

appearance는 suffix. `c_app=0`이면 예전 기하-only와 바이트 동일.

---

## 4. 전체 흐름 (지금 실제로 도는 것)

```
npz (씬마다 N ≈ 8만~50만+)
  data.py
    정규화 → (옵션) 증강, 앵커도 같이 변환
    drop_outside |xyz|≤1
    importance 선택 → 디코더 262,144 + 인코더 ≤2,097,152 (상위집합)
    k-means 앵커 1024칸에 spill 8 배정
    꽉 찬 칸은 Fibonacci 템플릿에 Hungarian 슬롯 정렬
  ▼
enc_in  (B, 2097152, 15)   xyz3 + attr11 + mask 경로
target  (B, 262144, 14)    xyz3+log_scale3+quat4+logit_op1+rgb3  (SH DC만, sh_dim=0)
  ▼
encoder  GroupPointPooler: 칸당 ≤2048점 → 쿼리 16 → Linear → patch 1024
         pack Linear(1024→1024) identity 초기화, pack_trainable=true
         + residual (encoder_residual_start부터)
  ▼
z_raw   (B, 32, 256, 128)
  ▼
compressor
  token_embed 32→448
  intra self-attn ×3  (칸 안 32 토큰)
  token_merge 32×448 → 448
  cell_mix (merge=1이라 거의 항등)
  window attn ×2  (32×32 격자, window 8, 짝수 층 shift)
  group_head + mid 256
  heads: cen4 | occ1 | shape3 | app8
  ▼
z_compact  (B, 16, 32, 32)     ★ 월드모델이 볼 텐서
  ▼
decompressor
  채널을 다시 자름
  folding: 점 = centroid + extent * R * diag(s) * (unit_ball + residual)
  z_raw_hat 재합성
  ▼
CodecDecoder  10층 self+cross, decoder_refine_alpha로 잔차 게이트
  → pred xyz, presence
  ▼
AttributeDecoder  (xyz는 기본 detach)
  → attr_pred = nudged xyz + scale/rot/opacity/color
  ▼
losses + (attr_start 이후) render_loss
```

`decode_compact(z)`는 decompressor + CodecDecoder만. **AttributeDecoder를 안 탄다.**
월드모델이 색까지 쓰려면 `forward`와 같이 attr 경로를 붙여야 한다. 지금 학습
렌더가 보는 것은 `attr_pred`이다.

---

## 5. compact 16채널이 의미하는 것 (코드)

`StagedCompressor.forward` (`compressor.py` ~499):

```
cen[0:3] = 인코더가 본 칸 표본 평균
           + tanh(head) * extent * 0.05     (CENTROID_DELTA)
           use_fixed_anchor_center=0 이므로 k-means npy가 아님
cen[3]   = encode_scale(extent)             log-extent. npy에 없음
occ[0]   = (2*count/256 - 1) + tanh(res)*0.1
shape[3] = head_shape(mid)                  자유
app[8]   = head_appear(mid)                 자유, suffix
```

`merge=1`이라 셀 = 그룹. z-order로 32×32 격자에 심음 (`latent_zorder`).

디코드 (`_shape_to_offsets`):

```
local = unit_ball(256) + tanh(residual / cap) * cap * folding_res_gain
local = local - mean
local = local * aniso(frame[:3])            log-ratio tanh cap 1.5
local = R(axis-angle frame[3:6]) @ local
xyz   = centroid + decode_scale(cen[3]) * local
```

`fold_head`와 `shape_xyz` **둘 다 shape 3 + ctx를 읽는다.** ctx는 decompressor가
만든 그룹 컨텍스트. 3숫자가 6-DoF 프레임과 256×3 잔차를 동시에 가리킴.

`folding_res_gain=0`이면 잔차는 곱 0 (E1은 step 8000까지).
`decoder_refine_alpha=0`이면 CodecDecoder xyz 잔차가 꺼짐.

---

## 6. 코드에 있으나 지금 꺼진 경로

학습이 안 탄다고 지워진 것은 아니다. `args`가 끈다.

| 모듈/플래그 | 지금 | 켜면 |
|---|---|---|
| `GenerativeDecoder` | `no_gen_branch` | compact에서 새 점 집합. WM용으로 설계됐으나 T16k는 끔 |
| `DirectJointGaussianDecoder` / `JointGaussianRefiner` | 0 | F3. ~9.5 dB로 닫힌 실험 |
| `structured_local_code` | 0 | 로컬 토큰을 compact에 유지 |
| `compress_merge_attn` | false | 쿼리 1개 attention pooling |
| `shared_free_head` | 0 | shape+app를 한 헤드 |
| `GroupAttributeEncoder` | 풀러가 켜지면 **미호출·동결** | identity pack 시절 aux 슬롯 |
| `partition_mode=morton` | 인자만 남음 | `scene_anchors`가 있으면 `_pack_slots`가 앵커 분기 |
| CodecDecoder 속성 헤드 | `attr_decoder_layers=4`면 출력 미사용 | 예전 기하 토큰에 매달린 헤드 |
| `use_fixed_anchor_center` | 0 | `forward`가 `group_anchor=None`으로 버림 |
| VAE / `w_kl` | deterministic, 0 | |
| `compress_wide_*` | 0 | compact 격자 ResBlock |

---

## 7. 루트 README가 틀린 부분 (혼동 방지)

| README에 적힌 것 | 지금 16k |
|---|---|
| `z_compact` 32×64×64 | **16×32×32** |
| 그룹 32점, merge=2, 셀당 두 그룹 | 그룹 **256**, merge=1 |
| occupancy 기본 0, shape 12 | occ **1**, shape **3**, app **8** |
| identity Morton pack이 학습 경로 | **풀러**. identity는 초기화일 뿐 |
| gen이 배포 경로 | gen 꺼짐. codec `decode_compact` |
| curriculum latent 0–2000 | E1은 `latent_end=0` — **처음부터 decode** |

구조·학습을 읽을 때는 이 파일과 `10`·`11`을 README보다 우선한다.
