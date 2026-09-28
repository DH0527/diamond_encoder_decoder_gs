# M4 완전 재현 명세

`runs/M4_20260818_105922` · held-out PSNR **17.75 dB** (step 16000, GT 천장 18.70, gap 0.96)

이 문서만 읽고 코드를 처음부터 작성해 같은 학습을 돌리고 같은 결과를 얻을 수 있도록 쓴 명세입니다. 근거·실패 기록은 `STRUCTURE_KR.md` / `HISTORY_KR.md`에 있고, 여기에는 **무엇을 어떻게 계산하는가**만 적습니다.

**재현 확인 기준**: 아래 §9의 held-out PSNR 궤적을 ±0.3 dB 안에서 재현하면 성공입니다.

---

## 0. 목차

| § | 내용 |
|---|---|
| 1 | 환경 |
| 2 | 데이터 |
| 3 | 공용 레이어 |
| 4 | 인코더 |
| 5 | 압축기 → z_compact |
| 6 | 복원기 |
| 7 | 디코더 2개 |
| 8 | 손실 · 스케줄 · 옵티마이저 |
| 9 | 실행 명령 · 기대 결과 |
| 10 | 재현 시 반드시 피해야 할 함정 |

---

## 1. 환경

```
python      3.11.15
torch       2.4.1+cu121   (cuda 12.1)
numpy       2.4.5
scipy       1.17.1        (linear_sum_assignment 필요)
GPU         NVIDIA RTX 5000 Ada Generation (32 GB)
platform    Linux-5.15.0-186-generic-x86_64-with-glibc2.35
래스터라이저  diff_gaussian_rasterization (3DGS 공식 구현)
```

- `--amp bf16` 사용. fp16 경로도 있으나 M4는 bf16입니다.
- 단일 GPU. `torchrun --nproc_per_node=1`. **DDP 2장 이상이면 유효 배치가 배수로 늘어나 결과가 달라집니다.**
- 메모리 사용량 약 18.9 GB.
- 속도 약 4.4 s/it → 20,000스텝 ≈ 24.5시간.

---

## 2. 데이터

### 2.1 원본

```
root        .../speedy-splat/output/train_colmap_seg_prune_scores/replay_seg
파일        step_000010.npz ... step_030000.npz   (3,000개)
```

각 npz는 `state_t`라는 dict 하나를 담습니다. 키: `camera`, `gaussians`, `ids`, `pruning_scores`, `segmentation`.

**중요**: 이 3,000개는 **한 장면(기차 야드)의 3DGS 최적화 스냅샷**입니다. 임의 두 스냅샷 사이의 공간 중첩이 99.9~100%(bbox 대각의 1% 이내), bounding box가 전부 189×68×127로 동일합니다. 스텝마다 카메라가 다르고 densify/prune으로 점 개수가 81,140~565,000으로 변합니다. **다중 장면 데이터셋이 아닙니다.**

`gaussians`에서 꺼내는 것: `xyz (N,3)`, `scaling (N,3)`, `rot (N,4)`, `opacity (N,1)`, `color (N,3)`, `sh (N,45)`.

### 2.2 분할과 정규화

```
split       assets/replay_seg_2700_300_seed42.json
            {"root":..., "n_total":3000, "n_train":2700, "n_val":300, "seed":42,
             "train":[정렬된 파일 목록의 인덱스 2700개], "val":[300개]}
```
파일 목록은 **경로 사전순 정렬** 기준입니다.

```
stats       assets/stats_replay_seg.json
center      [9.757812, -19.706543, -32.464844]
scale       47.418164062500004
            (sample_stride 50, quantile 0.02 로 산출된 bbox 기반)
```

### 2.3 성숙도 필터

`--min_snapshot_step 12000`: 파일명 숫자가 12000 미만인 것을 버립니다.

```
train 2700 -> 1628
val    300 -> 173
```

이유: GT 자체를 렌더한 상한이 스냅샷 성숙도에 따라 다릅니다 — early(<12k) 16.34 dB, late(≥12k) 20.13 dB. 섞으면 모델과 무관하게 천장이 내려갑니다.

### 2.4 타깃 채널 (`gaussian_target_channels`)

```python
xyz_norm  = (xyz - center) / scale                       # 3
log_scale = log(clip(scaling / scale, 1e-8, None))       # 3
rot       = quat / ||quat||                              # 4
opacity   = logit(opacity)                               # 1
color     = color (SH DC 계수 그대로)                      # 3
sh        = sh                                           # 45
target    = concat([...])                                # 59 채널
layout    = {xyz:0, scale:3, rot:6, opacity:10, color:11, sh:14}
```

**모델이 쓰는 것은 앞 14채널뿐입니다** (`cfg.target_dim = 14`, `sh_dim = 0`, `--w_sh 0`). 59채널은 로더가 그대로 들고 다닙니다.

### 2.5 씬 앵커

```
assets/scene_anchors.npy    (4096, 3)  float32
```
전체 데이터셋의 점 구름에 k-means를 돌려 만든 4,096개 중심점입니다. **world 좌표로 저장**되어 있으므로 로더가 자동 감지해 `(a - center)/scale`로 정규화합니다 (97.0%가 단위 큐브 안).

k-means여야 합니다 — FPS(farthest point sampling)는 밀집 영역에서 그룹 반지름 1.18을 만들고 k-means는 0.064입니다.

### 2.6 항목 하나를 만드는 절차 (`__getitem__`)

난수 시드: `rng = default_rng((seed*1000003 + epoch*7919 + index) & 0x7FFFFFFF)`, `--seed 42`.

**(a) 증강** (`--augment`, train만)
```
회전    각 축 uniform(-8°, +8°)  ->  R
스케일  uniform(1-0.03, 1+0.03)  ->  s
이동    uniform(-0.01, +0.01)^3  ->  shift
xyz_norm <- (xyz_norm @ R.T) * s + shift
target 의 log_scale += log(s),  rot <- quat_mul(R->quat, rot)
camera  <- transform_camera_vector(camera, world 변환)
anchors <- (anchors @ R.T) * s + shift      # ★ 앵커도 같이 변환해야 함
```
**앵커 변환을 빠뜨리면 슬롯의 81%가 빕니다.**

**(b) 큐브 밖 제거** (`--drop_outside`): `all(|xyz_norm| <= 1)`인 점만 유지. `keep_idx`로 원본 행 인덱스를 보존합니다.

**(c) 중요도**
```python
score = target[:, 10]                                    # logit(opacity)
if pruning_scores 존재:  score += 0.25 * zscore(pruning_scores)
if density_aware_sample: score += w * (1 - local_density)
```
`local_density`: xyz_norm을 32³ 격자로 양자화한 뒤 `log1p(cell count) / log1p(max count)`.

`pruning_scores`는 **샘플링에만** 씁니다. 인코더 입력 피처에 넣지 않습니다 — `decode(z_compact)`가 볼 수 없는 정보이므로.

**(d) 선택**: Morton 코드(`morton_bits`)로 정렬한 순서에 중요도 기반 계층 샘플링(`--sample_mode stratified`)을 적용해 최대 `max_points = 262144`개를 고릅니다.

**(e) 앵커 배정 (`_anchor_assign`)** — 이게 그룹 구성을 정합니다.

```
gsz = 64, ng = 4096, spill = 8
1) 각 점의 최근접 anchor 8개를 거리 순으로 구한다 (16384점씩 청킹, 전체 행렬은 4 GB)
2) 1순위 거리로 전체 점을 정렬 (가까운 것 우선)
3) 폴백 랭크 h = 0..7 에 대해 벡터화 라운드 1회:
     같은 anchor 를 고른 점들을 거리순 정렬 -> rank < (64 - 현재 적재량) 인 점만 배정
4) 남은 점은 버린다 (anchor_assignment=spill)
```
가까운 것 우선이라 밀려나는 점은 "늦게 도착한 점"이 아니라 "경쟁에서 진 점"입니다.

측정된 폐기율: 최근접만 26.1% → 8-스필오버 15.6% (step_005070), 15.9% → 6.1% (step_018020).

**(f) 슬롯 내부 정렬 (`--slot_sort template`)**

각 그룹의 64점을 고정 템플릿에 최적 배정합니다:
```
1) 그룹 중심 제거
2) 공분산 eigh -> 고유프레임으로 회전, sqrt(eigenvalue)로 화이트닝
   (얇은 축은 가장 굵은 축의 1e-3 로 하한)
3) 축별 skew 부호로 정준 부호 고정, det > 0 이 되도록 보정
4) scipy linear_sum_assignment 로 64x64 배정 (템플릿 = fibonacci_ball(64))
```
측정: Morton 순서 대비 평균 |잔차| 1.010 → **0.408**, rank-22 상한 1.176 → 0.679, chamfer 0.261 → 0.225.

**(g) 슬롯 텐서 채우기**
```
target      (262144, 59)   선택된 점의 타깃, 빈 슬롯은 0
input       (262144, 63)   = [target(59), 0, density, slot_rank, chunk_rank]
mask        (262144,)      1.0 = 살아있는 슬롯
source_index(262144,)      슬롯 -> npz 행 인덱스, 빈 슬롯 -1
```
- `input_dim = target_dim + 4 = 63`
- 63채널 중 60번째는 **의도적으로 상수 0**입니다 (과거 `pruning_score`가 있던 자리, 체크포인트 호환용)
- `slot_rank = (flat_pos % 64) / 63`, `chunk_rank = (flat_pos // chunk_size) / (ceil(262144/chunk_size) - 1)`, `chunk_size = 64`
- 순위는 **패킹된 레이아웃(그룹 우선)** 기준이어야 합니다

**`source_index`가 필수인 이유**: NN 조회로 슬롯을 원본 행에 되매핑하면 0.991만 단사입니다 — 이 장면 Gaussian의 0.16~0.17%가 정확히 겹치고, 추가 0.5%가 정규화 왕복의 float32 오차 3e−5보다 가깝습니다.

**(h) 인코더 입력 확대**
```
max_input_points = 589824    (= 4096 그룹 x 144점)
enc_input (589824, 63),  enc_mask (589824,)
```
같은 중요도·같은 앵커 규칙으로 더 많이 뽑되, **작은 집합(262144)의 상위집합**이어야 합니다. 다른 집합이면 world model의 셀에 안정된 Gaussian 동일성이 없어집니다.

**(i) 사진과 추가 시점**
```
photo       (3, 544, 977)   assets/npz_to_image.json 으로 매핑된 실제 사진
extra_cams  (4, 16)         assets/view_pool.json 에서 4개
extra_imgs  (4, 3, 272, 488)
camera      (16,)           fx,fy,cx,cy,R(9),T(3) 를 평탄화한 벡터
```
`--extra_real_views 4`. `fx == 0`이면 빈 슬롯 표시입니다.

**(j) 슬롯 캐시**: `--cache_slots`는 증강이 꺼진 경우에만 유효합니다(val). train은 증강 때문에 매번 재계산합니다.

---

## 3. 공용 레이어

```python
FourierFeatures(in_dim, num_freqs, include_input=True)
    freqs = 2^[0..F-1] * pi
    out   = [x, sin(x*f)..., cos(x*f)...]      out_dim = in*(1 + 2F)

MLP(dims, last_act, dropout, in_norm, hidden_norm, leaky)
    Linear -> [LayerNorm] -> GELU 또는 LeakyReLU(0.1) -> [Dropout]  반복
    마지막 층은 last_act=False 면 활성화 없음

residual_head(dims) = MLP(dims, in_norm=True, hidden_norm=True, leaky=True)

SelfAttentionBlock(dim, heads, dropout, mlp_ratio=4)   # pre-norm
    h = LN(x); a = MHA(h,h,h, key_padding_mask); x = x + a
    x = x + MLP([dim, 4*dim, dim])(LN(x))
    ★ key 가 전부 마스킹된 행은 NaN 이 되므로, 그 행은 자기 자신에 attend 시키고
      결과를 0 으로 덮어써야 합니다

CrossAttentionBlock(dim, heads, ...)                   # 동일, q/kv 분리 LN

ResBlock2d(C)
    h = GELU(conv1(GroupNorm(min(8,C), C)(x))); return x + conv2(h)

WindowSelfAttention(dim, heads, (H,W), window, shift)
    (H,W) 격자를 window x window 로 분할, 홀수 층은 window//2 시프트
```

스케일 인코딩:
```python
CENTROID_DELTA  = 0.05
SCALE_REF       = 0.02
SCALE_LOG_SPAN  = 3.0
encode_scale(e)  = log(e_mean / SCALE_REF) / SCALE_LOG_SPAN     # e_mean = 3축 평균
decode_scale(u)  = clamp(SCALE_REF * exp(SCALE_LOG_SPAN * clamp(u,-3,2)), 1e-5, 4.0)
axis_angle_to_matrix(v) = Rodrigues
```

속성 표준화 상수 (2.17M 점 실측):
```python
ATTR_MEAN = (-8.2933,-8.5553,-8.5972, 0.9118,-0.0164,-0.0319,-0.0113,
              0.7002, 0.1821, 0.1609,-0.0383)
ATTR_STD  = ( 1.5558, 1.4651, 1.6287, 0.1302, 0.2092, 0.2536, 0.2053,
              3.7138, 1.3148, 1.2916, 1.2338)
```
순서는 target 채널 3..14 (log_scale 3, quat 4, opacity 1, color 3) = 11개.

**LayerNorm 을 쓰면 안 됩니다.** 11채널은 비교 가능한 양이 아니고, 점별 평균을 빼면 그 점의 색·불투명도 오프셋이 함께 날아갑니다 — 실측 held-out R² 0.998 → 0.747.

---

## 4. 인코더 `PatchPackEncoder` (1,069,136 params)

### 레이아웃
```
group_size G        = 64
num_groups          = ceil(262144 / 64) = 4096
tokens_per_group T  = 8       (--local_tokens_per_group)
latent_channels C   = 32
patch_dim           = G * 4   = 256
token_dim           = T * C   = 256
raw_cells           = 256*128 = 32768   (--latent_hw 256 128)
group_in            = 589824 / 4096 = 144
```

### 4.1 `pooler` = `GroupPointPooler` (743,552) — group_in(144) > G(64) 이므로 활성

```python
in_dim  = 3 + attr_pack_dim = 3 + 11 = 14
d       = 128 (--pool_dim),  n_q = 16 (--pool_queries), heads = 4
embed   = MLP([14, 128, 128], last_act=True)
q       = Parameter(randn(1,16,128) * 0.02)
attn    = CrossAttentionBlock(128, heads=4)
out     = Linear(16*128 -> 256);  weight ~ N(0, 1e-2), bias = 0
```

forward (`_build_patch_pooled`):
```python
loc  = (xyz_g - centroid) * mask          # centroid: group_anchor 또는 표본 중심
attr = (x[..., 3:14] - ATTR_MEAN) / ATTR_STD
feats = cat([loc, attr])                  # (B, 4096, 144, 14)
patch = pooler(feats, mask)               # 그룹축 512개씩 청킹 (--pool_chunk 512)
```

### 4.2 `pack` (65,792) — Linear(256 -> 256), identity 초기화, **학습 가능**

```python
init: weight = 0, bias = 0, weight[:256,:256] = I
--pack_trainable  ->  requires_grad = True
```
동결하면 인코더 기하 경로에 학습 파라미터가 0개가 되고, 실측 `‖dp‖`가 모든 스텝에서 정확히 0이 됩니다.

### 4.3 `residual` (200,718) + `residual_gate`

```python
residual = residual_head([256 + 7, 512, 512, 256])   # --encoder_residual_hidden 512
  마지막 층 weight = 0, bias = 0
residual_gate = Parameter(full((1,), 1e-2))          # ★ 0 이 아님

z_tok = pack(patch) + tanh(residual_gate) * residual(cat([patch, centroid, extent, count/64]))
```

**`residual_gate`를 0으로 두면 정확한 zero-gradient 안장점입니다** — `d/d(residual weights) = tanh(0)*(...) = 0`, `d/d(gate) = sech²(0)*⟨upstream, residual(feat)⟩ = 0` (∵ `residual(feat) = 0`). 어떤 lr에서도 영원히 안 움직입니다. 실측 4,000스텝 후 gate·weight·bias 모두 정확히 0.0.

### 4.4 `attr_encoder` = `GroupAttributeEncoder` (59,073) — **동결**

pooler가 활성이면 `_build_patch_pooled`가 점에서 직접 patch를 만들므로 호출되지 않습니다. 체크포인트 호환을 위해 만들어 두고 `requires_grad_(False)`.

(구조: `embed = MLP([11,64,64])`, `slot_emb = randn(1,64,64)*0.02`, `mix = SelfAttentionBlock(64,heads=4)`, `out = Linear(64,1)` with `weight ~ N(0,1e-2)`)

### 4.5 해석적 앵커 (`patch_anchors`) — 파라미터 없음

```python
cnt      = mask_g.sum(-1)
sample_c = (xyz_g * mask).sum(2) / max(cnt,1)
var      = ((xyz_g - sample_c)² * mask).sum(2) / max(cnt,1)
centroid = group_anchor 또는 sample_c        # M4: --use_fixed_anchor_center 0 -> sample_c
extent   = sqrt(clamp(var, 0))
valid    = cnt > 0
```

### 4.6 출력

```python
z_map = tokens_to_map(z_tok)   # (B, 4096, 256) -> (B, 32768, 32) -> (B, 32, 256, 128)
map_blocks = Sequential()      # --map_blocks 0, 비어 있음
return z_map, anchors, patch
```

`z_raw`는 (B, 32, 256, 128) = **1,048,576개**입니다. 병목이 아닙니다.

---

## 5. 압축기 `StagedCompressor` (17,038,624) → z_compact

```
model_dim d = 448 (--model_dim),  heads = 8
compact_cells = 64*64 = 4096  (--compact_latent_hw 64 64)
merge M = ceil(4096/4096) = 1        -> 셀 1개 = 그룹 1개 = 앵커 1개
per_group = 32 / 1 = 32
예산: centroid 4 | occupancy 1 | shape 19 | appearance 8
```

### 5.1 forward 순서

```python
tokens = pad_groups(map_to_tokens(z_map))      # (B, 4096, 8, 32)

# pooled_input(=True) 이므로 extent 정규화는 SKIP
#   pooler 가 있으면 patch 는 계량적 의미가 없는 코드라, 854배 변동하는 양으로
#   나누면 정규화가 없애려던 스케일 변동을 오히려 주입합니다

x = token_embed(tokens)                        # Linear(32 -> 448),  14,784
x = x + token_pos                              # Parameter(1,1,8,448) ~ N(0,0.02)
for blk in intra_blocks:                       # SelfAttentionBlock x 3, 7,242,816
    x = blk(x, key_padding_mask = ~tok_valid)  #   그룹 내 8토큰

group_vec = token_merge(x.reshape(B,4096,8*448))
#   MLP([3584, 896, 448], hidden_norm=True)     3,615,808
#   --compress_merge_stages 0 -> 중간 4d 단계 없음
#   --compress_merge_attn False -> attention pooling 미사용

cell_vec = cell_mix(group_vec)                 # MLP([448,448,448], last_act) 402,304
cell_vec = window_stack(window_blocks, cell_vec, valid=cell_valid)
#   WindowSelfAttention x 2, window 8, 홀수층 shift, (64,64) 격자, 4,885,888

anchor_feat = cat([anchor_pe(centroid), extent, count/64])
#   anchor_pe = FourierFeatures(3, num_freqs_slot, include_input=True)
gfeat = group_head(cat([group_vec, cell_vec, anchor_feat]))
#   MLP([448*2 + pe_dim + 4, 448, 448], last_act)   616,896

mid = mid_net(gfeat)                           # 448 -> 256 -> 256, 181,760
#   Sequential(Linear, LayerNorm, LeakyReLU(0.1), Linear, LayerNorm, LeakyReLU(0.1))

n_used = count.sum(-1)
gstat  = [log1p(n_used)/log1p(262144),  clamp(n_used/262144, 0, 1)]
mid    = mid + global_to_mid(gstat)            # Sequential(Linear(2,256), LeakyReLU, Linear(256,256))
#   마지막 층 zero-init.  66,560
```

### 5.2 예산 헤드 (`--compress_head_stages` 비어 있음 → 전부 단일 Linear)

```python
cen_res = head_centroid(mid)                   # Linear(256, 4),  1,028
u_scale = encode_scale(extent).unsqueeze(-1)
s       = decode_scale(u_scale)
cen     = centroid + tanh(cen_res[...,:3]) * s * CENTROID_DELTA     # 0.05
cen     = cat([cen, u_scale, cen_res[...,4:]])                      # 4채널 (ch3 = 그룹 스케일)

occ_res    = head_occ(mid)                     # Linear(256, 1),  257
occ_anchor = 2*count/64 - 1
occ        = occ_anchor + tanh(occ_res) * 0.1                       # 1채널

shape      = head_shape(mid)                   # Linear(256, 19), 4,883
appearance = head_appear(mid)                  # Linear(256, 8),  2,056

pg        = cat([cen, occ, shape, appearance])                      # 32채널
z_cells   = pg.reshape(B, 4096, 32)
z_compact = 셀 -> 격자 배치 후 (B, 32, 64, 64)
```
`shared_free_head`, `structured_local_code`, `compress_wide_*` 는 모두 OFF입니다.

**z_compact = 32 × 64 × 64 = 131,072개.** Gaussian 262,144개당 0.5개 숫자.

앵커 채널(centroid 4 + occupancy 1 = 5개)은 물리량이라 극히 민감합니다 — 실측 1% 노이즈에 −0.97 dB. 자유 27채널은 30% 노이즈에 −0.19 dB.

### 5.3 aux 출력
```python
{"kl": 0, "cell_valid", "group_valid", "n_used", "empty_frac", "global_cond"}
```
`--latent_mode deterministic` 이므로 `head_logvar` 없음, KL = 0.

---

## 6. 복원기 `StagedDecompressor` (13,576,060)

### 6.1 앵커 읽기
```python
pg       = z_compact -> (B, 4096, 32)
centroid = pg[..., 0:3]
scale    = decode_scale(pg[..., 3:4])
count    = clamp((pg[...,4] + 1) * 0.5 * 64, 0, 64)
shape    = pg[..., 5:24]      # 19
appear   = pg[..., 24:32]     # 8
```

### 6.2 문맥 경로
```python
feat     = cat([pg, anchor_pe(pg[...,0:3])])
g        = group_embed(feat)                   # MLP, 229,942
cell_vec = cell_mix(g)                         # 402,304
cell_vec = window_stack(window_blocks, cell_vec)          # x2, 4,885,888
hh       = cat([cell_vec.unsqueeze(2), g])
for blk in intra_blocks: hh = blk(hh)          # x3, 7,242,816
cell_vec, g = hh[:,:,0], hh[:,:,1:]
deep     = token_out(cat([g, cell_vec]))       # -> token_dim 256,  519,488
```

### 6.3 Folding — 위치가 만들어지는 곳

```python
h     = cat([shape, ctx_tok])                  # ctx_tok = g (448)
frame = fold_head(h)                           # -> 6채널,  60,678
#   --folding_frame_channels 6 : aniso 3 + axis-angle 3

# 비등방성, log-ratio 상한
cap   = 1.5                                    # --folding_aniso_log_cap
la    = frame[...,:3] - mean(frame[...,:3])
aniso = exp(tanh(la / cap) * cap)               # 한 축 최대 e^1.5 = 4.5배

local = fibonacci_ball(64)                      # 고정 버퍼, 단위 공을 채우는 저불일치 점
local = local + bounded_residual(shape_xyz(h).reshape(B,4096,64,3))
#   shape_xyz: -> 192채널,  234,944
#   bounded_residual(r) = tanh(r / 1.0) * 1.0 * folding_res_gain
#     --folding_res_cap 1.0,  gain 은 스케줄 (§8.3)

local = local - local.mean(dim=2, keepdim=True) # ★ 평균 제거 (아래 참조)
local = local * aniso
local = R(frame[...,3:6]) @ local               # axis_angle_to_matrix

direct_xyz  = local * scale
deep_xyz    = bounded_residual(deep[..., :192].reshape(B,4096,64,3))
learned_xyz = (local + deep_xyz) * scale
```

**평균 제거가 필수인 이유**: 템플릿은 중심에 있지만 학습된 잔차는 아니고, 그 평균이 그룹 centroid에 얹힙니다. `residual_pack`에서 pack 타깃이 `xyz - centroid`이고 두 chamfer 항이 centroid 제거이므로 **어떤 활성 손실도 이걸 감독하지 않습니다.** 실측 centroid rmse가 step 500→2000에 0.00113 → 0.00374 → 0.00719 (제곱오차의 0.7% → 17.7%).

**`aniso` 상한이 필수인 이유**: 기하평균 정규화는 곱은 고정하지만 비율은 고정하지 않습니다. 실측 aniso 0.027~1411.5, offset 크기 0.620 → 7.976, effective rank 5.80 → 4.24.

**잔차 상한이 필수인 이유**: 무제한이면 192출력 잔차 헤드가 6출력 frame 헤드를 앞질러 비등방성을 자기가 재현합니다 — step 500에 lam2 0.999 → 0.243 (GT 0.579), frame 헤드는 zero init 그대로.

### 6.4 shortcut
```python
shortcut_xyz = zeros                            # residual_pack=True
shortcut_mask = clamp(count - slot_index, 0, 1)
tok = learned + alpha * shortcut,  alpha = 0.0  # --shortcut_alpha_start 999999999
```
alpha=0이 "모든 점을 셀 centroid에 두기"라는 공짜 답을 막습니다.

### 6.5 ctx 출력
```python
{"cell_vec", "group_vec": g, "appearance": appear,
 "attr_code": cat([shape(19), appear(8)]),     # ★ --attr_read_shape 1
 "shared_code": pg[...,5:],
 "centroid", "count", "scale",
 "res_ratio", "direct_frac", "direct_ratio",   # 진단용
 "learned_xyz", "direct_xyz", "shortcut_alpha",
 "fold_frame", "fold_local"}
z_raw_hat = tok -> (B, 32, 256, 128)
```

---

## 7. 디코더 2개

### 7.1 `CodecDecoder` (49,680,989) — 기하

```
decoder_layers = 10, model_dim 448, heads 8, patch_chunk 512
```
```python
patch  = unpack(tok)                            # Linear(256,256), 65,792
coarse = patch[..., :192].reshape(-1,64,3) + centroid       # residual_pack 복원
mask_v = patch[..., 192:256]

mem_tok = mem_proj(tok)                         # Linear(256, 8*448), 14,784
memory  = mem_tok.reshape(-1, 8, 448)
if decoder_neighbor_context and decoder_neighbor_group_tokens:
    memory = cat([memory, neighbor_group_tokens])           # 격자 이웃

offset = arange(64)/63
q = slot_query(offset_pe(offset))               # MLP, 205,632
q = q + slot_embedding.weight                   # Embedding(64, 448), 28,672
q = q + mem_tok.mean(2)
q = q + q_xyz(xyz_pe(coarse))                   # MLP, 219,072

h = q
for sa, ca in zip(self_blocks, blocks):         # 각 10개, 24.1M + 24.2M
    h = sa(h, key_padding_mask = mask_v <= 0.5)
    h = ca(h, memory)

residual = tanh(xyz_residual(cat([h, xyz_pe(coarse)])))     # 423,889
xyz = coarse + residual * scale * 0.6 * refine_alpha        # --residual_scale 0.6

presence = (mask_v - 0.5)*8 + presence_head(h) + (occ - 0.5)*4
#   occ = clamp(count - slot_index, 0, 1)
#   count 사전값이 없으면 padded frame 에서 "전부 존재"로 붕괴합니다
```

`_attributes`의 속성 헤드들(`attr_mlp`, `head_scale/rot/opacity/color`)은 **`attr_decoder_layers > 0` 이면 죽은 코드**입니다. `attr_pred`가 렌더·속성손실·eval·배포가 읽는 것이므로. 체크포인트 호환으로 남겨둡니다.

`checkpoint_decode` 사용 시 `use_reentrant=False`.

### 7.2 `AttributeDecoder` (3,624,720) — 속성

```
attr_decoder_dim d = 256, attr_decoder_layers = 4, heads 8
c_app = 8, c_shape_read = 19, c_code = 27
attr_cond = xattn, n_code_tok = 4, attr_nbr_window = 1
```
```python
xyz_pe = FourierFeatures(3, num_freqs_xyz, include_input=True)
loc_pe = FourierFeatures(3, num_freqs_xyz, include_input=True)     # --attr_local_pe 1
slot_emb = Embedding(64, 256);  weight ~ N(0, 0.02)                # 16,384
to_code_tok  = Linear(8,  4*256)                                   # 9,216
to_shape_tok = Linear(19, 4*256);  weight = 0, bias = 0            # 20,480  ★ zero-init
nbr_emb  = Parameter(randn(1, 2*1+1, 1, 256) * 0.02)
xattn    = MultiheadAttention(256, 8, batch_first=True)            # 263,168
xnorm    = LayerNorm(256)
in_proj  = MLP([256 + pe_dim, 256, 256], last_act=True)            # 151,552
blocks   = SelfAttentionBlock(256, heads=8) x 4                    # 3,159,040
head_scale/rot/opacity/color/nudge = Linear(256, 3/4/1/3/3)
scale_base_a = Parameter(tensor(0.7872))
```

forward (그룹 512개씩 청킹):
```python
tok = slot_emb.weight                                      # (64, 256)
shp   = code[..., :19];  app_o = code[..., 19:27]
feats = [tok, xyz_pe(p)]
loc   = p - p.mean(1, keepdim=True)
loc   = loc / clamp(mean(||loc||), 1e-6)
feats.append(loc_pe(loc))
h  = in_proj(cat(feats))

ct = to_code_tok(app_o) + to_shape_tok(shp)                # (W, 4, 256)
ct = ct + nbr_emb                                          # 이웃 순서 표시
h  = h + xattn(xnorm(h), ct, ct)
for blk in blocks: h = blk(h)

out = [ p + tanh(head_nudge(h)) * 0.15 * scale,            # --attr_nudge_cap 0.15
        bounded_log_scale(head_scale(h), scale),
        normalize(head_rot(h)),
        head_opacity(h),
        head_color(h) ]
```

**`bounded_log_scale` — 그룹 상대 비대칭 밴드**
```python
down = up = 3.0                                # --attr_scale_cap_down/_up
b    = head_scale.bias
g    = log(clamp(group_scale, 1e-6))
base = scale_base_a * g + b                    # 0.7872 초기값
r    = raw - b                                 # ★ base 가 아니라 bias 기준
r    = where(r < 0, down*tanh(r/down), up*tanh(r/up))
return base + r
```
`r = raw - b` 여야 하는 이유: `raw = W h + b` 이므로 `raw - base = W h - a·log(extent)`. zero init(W≈0)에서 extent<1인 모든 그룹이 상한에 포화하고 모든 Gaussian이 e^1.5배 크게 시작합니다. `raw - b = W h`는 init에서 0.

기준선은 실측 최소제곱 적합입니다: `median log_scale ≈ 0.787·log(extent) − 2.559`, R² = 0.759 (5 스냅샷 × 4096 그룹).

`xattn.out_proj`를 zero-init 하면 안 됩니다 — 이 코드베이스에서 같은 제로곱 안장점을 세 번 잡았습니다.

### 7.3 두 디코더 연결 (`model.forward`)

```python
z_raw, anchors, patch = encode(enc_x, enc_mask, group_anchor=None)   # use_fixed_anchor_center 0
z_compact, aux        = compress(z_raw, anchors)
z_raw_hat, ctx        = decompressor(z_compact, shortcut_alpha=0.0)

pred, presence = decoder(z_raw_hat, ctx["cell_vec"], ctx["scale"],
                         centroid=ctx["centroid"], count=ctx["count"],
                         group_vec=ctx["group_vec"], attr_xyz=attr_xyz,
                         appear=ctx["appearance"])

geo = pred[..., 0:3]
if attr_detach_geometry:  geo = geo.detach()          # 스케줄, step 6000 에 해제
if attr_xyz is not None and p_tf > 0:                 # teacher forcing, 점별 혼합
    geo = where(rand(B,N,1) < p_tf, attr_xyz[...,0:3], geo)

attr_pred = attr_decoder(geo, ctx["attr_code"], ctx["scale"].detach())["pred"]
```

**`appear`(=`attr_code`)를 detach 하면 안 됩니다.** 과거 이 detach 하나가 appearance가 latent에 못 들어간 이유였습니다 — 실측 `w_render_attr`가 attr_decoder에 L2 128.5를 주고 encoder/compressor/decompressor에는 정확히 0. 그 코드에서 그룹 평균 속성을 복원하는 held-out R²가 pack에서 −0.947.

`ctx["scale"]`은 detach 합니다. `geo`는 스케줄에 따릅니다.

---

## 8. 손실 · 스케줄 · 옵티마이저

### 8.1 활성 손실과 실측 기여도 (정상 상태)

| 항 | 값 | 실효 w | 기여 | 비중 |
|---|---|---|---|---|
| `render_attr` | 0.110 | 60 | 6.60 | 24% |
| `intra_chamfer` | 6.96 | 0.8 | 5.57 | 20% |
| `group_centroid` | 10.96 | 0.5 | 5.48 | 20% |
| `intra_sinkhorn` | 3.79 | 1.0 | 3.79 | 14% |
| `opacity` | 1.76 | 0.5 | 0.88 | 3% |
| `scale` | 0.99 | 0.5 | 0.50 | 1.8% |
| `color` | 0.60 | 0.5 | 0.30 | 1.1% |
| `splat_area` | 0.042 | 4.0 | 0.17 | 0.6% |
| `rot` | 0.098 | 0.25 | 0.025 | 0.1% |
| `chamfer` | 0.0035 | 5.0 | 0.018 | 0.1% |
| `coverage` | 0.0022 | 2.0 | 0.0045 | 0.02% |
| `latent_decorr` | 0.101 | 0.05 | 0.005 | 0.02% |
| `shape_sinkhorn` | — | 1.5 | — | — |
| `latent_std` | — | 1.0 | — | — |
| `presence` | — | 0.5 | — | — |
| `equiv` | — | 0.5 (step 4000+) | — | — |

`scale/rot/opacity/color`의 실효 가중치는 명목값 × `attr_param_mult` (= floor 0.5)입니다.

**`chamfer`(0.1%), `coverage`(0.02%), `latent_decorr`(0.02%), `rot`(0.1%)은 실질적으로 참가하지 않습니다.** 절대 정규화 씬 단위(중앙값 거리 ~0.002)와 extent 정규화 단위(~10)가 숫자로 ~3000배 차이 나기 때문입니다. **M4를 재현하려면 이 상태를 그대로 두어야 합니다.**

### 8.2 각 항의 정의

```python
masked_reduce(l, m)     = (l*m).sum() / (m.sum() + 1e-8)
masked_smooth_l1(p,t,m) = masked_reduce(smooth_l1(p,t,beta=0.02), m)

symmetric_chamfer(p, g):
    i_pg = argmin cdist(p.detach(), g.detach())      # no_grad
    i_gp = argmin cdist(g.detach(), p.detach())
    a = sqrt(((p - g[i_pg])**2).sum(-1) + 1e-12).mean()
    b = sqrt(((g - p[i_gp])**2).sum(-1) + 1e-12).mean()
    return 0.5*(a + b)
    # cdist 는 backward 용 N x M 을 저장하므로 argmin 만 no_grad 로 구하고
    # 거리 하나만 미분해야 합니다 (16k 샘플에서 방향당 ~1 GB)

multiscale_chamfer:  scales (4096, 16384, 65536), weights (0.2, 0.3, 0.5)
    각 스케일에서 GT·pred 를 각각 공간 균형 샘플링(32³ bins) 후 symmetric_chamfer
    ★ 같은 슬롯 인덱스를 쓰면 집합 거리가 아니라 대응점 비교가 됩니다

coverage_onesided_loss:  samples 16384, bins 32
    GT -> pred 단방향 최근접 거리 평균

intra_group_chamfer(pred, target, mask, 64, scale, chunk_groups=1024):
    그룹별로 pred/target 을 그룹 extent 로 나눈 뒤 (하한 = 배치 중앙값의 0.05)
    64x64 pairwise, 양방향 최근접 거리 평균
    return_parts=True 로 p2g / g2p / radius 도 계산 가능
    ★ M4 에서는 p2g / radius 가 codec 브랜치에 배선되지 않았습니다 (gen 전용)

intra_group_sinkhorn_loss:  epsilon 0.08, iterations 6, chunk 256
    같은 정규화 후 balanced Sinkhorn OT

group_centroid_loss:  그룹 중심 오차 / 그룹 extent

sinkhorn_parameter_target(pred_xyz, target, mask, 64, scale, eps=0.08, iters=6):
    cost = cdist(pred_xyz/scale, target_xyz/scale)      # ★ 위치만
    balanced Sinkhorn -> 그 수송계획으로 GT 속성 전체를 전달
    ★ 위치만 쓰므로, 같은 자리의 서로 다른 GT Gaussian 은 혼합값이 됩니다

_attr_losses:  scale/rot/opacity/color 각각 masked L1 (rot 은 quat_loss)
    가중치 a_pw = mask * where(render_visible, attr_anchor_covered=0.25, 1.0)

balanced_presence_loss(logits, mask):
    BCE, 양성/음성 평균을 0.5:0.5 로 합성

render_loss(...):
    양쪽을 같은 카메라·같은 래스터라이저로 렌더
    pred: idx_p = presence > 0 (--render_presence 1), target: GT mask
    ref = render(target Gaussians) 이지만, 실제 사진이 있으면 그것으로 대체
    시점: 프레임 자기 시점 1 + extra_real_views 4 + 합성 jitter (views 5)
    손실 = (1-lam)*L1 + lam*(1-SSIM),  lam_dssim 0.2
    커버리지 < 0.25 인 프레임은 건너뜀 (crop 으로 거의 검은 프레임이 되는 경우)
    splat_area: 투영 면적의 분위 0.80 초과분 힌지 (--splat_area_quantile 0.80)
    detach_xyz = False   (별도 attr_decoder 가 있으므로)

latent_decorr:  자유 27채널의 상관행렬 off-diagonal 제곱 평균
latent_std:  자유채널 std 의 [0.35, 3.0] 밴드 밖 벌점
equiv:  씬을 10° 회전시켜 latent 도 같이 회전해야 함, 4스텝마다 (--equiv_every 4)
```

### 8.3 스케줄 (`step`은 0부터)

```python
phase:  polish_start=1 이므로 step>=1 은 전부 "polish"
        (latent_end 0, geo_end 0)

m_geo = 1.0,  m_detail = ramp(step, 0, 100)        # --detail_ramp_steps 100

w_render_attr  = 60 * ramp(step, render_start=0, render_ramp_steps=400)
w_equiv        = 0.5 if step >= 1200 else 0        # equiv_start 1200, ramp 800
attr_param_mult= ramp(step, 0, 1000, 1.0 -> 0.5)   # floor 0.5, decay_steps 1000
                 -> scale/rot/opacity/color 에 곱

render_downscale = round(exp(log(8) + t*(log(2)-log(8))))
                   t = ramp(step, 0, 4000)         # 8 -> 2, 4000 스텝
folding_res_gain = ramp(step, 0, 400)              # 0 -> 1
decoder_refine_alpha = ramp(step, 400, 800)        # 0 -> 1
attr_teacher_prob    = ramp(step, 0+0, 1, 1.0 -> 0.0)
                       # attr_start 0, force_steps 0, anneal_steps 1
                       # -> step 0 에서만 1.0, 이후 0. 사실상 teacher forcing 없음
attr_detach_geometry = (step < 6000)               # --attr_detach_release 6000
shortcut_alpha       = 0.0
run_gen              = False                       # --no_gen_branch
```

`ramp(step, start, length, lo=0, hi=1) = lo + (hi-lo) * clamp((step-start)/length, 0, 1)`

### 8.4 옵티마이저

```python
AdamW(param_groups, weight_decay=1e-4)
grad_clip = 1.0   (clip_grad_norm_ 후 step)
amp = bf16
batch_size = 1,  workers = 6

global_lr(step):
    step < 200:  5e-5 * (step+1)/200
    else:        5e-6 + 0.5*(5e-5 - 5e-6)*(1 + cos(pi * (step-200)/(20000-200)))
```

파라미터 그룹과 polish 배율:

| 그룹 | 모듈 | 배율 | 실효 lr @step 5000 |
|---|---|---|---|
| `encoder` | encoder (attr_encoder 제외) | 0.1 | 4.4e−6 |
| `attr_enc` | encoder.attr_encoder | 0.5 | (동결, 그룹 비어 있음) |
| `compressor` | compressor | 0.3 | 1.3e−5 |
| `decompressor` | decompressor | 0.3 | 1.3e−5 |
| `decoder` | decoder | 0.5 | 2.2e−5 |
| `attr` | attr_decoder | 1.0 | 4.4e−5 |

**`attr_enc`는 반드시 `encoder`와 별도 그룹**이어야 합니다 — train.py가 residual 분기가 꺼져 있으면 `encoder` 그룹 lr을 0으로 만들고, 그게 appearance 인코딩 모듈을 동결시켜서는 안 됩니다.

**orphan 검사 필수**: 학습 가능한데 어떤 그룹에도 없는 파라미터가 있으면 `RuntimeError`. attr_decoder 3.325M이 런 전체에서 초기값에 앉아 있었고 유일한 증상이 "체크포인트 간 eval이 소수 셋째 자리까지 동일"이었습니다.

### 8.5 평가

```
eval_every 1000
eval_milestones 100,250,500,1000,2000,4000,6000,8000,12000,16000,20000
eval_val_indices 0,4,7,31,63,95,127,159
eval_view_count 8, eval_view_seed 1234
score_metric psnr   ->  score = -psnr (클수록 좋으므로 부호 반전)
save_every 2000
```

`run_eval`은 **반드시 `enc_x`/`enc_mask`를 넘겨야 합니다** — 안 넘기면 pooler eval이 학습과 다른 그룹화를 측정합니다 (rmse 0.675 vs 실제 0.0096).

---

## 9. 실행 명령과 기대 결과

### 9.1 M4는 워밍스타트입니다

```
--init_from runs/R8I_20260818_035640/ckpt_step00005000.pt
--init_skip __none__
```
R8I step 5000에서 출발했고, 그 체크포인트는 R8H에서, R8H는 R8에서 왔습니다. **완전 from-scratch로 20,000스텝을 돌리면 같은 결과가 나오지 않습니다.**

`--init_skip __none__`은 센티넬입니다 — 이름이 어떤 파라미터의 접두사도 아니므로 아무것도 건너뛰지 않습니다. (기반 런처가 `--init_skip attr_decoder.head_scale`을 들고 있어 이를 무효화해야 했습니다. 안 하면 학습된 scale 헤드를 버려 step 100 PSNR이 16.35 → 11.03이 됩니다.)

### 9.2 명령

기본값과 다른 인자 129개 전부:

```bash
torchrun --standalone --nproc_per_node=1 --master_port=29522 train.py \
 --root .../replay_seg --out_dir runs/M4 \
 --split_path assets/replay_seg_2700_300_seed42.json \
 --stats_path assets/stats_replay_seg.json --num_val_files 300 \
 --min_snapshot_step 12000 \
 --max_points 262144 --max_input_points 589824 --drop_outside \
 --sample_mode stratified --crop_prob 0.0 --density_aware_sample \
 --no_slot_redistribute --partition_mode morton --partition_block 32 \
 --slot_sort template --cache_slots \
 --scene_anchors assets/scene_anchors.npy --anchor_assignment spill --anchor_spill 8 \
 --use_fixed_anchor_center 0 \
 --augment --aug_rot_deg 8.0 --aug_scale_jitter 0.03 --aug_shift 0.01 \
 --workers 6 --batch_size 1 --chunk_size 64 --group_size 64 \
 --local_tokens_per_group 8 --latent_channels 32 --latent_hw 256 128 \
 --compact_latent_channels 32 --compact_latent_hw 64 64 \
 --budget_centroid 4 --budget_occupancy 1 --budget_shape 19 --budget_appearance 8 \
 --attr_pack_dim 11 --attr_pack_hidden 64 --pool_dim 128 \
 --attr_decoder_layers 4 --attr_decoder_dim 256 --attr_cond xattn --attr_local_pe 1 \
 --attr_scale_cap_down 3.0 --attr_scale_cap_up 3.0 --attr_nudge_cap 0.15 \
 --attr_nbr_window 1 --attr_read_shape 1 \
 --model_dim 448 --heads 8 \
 --compress_intra_layers 3 --compress_window_layers 2 --compress_window 8 \
 --compress_mid_channels 256 \
 --decompress_intra_layers 3 --decompress_window_layers 2 \
 --decoder_layers 10 --residual_scale 0.6 --patch_chunk 512 \
 --folding_decode --folding_res_cap 1.0 --folding_frame_channels 6 \
 --folding_aniso_log_cap 1.5 --folding_res_start 0 --folding_res_ramp_steps 400 \
 --no_gen_branch --stage geometry \
 --latent_end 0 --late_decode_steps 0 --geo_end 0 --geo_weak_scale 0.35 \
 --gen_start 999999999 --gen_end 999999999 --late_codec_start 999999999 \
 --encoder_residual --encoder_residual_start 0 \
 --decoder_refine_start 400 --decoder_refine_ramp_steps 800 \
 --shortcut_alpha_start 999999999 --pack_trainable \
 --shared_free_head 0 --structured_local_code 0 \
 --joint_shared_decoder 0 --joint_direct_decoder 0 \
 --w_chamfer 5.0 --w_coverage 2.0 --w_intra_chamfer 0.8 --w_intra_sinkhorn 1.0 \
 --w_group_centroid 0.5 --w_shape_sinkhorn 1.5 \
 --sinkhorn_epsilon 0.08 --sinkhorn_iterations 6 --sinkhorn_chunk 256 \
 --w_presence 0.5 --chamfer_scales 4096,16384,65536 \
 --chamfer_scale_weights 0.2,0.3,0.5 --balanced_chamfer \
 --intra_chamfer_chunk 1024 --detail_ramp_steps 100 \
 --w_scale 1.0 --w_rot 0.5 --w_opacity 1.0 --w_color 1.0 --w_sh 0.0 \
 --attr_start 0 --attr_force_steps 0 --attr_anneal_steps 1 \
 --attr_param_decay_steps 1000 --attr_param_floor 0.5 \
 --attr_responsibility 1.0 --attr_anchor_covered 0.25 \
 --attr_detach_geometry 1 --attr_detach_release 6000 \
 --attr_match_mode sinkhorn \
 --photo_map assets/npz_to_image.json --view_pool assets/view_pool.json \
 --extra_real_views 4 --render_views 5 --view_downscale 2 \
 --w_render 0 --w_render_attr 60.0 --w_splat_area 4.0 --splat_area_quantile 0.80 \
 --render_presence 1 --render_start 0 --render_ramp_steps 400 \
 --render_downscale 2 --render_downscale_start 8 --render_downscale_steps 4000 \
 --render_lam_dssim 0.2 --render_min_coverage 0.25 \
 --w_z_raw 0 --w_z_raw_mse 0 --w_z_hard 0 --w_z_token_var 0 --w_z_std_ratio 0 \
 --w_z_residual 0 --w_z_residual_hard 0 --w_z_intra_chamfer 0 \
 --w_learned_residual 0 --w_shape_direct 0 \
 --w_plane_chamfer 0 --w_proj_hist 0 --w_dispersion 0 --w_voxel_occ 0 \
 --w_teacher_cycle 0 --w_res_ratio 0 --w_kl 0 \
 --w_latent_std 1.0 --latent_std_floor 0.35 --latent_std_ceil 3.0 \
 --w_latent_decorr 0.05 \
 --w_equiv 0.5 --equiv_start 1200 --equiv_ramp_steps 800 --equiv_every 4 \
 --equiv_rot_deg 10.0 --equiv_shape_weight 0.0 \
 --lr 5e-5 --lr_min 5e-6 --lr_warmup_steps 200 --weight_decay 1e-4 \
 --grad_clip 1.0 --amp bf16 --seed 42 --encoder_warmup_steps 0 \
 --polish_start 1 --polish_encoder_scale 0.1 --polish_attr_encoder_scale 0.5 \
 --polish_compressor_scale 0.3 --polish_decompressor_scale 0.3 \
 --polish_decoder_scale 0.5 --polish_attr_scale 1.0 \
 --score_metric psnr --eval_view_count 8 --eval_view_seed 1234 \
 --max_steps 20000 --log_every 50 --eval_every 1000 --save_every 2000 \
 --eval_milestones 100,250,500,1000,2000,4000,6000,8000,12000,16000,20000 \
 --eval_val_indices 0,4,7,31,63,95,127,159 \
 --init_from <R8I ckpt_step00005000.pt> --init_skip __none__
```

### 9.3 기대 결과

**모델 크기 84,989,529 params.** 모듈별:

| 모듈 | params |
|---|---|
| encoder | 1,069,136 |
| compressor | 17,038,624 |
| decompressor | 13,576,060 |
| decoder (CodecDecoder) | 49,680,989 |
| attr_decoder | 3,624,720 |

**held-out PSNR 궤적** (GT 천장 18.70):

| step | PSNR | gap | SSIM |
|---|---|---|---|
| 100 | 12.65 | 6.06 | 0.286 |
| 250 | 13.61 | 5.09 | 0.320 |
| 500 | 14.64 | 4.06 | 0.357 |
| 1000 | 15.32 | 3.38 | 0.393 |
| 2000 | 16.21 | 2.50 | 0.449 |
| 4000 | 16.39 | 2.31 | 0.462 |
| 6000 | 16.24 | 2.46 | 0.477 |
| 8000 | 16.55 | 2.15 | 0.494 |
| 12000 | 17.36 | 1.34 | 0.526 |
| 14000 | 17.55 | 1.16 | 0.533 |
| 16000 | **17.75** | **0.96** | 0.543 |

- step 6000의 하락(−0.32)은 `attr_detach_geometry` 해제의 정상적 교란입니다.
- eval 간 변동 std는 0.44 dB이고 그 위에 +0.237 dB/1000스텝의 추세가 있습니다.
- 학습 loss는 씬 구성 때문에 14.4~42.5로 진동합니다. **정상입니다** — `corr(loss, n_points) = +0.45`이고, n 회귀로 분산의 20.3%가 설명되며, 잔차 std가 4.23 → 2.01로 **줄어듭니다**.

**렌더 분해** (step 16000, 4장면 평균):

| | PSNR |
|---|---|
| canon (선택된 K, GT 위치·속성) | 48.39 |
| codec_snap | 20.70 |
| codec (예측 위치 + GT 속성) | 19.12 |
| codec_own (실제 배포 출력) | 17.10 |
| nn_unique | 0.490 |

---

## 10. 재현 시 반드시 피해야 할 함정

이 코드베이스에서 **실제로 발생했고 조용히 실패했던** 것들입니다.

| # | 함정 | 증상 |
|---|---|---|
| 1 | 앵커를 증강과 함께 변환하지 않음 | 슬롯 81%가 빈다 |
| 2 | `scene_anchors.npy`가 world 좌표인 것을 모름 | 앵커가 −132..186 범위 |
| 3 | `run_eval`에 `enc_x`/`enc_mask`를 안 넘김 | eval rmse 0.675 vs 실제 0.0096 |
| 4 | `residual_gate`를 0으로 초기화 | 제로곱 안장점, 영원히 안 움직임 |
| 5 | `xattn.out_proj`를 0으로 초기화 | 같은 문제 (세 번째 발생) |
| 6 | `GroupAttributeEncoder.out`을 0으로 초기화 | 같은 문제 |
| 7 | `param_groups`에 모듈 추가를 잊음 | 3.325M이 초기값에 앉아 있고 유일한 증상이 "eval이 소수 셋째 자리까지 동일" |
| 8 | `--init_from`이 step을 0으로 리셋하는 것을 잊음 | 절대 step 게이트가 전부 닫혀 속성·렌더 손실이 꺼진 채로 수천 스텝 |
| 9 | `--init_skip`을 기본 런처에서 물려받음 | 학습된 scale 헤드를 버려 step 100에 −5.3 dB |
| 10 | 렌더 손실이 GT 마스크로 pred를 선택 | `presence`가 목적함수에 아예 안 들어감 |
| 11 | `aniso`를 무제한으로 둠 | 0.027~1411.5, offset 0.620 → 7.976 |
| 12 | folding 잔차의 평균을 제거하지 않음 | centroid rmse 0.00113 → 0.00719 |
| 13 | NN 조회로 슬롯↔npz 행 매핑 | 0.991만 단사 |
| 14 | `cdist` 전체를 미분 그래프에 둠 | 16k 샘플에서 방향당 ~1 GB |
| 15 | 속성 표준화에 LayerNorm 사용 | held-out R² 0.998 → 0.747 |
| 16 | pooler가 있는데 extent 정규화를 적용 | 없애려던 스케일 변동을 주입 |
| 17 | `multiscale_chamfer`에서 같은 슬롯 인덱스 사용 | 집합 거리가 아니라 대응점 비교가 됨 |
| 18 | DDP 2장 이상 | 유효 배치가 배수, 같은 스텝에서 결과 다름 |

---

## 부록 A. M4에 구현돼 있으나 꺼진 것

`gen_decoder`(`--no_gen_branch`), `joint_shared_decoder`, `joint_direct_decoder`, `structured_local_code`, `shared_free_head`, `compress_wide_channels/layers`, `compress_head_stages`, `compress_merge_attn`, `compress_merge_stages`, `teacher_cycle`, `w_cov3d`, `w_intra_spacing`, `w_p2g`, `w_radius`(codec), `latent_mode vae`.

## 부록 B. M4의 알려진 한계 (재현하면 같이 나옵니다)

| | 실측 |
|---|---|
| 커버리지 (`nn_unique`) | 49.0% — 26만개 중 서로 구별되는 것이 약 12.9만개 |
| 그룹 반지름 pred/GT | 0.670 (중심으로 수축) |
| 표면 이탈 거리 | 0.315 그룹반지름 |
| 그룹내 속성 다양성 | GT 대비 크기 9.7%, 방향 2.9%. 설명분산 −8.6%, 방향 일치도 0.1% |
| 투영 splat 반지름 | 모든 분위에서 GT의 0.40배 |
| latent effective rank | 8.7 / 32 |
| `opacity` nrmse | 1.054 (데이터셋 평균보다 나쁨) |
| `rot` nrmse | 1.096 (같음) |
| 300만개 → 26만개 추출 시 타깃 자체의 상한 | 11.11 dB |
