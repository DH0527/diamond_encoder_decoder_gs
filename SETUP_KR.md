# can3tok — 설치 · 데이터 · 학습 재현 가이드

3DGS 스냅샷(가우시안 최대 262,144개)을 **고정 크기 latent 16,384개 숫자**(`16 × 32 × 32`)로
압축하는 인코더-디코더입니다. latent 는 DIAMOND 월드모델의 입력으로 쓰기 위한 것이라
크기를 키우지 않는 것이 전제입니다.

이 문서는 다른 서버에서 **처음부터** 같은 결과를 내기 위해 필요한 모든 것을 적습니다.
원래 서버 기준 날짜는 2026-09-28 입니다.

---

## 0. 한눈에 보기

```
Tanks&Temples COLMAP (train 301장, truck 251장)
   │  ① 수정된 3DGS 로 30,000 iter 학습하면서 10 iter 마다 스냅샷 덤프     §4
   ▼
replay/step_000010.npz … step_030000.npz   씬당 3,000개, 합계 147 GB
   │  ② 정규화 통계 · k-means 앵커 1024 · split · 사진 매핑                 §5
   ▼
assets/*.json, *.npy
   │  ③ can3tok 학습 (GPU 3장 DDP, 80,000 step)                             §6
   ▼
z_compact ∈ R^{16×32×32}  →  디코더  →  가우시안 262,144개  →  렌더
```

**저장소에 있는 것**: 코드, 문서, 자산(`assets/`, 10 MB), 학습 설정 스냅샷(`configs/`),
3DGS 수정 패치(`data_pipeline/`), 환경 명세(`environment/`).

**저장소에 없는 것** (크기 때문에 서버에만 있음, §9): 체크포인트(`runs/`, 약 200 GB),
replay npz(147 GB), COLMAP 원본(2 GB), 분석 그림(`mdmd/*.png`, 220 MB).

---

## 1. 저장소 구성

| 경로 | 내용 |
|---|---|
| `can3tok/` | 모델 · 데이터 로더 · 손실 · 학습 루프 (패키지 본체) |
| `train.py` | 진입점 (`can3tok.train.main` 호출) |
| `configs/` | 재현용 학습 설정 스냅샷 (`args.json`) — §6.1 |
| `scripts/` | 실행 스크립트. 새 서버에서는 `launch_from_config.sh` 하나면 됨 |
| `tools/` | 자산 생성, 진단·오라클 도구, **경로 재배치 `relocate_assets.py`** |
| `assets/` | 정규화 통계, 앵커, split, 사진 매핑, view pool |
| `data_pipeline/3dgs_replay/` | Inria 3DGS 에 얹는 패치 + 추가 파일 (npz 덤프용) — §4 |
| `environment/` | conda / pip 명세 — §3 |
| `tests/` | 레이아웃 · 계약 단위 테스트 |
| `mdmd/` | 분석 문서 (그림 제외). 최신은 `17_detail_bottleneck_20260917.md`, `18_gradient_training_audit_20260925.md` |
| 루트 `*_KR.md`, `*.md` | 이전 단계 설계·진단 기록. `README.md` 본문은 옛 32×64×64 설계 설명임 |

---

## 2. 하드웨어와 용량

원래 서버:

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 5000 Ada Generation 32 GB × 4 (학습은 3장 DDP) |
| 드라이버 / CUDA | 580.82.09 / CUDA 12.1 툴킷 (`/usr/local/cuda-12.1`) |
| 학습 속도 | 약 4.7 ~ 5.4 s/step (3장, batch 1/GPU) → 80,000 step 에 약 4.5일 |
| GPU 메모리 | GPU 당 약 27 ~ 29 GB |

필요한 디스크:

| 항목 | 크기 |
|---|---|
| replay npz (vanilla train + truck) | 147 GB (train 50 GB, truck 95 GB) |
| COLMAP 원본 | 약 2 GB |
| 체크포인트 | 전체 상태 1.2 GB × `save_every` 마다, `ckpt_best.pt` 0.38 GB |

체크포인트를 1,000 step 마다 저장하면 80,000 step 에 약 96 GB 입니다. 원래 서버는
디스크 98% 까지 차서 중간 체크포인트를 지웠습니다. **`save_every` 를 늘리거나 정리
스크립트를 두는 것을 권합니다.**

---

## 3. 환경 설치

두 개의 conda 환경이 필요합니다.

| env | 용도 | Python | torch |
|---|---|---|---|
| `can3tok` | 인코더-디코더 학습 · 평가 | 3.11 | 2.4.1 + cu121 |
| `3dgs` | 3DGS 학습 + replay npz 덤프 | 3.10 | 2.2.0 + cu121 |

### 3.1 `can3tok` 환경

```bash
conda create -n can3tok python=3.11 -y
conda activate can3tok
pip install -r environment/requirements_can3tok.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

`requirements_can3tok.txt` 는 코드가 실제로 import 하는 것만 고정했습니다
(torch, torchvision, numpy, scipy, pillow, matplotlib, tqdm). 원래 서버의 전체 목록은
`environment/pip_freeze_can3tok_full.txt` 인데 다른 프로젝트 패키지가 섞여 있어
그대로 설치할 필요는 없습니다.

### 3.2 `3dgs` 환경

```bash
conda create -n 3dgs python=3.10 -y
conda activate 3dgs
pip install -r environment/requirements_3dgs.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### 3.3 CUDA 확장 빌드

래스터라이저는 PyPI 에 없으므로 소스에서 빌드합니다. 두 환경 모두 **같은 커밋**을
씁니다 (원래 서버에서 두 빌드 소스가 바이트 단위로 동일함을 확인했습니다).

| 확장 | 저장소 | 커밋 |
|---|---|---|
| diff-gaussian-rasterization | github.com/graphdeco-inria/diff-gaussian-rasterization | `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` (branch `dr_aa`) |
| simple-knn | gitlab.inria.fr/bkerbl/simple-knn | `86710c2d4b46680c02301765dd79e465819c8f19` |
| fused-ssim (3dgs 만) | github.com/rahul-goel/fused-ssim | `1272e21a282342e89537159e4bad508b19b34157` |

이 셋은 `graphdeco-inria/gaussian-splatting` 커밋 `54c035f` 가 서브모듈로 고정한 버전
그대로이므로, §4.2 에서 그 저장소를 받으면 함께 따라옵니다.

```bash
export CUDA_HOME=/usr/local/cuda-12.1          # torch 의 cu121 과 맞춘다
cd gaussian-splatting                          # §4.2 에서 받은 저장소

# can3tok 환경: 래스터라이저만 필요
conda activate can3tok
pip install ./submodules/diff-gaussian-rasterization

# 3dgs 환경: 셋 다
conda activate 3dgs
pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./submodules/fused-ssim
```

라이선스: 이 확장들과 Inria 3DGS 코드는 **Gaussian-Splatting License (비상업 연구용)**
입니다. 그래서 이 저장소에는 소스를 넣지 않고 커밋만 고정했습니다.

### 3.4 설치 확인

```bash
conda activate can3tok
python -c "import torch, diff_gaussian_rasterization; print(torch.__version__, torch.cuda.is_available())"
python -m tests.test_layout          # 레이아웃 / 계약 단위 테스트
```

---

## 4. 데이터 확보 — COLMAP → 3DGS → replay npz

### 4.1 원본 데이터

Tanks and Temples 의 **train**, **truck** 두 씬을 3DGS 의 `convert.py`(COLMAP)로 처리한
것입니다.

| 씬 | 원래 서버 경로 | 이미지 | 해상도 |
|---|---|---|---|
| train | `/data/daeho/train_colmap` | 301 장 | 977 × 544 |
| truck | `/data/daeho/truck_colmap` | 251 장 | 977 × 545 |

각 폴더는 `images/`, `sparse/0/{cameras,images,points3D}.bin` 을 갖는 표준 3DGS 입력
형식입니다. Inria 가 배포하는 전처리본(`tandt_db.zip`)을 쓰거나, 원본 프레임에
`python convert.py -s <scene>` 을 돌리면 됩니다.

### 4.2 수정된 3DGS 준비

원래 서버의 3DGS 는 **Inria 원본 커밋 `54c035f` (2024-10-30) 위에 커밋되지 않은 로컬
수정**을 얹은 것입니다. 그 수정분을 패치로 떠 두었고, 원본 커밋에 적용하면 서버의 수정본과
**바이트 단위로 같아지는 것을 확인**했습니다.

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting
git checkout 54c035f
git submodule update --init --recursive

# 수정 3개 파일 (train.py, scene/gaussian_model.py, utils/camera_utils.py)
git apply <can3tok>/data_pipeline/3dgs_replay/gaussian_splatting_54c035f.patch

# 새 파일 (replay 덤프 로직, 렌더 어댑터, 점검 도구)
cp -r <can3tok>/data_pipeline/3dgs_replay/overlay/* .
```

패치가 추가하는 것: `--log_every N`(N iter 마다 스냅샷), `--compact_npz`(float16 zip-npz
형식). 덤프 본체는 `replay_utils.py` 입니다.

### 4.3 npz 덤프

```bash
GS_ROOT=/path/to/gaussian-splatting \
PY=/path/to/envs/3dgs/bin/python \
OUT_ROOT=/path/to/vanilla-3dgs/output \
TRAIN_SRC=/path/to/train_colmap TRUCK_SRC=/path/to/truck_colmap \
bash data_pipeline/3dgs_replay/run_dump_compact_npz.sh
```

train 은 GPU 0, truck 은 GPU 1 에서 동시에 돕니다. 결과:

```
$OUT_ROOT/train/replay/step_000010.npz … step_030000.npz   (3,000개, 약 50 GB)
$OUT_ROOT/truck/replay/step_000010.npz … step_030000.npz   (3,000개, 약 95 GB)
```

### 4.4 npz 형식

한 파일 = 3DGS 학습의 한 시점 스냅샷 (zip 압축 npz).

| 키 | dtype | shape | 의미 |
|---|---|---|---|
| `it` | int32 | () | 3DGS iteration |
| `xyz` | float16 | (N, 3) | 월드 좌표 |
| `scaling` | float16 | (N, 3) | **선형** 스케일 (exp 적용 후) |
| `rot` | float16 | (N, 4) | 쿼터니언 |
| `opacity` | float16 | (N, 1) | **sigmoid 적용 후** [0, 1] |
| `color` | float16 | (N, 3) | SH DC 계수 |
| `ids` | uint32 | (N,) | 가우시안 영속 ID |
| `cam` | float32 | (16,) | 이 스냅샷의 카메라 벡터 |
| `image_name` | str | () | 그 카메라의 사진 이름 |
| `image_wh` | int32 | (2,) | 사진 크기 |
| `reward` | float32 | (8,) | 3DGS 학습 보상 신호 |

SH 고차항은 디스크 때문에 뺐습니다 (이 코덱은 DC 만 씁니다). 가우시안 수는 vanilla train
약 73만 ~ 78만, truck 약 137만 ~ 143만 개입니다.

**주의 — 규약이 로더 타깃과 다릅니다.** npz 는 `scaling` 이 선형, `opacity` 가 sigmoid
후입니다. 반면 `can3tok` 로더가 만드는 타깃은 log-scale, logit-opacity 입니다
(`io_utils.gaussian_target_channels`). npz 를 직접 렌더할 때 `exp()` 나 `sigmoid()` 를
한 번 더 씌우면 그림이 완전히 깨집니다 (실제로 겪었습니다: 23.8 dB 가 9.9 dB 로 보임).

`load_npz_state()` 가 이 형식과 옛 형식(`it / state_t / meta`)을 모두 읽습니다.

### 4.5 Speedy-Splat 데이터 (이전 실험)

9월 21일 이전 실험(T16k, C1 등)은 가지치기를 하는 **Speedy-Splat** 의 replay 를 썼습니다
(스냅샷당 8만 ~ 111만 개). 그 코드는 별도 저장소 `github.com/EDGEB1027/seondo` 의
`speedy-splat/` 에 있습니다. 현재 학습은 vanilla 데이터를 쓰므로 새로 시작할 때는
필요 없습니다. `assets/*speedy*` 는 옛 실험 재현용으로 남겨 두었습니다.

---

## 5. 자산 (`assets/`)

### 5.1 무엇이 있나 — 현재 학습이 쓰는 것

| 파일 | 내용 | 만드는 법 |
|---|---|---|
| `stats_vanilla_{train,truck}_cap.json` | 씬별 xyz 정규화 `center`, `scale` | §5.4 |
| `anchors_vanilla_{train,truck}_n1024.npy` | 씬별 k-means 앵커 1024개 = latent 셀 1024개 (월드 좌표) | `build_vanilla_c1_assets.sh` |
| `split_vanilla_both_blockB.json` | train 2,867 / val 408 스냅샷 전역 인덱스 | 〃 |
| `npz_to_image_vanilla_both.json` | npz 경로 → 그 카메라의 사진 | 〃 (COLMAP 카메라와 매칭) |
| `view_pool_vanilla_both.json` | 씬별 전체 사진 + 카메라 (보류 뷰 평가용) | 〃 |

`blockB` split 은 3DGS step 구간 네 곳을 통째로 val 로 빼고 양쪽에 200 step 여유를 둔
것입니다. 무작위 split 은 val 의 98% 가 train 스냅샷과 10 step 이내라 사실상 누수였습니다.

### 5.2 새 서버로 옮길 때 — 경로 재배치

자산 JSON 과 `configs/*.args.json` 에는 원래 서버의 절대경로가 박혀 있습니다. 접두어는
네 종류뿐이고, `tools/relocate_assets.py` 가 한 번에 바꿉니다 (키까지 바꿉니다 —
`npz_to_image_*.json` 은 npz 전체 경로를 키로 씁니다).

```bash
MAPS="--map /data/daeho/aaaa_proj/seondo/vanilla-3dgs=/새/경로/vanilla-3dgs \
      --map /data/daeho/aaaa_proj/seondo/speedy-splat=/새/경로/speedy-splat \
      --map /data/daeho/train_colmap=/새/경로/train_colmap \
      --map /data/daeho/truck_colmap=/새/경로/truck_colmap"

python tools/relocate_assets.py --dry-run  $MAPS                  # 바뀔 곳 확인
python tools/relocate_assets.py --backup assets_orig  $MAPS       # assets/ 적용
python tools/relocate_assets.py --assets configs --backup configs_orig $MAPS
```

31개 자산 파일에서 68,542 곳을 바꾸고 되돌려서 원본과 완전히 같아지는 것을 확인했습니다.
끝나면 "모든 절대경로가 새 경로 아래로 옮겨졌습니다" 가 나와야 합니다. speedy 데이터를
안 쓰면 그 `--map` 은 빼도 되고, 그 경우 남은 speedy 경로를 알려 줍니다.

### 5.3 자산을 새로 만들 때

replay npz 를 새로 만들었다면 (가우시안 분포가 달라지므로) 자산도 다시 만드는 게 맞습니다.

```bash
# scripts/build_vanilla_c1_assets.sh 상단의 경로 6개를 새 서버 경로로 바꾼 뒤
bash scripts/build_vanilla_c1_assets.sh
```

이 스크립트는 통계 → k-means 앵커(1024, GPU) → truck 사진 매칭 → 두 씬 병합 → blockB
split 순으로 만들고, 마지막에 7개 파일이 다 있는지 검사합니다. 이미 있는 파일은 재사용하므로
다시 만들려면 먼저 지우세요. 그 다음 §5.4 를 한 번 더 돌립니다.

### 5.4 배경 클리핑 수정 (`*_cap.json`) — 반드시 쓸 것

로더는 정규화 좌표가 `|xn| ≤ 1` 인 가우시안만 남깁니다 (`drop_outside`). 원래 통계는
2%/98% 분위수 상자로 `scale` 을 잡아서, **먼 배경(산 등)의 크고 불투명한 가우시안이 잘려
나갔습니다.** 전체의 0.5% 인데 배경이 보이는 뷰에서는 화질을 2.6 dB 좌우했고, 배경용으로
배치된 앵커 44 ~ 52개(= latent 셀의 4 ~ 5%)가 거의 빈 채로 학습되고 있었습니다.

`*_cap.json` 은 `scale` 에 `extent_cap = 2.0` 을 곱한 것입니다. 렌더 PSNR 이 포화하는 cap 을
재서 정했습니다 (스냅샷 3개 × 보류 뷰 4개):

| cap | train | truck |
|---|---|---|
| 1.0 (원래) | −0.74 dB | −1.00 dB |
| 1.5 | +0.01 | −0.02 |
| **2.0 (채택)** | **0.00** | −0.02 |

점 유지율 95.5% → 99.8%. 새 데이터로 다시 만들 때는 원래 `stats_*.json` 을 만든 뒤
`scale` 을 2 배 한 복사본을 `*_cap.json` 으로 저장하면 됩니다. 씬 종류가 달라지면
(§10 의 Mip-NeRF 360 같은 무한 배경) cap 을 씬마다 다시 재야 합니다.

---

## 6. 학습

### 6.1 설정 스냅샷 (`configs/`)

| 파일 | 설명 | 결과 |
|---|---|---|
| `B1_bgcap.args.json` | 배경 수정 설정, **처음부터** 학습 | 31,000 step 에서 계보 최고 PSNR 20.88 / SSIM 0.782. 약 34,000 step 에서 그래디언트 폭주 |
| `B1g_stable.args.json` | 위 설정 + 안정화 장치 3종, 30,000 체크포인트에서 **재개**용 | 폭주 없음. 대신 약 19.2 dB 에 머물다 서서히 하락 (75k 에서 18.84) |

두 설정의 공통 핵심값:

| 구분 | 값 |
|---|---|
| 데이터 | vanilla train+truck, `min_snapshot_step 12000`, `split_vanilla_both_blockB` |
| 점 예산 | 인코더 입력 최대 2,097,152 · 디코더 출력 262,144 (셀 1024 × 256 슬롯) |
| latent | `16 × 32 × 32` = 16,384. 셀당 16채널 = centroid 4 · occupancy 1 · **shape 3** · **appearance 8** |
| 디코더 | 폴딩(피보나치 공 템플릿 + 6-number 프레임 + 잔차), 속성 헤드는 디코딩된 위치를 읽음 |
| 학습 | 3 GPU DDP, batch 1/GPU, bf16, lr 2e-4 → 1e-5 코사인, 80,000 step |
| 렌더 손실 | `w_render_attr 60`, `w_sobel 1`, `w_render_perc 0.05`, `edge_gain 4`, `detail_start 40000` |
| 주요 기하·속성 손실 | `w_coverage 1200`, `w_intra_sinkhorn 4`, `w_attr_set 3`, `w_attr_local_set 3`, `w_proj_hist 2.5`, `w_plane_chamfer 2.5` |
| 계약 플래그 | `normalize_pooler_xyz 1`, `holdout_own_photo 1`, `keep_extra_fullres 1`, `eval_pred_mask 1`, 나머지 0 |

`B1g_stable` 에만 있는 것: `grad_spike_mult 3`, `grad_spike_abs 6000`(비정상 기울기 스텝
건너뛰기), `near_detach_frac 0.1`(카메라 근처 점의 1/z 기울기 차단), `w_splat_area 1.0`
(GT 화면반경 상위 20% 보다 큰 가우시안 벌점).

### 6.2 실행

`scripts/launch_from_config.sh` 하나로 처음부터·재개 모두 됩니다. 실행 전에 설정에
계약 플래그 8개가 모두 있는지 검사하고, 없으면 거부합니다 (§8.1).

```bash
conda activate can3tok
cd <can3tok>

# 먼저 확인만 (GPU 안 씀): 계약 플래그 + 데이터·자산 경로 존재 여부 + 최종 명령
bash scripts/launch_from_config.sh configs/B1_bgcap.args.json --dry-run

# 처음부터
GPUS=0,1,2 NPROC=3 bash scripts/launch_from_config.sh configs/B1_bgcap.args.json --detached

# 재개 (안정화 설정)
bash scripts/launch_from_config.sh configs/B1g_stable.args.json \
     --resume runs/B1_bgcap_xxx/ckpt_step00030000.pt --detached

# 값 덮어쓰기
bash scripts/launch_from_config.sh configs/B1_bgcap.args.json \
     --set max_steps=40000 --set save_every=2000 --detached
```

결과는 `runs/<설정이름>_<시각>/` 에 체크포인트, `runs/<설정이름>_<시각>.log` 에 로그.
환경변수: `GPUS`, `NPROC`, `TAG`, `MASTER_PORT`, `PYTHON`, `TORCHRUN`.

GPU 가 3장 미만이면 `NPROC` 를 줄이면 됩니다. batch 는 GPU 당 1 이라 유효 batch 가
바뀌므로 결과가 원래 서버와 정확히 같지는 않습니다.

### 6.3 로그 읽기

학습 줄 (50 step 마다):

```
[joint] step 30000/80000 loss 25.65 … gn 854.3 … TOP[codec_proj_hist:21% codec_attr_set:11% …] … 4.70s/it 27.2GB
```

- `gn` — 클리핑 전 기울기 노름. **수천을 넘기 시작하면 폭주 전조** (§8.4)
- `TOP[…]` — 목적함수에서 비중이 큰 항 4개
- `[GSKIP] step …` — 안정화 설정에서 스파이크 스텝을 건너뛴 기록

평가 줄 (`eval_every` 마다):

```
eval step 31000: PSNR photo=20.88 (gt 17.09, gap -3.79) … ssim=0.782 | s0 19.57(…) | s1 22.19(…) | … uniq=0.558 …
```

- `PSNR photo` — 모델 렌더 vs **보류된 사진**, val 스냅샷 8개 평균. 주 지표
- `gt` — GT 가우시안(로더 타깃)을 그대로 렌더한 값. 기준선
- `s0`, `s1` — train, truck 씬별
- `gen_rmse=nan` 은 생성 분기가 꺼져 있어서이며 정상

`ckpt_best.pt` 는 `PSNR photo` 최고점 체크포인트입니다 (모델 가중치만, 0.38 GB).

---

## 7. 평가 · 렌더

학습 중 자동 평가 외에 한 스냅샷을 크게 렌더해 볼 때:

```bash
# val 스냅샷 하나를 그 자신의 카메라에서: 사진 / GT 가우시안 / 모델 (세로 3단 PNG)
REPO=$PWD python tools/render_one.py --ckpt runs/…/ckpt_best.pt --index 166 --out out.png
```

`--index` 는 val 셋 안의 순번입니다. 옛 체크포인트(계약 플래그가 없는 `args.json`)도 없는 키를
0 으로 채워서 제대로 렌더합니다 (§8.1). 여러 체크포인트를 비교할 때는 `tools/render_compare.py`,
평가 그림은 `tools/render_eval_figs.py`.

**보류 사진을 쓸 것.** `view_pool` 인덱스 3번(`train_colmap/images/00004.jpg`)은 학습에서
빠진 사진입니다. val 스냅샷 166번(`step_028630.npz`)의 자기 사진이 이것이므로, 자기 카메라
렌더가 곧 일반화 평가가 됩니다. 단 `holdout_own_photo=1` 인 설정에서만 그렇습니다 (§8.5).

---

## 8. 반드시 알아야 할 함정

모두 실제로 겪고 결론을 틀리게 낸 것들입니다.

### 8.1 계약 플래그와 파서 기본값

`train.py` 파서는 나중에 추가된 플래그 몇 개의 기본값이 1 입니다 (`normalize_pooler_xyz`,
`count_aware_template`, `attr_slot_mask`, `shared_cell_owner`, `holdout_own_photo`,
`keep_extra_fullres`, `eval_pred_mask`). **옛 런의 `args.json` 에는 이 키가 없어서**, 그걸로
인자를 복원하면 기본값 1 이 조용히 들어가고 체크포인트가 학습된 적 없는 코드 경로가 켜집니다.

겪은 일: 21.5 dB 모델을 이렇게 재개했더니 500 step 만에 16 dB 로 무너졌고, 처음엔 학습률
탓으로 오판했습니다. 렌더 스크립트에서도 같은 이유로 22.06 dB 모델이 12.06 dB 로 찍혔습니다.

대책:
- `configs/` 의 두 설정은 8개 키가 모두 명시돼 있습니다.
- `launch_from_config.sh` 는 키가 빠진 설정을 거부합니다.
- 옛 체크포인트를 렌더할 때는 없는 키를 0 으로 채웁니다 (`render_one.py` 의 compat 블록).

### 8.2 npz 규약

npz 의 `scaling` 은 선형, `opacity` 는 sigmoid 후입니다 (§4.4). 로더 타깃은 log · logit
입니다. 섞으면 렌더가 완전히 깨집니다.

### 8.3 배경 클리핑

원래 `stats_*.json` 을 쓰면 먼 배경이 로더에서 삭제됩니다. **`*_cap.json` 을 쓰세요** (§5.4).
그런데 이것만으로는 배경이 다 돌아오지 않습니다 — 배경 앵커 44개가 셀당 256 슬롯에 막혀
소유 점의 34% 만 담습니다 (§10).

### 8.4 그래디언트 폭주 (약 34,000 step)

`B1_bgcap` 설정은 34,000 step 부근에서 `gn` 이 800 → 9,425 → 29,919 → 269,054 로 치솟고
PSNR 이 20.8 에서 12 로 떨어졌습니다. `near_detach_frac`, `grad_spike_*` 가 이를 막지만
(`B1g_stable`), 함께 들어간 `w_splat_area` 때문인지 최고 품질이 약 2 dB 낮아졌습니다.

### 8.5 자기 사진 누수

`holdout_own_photo=0` 이면 스냅샷의 자기 사진이 보류 목록을 우회해 학습에 들어갑니다.
`blockB` 학습셋에도 `00004.jpg` 를 자기 사진으로 갖는 스냅샷이 7개 있습니다
(016190, 021160, 021960, 022570, 024230, 024470, 024900). `holdout_own_photo=1` 이 런타임에
이를 막습니다. 옛 런(T16k 등)은 이 플래그가 생기기 전이라 새었습니다 — 그 런들의 `00004.jpg`
수치는 일반화가 아니라 적합입니다.

### 8.6 PSNR 은 가는 구조를 못 봅니다

난간이 통째로 사라져도 PSNR 은 GT 가우시안보다 높게 나옵니다. 디테일 판정은 반드시
난간 크롭을 육안으로 함께 보세요 (`mdmd/17_detail_bottleneck_20260917.md` 의 크롭 좌표).

### 8.7 GPU 공용 사용

원래 서버는 다른 사용자와 GPU 를 나눠 씁니다. 프로세스를 멈출 때는 `ps -o user,args` 와
`--out_dir` 를 먼저 확인하고 본인 것만 PID 로 멈추세요. `pkill -f` 는 쓰지 마세요.

---

## 9. 체크포인트 · 데이터 옮기기

저장소에 없는 것을 새 서버로 가져가는 방법입니다.

| 대상 | 원래 서버 경로 | 크기 |
|---|---|---|
| 계보 최고 모델 | `runs/B1_bgcap_20260922_024641/ckpt_best.pt` (31k, 20.88 dB) | 0.38 GB |
| 재개용 전체 상태 | `runs/B1_bgcap_20260922_024641/ckpt_step00029000.pt` 등 | 1.2 GB / 개 |
| replay npz | `/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/{train,truck}/replay/` | 147 GB |
| COLMAP 원본 | `/data/daeho/{train,truck}_colmap/` | 2 GB |

```bash
# 예: rsync (재개 가능, 압축 불필요 — npz 는 이미 압축돼 있음)
rsync -avP user@old-server:/data/daeho/aaaa_proj/seondo/vanilla-3dgs/output/train/replay/ /new/vanilla-3dgs/output/train/replay/
```

npz 는 다시 만드는 것이 복사보다 느립니다 (3DGS 30,000 iter × 2 씬, 씬당 몇 시간). 네트워크가
되면 복사하세요. 가져온 뒤에는 §5.2 로 경로를 바꿉니다.

체크포인트를 GitHub 에 올리지 마세요 (파일당 100 MB 제한). 공유가 필요하면 GitHub Release
첨부(파일당 2 GB)나 클라우드 드라이브를 쓰세요.

---

## 10. 현재 상태와 다음 할 일 (2026-09-28)

### 확정된 것

| 문제 | 원인 | 조치 | 상태 |
|---|---|---|---|
| 배경(산)이 뭉개짐 | 전처리가 먼 가우시안 삭제 | `*_cap.json` | 적용됨. 초반 +3.9 dB (같은 step 대비) |
| 학습 폭주 | 34k 부근 기울기 폭주 | `near_detach` + `grad_spike` | 적용됨 (`B1g_stable`) |
| 블러·질감 | **셀당 256 슬롯 상한** — train 67%, truck 82% 의 점이 셀 오버플로로 버려짐 | 디코더 단 선택적 densification | 설계만 됨 |
| 난간 같은 가는 선 | 셀 내부 패턴이 GT 와 무상관 (상관 0.06), 셀 프레임 방향 오차 29° | 코드→패턴 감독 | 미해결 |

### 가장 유망한 다음 실험

1. **`w_splat_area` 완화** — `B1_bgcap` 30k 에서 `B1g_stable` 로 재개하되 `--set w_splat_area=0.1`
   (또는 분위수 0.8 → 0.99). 안정성과 20.8 dB 품질을 같이 얻는지 판정.
2. **선택적 densification** — 오버플로 셀에만 가우시안 1만 ~ 2만 개(출력의 4 ~ 8%) 추가.
   화면 기여도(투영면적 × 불투명도)로 고르면 오라클 기준 train +1.4 dB, truck +1.9 dB.
   로더 importance 로 고르면 같은 개수로 1/6 밖에 안 됩니다 — **무엇을 고르느냐가 핵심**.
   latent 16,384 는 그대로입니다.
3. **씬 확장** — Mip-NeRF 360(`bicycle, bonsai, counter, garden, kitchen, room, stump`),
   360 extra(`flowers, treehill`), Deep Blending(`drjohnson, playroom`). COLMAP 희소점 기준으로
   필요한 cap 이 1.22 ~ 5.58 로 씬마다 4.6 배 다르므로, 고정 cap 대신 **씬별 cap 또는 scene
   contraction(Mip-NeRF 360 방식)** 이 필요합니다. npz 를 만든 뒤 실제 가우시안 분포로 다시 재세요.

### 분석 기록

자세한 근거는 `mdmd/` 에 있습니다. 특히:

- `17_detail_bottleneck_20260917.md` — 난간이 어디서 죽는가 (오라클 사다리, 셀 고유차원, 속성 붕괴)
- `18_gradient_training_audit_20260925.md` — 폭주와 안정화
- `12_code_audit_and_improvement_20260915.md` — 코드 감사
