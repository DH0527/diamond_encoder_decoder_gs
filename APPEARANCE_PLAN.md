# 구조 수정 계획 — full-Gaussian 경로

코드를 처음부터 끝까지 읽고 찾은 문제와 수정 설계. 측정 근거는 [MODEL.md](MODEL.md) §7.

---

## 1. 발견된 구조 문제 7개

| | 문제 | 위치 | 결과 |
|---|---|---|---|
| **S1** | pack이 `x[..., 0:3]`만 가져감 | `encoder.build_patch:92` | appearance가 `z_raw`에 안 들어감 → `z_compact`에 0비트 |
| **S2** | 채널 예산에 appearance 칸 없음 | `config.channel_budget` | `centroid+occupancy+shape == per_group` 강제 |
| **S3** | decompressor가 `shape` 전체를 기하로만 해석 | `compressor.py:608` | appearance 채널이 있어도 읽는 곳이 없음 |
| **S4** | codec attribute 헤드가 기하 파생 토큰만 읽음 | `decoder._attributes` | `h`는 `z_raw_hat` ← 기하 경로 |
| **S5** | **gen이 attribute를 zeros로 채움** | `gen_decoder.py:316` | **배포 경로가 가우시안을 못 만듦** |
| **S6** | z-공간 손실이 pack **전체**를 타깃으로 함 | `losses.z_raw_losses` | pack이 학습형이 되면 최강 항(51.8%)의 타깃이 움직임 |
| **S7** | mask 채널이 `z_raw`의 25%를 쓰고 0.40 bit/slot만 나름 | `build_patch`, 측정됨 | 쓸 수 있는 공간이 놀고 있음 |

S5가 가장 치명적입니다. `gen`은 world model이 `z_compact`를 업데이트한 뒤 디코딩하는 **유일한** 경로인데, 지금은 xyz만 내고 나머지는 리터럴 0입니다.

---

## 2. 설계 원칙

**측정으로 검증된 것은 하나도 건드리지 않고, appearance를 분리 가능한 병렬 경로로 추가합니다.**

건드리지 않을 것 (전부 측정 근거 있음):
* 결정론적 xyz pack과 그 위의 `w_z_residual` (그룹 내 그래디언트의 51.8%)
* folding 디코더 (고정 템플릿 / 채워진 공 / 회전 프레임 / det=+1 / cap+ramp)
* 템플릿 정렬 슬롯, local kd, Morton
* `z_raw` 크기 (1,048,576) — 3.75배로 키우면 15.7 GB/rank에 안 들어감 (측정)
* `z_compact` 크기 (131,072) — 고정 제약

---

## 3. 설계

### 3.1 `z_raw` 토큰 레이아웃 — 크기 그대로, mask 자리를 재사용

```
그룹당 256 dim (= token_dim = tpg 8 × latent_channels 32)

기존:  [ 0 : g*3 ]  xyz 오프셋, 항등 pack   (192)
       [ g*3 : g*4 ]  mask                   (64)

변경:  [ 0 : g*3 ]  xyz 오프셋, 항등 pack   (192)   ← 그대로, 결정론적
       [ g*3 : g*4 ]  appearance 코드, 학습형 (64)   ← mask 대체
```

**mask를 버려도 되는 이유** (측정): prefix 패킹 때문에 그룹은 꽉 차거나 비어 있고,
**부분적으로 찬 그룹은 장면당 정확히 1개**입니다. mask 엔트로피는 fp16 16비트 중
**0.40비트**. 비었는지 여부는 log-extent 앵커가 이미 알고 있고
(`alive = scale > SCALE_REF*0.05`), presence 헤드는 count prior를 따로 받습니다.

**64 dim이 충분한 이유** (측정): `z_compact`가 appearance에 줄 수 있는 것이 그룹당
8채널이고 거기서 이미 포화합니다 (§7.7). `z_raw` 64는 그 **8배**입니다.

### 3.2 appearance pack 인코더 (신규, S1·S7 해결)

```
그룹의 attribute (g × A_in)          A_in = 11 (log_scale 3, quat 4, logit_opacity 1, SH DC 3)
    │  채널별 표준화 (양자가 스케일이 전혀 다름)
    │  per-point MLP: A_in → d_a
    │  그룹 내 self-attention 1층      ← 문맥을 보고 무엇을 남길지 결정
    │  per-point 투영: d_a → 1
    ▼
g 개 숫자 = pack 의 aux 블록
```

손실 압축을 **문맥이 있는 곳에서** 합니다. 점당 독립 MLP로 11→1을 미리 뭉개면
그룹 내 중복을 볼 기회를 버립니다.

### 3.3 `z_compact` 채널 예산 (S2 해결)

```
32 = centroid 3 + log-extent 1 + shape S + appearance A
```

| | S | A | 근거 |
|---|---|---|---|
| 기본값 | 28 | **0** | 기존 런과 완전 호환 |
| 새 런 | 20 | **8** | A=8이 §7.7 곡선의 무릎, S=20은 §7.1에서 ~16.9 dB |

거래: 기하 **−0.5 dB** ↔ appearance **+4.4 dB** (둘 다 오라클 상한).

### 3.4 appearance 를 두 디코더에 배선 (S3·S4·S5 해결)

```
z_compact
  └─ pg[..., c_cen+c_occ+c_shape : ]   = appearance (A)
        │
        ├─→ decompressor: ctx["appearance"] 로 발행
        │      └─→ codec decoder attribute 헤드
        │            attr = attr_mlp(cat([h, xyz_pe(xyz_cond), appearance]))
        │
        └─→ gen_decoder: 같은 슬라이스를 자기 attribute 헤드로
               (신규 — gen 은 attribute 헤드가 아예 없었음)
```

gen 은 codec 과 **같은 헤드 구조**를 갖습니다. 기하에서 basis 를 공유시킨 것과 같은
이유입니다 — 교사/학생이 다른 방식으로 만들면 distillation 이 표면만 맞춥니다.

### 3.5 z-공간 손실을 기하 prefix 로 제한 (S6 해결)

`z_raw_losses` / `group_residual_z_loss` 가 지금은 토큰 **전체**를 비교합니다.
appearance 블록이 학습형이 되면 그 타깃이 매 스텝 움직입니다.

→ `A > 0` 일 때 두 손실 모두 `[0 : g*3]` 만 봅니다. `w_z_residual` 의 정확한
결정론적 타깃이 그대로 보존됩니다.

appearance 쪽 감독은 **디코딩된 attribute** 에서만 옵니다 (파라미터 손실 + 렌더 손실).

---

## 4. 파일별 변경

| 파일 | 변경 |
|---|---|
| `config.py` | `attr_pack_dim`, `budget_appearance` 필드. `channel_budget` 에 appearance 칸 |
| `encoder.py` | `GroupAttributeEncoder`. `build_patch` 가 mask 대신 appearance 코드를 aux 로 |
| `compressor.py` | `head_appearance`. decompressor 에서 슬라이스 → `ctx["appearance"]` |
| `decoder.py` | `_attributes` 가 appearance 를 읽음 |
| `gen_decoder.py` | **attribute 헤드 신규**. appearance 슬라이스 읽음 |
| `model.py` | 배선 |
| `losses.py` | `A>0` 이면 z-공간 손실을 기하 prefix 로 |
| `train.py` | `--budget_appearance`, `--attr_pack_dim` |

---

## 5. 검증 순서 (각 단계가 다음의 전제)

1. `A=0` 으로 기존과 **bit-identical** 한지 — 회귀 없음 확인
2. `A=8`, 8192점 smoke — 배선 확인
3. **appearance 정보가 실제로 흐르는지**: `z_compact` 의 appearance 채널을 셔플했을 때
   디코딩된 attribute 가 바뀌는가. 안 바뀌면 경로가 죽은 것
4. 262k 실규모, `--stage geometry`
   * 게이트: attribute R² > 0 (현재는 −0.99 ~ −0.23, 즉 데이터셋 평균만도 못함)
   * 게이트: held-out 3뷰 PSNR > 20 dB (현재 17.05 오라클 / 17.35 모델 1뷰)

---

## 6. 하지 않는 것

* `z_raw` / `z_compact` 크기 변경 — 제약이고 측정상 불가
* SH rest 45채널 — 렌더러가 `sh_degree=0` 이라 손실이 볼 수 없음. `--stage geometry` 14채널만
* canonicalizer, 계층적 cell attention, 시간적 안정성 — 이 변경 뒤로
* 기하 최적화 — 상한의 93%, 남은 여유 +0.3 dB
