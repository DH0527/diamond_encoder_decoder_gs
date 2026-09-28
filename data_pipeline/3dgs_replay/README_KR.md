# 3DGS replay npz 덤프

can3tok 의 학습 데이터(`step_*.npz`)를 만드는 수정된 3DGS. 전체 절차는 `SETUP_KR.md` §4.

| 파일 | 내용 |
|---|---|
| `gaussian_splatting_54c035f.patch` | `graphdeco-inria/gaussian-splatting` 커밋 `54c035f` 에 대한 수정 (train.py, scene/gaussian_model.py, utils/camera_utils.py). 적용 결과가 원래 서버의 수정본과 바이트 단위로 같음을 확인함 |
| `overlay/replay_utils.py` | 스냅샷 덤프 본체 (`--compact_npz`: float16 zip-npz) |
| `overlay/render_adapter.py` | 렌더 어댑터 |
| `overlay/tools/` | `inspect_npz.py`(내용 확인), `match_replay_npz_to_colmap.py`(npz 카메라 ↔ COLMAP 사진 매칭), `run_dequant_check.py`(float16 오차 확인) |
| `run_dump_compact_npz.sh` | train(GPU 0) · truck(GPU 1) 동시 학습 + 덤프. 경로는 환경변수 |

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting && git checkout 54c035f && git submodule update --init --recursive
git apply <can3tok>/data_pipeline/3dgs_replay/gaussian_splatting_54c035f.patch
cp -r <can3tok>/data_pipeline/3dgs_replay/overlay/* .
```

패치 적용 시 `trailing whitespace` 경고가 6 줄 나오는데, 원본 코드의 줄끝 공백이라 무해합니다.

라이선스: Inria 3DGS 는 Gaussian-Splatting License (비상업 연구용). 이 패치는 그 파생물이므로
같은 조건을 따릅니다.
