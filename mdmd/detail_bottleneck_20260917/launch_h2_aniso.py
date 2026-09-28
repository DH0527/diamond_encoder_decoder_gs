"""H2: T16k 70k resume + 이방성 항 + Hungarian scale, 쓸 만한 LR. GPU 3.

왜 이 조합인가 (모두 T16k step 70000 에서 측정).
  1. 위치는 이미 난간을 덮고 있다. 예측 위치에 GT 속성을 최근접으로 옮기면 핸드레일이
     선으로 나온다. 실패는 속성이다.
  2. 회전은 무감독이다. w_rot=0 이고, 유일한 경로인 covariance3d_loss 는 회전에
     0.0% 만 반응한다 (GT 회전을 줘도 손실 불변, GT 스케일을 주면 99.1% 감소).
  3. 손실 형태를 바꿔도 안 된다. trace 정규화해도 회전 민감도 1.6%. 이유는 예측이
     구에 가깝다는 것 (이방성 p50 3.73 vs GT 15.51, 4배 미만이 52.4% vs GT 13.2%).
     R S S^T R^T 는 S 가 구면이면 R 을 잃는다. 구를 돌려도 구다.
  => 그래서 순서가 강제된다. 이방성을 먼저 세우고, 그 다음에 회전을 켠다.
     w_attr_hung_rot 은 여기서 의도적으로 0 이다.

H1 이 아무것도 못 움직인 이유도 여기서 고친다. LR 은 max_steps 기준 코사인이라
70000/72000 에서 이미 lr_min(1e-5) 이었다. max_steps 는 schedule.py 에서 LR 코사인과
루프 종료에만 쓰이고 손실 램프는 전부 명시적 스텝을 쓰므로, 늘려도 다른 스케줄을
되감지 않는다. 140000 이면 step 70000 에서 약 1.05e-4 다.

게이트는 PSNR 이 아니다. PSNR 은 이 문제에 눈이 멀었다 -- 난간을 제대로 그리는 오라클
렌더가 base 보다 1.5dB 낮게 나온다. 대신 체크포인트마다:
  1단계  covcheck.py : 예측 이방성 p50 3.73 -> GT 15.51 쪽으로, 4배 미만 52.4% -> 13.2% 쪽으로
  2단계  rotcheck.py : 그 다음에 이방성 8배 초과 구간 방향 오차 p50 41.5도 -> 평균 기준선 35.0도 아래로
"""
import json, os, sys
from datetime import datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TARGS = os.path.join(REPO, "runs/T16k_20260913_093332/args.json")
SKIP = {"resume", "init_from", "eval_only", "out_dir", "init_skip"}
cfg = json.load(open(TARGS))
argv = []
for k, v in cfg.items():
    if k in SKIP:
        continue
    if isinstance(v, bool):
        if v:
            argv.append("--" + k)
    elif isinstance(v, list):
        if v:
            argv += ["--" + k] + [str(x) for x in v]
    elif v is None:
        continue
    else:
        argv += ["--" + k, str(v)]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = os.environ.get("H2_OUT", f"runs/H2_aniso_{stamp}")
os.makedirs(os.path.join(REPO, out), exist_ok=True)
argv += [
    "--out_dir", out,
    "--resume", "runs/T16k_20260913_093332/ckpt_step00070000.pt",
    # LR 코사인에만 영향. 손실 램프는 건드리지 않는다.
    "--max_steps", os.environ.get("H2_MAX_STEPS", "140000"),
    "--lr", "0.0002",
    "--save_every", "500",
    "--eval_every", "10000",
    "--log_every", "50",
    # R1/H1 과 같은 compat 계약
    "--shared_cell_owner", "0",
    "--normalize_pooler_xyz", "0",
    "--count_aware_template", "0",
    "--attr_slot_mask", "0",
    "--holdout_own_photo", "0",
    "--keep_extra_fullres", "0",
    "--reuse_prev_vis", "0",
    "--eval_pred_mask", "0",
    "--w_attr_spread", "0.0",
    "--w_attr_slope", "0.0",
    # 새 항: 크기와 방향을 나눠낸 모양. 구에서도 기울기가 산다.
    "--w_aniso", os.environ.get("H2_W_ANISO", "5.0"),
    # 1:1 매칭. sinkhorn 이 "가장 잘 맞는 상대" 를 골라 평균으로 수렴하는 것을 막는다.
    "--w_attr_hung_scale", os.environ.get("H2_W_HUNG_SCALE", "5.0"),
    "--w_attr_hung_opacity", os.environ.get("H2_W_HUNG_OPACITY", "5.0"),
    "--w_attr_hung_rot", os.environ.get("H2_W_HUNG_ROT", "0.0"),
    # 같은 연산을 몇 조각으로 나눠 도느냐일 뿐이다. 기울기도 궤적도 그대로이고
    # 메모리만 줄고 속도만 느려진다. GPU 3 을 다른 사용자의 작업(6.2GB)과 나눠 쓰는
    # 동안 필요하다: 32.8GB 중 25.2GB 만 남아서 T16k 의 25.8GB 가 들어가지 않는다.
    "--patch_chunk", os.environ.get("H2_PATCH_CHUNK", "64"),
]
print("OUT_DIR=" + out, flush=True)
os.chdir(REPO)
os.execv(sys.executable, [sys.executable, "-u", os.path.join(REPO, "train.py")] + argv)
