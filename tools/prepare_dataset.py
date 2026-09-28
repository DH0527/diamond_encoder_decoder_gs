"""replay npz 에서 학습에 필요한 자산과 실행 설정을 한 번에 만든다.

3DGS 를 새로 학습해 npz 를 만들었다면 가우시안 분포가 달라지므로 자산을 다시 만들어야
한다 (SETUP_KR.md §5). 예전에는 build_assets.py + merge_assets.py + 손으로 만든 blockB
split + 손으로 만든 *_cap 통계 + config 수정의 다섯 단계였고, 기본값 두 개(앵커 4096,
무작위 split)가 현재 학습 설정과 어긋났다. 이 스크립트는 그걸 한 번에, 현재 설정에 맞게
한다.

    python tools/prepare_dataset.py \\
        --names   train,truck \\
        --replays /new/vanilla-3dgs/output/train/replay,/new/vanilla-3dgs/output/truck/replay \\
        --colmaps /new/train_colmap,/new/truck_colmap \\
        --gs_root /new/gaussian-splatting \\
        --tag     v2 \\
        --check

만드는 것 (--tag v2, 씬 이름 S):

    assets/stats_v2_S.json            원래 공식 (2%/98% 분위수 상자, x1.05)
    assets/stats_v2_S_cap.json        scale x extent_cap. 학습은 이걸 쓴다 (§5.4 배경 클리핑)
    assets/anchors_v2_S_n1024.npy     k-means 앵커 = latent 셀 (월드 좌표)
    assets/npz_to_image_v2_S.json     스냅샷 -> 그 카메라의 사진 (파일명이 아니라 카메라 외부행렬로 매칭)
    assets/view_pool_v2_S.json        씬의 모든 사진 + 카메라
    assets/npz_to_image_v2_both.json  npz 전체 경로를 키로 병합 (씬끼리 파일명이 겹치므로)
    assets/view_pool_v2_both.json
    assets/split_v2_both_blockB.json  blockB split (아래)
    assets/dataset_v2.json            무엇을 어떤 파라미터로 만들었는지 기록
    configs/v2.args.json              --config_src 에서 데이터 경로만 바꾼 실행 설정

blockB split 규칙 (원래 서버의 split_*_blockB.json 을 정확히 재현함을 확인):

    val   : 3DGS step 이 --blocks 구간 안인 스냅샷
    제외  : 구간 양쪽 --guard step 이내 (val 과 거의 같은 스냅샷이 train 에 들어가는 누수 방지)
    제외  : --min_step 미만 (학습이 어차피 안 씀)
    제외  : train 중에서 자기 사진이 --holdout_photo 인 스냅샷 (보류 사진 누수 방지)
    train : 나머지

무작위 split 을 쓰지 않는 이유: 스냅샷이 10 step 간격이라, 무작위로 뽑으면 val 의 98% 가
train 스냅샷과 10 step 이내였다. 사실상 같은 장면을 외워서 맞히는 셈이다.

이미 있는 통계·앵커는 재사용한다 (앵커는 GPU k-means 라 비싸다). 다시 만들려면 --force.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from can3tok.data import list_npz_files  # noqa: E402  (학습 로더와 같은 파일 순서)
from can3tok.io_utils import load_npz_state  # noqa: E402

CONTRACT = ["normalize_pooler_xyz", "count_aware_template", "attr_slot_mask",
            "shared_cell_owner", "holdout_own_photo", "keep_extra_fullres",
            "reuse_prev_vis", "eval_pred_mask"]


def _load_build_assets():
    spec = importlib.util.spec_from_file_location(
        "build_assets", os.path.join(REPO, "tools", "build_assets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(path):
    m = re.search(r"step_(\d+)\.npz$", path)
    if not m:
        raise SystemExit(f"파일명에서 step 을 읽을 수 없습니다: {path}")
    return int(m.group(1))


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _fast_cam(path):
    """카메라 16-벡터만 읽는다.

    zip-npz 는 멤버 단위로 지연 로딩되므로 'cam' 만 꺼내면 가우시안 배열을 풀지 않는다.
    load_npz_state 로 전부 읽으면 vanilla 6,000 개 파일에 약 145 GB 를 읽게 된다.
    """
    with np.load(path, allow_pickle=True) as z:
        if "cam" in z.files:
            return np.asarray(z["cam"], np.float32).reshape(-1)
    c = load_npz_state(path).get("camera")          # 옛 형식 (it / state_t / meta)
    return None if c is None else np.asarray(c, np.float32).reshape(-1)


def _xyz(path):
    with np.load(path, allow_pickle=True) as z:
        if "xyz" in z.files:
            return np.asarray(z["xyz"], np.float32)
    return np.asarray(load_npz_state(path)["xyz"], np.float32)


def match_photos(files, views, tol):
    V = np.stack([v["cam"] for v in views])[:, 4:]   # R(9)+T(3) = pose
    mapping, worst, unmatched = {}, 0.0, []
    for k, p in enumerate(files):
        c = _fast_cam(p)
        if c is None:
            unmatched.append(os.path.basename(p))
            continue
        d = np.abs(V - c[4:]).max(axis=1)
        j = int(d.argmin())
        worst = max(worst, float(d[j]))
        if d[j] > tol:
            unmatched.append(os.path.basename(p))
            continue
        mapping[os.path.basename(p)] = j
        if (k + 1) % 1000 == 0:
            print(f"      {k + 1}/{len(files)}  최대 자세 오차 {worst:.2e}", flush=True)
    return mapping, worst, unmatched


def block_split(files, photo_of, blocks, guard, min_step, holdouts):
    """files: 전역 순서의 npz 경로. photo_of[i] = (scene_name, image_basename) 또는 None."""
    train, val, n_young, n_guard, dropped = [], [], 0, 0, []
    for i, p in enumerate(files):
        s = _step(p)
        if s < min_step:
            n_young += 1
            continue
        if any(a <= s <= b for a, b in blocks):
            val.append(i)
        elif any(a - guard <= s <= b + guard for a, b in blocks):
            n_guard += 1
        elif photo_of[i] is not None and photo_of[i] in holdouts:
            dropped.append(i)
        else:
            train.append(i)
    return train, val, n_young, n_guard, dropped


def retention(files, center, scale, cap, n=6):
    """정규화 큐브 안에 남는 점 비율 (cap 1.0 과 cap 적용 후)."""
    pick = files[:: max(len(files) // n, 1)][:n]
    m = np.concatenate([np.abs((_xyz(p) - center) / scale).max(-1) for p in pick])
    return float((m <= 1.0).mean()), float((m <= cap).mean())


def check_datasets(config_path):
    """학습 로더를 실제로 만들어 스냅샷 하나씩 읽어 본다 (계약 전체 검증)."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from argv_from_argsjson import ns_to_argv, load_ns  # noqa
    from can3tok.train import build_parser, make_datasets
    ns = load_ns(config_path, {"out_dir": "/tmp/prepare_dataset_check", "resume": "", "init_from": ""})
    args = build_parser().parse_args(ns_to_argv(ns))
    tr, va = make_datasets(args)
    a, b = tr[0], va[0]
    print(f"   로더: train {len(tr)} / val {len(va)}   "
          f"train[0] 점 {int(a['mask'].sum()):,}  val[0] 점 {int(b['mask'].sum()):,}  "
          f"사진 {tuple(a['photo'].shape) if 'photo' in a else '없음'}")
    return len(tr), len(va)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", required=True, help="씬 이름, 쉼표 구분 (예: train,truck)")
    ap.add_argument("--replays", required=True, help="씬별 replay 디렉터리 (step_*.npz), --names 와 같은 순서")
    ap.add_argument("--colmaps", required=True, help="씬별 COLMAP 디렉터리 (images/, sparse/0/)")
    ap.add_argument("--tag", required=True, help="자산 파일명 접두어 (예: v2)")
    ap.add_argument("--gs_root", default=os.environ.get("GS_ROOT", ""),
                    help="gaussian-splatting 저장소 (COLMAP 로더용). 기본 $GS_ROOT")
    ap.add_argument("--config_src", default="configs/B1_bgcap.args.json",
                    help="데이터 경로만 바꿔 쓸 원본 설정")
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--config_out", default="", help="기본 configs/<tag>.args.json")
    ap.add_argument("--anchors", type=int, default=0,
                    help="앵커 수. 0 이면 config 의 max_points / group_size (현재 1024)")
    ap.add_argument("--extent_cap", default="2.0",
                    help="씬별 scale 배수 (하나면 전 씬 공통). SETUP_KR.md §5.4")
    ap.add_argument("--blocks", default="13000:13500,18000:18500,23500:24000,28500:29000",
                    help="val 로 쓸 3DGS step 구간 (a:b, 양끝 포함)")
    ap.add_argument("--guard", type=int, default=200, help="구간 양쪽에서 버릴 step 폭")
    ap.add_argument("--min_step", type=int, default=-1,
                    help="이보다 어린 스냅샷 제외. -1 이면 config 의 min_snapshot_step")
    ap.add_argument("--holdout_photo", action="append", default=None,
                    help="SCENE:BASENAME. 자기 사진이 이것인 train 스냅샷을 뺀다. "
                         "여러 번 지정 가능. 기본 train:00004.jpg (config 의 eval_view_force 3)")
    ap.add_argument("--stats_stride", type=int, default=50)
    ap.add_argument("--quantile", type=float, default=0.02)
    ap.add_argument("--anchor_stride", type=int, default=100)
    ap.add_argument("--anchor_iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cam_tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="있는 통계·앵커도 다시 계산")
    ap.add_argument("--check", action="store_true", help="끝나고 학습 로더를 실제로 만들어 검증")
    a = ap.parse_args(argv)

    os.chdir(REPO)
    names, replays, colmaps = _csv(a.names), _csv(a.replays), _csv(a.colmaps)
    if not (len(names) == len(replays) == len(colmaps)):
        raise SystemExit(f"--names {len(names)} / --replays {len(replays)} / --colmaps {len(colmaps)} 개수가 다릅니다")
    if len(set(names)) != len(names):
        raise SystemExit(f"씬 이름이 겹칩니다: {names}")
    replays = [os.path.abspath(r) for r in replays]
    colmaps = [os.path.abspath(c) for c in colmaps]
    caps = [float(x) for x in _csv(a.extent_cap)]
    if len(caps) == 1:
        caps = caps * len(names)
    if len(caps) != len(names):
        raise SystemExit(f"--extent_cap 은 1 개 또는 씬 수({len(names)}) 만큼")
    if a.gs_root:
        os.environ["GS_ROOT"] = os.path.abspath(a.gs_root)

    # 비싼 단계(통계·GPU k-means) 전에 입력부터 확인한다
    errs = []
    gsr = os.environ.get("GS_ROOT", "/data/daeho/gaussian-splatting")
    if not os.path.isfile(os.path.join(gsr, "scene", "colmap_loader.py")):
        errs.append(f"--gs_root: {gsr}/scene/colmap_loader.py 가 없습니다 (SETUP_KR.md §4.2 의 gaussian-splatting 저장소)")
    for name, rep, col in zip(names, replays, colmaps):
        if not glob.glob(os.path.join(rep, "step_*.npz")):
            errs.append(f"[{name}] replay 에 step_*.npz 가 없습니다: {rep}")
        if not os.path.isdir(os.path.join(col, "images")):
            errs.append(f"[{name}] COLMAP images/ 가 없습니다: {col}")
        sp0 = os.path.join(col, "sparse", "0")
        if not any(os.path.isfile(os.path.join(sp0, f)) for f in ("images.bin", "images.txt")):
            errs.append(f"[{name}] COLMAP sparse/0/images.bin 이 없습니다: {sp0}")
    if errs:
        raise SystemExit("입력을 확인하세요:\n  " + "\n  ".join(errs))

    src = json.load(open(a.config_src))
    miss = [k for k in CONTRACT if k not in src]
    if miss:
        raise SystemExit(f"{a.config_src} 에 계약 플래그가 없습니다: {miss} (SETUP_KR.md §8.1)")
    n_anch = a.anchors or int(src["max_points"]) // int(src["group_size"])
    min_step = a.min_step if a.min_step >= 0 else int(src.get("min_snapshot_step", 0))
    blocks = [tuple(int(v) for v in b.split(":")) for b in _csv(a.blocks)]
    holdouts = set()
    for h in (a.holdout_photo if a.holdout_photo is not None else ["train:00004.jpg"]):
        if not h:
            continue
        sc, _, base = h.partition(":")
        if sc not in names:
            print(f"!! --holdout_photo {h}: 씬 '{sc}' 이 --names 에 없어 무시합니다")
            continue
        holdouts.add((sc, base))
    ba = _load_build_assets()
    os.makedirs(a.assets, exist_ok=True)
    T = a.tag

    print(f"=== prepare_dataset  tag={T}  씬 {names}  앵커 {n_anch}  cap {caps}  min_step {min_step}")
    print(f"    blocks {blocks}  guard {a.guard}  보류 사진 {sorted(holdouts) or '없음'}")

    per_scene, all_files, photo_of, pm_both, vp_both = [], [], [], {}, {}
    for si, (name, rep, col, cap) in enumerate(zip(names, replays, colmaps, caps)):
        files = list_npz_files(rep)
        if not files:
            raise SystemExit(f"[{name}] {rep} 에 step_*.npz 가 없습니다")
        steps = [_step(p) for p in files]
        scene_key = os.path.dirname(rep)
        print(f"\n[{name}] {len(files)} 스냅샷  step {min(steps)}..{max(steps)}  ({rep})")

        # 1) 정규화 통계
        p_raw = os.path.join(a.assets, f"stats_{T}_{name}.json")
        if os.path.exists(p_raw) and not a.force:
            stats = json.load(open(p_raw))
            print(f"   통계: 재사용 {p_raw}")
        else:
            print(f"   통계 계산 (stride {a.stats_stride}, 분위수 {a.quantile}) ...", flush=True)
            stats = ba.build_stats(files, a.stats_stride, a.quantile)
            stats["root"] = rep
            json.dump(stats, open(p_raw, "w"), indent=2)
        cen, sc0 = np.asarray(stats["center"], np.float32), float(stats["scale"])
        cap_stats = dict(stats)
        cap_stats.update(scale=sc0 * cap, raw_scale=sc0, extent_cap=cap,
                         note="scale = raw_scale * extent_cap (tools/prepare_dataset.py, SETUP_KR.md §5.4)")
        p_cap = os.path.join(a.assets, f"stats_{T}_{name}_cap.json")
        json.dump(cap_stats, open(p_cap, "w"), indent=1)
        keep0, keep1 = retention(files, cen, sc0, cap)
        print(f"   scale {sc0:.3f} -> {sc0 * cap:.3f}  (cap {cap})   큐브 안 점 {100 * keep0:.1f}% -> {100 * keep1:.1f}%")

        # 2) 앵커
        p_anc = os.path.join(a.assets, f"anchors_{T}_{name}_n{n_anch}.npy")
        if os.path.exists(p_anc) and not a.force:
            anch = np.load(p_anc)
            print(f"   앵커: 재사용 {p_anc} {anch.shape}")
        else:
            print(f"   앵커 k-means {n_anch} (stride {a.anchor_stride}, iters {a.anchor_iters}, {a.device}) ...", flush=True)
            anch, cnt = ba.build_anchors(files, n_anch, a.anchor_stride, a.anchor_iters, a.seed, a.device)
            np.save(p_anc, anch)
            print(f"   빈 셀 {int((cnt == 0).sum())}  셀 크기 중앙값 {int(np.median(cnt))}  최대 {int(cnt.max())}")
        if anch.shape != (n_anch, 3):
            raise SystemExit(f"[{name}] 앵커 shape {anch.shape} != ({n_anch}, 3)")
        out_cube = int((np.abs((anch - cen) / (sc0 * cap)).max(-1) > 1.0).sum())
        print(f"   cap 적용 후 큐브 밖 앵커 {out_cube}/{n_anch}")

        # 3) 사진 매칭
        print(f"   COLMAP 카메라 ({col}) ...", flush=True)
        views = ba.read_colmap_cameras(col)
        print(f"   사진 {len(views)} 장. 스냅샷 카메라와 매칭 ...", flush=True)
        mapping, worst, unmatched = match_photos(files, views, a.cam_tol)
        print(f"   매칭 {len(mapping)}/{len(files)}  최대 자세 오차 {worst:.2e}")
        if unmatched:
            raise SystemExit(f"[{name}] 사진과 매칭 안 된 스냅샷 {len(unmatched)}: {unmatched[:5]} ... "
                             f"COLMAP 디렉터리가 이 3DGS 학습에 쓴 것과 같은지 확인하세요.")
        pm = {os.path.basename(p): {"image": views[mapping[os.path.basename(p)]]["image"], "scene": scene_key}
              for p in files}
        vp = {scene_key: [{"image": v["image"], "cam": v["cam"].tolist()} for v in views]}
        json.dump(pm, open(os.path.join(a.assets, f"npz_to_image_{T}_{name}.json"), "w"), indent=1)
        json.dump(vp, open(os.path.join(a.assets, f"view_pool_{T}_{name}.json"), "w"))
        vp_both.update(vp)
        for p in files:
            pm_both[os.path.join(rep, os.path.basename(p))] = pm[os.path.basename(p)]
            photo_of.append((name, os.path.basename(pm[os.path.basename(p)]["image"])))
        all_files += files

        hold_idx = [j for j, v in enumerate(views) if (name, os.path.basename(v["image"])) in holdouts]
        for j in hold_idx:
            print(f"   보류 사진 {os.path.basename(views[j]['image'])} = view_pool 인덱스 {j}")
        per_scene.append(dict(name=name, replay=rep, colmap=col, scene_key=scene_key,
                              n_files=len(files), step_min=min(steps), step_max=max(steps),
                              stats=p_raw, stats_cap=p_cap, anchors=p_anc, extent_cap=cap,
                              scale=sc0, keep_cube_before=keep0, keep_cube_after=keep1,
                              n_views=len(views), holdout_view_index=hold_idx))

    # 4) 병합
    p_pm = os.path.join(a.assets, f"npz_to_image_{T}_both.json")
    p_vp = os.path.join(a.assets, f"view_pool_{T}_both.json")
    json.dump(pm_both, open(p_pm, "w"), indent=1)
    json.dump(vp_both, open(p_vp, "w"))

    # 5) blockB split
    tr, va, n_young, n_guard, dropped = block_split(all_files, photo_of, blocks, a.guard, min_step, holdouts)
    p_sp = os.path.join(a.assets, f"split_{T}_both_blockB.json")
    json.dump({"roots": replays, "n_total": len(all_files), "n_train": len(tr), "n_val": len(va),
               "scheme": "contiguous maturity blocks + guard bands + own-photo holdout",
               "blocks": [list(b) for b in blocks], "guard_band": a.guard,
               "min_snapshot_step_assumed": min_step,
               "holdout_photos": [f"{s}:{b}" for s, b in sorted(holdouts)],
               "dropped_own_photo": [all_files[i] for i in dropped],
               "train": tr, "val": va}, open(p_sp, "w"))
    print(f"\n[split] train {len(tr)} / val {len(va)}   "
          f"(어려서 제외 {n_young}, 여유구간 제외 {n_guard}, 보류사진 제외 {len(dropped)})")
    for s in per_scene:
        lo = sum(x["n_files"] for x in per_scene[:per_scene.index(s)])
        hi = lo + s["n_files"]
        print(f"   {s['name']:8s} train {sum(lo <= i < hi for i in tr):5d}  val {sum(lo <= i < hi for i in va):4d}")
    if dropped:
        print("   보류 사진 때문에 뺀 train 스냅샷: " + ", ".join(os.path.basename(all_files[i]) for i in dropped))

    # 6) 실행 설정
    cfg = dict(src)
    cfg.update(root=",".join(replays),
               stats_path=",".join(s["stats_cap"] for s in per_scene),
               scene_anchors=",".join(s["anchors"] for s in per_scene),
               split_path=p_sp, photo_map=p_pm, view_pool=p_vp,
               out_dir="", resume="", init_from="")
    p_cfg = a.config_out or os.path.join("configs", f"{T}.args.json")
    os.makedirs(os.path.dirname(p_cfg) or ".", exist_ok=True)
    json.dump(cfg, open(p_cfg, "w"), indent=2)
    if str(src.get("eval_view_force", "")).strip():
        forced = [int(x) for x in _csv(src["eval_view_force"])]
        for s in per_scene:
            if s["holdout_view_index"] and not set(s["holdout_view_index"]) & set(forced):
                print(f"!! {s['name']}: 보류 사진 view 인덱스 {s['holdout_view_index']} 가 "
                      f"config eval_view_force {forced} 와 다릅니다. --set eval_view_force=... 로 맞추세요.")

    manifest = dict(tag=T, created=datetime.datetime.now().isoformat(timespec="seconds"),
                    config_src=a.config_src, config_out=p_cfg, scenes=per_scene,
                    n_anchors=n_anch, blocks=blocks, guard=a.guard, min_step=min_step,
                    holdout_photos=sorted(holdouts), split=p_sp, photo_map=p_pm, view_pool=p_vp,
                    n_train=len(tr), n_val=len(va),
                    params=dict(stats_stride=a.stats_stride, quantile=a.quantile,
                                anchor_stride=a.anchor_stride, anchor_iters=a.anchor_iters,
                                seed=a.seed, cam_tol=a.cam_tol))
    p_man = os.path.join(a.assets, f"dataset_{T}.json")
    json.dump(manifest, open(p_man, "w"), indent=2)

    print(f"\n실행 설정 -> {p_cfg}")
    print(f"기록      -> {p_man}")
    if a.check:
        print("\n[check] 학습 로더로 검증 ...", flush=True)
        n_tr, n_va = check_datasets(p_cfg)
        if n_tr + n_va == 0:
            raise SystemExit("로더가 스냅샷을 하나도 못 읽었습니다")
    print("\n다음:")
    print(f"   bash scripts/launch_from_config.sh {p_cfg} --dry-run")
    print(f"   bash scripts/launch_from_config.sh {p_cfg} --detached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
