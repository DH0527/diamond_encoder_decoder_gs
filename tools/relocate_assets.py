"""assets/*.json 에 박힌 절대경로를 새 서버 경로로 바꾼다.

자산 JSON 은 원래 서버의 절대경로를 그대로 담고 있다. 접두어는 네 종류뿐이다.

    /data/daeho/aaaa_proj/seondo/vanilla-3dgs   vanilla 3DGS replay npz  (현재 학습)
    /data/daeho/aaaa_proj/seondo/speedy-splat   Speedy-Splat replay npz  (이전 실험)
    /data/daeho/train_colmap                    Tanks&Temples train COLMAP + 사진
    /data/daeho/truck_colmap                    Tanks&Temples truck COLMAP + 사진

값만이 아니라 **키**도 바꾼다. npz_to_image_*.json 은 npz 전체 경로를 키로 쓰기 때문에
(data.py 가 씬이 여러 개일 때 basename 충돌을 막으려고 그렇게 한다) 값만 바꾸면
로더가 KeyError 를 낸다.

    # 무엇이 바뀌는지만 본다
    python tools/relocate_assets.py --dry-run \
        --map /data/daeho/aaaa_proj/seondo/vanilla-3dgs=/mnt/data/vanilla-3dgs \
        --map /data/daeho/train_colmap=/mnt/data/train_colmap \
        --map /data/daeho/truck_colmap=/mnt/data/truck_colmap

    # 실제로 쓴다 (원본은 --backup 디렉터리에 복사)
    python tools/relocate_assets.py --backup assets_orig --map ... --map ...

--map 은 여러 번 줄 수 있고, 긴 접두어부터 먼저 적용한다. 접두어 뒤가 경로 경계
('/' 또는 문자열 끝) 일 때만 바꾸므로 /data/x 가 /data/xy 를 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import Counter


def parse_maps(items):
    maps = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--map 형식은 OLD=NEW 입니다: {it!r}")
        old, new = it.split("=", 1)
        old, new = old.rstrip("/"), new.rstrip("/")
        if not old:
            raise SystemExit(f"빈 OLD 접두어: {it!r}")
        maps.append((old, new))
    # 긴 접두어 먼저: /a/b 가 /a 보다 먼저 맞아야 한다
    return sorted(maps, key=lambda m: len(m[0]), reverse=True)


def _rewrite_one(s, maps, hits):
    for old, new in maps:
        if s == old or s.startswith(old + "/"):
            hits[old] += 1
            return new + s[len(old):]
    return s


def rewrite_str(s, maps, hits):
    # train.py 의 --root / --stats_path / --scene_anchors 는 쉼표로 이은 경로 목록이다
    # (configs/*.args.json). 앞부분만 보면 두 번째 씬 경로를 놓친다.
    if "," in s and s.startswith("/"):
        return ",".join(_rewrite_one(p, maps, hits) for p in s.split(","))
    return _rewrite_one(s, maps, hits)


def rewrite(obj, maps, hits):
    if isinstance(obj, str):
        return rewrite_str(obj, maps, hits)
    if isinstance(obj, list):
        return [rewrite(v, maps, hits) for v in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = rewrite_str(k, maps, hits) if isinstance(k, str) else k
            if nk in out:
                raise SystemExit(f"키 충돌: 두 키가 같은 새 경로 {nk!r} 로 바뀝니다")
            out[nk] = rewrite(v, maps, hits)
        return out
    return obj


def leftover_prefixes(obj, acc, new_roots):
    """재배치 후에도 새 경로 아래에 있지 않은 절대경로 = --map 을 빠뜨린 것."""
    if isinstance(obj, str):
        for part in (obj.split(",") if "," in obj else [obj]):
            if part.startswith("/") and not any(part == r or part.startswith(r + "/") for r in new_roots):
                acc["/".join(part.split("/")[:4])] += 1
    elif isinstance(obj, list):
        for v in obj:
            leftover_prefixes(v, acc, new_roots)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            leftover_prefixes(k, acc, new_roots)
            leftover_prefixes(v, acc, new_roots)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--assets", default="assets", help="자산 디렉터리 (기본 assets)")
    ap.add_argument("--map", action="append", default=[], metavar="OLD=NEW",
                    help="경로 접두어 치환. 여러 번 지정 가능")
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 바뀔 개수만 출력")
    ap.add_argument("--backup", default="", help="쓰기 전에 원본 JSON 을 이 디렉터리로 복사")
    a = ap.parse_args(argv)

    if not a.map:
        ap.error("--map 을 하나 이상 주세요")
    maps = parse_maps(a.map)
    files = sorted(glob.glob(os.path.join(a.assets, "*.json")))
    if not files:
        raise SystemExit(f"{a.assets}/*.json 이 없습니다")

    if a.backup and not a.dry_run:
        os.makedirs(a.backup, exist_ok=True)

    total = Counter()
    changed_files = 0
    for p in files:
        with open(p) as f:
            data = json.load(f)
        hits = Counter()
        new = rewrite(data, maps, hits)
        n = sum(hits.values())
        total.update(hits)
        if n == 0:
            continue
        changed_files += 1
        print(f"  {os.path.basename(p):42s} {n:>7,} 곳")
        if not a.dry_run:
            if a.backup:
                shutil.copy2(p, os.path.join(a.backup, os.path.basename(p)))
            tmp = p + ".tmp"
            with open(tmp, "w") as f:
                # 원본과 같은 형태로: 들여쓰기 있던 파일은 유지, 없던 파일은 압축
                with open(p) as g:
                    indented = "\n " in g.read(4096)
                json.dump(new, f, indent=1 if indented else None, ensure_ascii=False)
            os.replace(tmp, p)

    print()
    print(f"{'(dry-run) ' if a.dry_run else ''}파일 {changed_files}/{len(files)} 개, 접두어별 치환 횟수:")
    for old, new in maps:
        print(f"  {total[old]:>8,}  {old}  ->  {new}")

    if not a.dry_run:
        left = Counter()
        new_roots = [new for _, new in maps]
        for p in files:
            with open(p) as f:
                leftover_prefixes(json.load(f), left, new_roots)
        if left:
            print("\n새 경로로 옮겨지지 않은 절대경로 (쓰는 자산이면 --map 을 추가하세요):")
            for k, v in left.most_common(10):
                print(f"  {v:>8,}  {k}")
        else:
            print("\n모든 절대경로가 새 경로 아래로 옮겨졌습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
