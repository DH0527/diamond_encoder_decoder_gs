"""Rebuild a train.py argv list from a saved args.json (last-wins already resolved)."""
from __future__ import annotations

import argparse
import json
import sys
from argparse import _HelpAction, _StoreFalseAction, _StoreTrueAction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from can3tok.train import build_parser  # noqa: E402


def ns_to_argv(ns: argparse.Namespace, skip: set[str] | None = None) -> list[str]:
    skip = skip or set()
    parser = build_parser()
    argv: list[str] = []
    for act in parser._actions:
        if isinstance(act, _HelpAction) or not act.option_strings:
            continue
        dest = act.dest
        if dest in skip:
            continue
        opt = act.option_strings[0]
        val = getattr(ns, dest, None)
        if isinstance(act, _StoreTrueAction):
            if val:
                argv.append(opt)
            continue
        if isinstance(act, _StoreFalseAction):
            if not val:
                argv.append(opt)
            continue
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            argv.append(opt)
            argv.extend(str(x) for x in val)
        else:
            argv.extend([opt, str(val)])
    return argv


def load_ns(path: str, overrides: dict) -> argparse.Namespace:
    raw = json.loads(Path(path).read_text())
    raw.update(overrides)
    return argparse.Namespace(**raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--set", action="append", default=[],
                    help="dest=value overrides, repeatable")
    a = ap.parse_args()
    ov: dict = {}
    for item in a.set:
        k, _, v = item.partition("=")
        ov[k] = v
    print("\x00".join(ns_to_argv(load_ns(a.src, ov))))


if __name__ == "__main__":
    main()
