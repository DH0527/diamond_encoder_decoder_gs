#!/usr/bin/env python3
"""torchrun entry point: torchrun --nproc_per_node=4 train.py --root ... --out_dir ..."""

from can3tok.train import main

if __name__ == "__main__":
    main()
