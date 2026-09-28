"""decode_compact 경로의 모듈 트리 + 파라미터 수. 부분 해동 사다리의 후보를 고르기 위한 것."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

D = load(sys.argv[1], "val", 166, 1)
m = D["model"]
tot = sum(p.numel() for p in m.parameters())
print(f"[model] {type(m).__name__}  total {tot/1e6:.2f}M")
for n, c in m.named_children():
    k = sum(p.numel() for p in c.parameters())
    if k == 0: continue
    print(f"  {n:22s} {type(c).__name__:24s} {k/1e6:8.3f}M")
    for n2, c2 in c.named_children():
        k2 = sum(p.numel() for p in c2.parameters())
        if k2 == 0: continue
        print(f"    {n2:20s} {type(c2).__name__:24s} {k2/1e6:8.3f}M")
