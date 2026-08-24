#!/usr/bin/env python3
"""Did a speed change alter the labels? Compare two runs pixel by pixel.

Speed work is only allowed to be free if it is free, so this reports the exact number of
pixels that changed class and which way they went -- not just the summary shares, which
can hide two errors that cancel.

Usage: compare_masks.py <run_a> <run_b>
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

R = Path("/home/paul/SyncAI-Lib-HydraNet/runs/site30k_qa")
A, B = R / sys.argv[1] / "masks", R / sys.argv[2] / "masks"
NAMES = {0: "void", 1: "floor", 2: "wall", 3: "column", 4: "display_table", 5: "shelf",
         6: "person", 7: "laptop", 8: "tablet", 9: "phone", 10: "boxed_stock", 255: "unlabelled"}

files = sorted(f.name for f in A.glob("*.png"))
missing = [f for f in files if not (B / f).exists()]
if missing:
    print(f"!! {len(missing)} frames missing from {B}")
total = changed = 0
moves: Counter = Counter()
per_frame = []
for f in files:
    if not (B / f).exists():
        continue
    a = np.asarray(Image.open(A / f))
    b = np.asarray(Image.open(B / f))
    diff = a != b
    total += a.size
    changed += int(diff.sum())
    per_frame.append((f, int(diff.sum())))
    if diff.any():
        for (x, y), n in Counter(zip(a[diff].tolist(), b[diff].tolist())).items():
            moves[(x, y)] += n

print(f"{len(per_frame)} frames compared: {changed} of {total} pixels differ "
      f"({100 * changed / max(total, 1):.4f}%)")
per_frame.sort(key=lambda r: -r[1])
for name, n in per_frame[:5]:
    print(f"   {name:52s} {n:7d} px  ({100 * n / (1920 * 1080):.3f}% of the frame)")
if moves:
    print("\n   biggest class moves (old -> new):")
    for (x, y), n in moves.most_common(8):
        print(f"     {NAMES.get(x, x):14s} -> {NAMES.get(y, y):14s} {n:7d} px")
