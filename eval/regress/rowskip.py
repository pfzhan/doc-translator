"""两版计划按单元顺序比：单元数变了的文件，和同一位置上跳过与否变了的单元。用法: rowskip.py tagA tagB [显示几条]"""
import os
from pathlib import Path
import json, glob, sys
E = Path(os.environ.get("EVALSET", Path.home() / "doc-translator-evalset"))
a, b = sys.argv[1:3]
show = int(sys.argv[3]) if len(sys.argv) > 3 else 5
for fa in sorted(glob.glob(f"{E}/regress/plans/*.{a}.json")):
    name = fa.split("/")[-1][: -len(f".{a}.json")]
    A, B = json.load(open(fa)), json.load(open(fa.replace(f".{a}.json", f".{b}.json")))
    if len(A) != len(B):
        print(name, "units", len(A), "->", len(B))
        continue
    pages = sum(1 for x, y in zip(A, B) if x[0] != y[0])
    flips = [(x, y) for x, y in zip(A, B) if x[1] != y[1]]
    print(name, "units", len(A), "page moves", pages, "skip changes", len(flips))
    for x, y in flips[:show]:
        print("   p%d %s -> %s %r" % (x[0], x[1], y[1], y[2][:100]))
