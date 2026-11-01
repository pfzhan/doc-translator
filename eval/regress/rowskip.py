"""Compare two versions of plans unit by unit: files whose unit count changed, and units at the same position whose
skip flag changed. Usage: rowskip.py tagA tagB [how many to show]"""

import glob
import json
import os
import sys
from pathlib import Path

E = Path(os.environ.get("EVALSET", Path.home() / "doc-translator-evalset"))
a, b = sys.argv[1:3]
show = int(sys.argv[3]) if len(sys.argv) > 3 else 5
for fa in sorted(glob.glob(f"{E}/regress/plans/*.{a}.json")):
    name = fa.split("/")[-1][: -len(f".{a}.json")]
    A, B = json.loads(Path(fa).read_text()), json.loads(Path(fa.replace(f".{a}.json", f".{b}.json")).read_text())
    if len(A) != len(B):
        print(name, "units", len(A), "->", len(B))
        continue
    pages = sum(1 for x, y in zip(A, B) if x[0] != y[0])
    flips = [(x, y) for x, y in zip(A, B) if x[1] != y[1]]
    print(name, "units", len(A), "page moves", pages, "skip changes", len(flips))
    for x, y in flips[:show]:
        print(f"   p{x[0]} {x[1]} -> {y[1]} {y[2][:100]!r}")
