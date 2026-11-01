"""Compare two versions of plans unit by unit and print only the differing parts.
Usage: segdiff.py a_tag b_tag [name]"""

import difflib
import glob
import json
import os
import sys
from pathlib import Path

E = Path(os.environ.get("EVALSET", Path.home() / "doc-translator-evalset"))
a_tag, b_tag = sys.argv[1], sys.argv[2]
only = sys.argv[3] if len(sys.argv) > 3 else None


def fmt(r):
    # Plans with unit boxes (the seg format of plandump_mfdcache.py) have an extra column of boxes, not compared
    p, s, t, fs = r[:4]
    return f"{t} || " + " ; ".join(x[0].replace("\x01", "").replace("\x02", "") for x in fs), p, s


for old in sorted(glob.glob(f"{E}/regress/plans/*.{a_tag}.json")):
    name = old.split("/")[-1][: -len(f".{a_tag}.json")]
    if only and name != only:
        continue
    a = json.loads(Path(old).read_text())
    b = json.loads(Path(old.replace(f".{a_tag}.json", f".{b_tag}.json")).read_text())
    if a == b:
        continue
    fa = [fmt(r) for r in a]
    fb = [fmt(r) for r in b]
    sm = difflib.SequenceMatcher(None, [x[0] for x in fa], [x[0] for x in fb], autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            for x, y in zip(fa[i1:i2], fb[j1:j2]):
                m = difflib.SequenceMatcher(None, x[0], y[0], autojunk=False)
                parts = []
                for o, a1, a2, b1, b2 in m.get_opcodes():
                    if o != "equal":
                        parts.append(f"[{x[0][max(0, a1 - 25) : a2 + 10]!r} -> {y[0][max(0, b1 - 25) : b2 + 10]!r}]")
                print(f"{name} p{x[1] + 1} skip {x[2]}->{y[2]}: " + " ".join(parts)[:600])
        else:
            for x in fa[i1:i2]:
                print(f"{name} - p{x[1] + 1} {x[2]} {x[0][:200]!r}")
            for y in fb[j1:j2]:
                print(f"{name} + p{y[1] + 1} {y[2]} {y[0][:200]!r}")
