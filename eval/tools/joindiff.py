"""两个标签的计划逐篇比，列出新并上的单元（后一版少了的边界）。用法: joindiff.py a b"""
import os
import json, sys, re
from pathlib import Path
from difflib import SequenceMatcher
E = Path(os.environ.get("EVALSET", Path.home() / "doc-translator-evalset")); a, b = sys.argv[1:3]
for pa in sorted((E / "plans").glob(f"*.{a}.json")):
    id = pa.name.split(f".{a}.")[0]; pb = E / "plans" / f"{id}.{b}.json"
    if not pb.exists():
        continue
    ua = [r[2] for r in json.load(open(pa))]; ub = [r[2] for r in json.load(open(pb))]
    for op, i1, i2, j1, j2 in SequenceMatcher(None, ua, ub, autojunk=False).get_opcodes():
        if op != "equal":
            print(id, op, "|".join(t[-50:] + " ‖ " for t in ua[i1:i2])[:300].replace("\n", " "))
