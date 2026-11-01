"""Compare the plans of two tags paper by paper and list newly joined units (boundaries missing in the second).
Usage: joindiff.py a b"""

import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

E = Path(os.environ.get("EVALSET", Path.home() / "doc-translator-evalset"))
a, b = sys.argv[1:3]
for pa in sorted((E / "plans").glob(f"*.{a}.json")):
    id = pa.name.split(f".{a}.")[0]
    pb = E / "plans" / f"{id}.{b}.json"
    if not pb.exists():
        continue
    ua = [r[2] for r in json.loads(Path(pa).read_text())]
    ub = [r[2] for r in json.loads(Path(pb).read_text())]
    for op, i1, i2, _j1, _j2 in SequenceMatcher(None, ua, ub, autojunk=False).get_opcodes():
        if op != "equal":
            print(id, op, "|".join(t[-50:] + " ‖ " for t in ua[i1:i2])[:300].replace("\n", " "))
