"""逐页比两个版面框目录（a 当参照）：每类 a 有几个、b 里对上几个（同类 IoU>=0.5）、b 多出几个。
用法: boxcmp.py <a 目录> <b 目录>"""
import json, sys
from collections import Counter
from pathlib import Path
a_dir, b_dir = map(Path, sys.argv[1:3])

def iou(r, s):
    w = min(r[2], s[2]) - max(r[0], s[0]); h = min(r[3], s[3]) - max(r[1], s[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((r[2] - r[0]) * (r[3] - r[1]) + (s[2] - s[0]) * (s[3] - s[1]) - inter)

ref, hit, extra = Counter(), Counter(), Counter()
files = 0
for fa in sorted(a_dir.glob("*.json")):
    fb = b_dir / fa.name
    if not fb.exists():
        continue
    files += 1
    for pa, pb in zip(json.load(open(fa)), json.load(open(fb))):
        used = set()
        for la, ra, *_ in pa:
            ref[la] += 1
            best = max(((iou(ra, rb), k) for k, (lb, rb, *_) in enumerate(pb) if lb == la and k not in used), default=(0, -1))
            if best[0] >= 0.5:
                hit[la] += 1; used.add(best[1])
        for k, (lb, *_rest) in enumerate(pb):
            if k not in used:
                extra[lb] += 1
print(files, "files")
for label in sorted(ref | extra, key=lambda l: -ref[l]):
    print(f"{label:16s} ref {ref[label]:6d}  matched {hit[label]:6d} ({100*hit[label]/max(ref[label],1):5.1f}%)  extra {extra[label]:5d}")
t = sum(ref.values()); print(f"{'all':16s} ref {t:6d}  matched {sum(hit.values()):6d} ({100*sum(hit.values())/max(t,1):5.1f}%)  extra {sum(extra.values()):5d}")
