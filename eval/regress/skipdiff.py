"""两份计划里文本相同、skip 不同的单元。用法: skipdiff.py tagA tagB"""
import os
from pathlib import Path
import json, glob, sys, collections
E = Path(os.environ.get("EVALSET", Path.home() / "doc-translator-evalset"))
a, b = sys.argv[1:3]
for fa in sorted(glob.glob(f"{E}/regress/plans/*.{a}.json")):
    name = fa.split("/")[-1][: -len(f".{a}.json")]
    fb = fa.replace(f".{a}.json", f".{b}.json")
    A = collections.Counter((u[0], u[2], u[1]) for u in json.load(open(fa)))
    B = collections.Counter((u[0], u[2], u[1]) for u in json.load(open(fb)))
    for key in sorted(set(A) | set(B)):
        page, text, skip = key
        if A[key] != B[key] and (page, text, not skip) in (B if A[key] > B[key] else A):
            if A[key] > B[key]:
                print(name, f"p{page + 1}", f"{skip}->{not skip}", repr(text[:90]))
