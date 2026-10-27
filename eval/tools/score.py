"""按标准答案给一份分段计划打分。用法: score.py plan.json gold.json [列出错误]
每段源文在计划里找开头和结尾：段首、段尾 12 个词以内、全文只出现一次的 4 词片段。
并错：相邻两段（段落边界）的前段尾和后段首落在同一个单元。
拆错：一段的段首和段尾落在不同单元（行间公式两边拆开不算，见 gold.py）。"""
import json, re, sys, unicodedata
from collections import defaultdict

plan = json.load(open(sys.argv[1]))
gold = json.load(open(sys.argv[2]))["segments"]
show = len(sys.argv) > 3

stream, unit_of, unit_text = [], [], []
for index, row in enumerate(plan):
    text = re.sub(r"\{/?[a-z]\}|\{[a-z]?\d+\}|\{z\d+\}|\{c[0-9a-f]{6}\}|\{/[cz]\}", " ", row[2])
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    words = [w for w in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower()) if len(w) >= 2]
    stream.extend(words)
    unit_of.extend([index] * len(words))
    unit_text.append(row[2])

grams = defaultdict(list)
for i in range(len(stream) - 3):
    grams[tuple(stream[i:i + 4])].append(i)

def locate(words, from_end):
    offsets = range(len(words) - 4, max(len(words) - 16, -1), -1) if from_end else range(0, min(12, len(words) - 3))
    for k in offsets:
        hits = grams.get(tuple(words[k:k + 4]))
        if hits and len(hits) == 1:
            return unit_of[hits[0] + (3 if from_end else 0)]
    return None

spans = [(locate(seg["words"], False), locate(seg["words"], True)) for seg in gold]
found = sum(1 for a, b in spans if a is not None and b is not None)
split = merged = boundaries = 0
causes = defaultdict(int)
errors = []
for i, ((a, b), seg) in enumerate(zip(spans, gold)):
    if a is not None and b is not None and a != b:
        split += 1
        cause = "other"
        first, second = plan[b - 1] if b > a else plan[a], plan[b]
        if len(second) > 4 and len(first) > 4:
            last_page, last_rect = first[4][0], first[4][2]
            head_rect = second[4][1]
            if second[0] != last_page:
                cause = "page"
            elif head_rect[0] > last_rect[2] - 1 or head_rect[1] < last_rect[1] - 1:
                cause = "column"
        if cause == "other" and unit_text[b].lstrip().startswith("{v"):
            cause = "formula"
        causes[cause] += 1
        errors.append(("SPLIT", cause, " ".join(seg["words"][:8]), unit_text[b - 1][-60:] if b > a else "", unit_text[b][:60]))
    if seg["next"] == "par" and i + 1 < len(gold):
        nxt = spans[i + 1][0]
        if b is not None and nxt is not None:
            boundaries += 1
            if b == nxt:
                merged += 1
                errors.append(("MERGE", " ".join(seg["words"][-6:]), " ".join(gold[i + 1]["words"][:6]), unit_text[b][:80]))
print(json.dumps({"segments": len(gold), "found": found, "split": split, "boundaries": boundaries, "merged": merged, **{"split_" + k: v for k, v in causes.items()}}))
if show:
    for e in errors:
        print("  ", *[repr(x) for x in e])
