"""Inline formula score: locate formulas in the plans under mplans/ by the anchors in mathgold/ and check whether
each became a placeholder.
Usage: mathscore.py <tag> [show|lost [paper]]
Each formula is recorded as full (entirely in a placeholder), partial (characters left outside the placeholder),
missed (no placeholder), empty (nothing comparable between the two anchors) or lost (anchors not found). When only
whole words such as numbers and units (50 Myr, 250 μm) are left outside, it is recorded as minor instead, since
set as body text they do not hurt the translation; only a single variable, Greek letter or math symbol left
outside is an error (bad_partial, bad_missed). Anchor words turned into placeholders are counted as false
positives. Results are split by body font into two groups: Computer Modern, and other fonts (Times, Nimbus and so
on), in which italic variables use the same font as italic body text."""

import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

E = Path(os.environ.get("EVALSET", Path.home() / "doc-translator-evalset"))
tag = sys.argv[1]
show = len(sys.argv) > 2 and sys.argv[2] == "show"
lostlist = len(sys.argv) > 2 and sys.argv[2] == "lost"
only = sys.argv[3] if len(sys.argv) > 3 else None
MARK = re.compile(r"\{/?(?!v\d+\})[a-z]\w*\}|\{\|\}|\x01/?\w\x02")
WINDOW = 40  # maximum tokens between the left and right anchors
KINDS = ("full", "minor", "empty", "bad_partial", "bad_missed")


def body_fonts() -> dict[str, str]:
    cache = E / "mathgold" / "fonts.json"
    if cache.is_file():
        return json.load(cache.open())
    import pymupdf

    out = {}
    for g in sorted((E / "mathgold").glob("2*.json")):
        doc = pymupdf.open(E / "raw" / g.stem / "paper.pdf")
        counts: Counter = Counter()
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        counts[span["font"].split("+")[-1]] += len(span["text"].strip())
        out[g.stem] = counts.most_common(1)[0][0] if counts else ""
    json.dump(out, cache.open("w"))
    return out


def tokens(rows) -> list[tuple[str, str, str]]:
    """(label t/f, token, kind w/s): w is an alphanumeric string, s a single math symbol. A placeholder adds
    ("f", "§", "s")"""
    out = []

    def add(text: str, label: str) -> None:
        for match in re.finditer(r"[^\W_]+|[^\w\s]", unicodedata.normalize("NFKC", text)):
            piece = match.group()
            if piece[0].isalnum():
                out.append((label, piece.lower(), "w"))
            elif unicodedata.category(piece) == "Sm" or piece in "^_|":
                out.append((label, piece, "s"))

    for row in rows:
        if row[1]:
            continue
        text, formulas = row[2], row[3]
        for part in re.split(r"(\{v\d+\})", MARK.sub(" ", text)):
            m = re.fullmatch(r"\{v(\d+)\}", part)
            if m:
                k = int(m.group(1)) - 1
                out.append(("f", "§", "s"))
                add(MARK.sub(" ", formulas[k]) if k < len(formulas) else "", "f")
            else:
                add(part, "t")
    return out


fonts = body_fonts()
totals = defaultdict(Counter)
for g in sorted((E / "mathgold").glob("2*.json")):
    pid = g.stem
    if only and pid != only:
        continue
    plan = E / "mplans" / f"{pid}.{tag}.json"
    if not plan.is_file():
        continue
    gold = json.load(g.open())
    stream = tokens(json.load(plan.open()))
    eligible = [i for i, tok in enumerate(stream) if tok[2] == "w" and re.fullmatch(r"[a-z0-9]+", tok[1])]
    words = [stream[i][1] for i in eligible]
    starts = defaultdict(list)
    for k in range(len(words)):
        starts[words[k]].append(k)
    group = "cm" if re.match(r"(?:SF|CM|LM|Latin Modern)", fonts.get(pid, ""), re.I) else "other"
    stat = Counter()
    for item in gold["items"]:
        if item["trivial"]:
            continue
        left, right = item["left"], item["right"]
        best = None
        for k in starts.get(left[0], ()):
            if words[k : k + len(left)] != left:
                continue
            end = eligible[k + len(left) - 1]
            for r in starts.get(right[0], ()):
                if r < k + len(left) or words[r : r + len(right)] != right:
                    continue
                gap = eligible[r] - end
                if gap <= WINDOW and (best is None or gap < best[2]):
                    best = (k, r, gap)
                break
        if best is None:
            stat["lost"] += 1
            if lostlist:
                print("LOST", pid, item["src"][:40], "|", " ".join(left), "_", " ".join(right))
            continue
        k, r, _ = best
        anchors = [eligible[i] for i in (*range(k, k + len(left)), *range(r, r + len(right)))]
        stat["anchor_words"] += len(anchors)
        stat["anchor_fp"] += sum(1 for i in anchors if stream[i][0] == "f")
        region = stream[eligible[k + len(left) - 1] + 1 : eligible[r]]
        has_f = any(tok[0] == "f" for tok in region)
        leak = [tok for tok in region if tok[0] == "t"]
        serious = any(tok[2] == "s" or not re.fullmatch(r"\d+|[a-z]{2,}", tok[1]) for tok in leak)
        if has_f and not leak:
            kind = "full"
        elif not leak:
            kind = "empty"
        elif not serious:
            kind = "minor"
        else:
            kind = "bad_partial" if has_f else "bad_missed"
        stat[kind] += 1
        if show and kind.startswith("bad"):
            shown = " ".join(("[" + tok[1] + "]") if tok[0] == "f" else tok[1] for tok in region)
            print(f"{pid} {kind:7} ${item['src'][:50]}$  ->  {shown[:90]}")
    totals[group].update(stat)
    totals["all"].update(stat)
    if not show and not lostlist:
        scored = sum(stat[k] for k in KINDS)
        print(
            f"{pid} {group:5} {fonts.get(pid, '')[:22]:22} n={scored:4} full={stat['full']:4} minor={stat['minor']:3} "
            f"bad_partial={stat['bad_partial']:3} bad_missed={stat['bad_missed']:3} lost={stat['lost']:3} "
            f"fp={stat['anchor_fp']}/{stat['anchor_words']}"
        )
for group in ("cm", "other", "all"):
    s = totals[group]
    scored = sum(s[k] for k in KINDS)
    if scored:
        ok = s["full"] + s["minor"] + s["empty"]
        print(
            f"{tag} {group:5} n={scored} ok={ok} ({100 * ok / scored:.1f}%) full={s['full']} minor={s['minor']} "
            f"bad_partial={s['bad_partial']} bad_missed={s['bad_missed']} lost={s['lost']} "
            f"fp={s['anchor_fp']}/{s['anchor_words']}"
        )
