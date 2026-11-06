"""Recognize the letters of bitmap fonts with tesseract and vote on the label of each glyph shape."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

import pymupdf

from .pdf_type3_glyphs import Glyph, Key, Label, OcrChar, OcrWord, _GlyphCenters, _sure

_OCR_SCALE = 3.0


_OCR_TIMEOUT = 90


_LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


# ---- Recognition ----


def _ocr_pages(doc: pymupdf.Document, numbers: list[int]) -> dict[int, list[OcrWord]]:
    """Recognize the whole document together. Rendering runs on the main thread and tesseract in parallel; each page
    is recognized as soon as it is rendered. A page that times out casts no votes."""
    if shutil.which("tesseract") is None:
        raise ValueError("此 PDF 的文字为不含 Unicode 编码的点阵字体，需要安装 tesseract 才能识别原文")

    def read(path: Path) -> list[OcrWord]:
        try:
            return ocr_image(path)
        except subprocess.TimeoutExpired:
            return []

    workers = max(1, min(4, os.cpu_count() or 1, len(numbers)))
    with tempfile.TemporaryDirectory(prefix="type3-") as tmp, ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for pno in numbers:
            path = Path(tmp) / f"p{pno}.png"
            doc[pno].get_pixmap(matrix=pymupdf.Matrix(_OCR_SCALE, _OCR_SCALE), alpha=False).save(path)
            pending[pno] = pool.submit(read, path)
        return {pno: future.result() for pno, future in pending.items()}


def ocr_image(path: Path, scale: float = _OCR_SCALE) -> list[OcrWord]:
    exe = shutil.which("tesseract")
    if exe is None:
        return []
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    proc = subprocess.run(
        [exe, str(path), "stdout", "--psm", "3", "-l", "eng", "-c", "hocr_char_boxes=1", "hocr"],
        capture_output=True,
        text=True,
        timeout=_OCR_TIMEOUT,
        check=False,
        env=env,
    )
    if proc.returncode != 0 and "traineddata" in proc.stderr:
        raise ValueError("点阵字体页面未识别出文字，请确认已安装 tesseract 英文语言包")
    parser = _HocrWords(scale)
    parser.feed(proc.stdout)
    return parser.words


class _HocrWords(HTMLParser):
    """Words and per-character boxes in hOCR. Character boxes and confidences are on ocrx_cinfo."""

    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.words: list[OcrWord] = []
        self._stack: list[str] = []
        self._word: tuple[pymupdf.Rect, float] | None = None
        self._chars: list[OcrChar] = []
        self._char: tuple[pymupdf.Rect, float] | None = None
        self._text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "span":
            return
        info = dict(attrs)
        kind = info.get("class") or ""
        title = info.get("title") or ""
        self._stack.append(kind)
        if kind == "ocrx_word":
            box = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", title)
            conf = re.search(r"x_wconf ([\d.]+)", title)
            self._word = (self._rect(box), float(conf.group(1)) if conf else 0.0) if box else None
            self._chars = []
        elif kind == "ocrx_cinfo":
            box = re.search(r"x_bboxes (\d+) (\d+) (\d+) (\d+)", title)
            conf = re.search(r"x_conf ([\d.]+)", title)
            self._char = (self._rect(box), float(conf.group(1)) if conf else 0.0) if box else None
            self._text = ""

    def handle_data(self, data: str) -> None:
        if self._char is not None:
            self._text += data

    def handle_endtag(self, tag: str) -> None:
        if tag != "span" or not self._stack:
            return
        kind = self._stack.pop()
        if kind == "ocrx_cinfo" and self._char is not None:
            text = _LIGATURES.get(self._text.strip(), self._text.strip())
            if text:
                self._chars.append(OcrChar(text, *self._char))
            self._char = None
        elif kind == "ocrx_word" and self._word is not None:
            text = "".join(char.text for char in self._chars)
            if text:
                self.words.append(OcrWord(text, self._word[0], self._word[1], tuple(self._chars)))
            self._word = None

    def _rect(self, box: re.Match[str]) -> pymupdf.Rect:
        return pymupdf.Rect(*(int(box.group(index)) / self.scale for index in range(1, 5)))


def _vote_labels(glyphs: dict[int, list[Glyph]], words: dict[int, list[OcrWord]]) -> dict[Key, Label]:
    """Recognized characters vote on the glyphs they fall on by position.

    Words whose character count matches the glyph count are paired one by one, which settles most glyphs. Words that
    do not match (the ligature ffi is one glyph, OCR missed or added a character) are then aligned by the settled
    glyphs and by position: a reliably recognized glyph takes its own character, and the remaining characters go to
    the unsettled glyphs.
    """
    # Votes are confidences, not counts: a plain float dict, since Counter is typed with int values
    votes: dict[Key, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    uneven: list[tuple[list[Glyph], tuple[OcrChar, ...]]] = []
    for pno, page_words in words.items():
        centers = _GlyphCenters(glyphs.get(pno, []))
        for word in page_words:
            box = pymupdf.Rect(word.rect.x0 - 0.7, word.rect.y0 - 0.7, word.rect.x1 + 0.7, word.rect.y1 + 0.7)
            inside = sorted(centers.inside(box), key=lambda glyph: glyph.rect.x0)
            if not inside or not word.chars:
                continue
            # When the counts match, pair in order: the OCR character order is reliable, while per-character boxes
            # are often two or three points off
            if len(inside) == len(word.chars):
                for glyph, char in zip(inside, word.chars):
                    votes[glyph.key][char.text] += char.conf
            else:
                uneven.append((inside, word.chars))
    first = _best_labels(votes)
    for inside, chars in uneven:
        for glyph, taken in zip(inside, _align(inside, chars, first)):
            if taken and not _sure(first.get(glyph.key)):
                votes[glyph.key]["".join(char.text for char in taken)] += min(char.conf for char in taken) / 2
    return _best_labels(votes)


def _best_labels(votes: dict[Key, dict[str, float]]) -> dict[Key, Label]:
    labels: dict[Key, Label] = {}
    for key, counter in votes.items():
        total = sum(counter.values())
        if total > 0:
            # The first highest entry, as Counter.most_common(1) picked
            text, weight = max(counter.items(), key=lambda item: item[1])
            labels[key] = Label(text, weight / total)
    return labels


def _align(
    glyphs: list[Glyph],
    chars: tuple[OcrChar, ...],
    labels: dict[Key, Label],
) -> list[tuple[OcrChar, ...]]:
    """Assign the recognized characters to the glyphs in order, 0 to 3 characters per glyph, with the lowest cost.

    A settled glyph should get its own character; a character that does not overlap its glyph horizontally adds cost,
    so an extra dot recognized by OCR does not push all following characters out of place.
    """
    count, total = len(glyphs), len(chars)
    inf = float("inf")
    cost = [[inf] * (total + 1) for _ in range(count + 1)]
    back = [[0] * (total + 1) for _ in range(count + 1)]
    cost[0][0] = 0.0
    for index in range(1, count + 1):
        label = labels.get(glyphs[index - 1].key)
        for used in range(total + 1):
            for take in range(0, min(3, used) + 1):
                prev = cost[index - 1][used - take]
                if prev == inf:
                    continue
                taken = chars[used - take : used]
                text = "".join(char.text for char in taken)
                if _sure(label):
                    step = 0.0 if text == label.text else 2.0
                else:
                    step = 1.5 if take == 0 else 0.3 * (take - 1)
                # Characters assigned to a glyph must overlap it horizontally
                step += sum(1.0 for char in taken if _overlap(glyphs[index - 1].rect, char.rect) <= 0)
                if prev + step < cost[index][used]:
                    cost[index][used] = prev + step
                    back[index][used] = take
    if cost[count][total] == inf:
        return [() for _ in glyphs]
    assigned: list[tuple[OcrChar, ...]] = []
    used = total
    for index in range(count, 0, -1):
        take = back[index][used]
        assigned.append(tuple(chars[used - take : used]))
        used -= take
    return assigned[::-1]


def _overlap(left: pymupdf.Rect, right: pymupdf.Rect) -> float:
    return min(left.x1, right.x1) - max(left.x0, right.x0)
