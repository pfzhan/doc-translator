"""合并 PDF 里内容相同的对象。

双语版左边引用原书的页面，右边引用译文文档的页面。译文文档是原书的副本，图片、字体、
表单各有一份，原样存下来文件大一倍。MuPDF 的 garbage=4 能合并它们，但它两两比较所有
对象，一本几百页的书要十几分钟。这里按内容的哈希分组，一遍就找出重复。

一个对象合并到另一个以后，引用它的对象也可能变得相同（图片的蒙版合并后，两张图片的
字典就一样了），所以反复找，直到没有新的重复。被合并的对象不再有引用，保存时用
garbage=1 去掉。
"""
from __future__ import annotations

import hashlib
import re

import pymupdf

# 有身份的对象：同样内容的两页仍是两页。树里的节点带 /Parent，合并会弄乱树
_IDENTITY_RE = re.compile(r"/Type\s*/(?:Page|Pages|Catalog|Annot|Outlines|StructTreeRoot|StructElem)\b|/Parent\b")
_REF_RE = re.compile(r"(?<![\w.])(\d+) 0 R(?!\w)")


def merge_duplicates(doc: pymupdf.Document) -> int:
    """把引用指向内容相同的第一份对象，返回合并掉的对象数。"""
    gone: set[int] = set()
    while True:
        first_of: dict[tuple[str, bytes | None], int] = {}
        remap: dict[int, int] = {}
        for xref in range(1, doc.xref_length()):
            if xref in gone:
                continue
            key = _content_key(doc, xref)
            if key is None:
                continue
            first = first_of.setdefault(key, xref)
            if first != xref:
                remap[xref] = first
        if not remap:
            return len(gone)
        gone.update(remap)
        for xref in range(1, doc.xref_length()):
            if xref not in gone:
                _redirect(doc, xref, remap)


def _content_key(doc: pymupdf.Document, xref: int) -> tuple[str, bytes | None] | None:
    text = doc.xref_object(xref, compressed=True)
    if doc.xref_is_stream(xref):
        return text, hashlib.sha1(doc.xref_stream_raw(xref) or b"").digest()
    if not text.startswith(("<<", "[")) or _IDENTITY_RE.search(text):
        return None
    return text, None


def _redirect(doc: pymupdf.Document, xref: int, remap: dict[int, int]) -> None:
    if not doc.xref_is_stream(xref):
        text = doc.xref_object(xref, compressed=True)
        new = replace_refs(text, remap)
        if new != text:
            doc.update_object(xref, new)
        return
    # update_object 会丢掉从文件读进来的流，流对象逐项改字典
    for key in doc.xref_get_keys(xref):
        kind, value = doc.xref_get_key(xref, key)
        if kind in ("xref", "dict", "array"):
            new = replace_refs(value, remap)
            if new != value:
                doc.xref_set_key(xref, key, new)


def replace_refs(text: str, remap: dict[int, int]) -> str:
    """改对象文本里的间接引用。字符串里写着的 "12 0 R" 不是引用，不改。"""
    parts: list[str] = []
    for piece, is_code in _split_strings(text):
        if is_code:
            piece = _REF_RE.sub(lambda m: f"{remap.get(int(m.group(1)), int(m.group(1)))} 0 R", piece)
        parts.append(piece)
    return "".join(parts)


def _split_strings(text: str) -> list[tuple[str, bool]]:
    """切成 (片段, 是否在字符串外)。字面串 (...) 可以嵌套括号、用反斜杠转义；
    十六进制串 <...> 和字典的 << 区分开。"""
    out: list[tuple[str, bool]] = []
    start = index = 0
    size = len(text)
    while index < size:
        char = text[index]
        if char == "(":
            end, depth = index + 1, 1
            while end < size and depth:
                if text[end] == "\\":
                    end += 2
                    continue
                depth += {"(": 1, ")": -1}.get(text[end], 0)
                end += 1
        elif char == "<" and text[index + 1:index + 2] != "<" and text[index - 1:index] != "<":
            end = text.find(">", index)
            end = size if end < 0 else end + 1
        else:
            index += 1
            continue
        out.append((text[start:index], True))
        out.append((text[index:end], False))
        start = index = end
    out.append((text[start:], True))
    return out
