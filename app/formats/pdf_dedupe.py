"""Merge objects with identical content in a PDF.

The left side of the bilingual version references pages of the original book, the right side pages of the
translated document. The translated document is a copy of the original, so images, fonts and forms each exist
twice, and saved as is the file doubles in size. MuPDF's garbage=4 can merge them, but it compares all objects
pairwise and takes over ten minutes for a book of a few hundred pages. Here objects are grouped by a hash of
their content, which finds duplicates in one pass.

After one object is merged into another, objects referencing it may become identical too (after the masks of
two images are merged, the image dictionaries are the same), so the search repeats until no new duplicates
appear. Merged objects are no longer referenced and are removed with garbage=1 on save.
"""

from __future__ import annotations

import hashlib
import re

import pymupdf

# Objects with identity: two pages with the same content are still two pages. Tree nodes carry /Parent, and
# merging them would corrupt the tree
_IDENTITY_RE = re.compile(r"/Type\s*/(?:Page|Pages|Catalog|Annot|Outlines|StructTreeRoot|StructElem)\b|/Parent\b")
_REF_RE = re.compile(r"(?<![\w.])(\d+) 0 R(?!\w)")


def merge_duplicates(doc: pymupdf.Document) -> int:
    """Point references to the first of the objects with identical content and return the number of objects
    merged away."""
    gone: set[int] = set()
    # Merging changes only references in dictionaries and leaves stream content unchanged, so each hash is
    # computed once
    digests: dict[int, bytes] = {}
    while True:
        first_of: dict[tuple[str, bytes | None], int] = {}
        remap: dict[int, int] = {}
        texts: dict[int, str] = {}
        for xref in range(1, doc.xref_length()):
            if xref in gone:
                continue
            text = texts[xref] = doc.xref_object(xref, compressed=True)
            key = _content_key(doc, xref, text, digests)
            if key is None:
                continue
            first = first_of.setdefault(key, xref)
            if first != xref:
                remap[xref] = first
        if not remap:
            return len(gone)
        gone.update(remap)
        # Change only objects that reference a merged object. A "12 0 R" written inside a string is also picked
        # here, and _redirect leaves it unchanged
        for xref, text in texts.items():
            if xref not in gone and any(int(m.group(1)) in remap for m in _REF_RE.finditer(text)):
                _redirect(doc, xref, remap, text)


def _content_key(
    doc: pymupdf.Document,
    xref: int,
    text: str,
    digests: dict[int, bytes],
) -> tuple[str, bytes | None] | None:
    if doc.xref_is_stream(xref):
        if xref not in digests:
            digests[xref] = hashlib.sha1(doc.xref_stream_raw(xref) or b"").digest()
        return text, digests[xref]
    if not text.startswith(("<<", "[")) or _IDENTITY_RE.search(text):
        return None
    return text, None


def _redirect(doc: pymupdf.Document, xref: int, remap: dict[int, int], text: str) -> None:
    """text is the current text of the object (for a stream object, its dictionary)."""
    if not doc.xref_is_stream(xref):
        new = replace_refs(text, remap)
        if new != text:
            doc.update_object(xref, new)
        return
    # update_object drops a stream read from the file, so for stream objects the dictionary is changed key by key
    for key in doc.xref_get_keys(xref):
        kind, value = doc.xref_get_key(xref, key)
        if kind in ("xref", "dict", "array"):
            new = replace_refs(value, remap)
            if new != value:
                doc.xref_set_key(xref, key, new)


def replace_refs(text: str, remap: dict[int, int]) -> str:
    """Change indirect references in the object text. A "12 0 R" written inside a string is not a reference
    and is left unchanged."""
    parts: list[str] = []
    for piece, is_code in _split_strings(text):
        if is_code:
            piece = _REF_RE.sub(lambda m: f"{remap.get(int(m.group(1)), int(m.group(1)))} 0 R", piece)
        parts.append(piece)
    return "".join(parts)


def _split_strings(text: str) -> list[tuple[str, bool]]:
    """Split into (piece, whether outside a string). A literal string (...) can nest parentheses and escape
    with backslashes; a hex string <...> is told apart from the << of a dictionary."""
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
        elif char == "<" and text[index + 1 : index + 2] != "<" and text[index - 1 : index] != "<":
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
