import asyncio
import zipfile
from pathlib import Path

import pymupdf
import pytest
from bs4 import BeautifulSoup

from app.formats import translate_file
from app.formats.markdown import collect_segments, render, split_blocks
from app.runner import Runner
from app.translators import MockTranslator, OpenAITranslator

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


class MemCache:
    def __init__(self):
        self.d = {}

    def get_many(self, prefix, texts):
        return {t: self.d[(prefix, t)] for t in texts if (prefix, t) in self.d}

    def put_many(self, prefix, pairs):
        for s, t in pairs.items():
            self.d[(prefix, s)] = t


def runner():
    return Runner(MockTranslator("zh-CN"), cache=MemCache())


def run(coro):
    return asyncio.run(coro)


MD = """---
title: Demo
---

# Hello World

This is a paragraph
spanning two lines.

```python
print("keep me")
```

- item one
- item two
  continued

> quoted text

| Name | Desc |
| --- | --- |
| foo | bar baz |

![img](a.png)
"""


def test_markdown_translated_only():
    blocks = split_blocks(MD)
    texts = collect_segments(blocks)
    assert "Hello World" in texts
    assert "This is a paragraph spanning two lines." in texts
    assert "item two continued" in texts
    assert not any("keep me" in t for t in texts)
    assert not any("title" in t for t in texts)
    out = render(blocks, {t: "T:" + t for t in texts}, bilingual=False)
    assert "# T:Hello World" in out
    assert 'print("keep me")' in out
    assert "- T:item two continued" in out
    assert "> T:quoted text" in out
    assert "| T:foo | T:bar baz |" in out
    assert "![img](a.png)" in out
    assert out.startswith("---\ntitle: Demo\n---")


def test_markdown_bilingual_keeps_original():
    blocks = split_blocks(MD)
    texts = collect_segments(blocks)
    out = render(blocks, {t: "T:" + t for t in texts}, bilingual=True)
    assert "# Hello World" in out and "# T:Hello World" in out
    assert "This is a paragraph\nspanning two lines.\n\nT:This is a paragraph" in out
    assert "foo<br>T:foo" in out


def _epub_texts(path):
    z = zipfile.ZipFile(path)
    assert z.infolist()[0].filename == "mimetype"
    assert z.infolist()[0].compress_type == zipfile.ZIP_STORED
    html = "".join(z.read(n).decode() for n in z.namelist() if n.endswith((".xhtml", ".html", ".htm")))
    return BeautifulSoup(html, "lxml").get_text()


@pytest.mark.parametrize("name", ["alice.epub", "alice.mobi", "alice.azw3"])
@pytest.mark.parametrize("bilingual", [True, False])
def test_ebooks(tmp_path, name, bilingual):
    src = SAMPLES / name
    if not src.exists():
        pytest.skip("sample missing")
    outputs = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
    epub = next(p for p in outputs if p.suffix == ".epub")
    text = _epub_texts(epub)
    assert "[zh-CN] Alice was beginning to get very tired" in text
    # 双语保留原文；仅译文模式下原文只会以“[zh-CN] 原文”形式出现
    plain = text.replace("[zh-CN] Alice was beginning", "")
    assert ("Alice was beginning to get very tired" in plain) == bilingual


def test_pdf(tmp_path):
    src = SAMPLES / "attention.pdf"
    if not src.exists():
        pytest.skip("sample missing")
    for bilingual in (False, True):
        [out] = run(translate_file(src, tmp_path, runner(), bilingual, "zh-CN"))
        doc = pymupdf.open(out)
        assert len(doc) == len(pymupdf.open(src))
        text = doc[0].get_text()
        assert "[zh-CN]" in text
        if bilingual:
            assert doc[0].rect.width == pytest.approx(pymupdf.open(src)[0].rect.width * 2)


def test_llm_unparseable_batch_falls_back_to_single():
    tr = OpenAITranslator("zh-CN", api_key="x", base_url="http://localhost", model="m")
    calls = []

    async def fake_complete(system, content):
        calls.append(content)
        return "garbage" if len(calls) == 1 else "你好"

    tr._complete = fake_complete
    assert run(tr.translate_batch(["a", "b"])) == ["你好", "你好"]  # 批量结果无法解析，逐条补翻
    assert len(calls) == 3
