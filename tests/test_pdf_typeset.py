import pymupdf

from app.formats.pdf_typeset import FormulaMetric, FormulaPiece, TextPiece, typeset_lines

_SENTINEL = "\x01i\x021\x01/i\x02"


def test_formula_slot_sits_after_the_preceding_text():
    metrics = {1: FormulaMetric(slot_width=30, body_width=26, height=10, below=2)}
    lines = typeset_lines(f"译文 {_SENTINEL} 甲", em=10, max_width=200, metrics=metrics)
    assert len(lines) == 1
    text, formula, tail = lines[0].pieces
    assert isinstance(text, TextPiece) and isinstance(formula, FormulaPiece) and isinstance(tail, TextPiece)
    assert formula.x == text.width
    assert formula.index == 1 and formula.below == 2
    assert tail.x == formula.x + formula.slot_width
    assert _SENTINEL not in text.text and _SENTINEL not in tail.text


def test_wide_formula_is_its_own_line():
    metrics = {1: FormulaMetric(slot_width=40, body_width=36, height=12, below=1)}
    lines = typeset_lines(f"甲{_SENTINEL}乙", em=10, max_width=20, metrics=metrics)
    assert any(isinstance(piece, FormulaPiece) and piece.x == 0 for line in lines for piece in line.pieces)
    assert all(
        _SENTINEL not in piece.text
        for line in lines for piece in line.pieces if isinstance(piece, TextPiece)
    )


def test_typeset_formula_tracks_the_line_baseline(tmp_path):
    """公式下沿贴在同一行文字的基线上，不跟盒子顶走。"""
    from app.formats.pdf import TextBlock, _render_translated

    src = tmp_path / "src.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=200)
    page.insert_text((50, 50), "alpha BETA gamma", fontsize=10)
    doc.save(src)
    doc.close()

    opened = pymupdf.open(src)
    rect = opened[0].search_for("BETA")[0]
    formula = {
        "bbox": rect, "name": "a_1.png", "mid": 0,
        "w": round(rect.width, 1), "h": round(rect.height, 1), "d": 2.0,
        "text": "x", "has_img": True,
        "png": opened[0].get_pixmap(dpi=150, clip=rect).tobytes("png"),
        "spans": [{"bbox": tuple(rect), "text": "x", "size": 10, "origin": (rect.x0, rect.y1 - 2)}],
    }
    opened.close()
    block = TextBlock(
        page=0, rect=pymupdf.Rect(40, 40, 200, 55), line_rects=[pymupdf.Rect(40, 40, 200, 55)],
        text="alpha BETA gamma", size=10, color="#000000", bold=False,
    )
    out = _render_translated(
        src, [block], [f"译文 {_SENTINEL} 甲"], "zh-CN", {id(block): [formula]},
    )
    page = out[0]
    text_spans = [
        span
        for blk in page.get_text("dict")["blocks"] if blk.get("type") == 0
        for ln in blk["lines"] for span in ln["spans"]
        if span.get("origin") and any("\u4e00" <= ch <= "\u9fff" for ch in span["text"])
    ]
    assert text_spans
    hit = page.search_for("BETA")
    assert len(hit) == 1
    baseline = text_spans[0]["origin"][1]
    # 记录的下沿是 2pt，再乘渲染缩放，公式底应靠近文字基线而不是盒子顶
    assert abs((hit[0].y1 - baseline)) < 8
    assert hit[0].x0 > text_spans[0]["bbox"][0]
