import concurrent.futures
import time

import pymupdf

from app.formats import pdf_parts
from app.formats.pdf_parts import plan_parts, render_in_parts


def test_parts_are_balanced_and_never_split_a_paragraph_across_pages(monkeypatch):
    monkeypatch.setattr(pdf_parts, "_MIN_PART_PAGES", 2)
    weights = {pno: 1 for pno in range(12)}
    # 第 5 页的段落续到第 6 页，第 6、7 页之间不能切；第 8 页的续到第 9 页
    parts = plan_parts(weights, [(5, 6), (8, 9)], 3)
    assert sum(parts, []) == list(range(12))
    assert len(parts) == 3
    for host, tail in ((5, 6), (8, 9)):
        assert any(host in part and tail in part for part in parts)


def test_few_pages_are_written_in_one_part():
    assert plan_parts({pno: 3 for pno in range(10)}, [], 4) == [list(range(10))]


def _book(path) -> None:
    doc = pymupdf.open()
    for pno in range(4):
        doc.new_page(width=300, height=200).insert_text((40, 60), f"Original page {pno}", fontsize=12)
    # 指向别的页的链接：拼回去以后还要在
    for pno, page in enumerate(doc):
        page.insert_link({"kind": pymupdf.LINK_GOTO, "from": pymupdf.Rect(40, 50, 140, 64), "page": 3 - pno})
    doc.save(path)


def test_parts_written_in_other_processes_are_put_back_in_place(tmp_path):
    src = tmp_path / "book.pdf"
    _book(src)

    def render(pages, on_page):
        doc = pymupdf.open(src)
        for pno in pages if pages is not None else range(len(doc)):
            if on_page is not None:
                on_page(0, 0)
            # 写回只改内容流，不碰注释
            page = doc[pno]
            page.draw_rect(pymupdf.Rect(30, 40, 200, 70), color=None, fill=(1, 1, 1))
            page.insert_text((40, 90), f"Written page {pno}", fontsize=12)
        return doc

    progress: list[tuple[int, int]] = []
    # 服务里写回在工作线程中运行，子进程从那里 fork
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        out = pool.submit(
            render_in_parts, render, [[0, 1], [2, 3]], lambda done, total: progress.append((done, total)),
        ).result()
    assert out is not None and out.page_count == 4
    assert [page.get_text().strip().splitlines()[-1] for page in out] == [f"Written page {pno}" for pno in range(4)]
    assert [[link["page"] for link in page.get_links()] for page in out] == [[3], [2], [1], [0]]
    # 和一个进程整本写出来的一样
    whole = render(None, None)
    assert [page.get_pixmap(dpi=72).samples for page in out] == [page.get_pixmap(dpi=72).samples for page in whole]
    assert progress and all(total == 4 for _done, total in progress)


def test_a_stuck_part_falls_back_to_writing_the_whole_book(tmp_path, monkeypatch):
    src = tmp_path / "book.pdf"
    _book(src)
    monkeypatch.setattr(pdf_parts, "_STALL_SECONDS", 1)

    def render(pages, on_page):
        if 2 in pages:
            time.sleep(60)
        return pymupdf.open(src)

    started = time.monotonic()
    assert render_in_parts(render, [[0, 1], [2, 3]]) is None
    assert time.monotonic() - started < 10
