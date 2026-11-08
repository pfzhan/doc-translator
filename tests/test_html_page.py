"""HTML documents (app/formats/html_page.py) and how their translations are served."""

from pathlib import Path
from types import ModuleType

from bs4 import BeautifulSoup, Tag
from fastapi.testclient import TestClient

from app.formats import translate_file
from tests.helpers import run, runner
from tests.test_history import submit, wait_done

PAGE = (
    '<!DOCTYPE html><html lang="en"><head><meta charset="windows-1252"><title>My Page</title>'
    "<style>p{color:red}</style></head><body><h1>Welcome home</h1>"
    '<p>This is <b>bold</b> and a <a href="x.html">link</a>.</p>'
    '<script>var a="<p>no</p>";</script><pre>code here</pre><p>Caf\xe9 culture is nice.</p>'
    "<h2>References</h2><p>Smith, J. (2020). Some book.</p></body></html>"
).encode("cp1252")


def _tag(found: Tag | None) -> Tag:
    """The element, which the test expects the page to have."""
    assert found is not None
    return found


def _attr(tag: Tag, name: str) -> str:
    """A single-valued attribute (bs4 gives a list only for multi-valued ones such as class)."""
    value = tag[name]
    assert isinstance(value, str)
    return value


def test_html_translated_only(tmp_path: Path) -> None:
    src = tmp_path / "page.html"
    src.write_bytes(PAGE)
    out = run(translate_file(src, tmp_path, runner(), False, "zh-CN"))[0]
    assert out.name == "page.translated.html"
    soup = BeautifulSoup(out.read_bytes(), "lxml")
    # Written as UTF-8 and the charset declaration says so
    assert _attr(_tag(soup.find("meta")), "charset").lower() == "utf-8"
    assert _attr(_tag(soup.html), "lang") == "zh-CN"
    assert _tag(soup.title).string == "[zh-CN] My Page"
    assert _tag(soup.h1).get_text() == "[zh-CN] Welcome home"
    p = soup.find_all("p")[0]
    assert p.get_text() == "[zh-CN] This is bold and a link." and _attr(_tag(p.a), "href") == "x.html"
    assert _tag(soup.pre).get_text() == "code here" and "<p>no</p>" in (_tag(soup.script).string or "")
    assert "Smith, J. (2020). Some book." in soup.get_text() and "[zh-CN] Smith" not in soup.get_text()
    assert "[zh-CN] Café culture is nice." in soup.get_text()


def test_html_bilingual_keeps_the_original(tmp_path: Path) -> None:
    src = tmp_path / "page.htm"
    src.write_bytes(PAGE)
    out = run(translate_file(src, tmp_path, runner(), True, "zh-CN"))[0]
    soup = BeautifulSoup(out.read_bytes(), "lxml")
    assert _attr(_tag(soup.html), "lang") == "en"
    assert _tag(soup.title).string == "My Page / [zh-CN] My Page"
    assert [p.get_text() for p in soup.select("p.dt-translation")][0] == "[zh-CN] This is bold and a link."
    assert "Café culture is nice." in [p.get_text() for p in soup.find_all("p")]


def test_html_output_never_runs_scripts_from_this_site(app_env: ModuleType) -> None:
    with TestClient(app_env.app) as client:
        job = wait_done(client, submit(client, name="page.html", data=PAGE)["id"])
        assert job["status"] == "done", job
        url = job["outputs"][0]["url"]
        for query in ("", "?inline=1"):
            r = client.get(url + query)
            assert r.status_code == 200
            assert "sandbox" in r.headers["content-security-policy"]
            assert "script-src 'none'" in r.headers["content-security-policy"]
