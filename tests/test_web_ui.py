"""The web page (static/index.html and app.js) in a real Chromium, against the server with a mock translation service.

Needs the packages in requirements-dev.txt and `playwright install chromium`; skipped without them.
"""

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import httpx
import pytest

pytest.importorskip("playwright.sync_api")

import uvicorn
from playwright.sync_api import Browser, ConsoleMessage, Error, Page, sync_playwright

WAIT_MS = 15000
MD = "# A Title\n\nThe first paragraph of the document is here.\n\nThe second paragraph comes right after it.\n"


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Error as e:
            pytest.skip(f"Chromium is not installed (playwright install chromium): {e}")
        yield browser
        browser.close()


@pytest.fixture
def server(app_env: ModuleType) -> Iterator[str]:
    """The app served by uvicorn in a thread of this process, so it uses the stores app_env put in place."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    instance = uvicorn.Server(uvicorn.Config(app_env.app, log_level="warning"))
    thread = threading.Thread(target=instance.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(10)
    sock.close()


@pytest.fixture
def page(browser: Browser, server: str) -> Iterator[Page]:
    """A page of the app. Script errors and console errors on it fail the test."""
    context = browser.new_context(base_url=server)
    page = context.new_page()
    problems: list[str] = []
    page.on("pageerror", lambda error: problems.append(f"page error: {error}"))

    def console(message: ConsoleMessage) -> None:
        if message.type == "error":
            problems.append(f"console error: {message.text}")

    page.on("console", console)
    page.goto("/")
    yield page
    context.close()
    assert problems == []


def test_page_loads_its_services_and_languages(page: Page) -> None:
    page.wait_for_selector("#service_id option[value='mock']", state="attached", timeout=WAIT_MS)
    assert page.locator("#target_lang option").count() > 10
    page.get_by_role("tab", name="翻译服务").click()
    page.wait_for_selector("#service-list >> text=Mock", timeout=WAIT_MS)


def test_a_document_is_translated_from_the_page(page: Page, server: str, tmp_path: Path) -> None:
    """Choose a file and the mock service, start, and download the translation from the recent jobs."""
    source = tmp_path / "book.md"
    source.write_text(MD, encoding="utf-8")
    page.wait_for_selector("#service_id option[value='mock']", state="attached", timeout=WAIT_MS)
    page.select_option("#service_id", "mock")
    page.select_option("#target_lang", "zh-CN")
    page.set_input_files("#file", str(source))
    page.click("#submit")

    link = page.locator("#recent-list a[download]").first
    link.wait_for(timeout=WAIT_MS)
    href = link.get_attribute("href")
    assert href and "book" in (link.get_attribute("download") or "")
    translated = httpx.get(server + href).text
    assert "[zh-CN] The first paragraph of the document is here." in translated
    assert page.locator("#submit-error").is_hidden()


def test_another_machine_signs_in_and_comes_back_to_its_view(
    browser: Browser, server: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app import access

    token = "a-token-of-sixteen-chars"
    monkeypatch.setattr(access, "is_local", lambda client, host, forwarded: False)
    monkeypatch.setattr(access, "TOKEN", token)
    context = browser.new_context(base_url=server)
    page = context.new_page()
    page.goto("/#read/abc")
    page.wait_for_url("**/login.html#read/abc", timeout=WAIT_MS)

    page.fill("#token", "wrong-token-of-length")
    page.click("button[type=submit]")
    page.wait_for_url("**/login.html?failed=1#read/abc", timeout=WAIT_MS)
    assert page.locator("#failed").is_visible()

    page.fill("#token", token)
    page.click("button[type=submit]")
    page.wait_for_url(f"{server}/#read/abc", timeout=WAIT_MS)
    page.wait_for_selector("#service_id option[value='mock']", state="attached", timeout=WAIT_MS)

    # A new token on the server signs the browser out; the page's next request takes it to the login page
    monkeypatch.setattr(access, "TOKEN", "the-next-token-of-length")
    page.evaluate("api('/api/jobs').catch(() => {})")
    page.wait_for_url("**/login.html#read/abc", timeout=WAIT_MS)
    context.close()
