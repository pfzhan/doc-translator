"""The browser extension running in a real Chromium, against a stand-in for the translation server.

Needs the packages in requirements-dev.txt and `playwright install chromium`; skipped without them.
"""

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")

from playwright.sync_api import BrowserContext, Error, Page, Worker, sync_playwright

EXTENSION = Path(__file__).resolve().parent.parent / "extension"
VISIBLE = "This paragraph is on the screen from the start."
TRANSLATED = "译文："
# The server's limit on the page title (WebTranslateRequest.title)
MAX_TITLE = 1000
WAIT_MS = 10000


@dataclass
class TranslateRequest:
    """What the extension sent to POST /api/web/translate."""

    texts: list[str]
    title: str
    service_id: str


@dataclass
class FakeServer:
    """Serves test pages and answers the API calls the extension makes, each translation marked with TRANSLATED."""

    port: int = 0
    pages: dict[str, tuple[str, str]] = field(default_factory=dict)  # path -> content type, body
    requests: list[TranslateRequest] = field(default_factory=list)

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def add_page(self, path: str, body: str, content_type: str = "text/html; charset=utf-8") -> str:
        self.pages[path] = (content_type, body)
        return self.origin + path

    def sent_texts(self) -> list[str]:
        return [text for request in self.requests for text in request.texts]


API_ANSWERS: dict[str, object] = {
    "/api/services": {"services": [{"id": "mock", "name": "Mock", "enabled": True}], "default": "mock"},
    "/api/languages": [
        {"code": "auto", "native": "auto", "name": "auto"},
        {"code": "zh-CN", "native": "简体中文", "name": "Chinese"},
        {"code": "en", "native": "English", "name": "English"},
    ],
    "/api/web/cache": {"entries": 0},
}


def make_handler(fake: FakeServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def answer(self, status: int, content_type: str, body: str) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path in API_ANSWERS:
                self.answer(200, "application/json", json.dumps(API_ANSWERS[self.path]))
            elif self.path in fake.pages:
                self.answer(200, *fake.pages[self.path])
            else:
                self.answer(404, "text/plain", "not found")

        def do_POST(self) -> None:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts: list[str] = data["texts"]
            title: str = data["title"]
            fake.requests.append(TranslateRequest(texts, title, data["service_id"]))
            if len(title) > MAX_TITLE:
                detail = [{"msg": "String should have at most 1000 characters"}]
                self.answer(422, "application/json", json.dumps({"detail": detail}))
                return
            answer = {
                "translations": [TRANSLATED + text for text in texts],
                "langs": [data["target_lang"]] * len(texts),
                "detected_lang": "en",
            }
            self.answer(200, "application/json", json.dumps(answer))

    return Handler


@pytest.fixture(scope="module")
def fake_server() -> Iterator[FakeServer]:
    fake = FakeServer()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake))
    fake.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture(scope="module")
def browser(tmp_path_factory: pytest.TempPathFactory) -> Iterator[BrowserContext]:
    with sync_playwright() as playwright:
        try:
            context = playwright.chromium.launch_persistent_context(
                str(tmp_path_factory.mktemp("profile")),
                channel="chromium",  # the full browser: the headless shell has no extensions
                headless=True,
                args=[f"--disable-extensions-except={EXTENSION}", f"--load-extension={EXTENSION}"],
            )
        except Error as e:
            pytest.skip(f"Chromium is not available, run `playwright install chromium`: {e}")
        yield context
        context.close()


@pytest.fixture
def worker(browser: BrowserContext, fake_server: FakeServer) -> Worker:
    """The extension's service worker, with default settings pointing at the stand-in server."""
    worker = browser.service_workers[0] if browser.service_workers else browser.wait_for_event("serviceworker")
    worker.evaluate(
        """async (server) => {
          await chrome.storage.sync.clear();
          await chrome.storage.sync.set({ server });
        }""",
        fake_server.origin,
    )
    fake_server.requests.clear()
    return worker


@pytest.fixture
def page(browser: BrowserContext) -> Iterator[Page]:
    page = browser.new_page()
    yield page
    page.close()


def press_shortcut(worker: Worker, page: Page) -> bool:
    """Do what the shortcut does on the page's tab. False when that got no answer in time, which leaves it waiting."""
    finished: bool = worker.evaluate(
        """async ([url, wait]) => {
          const [tab] = await chrome.tabs.query({ url });
          const pressed = toggle(tab).then(() => true);
          return Promise.race([pressed, new Promise((resolve) => setTimeout(() => resolve(false), wait))]);
        }""",
        [page.url, 3000],
    )
    return finished


def wait_for_translations(page: Page, count: int) -> None:
    page.wait_for_function(
        "(count) => document.querySelectorAll('dt-trans').length >= count", arg=count, timeout=WAIT_MS
    )


def translations(page: Page) -> list[str]:
    return page.locator("dt-trans").all_text_contents()


def test_translates_a_paragraph_below_its_original(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    page.goto(fake_server.add_page("/plain", f"<!doctype html><title>t</title><p>{VISIBLE}</p>"))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert translations(page) == [TRANSLATED + VISIBLE]
    assert page.locator("p").text_content() == VISIBLE + TRANSLATED + VISIBLE


def test_screen_reader_text_is_not_sent(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    clipped = "Text only screen readers get to read."
    html = f"""<!doctype html><title>t</title><p>{VISIBLE}</p>
      <div style="position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0)">{clipped}</div>"""
    page.goto(fake_server.add_page("/clipped", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert fake_server.sent_texts() == [VISIBLE]


def open_options(browser: BrowserContext, worker: Worker) -> Page:
    extension_id = worker.url.split("/")[2]
    options = browser.new_page()
    options.goto(f"chrome-extension://{extension_id}/options.html")
    # Not wait_for_function: the CSP of extension pages does not allow evaluating strings
    options.locator("#status", has_text="已连接").wait_for(timeout=WAIT_MS)
    return options


def saved_service(worker: Worker) -> str:
    service: str = worker.evaluate("async () => (await loadSettings()).serviceId")
    return service


def test_options_keep_a_service_the_server_has(browser: BrowserContext, worker: Worker) -> None:
    worker.evaluate("chrome.storage.sync.set({ serviceId: 'mock' })")
    options = open_options(browser, worker)
    assert options.locator("#service").input_value() == "mock"
    assert saved_service(worker) == "mock"
    assert options.locator("#status").text_content() == "已连接"
    options.close()


def test_a_document_without_html_elements_shows_a_failure_on_the_button(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    page.goto(fake_server.add_page("/feed", "<?xml version='1.0'?><feed><a>Text</a></feed>", "application/xml"))
    assert press_shortcut(worker, page), "the press got no answer"
    badge: str = worker.evaluate(
        """async (url) => {
          const [tab] = await chrome.tabs.query({ url });
          return chrome.action.getBadgeText({ tabId: tab.id });
        }""",
        page.url,
    )
    assert badge == "×"
