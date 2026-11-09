"""The browser extension running in a real Chromium, against a stand-in for the translation server.

Needs the packages in requirements-dev.txt and `playwright install chromium`; skipped without them.
"""

import json
import threading
import time
from collections.abc import Callable, Iterator
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
    authorization: str | None


@dataclass
class FakeServer:
    """Serves test pages and answers the API calls the extension makes, each translation marked with TRANSLATED."""

    port: int = 0
    pages: dict[str, tuple[str, str]] = field(default_factory=dict)  # path -> content type, body
    requests: list[TranslateRequest] = field(default_factory=list)
    # What the stand-in model does to each text before marking it; reset for each test
    rewrite: Callable[[str], str] | None = None
    delay: float = 0  # seconds each translation takes
    # Requests with a text containing fail_on are answered with this HTTP status
    fail_on: str = ""
    fail_status: int = 502

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
            fake.requests.append(TranslateRequest(texts, title, data["service_id"], self.headers["Authorization"]))
            time.sleep(fake.delay)
            if fake.fail_on and any(fake.fail_on in text for text in texts):
                self.answer(fake.fail_status, "application/json", json.dumps({"detail": "服务出错"}))
                return
            if len(title) > MAX_TITLE:
                detail = [{"msg": "String should have at most 1000 characters"}]
                self.answer(422, "application/json", json.dumps({"detail": detail}))
                return
            answer = {
                "translations": [TRANSLATED + (fake.rewrite(text) if fake.rewrite else text) for text in texts],
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
          await chrome.storage.local.remove('token');
          await chrome.storage.sync.set({ server });
        }""",
        fake_server.origin,
    )
    fake_server.requests.clear()
    fake_server.rewrite = None
    fake_server.delay = 0
    fake_server.fail_on = ""
    fake_server.fail_status = 502
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


def test_a_formatted_paragraph_keeps_the_pages_own_link(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    html = """<!doctype html><title>t</title>
      <p>Read <a href="/guide" id="guide" class="nav">the <b>full</b> guide</a> before you start.</p>"""
    page.goto(fake_server.add_page("/rich", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    # Sent as HTML with only the formatting tags, numbered and without their attributes
    assert fake_server.sent_texts() == ['Read <a i="1">the <b i="0">full</b> guide</a> before you start.']
    # Rebuilt from copies of the page's elements: the href comes from the page, the id stays on the original only
    link = page.locator("dt-trans a")
    assert link.get_attribute("href") == "/guide" and link.get_attribute("class") == "nav"
    assert link.get_attribute("id") is None and link.get_attribute("i") is None
    assert page.locator("dt-trans").text_content() == TRANSLATED + "Read the full guide before you start."


def test_inline_code_is_copied_from_the_page(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    html = f"""<!doctype html><title>t</title><p>{VISIBLE}</p>
      <p>Run <code class="cmd">npm <b>install</b></code> and press <kbd>Ctrl</kbd>+<kbd>C</kbd> to stop
        <var>count</var> workers.</p>
      <ul><li><code>--verbose</code></li></ul>"""
    page.goto(fake_server.add_page("/inline-code", html))
    # A model that translates what it should not
    fake_server.rewrite = lambda text: text.replace("count", "计数").replace("install", "安装")
    press_shortcut(worker, page)
    wait_for_translations(page, 2)
    sent = fake_server.sent_texts()
    assert sent == [
        VISIBLE,
        'Run <span translate="no" i="0">npm install</span> and press <span translate="no" i="1">Ctrl</span>+'
        '<span translate="no" i="2">C</span> to stop <span translate="no" i="3">count</span> workers.',
    ]
    # Code alone (an option name) is not sent; in the sentence, each piece of code is the page's own
    code = "(selector) => [...document.querySelectorAll(selector)].map((el) => el.outerHTML)"
    assert page.evaluate(code, "dt-trans code, dt-trans kbd, dt-trans var") == page.evaluate(
        code, "p > code, p > kbd, p > var"
    )


def test_translation_only_mode_hides_the_originals_of_what_it_translated(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    worker.evaluate("chrome.storage.sync.set({ mode: 'translated' })")
    html = f"""<!doctype html><title>t</title><p>{VISIBLE}</p>
      <input placeholder="Search the whole archive">"""
    page.goto(fake_server.add_page("/translated-only", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    original = "() => document.querySelector('p').firstChild.data"
    placeholder = "() => document.querySelector('input').placeholder"
    page.wait_for_function(f"({placeholder})() !== 'Search the whole archive'", timeout=WAIT_MS)
    assert page.evaluate(original) == "" and translations(page) == [TRANSLATED + VISIBLE]
    assert page.evaluate(placeholder) == TRANSLATED + "Search the whole archive"

    # The next press hides the translations, and the originals come back
    press_shortcut(worker, page)
    assert page.evaluate(original) == VISIBLE
    assert page.evaluate(placeholder) == "Search the whole archive"


def test_text_in_a_collapsed_box_is_translated_once_expanded(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    hidden = "This text sits directly in a panel that starts collapsed."
    html = f"""<!doctype html><title>t</title><p>{VISIBLE}</p>
      <div id="panel" style="height: 0; overflow: hidden">{hidden}</div>"""
    page.goto(fake_server.add_page("/collapsed", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert hidden not in fake_server.sent_texts()

    page.evaluate("document.getElementById('panel').style.height = 'auto'")
    wait_for_translations(page, 2)
    assert TRANSLATED + hidden in translations(page)


def test_screen_reader_text_is_not_sent(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    clipped = "Text only screen readers get to read."
    html = f"""<!doctype html><title>t</title><p>{VISIBLE}</p>
      <div style="position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0)">{clipped}</div>"""
    page.goto(fake_server.add_page("/clipped", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert fake_server.sent_texts() == [VISIBLE]


def test_screen_reader_text_inside_a_sentence_does_not_split_it(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    # Positioning makes these blocks, which used to end the paragraph around them
    html = """<!doctype html><title>t</title><style>
        .sr-only { position: absolute; width: 1px; height: 1px; margin: -1px; overflow: hidden;
          clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
        .clipped { position: absolute; width: 1px; height: 1px; clip-path: inset(50%); }
      </style>
      <p>Read the <a href="/guide">guide<span class="sr-only"> to installing the tool</span></a> before you
        start<span class="clipped"> (opens a new window)</span>, then come back here.</p>"""
    page.goto(fake_server.add_page("/sr-inline", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert fake_server.sent_texts() == ['Read the <a i="0">guide</a> before you start, then come back here.']
    assert translations(page) == [TRANSLATED + "Read the guide before you start, then come back here."]


KATEX = """<style>
    .katex-mathml { position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0; border: 0; height: 1px;
      width: 1px; overflow: hidden; }
    .katex .base { display: inline-block; }
    .katex .vlist { display: table-cell; }
  </style><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>E</mi><mo>=</mo><mi>m</mi>
  <msup><mi>c</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">E=mc^2</annotation></semantics>
  </math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="mord">E</span>
  <span class="mrel">=</span><span class="mord">m</span><span class="mord">c<span class="vlist">2</span></span></span>
  </span></span>"""
MATHJAX = """<style>
    mjx-container { display: inline-block; }
    mjx-assistive-mml { position: absolute !important; top: 0; left: 0; clip: rect(1px, 1px, 1px, 1px) !important;
      padding: 1px 0 0 0 !important; display: block !important; width: auto !important; overflow: hidden !important; }
  </style><mjx-container class="MathJax" jax="CHTML"><mjx-math aria-hidden="true"><mjx-mi>E</mjx-mi><mjx-mo>=</mjx-mo>
  <mjx-mi>m</mjx-mi><mjx-msup><mjx-mi>c</mjx-mi><mjx-script style="display: block">2</mjx-script></mjx-msup>
  </mjx-math><mjx-assistive-mml><math alttext="E=mc^2"><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn>
  </msup></math></mjx-assistive-mml></mjx-container>"""
WIKIPEDIA_MATH = """<span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y"
  style="display: none;"><math alttext="{\\displaystyle E=mc^{2}}"><mi>E</mi></math></span><img
  class="mwe-math-fallback-image-inline" aria-hidden="true" alt="{\\displaystyle E=mc^{2}}"
  src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" style="width: 6em; height: 1em"></span>"""
MATHML = "<math><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></math>"


@pytest.mark.parametrize(
    ("formula", "source"),
    [
        (KATEX, "E=mc^2"),
        (MATHJAX, "E=mc^2"),
        (WIKIPEDIA_MATH, "{\\displaystyle E=mc^{2}}"),
        (MATHML, "E=mc2"),
    ],
    ids=["katex", "mathjax", "wikipedia", "mathml"],
)
def test_a_formula_stays_in_its_sentence_and_is_copied_into_the_translation(
    fake_server: FakeServer, worker: Worker, page: Page, formula: str, source: str
) -> None:
    html = f"""<!doctype html><title>t</title>
      <p>Mass and energy are related by {formula} in special relativity.</p>
      <math display="block"><mi>x</mi><mo>=</mo><mn>1</mn></math>"""
    page.goto(fake_server.add_page("/formula", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    # One sentence, the formula sent as its source; the display formula on its own line is not sent
    assert fake_server.sent_texts() == [
        f'Mass and energy are related by <span translate="no" i="0">{source}</span> in special relativity.'
    ]
    # The formula in the translation is a copy of the page's, not text from the model
    original = page.evaluate("() => document.querySelector('p > :not(style)').outerHTML")
    copied = page.evaluate("() => document.querySelector('dt-trans > *').outerHTML")
    assert copied == original


# Paragraphs of a screen-sized page, two to a request (the extension sends at most 3000 characters at once)
FILLER = "Each of these paragraphs is long enough to fill half a request. " * 22
TALL = "<style>p { height: 300px; margin: 0; overflow: hidden; font-size: 4px; }</style>"


def tall_page(count: int) -> str:
    return f"<!doctype html><title>t</title>{TALL}" + "".join(f"<p>P{i}. {FILLER}</p>" for i in range(count))


def sent_numbers(fake: FakeServer) -> list[list[int]]:
    """The paragraphs of tall_page each request carried, by number."""
    return [[int(text[1 : text.index(".")]) for text in request.texts] for request in fake.requests]


def test_paragraphs_in_view_are_sent_first(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    page.goto(fake_server.add_page("/tall", tall_page(20)))
    # P10 to P12 are in view (720 pixels high); P7 to P14 are within a screen
    page.evaluate("scrollTo(0, 3000)")
    press_shortcut(worker, page)
    wait_for_translations(page, 8)
    sent = sent_numbers(fake_server)
    assert sorted(n for batch in sent for n in batch) == list(range(7, 15))
    # In view first, then the nearest
    assert {10, 11, 12} <= set(sent[0] + sent[1])


def test_paragraphs_scrolled_away_from_wait_for_the_viewport(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    fake_server.delay = 1
    page.goto(fake_server.add_page("/tall-scrolled", tall_page(30)))
    # P0 to P4 are within a screen: two requests go out and P4 waits for its turn
    press_shortcut(worker, page)
    deadline = time.monotonic() + WAIT_MS / 1000
    while len(fake_server.requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    page.evaluate("scrollTo(0, document.body.scrollHeight)")
    wait_for_translations(page, 4 + 5)
    sent = [n for batch in sent_numbers(fake_server) for n in batch]
    # The bottom of the page was translated, and P4, now far above, was not sent
    assert 4 not in sent and {25, 26, 27, 28, 29} <= set(sent)

    page.evaluate("scrollTo(0, 0)")
    page.wait_for_function(
        "() => [...document.querySelectorAll('dt-trans')].some((el) => el.textContent.includes('P4.'))",
        timeout=WAIT_MS,
    )


def three_requests_page() -> str:
    """Three paragraphs in view, each too long to share a request."""
    return f"<!doctype html><title>t</title>{TALL}" + "".join(
        f"<p>Part {name}. {FILLER * 2}</p>" for name in ("one", "two", "three")
    )


def test_a_failed_request_marks_its_paragraph_and_the_rest_go_on(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    fake_server.fail_on = "Part two."
    page.goto(fake_server.add_page("/partial-failure", three_requests_page()))
    press_shortcut(worker, page)
    page.wait_for_function(
        "() => document.querySelectorAll('dt-trans:not([data-dt-failed])').length === 2", timeout=WAIT_MS
    )
    mark = page.locator("p:nth-of-type(2) > dt-trans[data-dt-failed]")
    assert mark.text_content() == "翻译失败 重试"

    # The button sends that paragraph again, and its translation replaces the notice
    fake_server.fail_on = ""
    page.locator("dt-trans[data-dt-failed] button").click()
    page.wait_for_function(
        "() => document.querySelectorAll('dt-trans:not([data-dt-failed])').length === 3", timeout=WAIT_MS
    )
    assert page.locator("dt-trans[data-dt-failed]").count() == 0
    assert page.locator("p:nth-of-type(2) > dt-trans").text_content() == TRANSLATED + f"Part two. {FILLER * 2}".strip()


def test_an_error_of_the_settings_stops_until_the_shortcut_retries(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    fake_server.fail_on = "Part"
    fake_server.fail_status = 401  # a wrong access token: every request would fail
    page.goto(fake_server.add_page("/settings-failure", three_requests_page()))
    press_shortcut(worker, page)
    deadline = time.monotonic() + WAIT_MS / 1000
    while len(fake_server.requests) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    # The two requests sent at once failed, and the third was not sent; no paragraph is marked
    assert len(fake_server.requests) == 2
    assert page.locator("dt-trans").count() == 0

    fake_server.fail_on = ""
    press_shortcut(worker, page)
    wait_for_translations(page, 3)


def test_a_long_page_title_does_not_fail_the_translation(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    html = f"<!doctype html><title>{'long title ' * 200}</title><p>{VISIBLE}</p>"
    page.goto(fake_server.add_page("/long-title", html))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert all(len(request.title) <= MAX_TITLE for request in fake_server.requests)
    assert fake_server.requests[0].title.startswith("long title")


def test_the_access_token_goes_with_each_translation(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    page.goto(fake_server.add_page("/plain", f"<!doctype html><title>t</title><p>{VISIBLE}</p>"))
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert fake_server.requests[0].authorization is None

    worker.evaluate("chrome.storage.local.set({ token: 'a-token-of-sixteen-chars' })")
    page.reload()
    press_shortcut(worker, page)
    wait_for_translations(page, 1)
    assert fake_server.requests[-1].authorization == "Bearer a-token-of-sixteen-chars"


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


def test_options_drop_a_service_the_server_no_longer_has(browser: BrowserContext, worker: Worker) -> None:
    worker.evaluate("chrome.storage.sync.set({ serviceId: 'gone' })")
    options = open_options(browser, worker)
    assert options.locator("#service").input_value() == ""
    # What the page shows is what translations use
    assert saved_service(worker) == ""
    assert "默认服务" in (options.locator("#status").text_content() or "")
    options.close()


def test_options_keep_a_service_the_server_has(browser: BrowserContext, worker: Worker) -> None:
    worker.evaluate("chrome.storage.sync.set({ serviceId: 'mock' })")
    options = open_options(browser, worker)
    assert options.locator("#service").input_value() == "mock"
    assert saved_service(worker) == "mock"
    assert options.locator("#status").text_content() == "已连接"
    options.close()


def test_a_press_before_the_page_has_a_body_can_be_repeated(
    fake_server: FakeServer, worker: Worker, page: Page
) -> None:
    html = f"""<!doctype html><title>t</title><script>
      window.addEventListener('DOMContentLoaded', () => document.body.remove());
    </script><body><p>{VISIBLE}</p></body>"""
    page.goto(fake_server.add_page("/no-body", html))
    assert page.evaluate("document.body === null")
    assert press_shortcut(worker, page), "the press got no answer"

    page.evaluate(
        """(text) => {
          const body = document.createElement('body');
          body.innerHTML = `<p>${text}</p>`;
          document.documentElement.append(body);
        }""",
        VISIBLE,
    )
    assert press_shortcut(worker, page), "the second press got no answer"
    wait_for_translations(page, 1)
    assert translations(page) == [TRANSLATED + VISIBLE]


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


def test_options_keep_the_access_token_out_of_synced_settings(browser: BrowserContext, worker: Worker) -> None:
    options = open_options(browser, worker)
    options.locator("#token").fill(" a-token-of-sixteen-chars ")
    options.locator("#connect").click()
    # The status already said 已连接 before the click, so wait for the token itself
    worker.evaluate(
        """async (wait) => {
          const end = Date.now() + wait;
          while (!(await chrome.storage.local.get('token')).token && Date.now() < end) {
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
        }""",
        WAIT_MS,
    )
    stored = worker.evaluate(
        "async () => [(await chrome.storage.local.get('token')).token, (await chrome.storage.sync.get(null)).token]"
    )
    # A secret must not sync to the other browsers of the account
    assert stored == ["a-token-of-sixteen-chars", None]
    options.close()


def translate_selection(worker: Worker, page: Page) -> None:
    """Do what the shortcut for selected text does on the page's tab."""
    worker.evaluate(
        "async (url) => { const [tab] = await chrome.tabs.query({ url }); await translateSelection(tab); }", page.url
    )


def test_selected_text_is_sent_for_translation(fake_server: FakeServer, worker: Worker, page: Page) -> None:
    password = "hunter2-is-a-secret"
    html = f"""<!doctype html><title>t</title><input id="secret" type="password" value="{password}">
      <p id="chosen">{VISIBLE}</p><p>Text that was not selected.</p>"""
    page.goto(fake_server.add_page("/selection", html))
    # Text selected in a password field never leaves the page
    page.focus("#secret")
    page.evaluate("document.getElementById('secret').select()")
    translate_selection(worker, page)

    page.evaluate(
        """() => {
          document.activeElement.blur();
          getSelection().selectAllChildren(document.getElementById('chosen'));
        }"""
    )
    translate_selection(worker, page)
    deadline = time.monotonic() + WAIT_MS / 1000
    while not fake_server.requests and time.monotonic() < deadline:
        time.sleep(0.05)
    assert fake_server.sent_texts() == [VISIBLE]
