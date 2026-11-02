from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import _origin_allowed


def request(main: ModuleType, method: str, path: str, **headers: str) -> httpx.Response:
    client = TestClient(main.app)
    resp: httpx.Response = client.request(method, path, headers=headers)
    return resp


@pytest.mark.parametrize(
    "origin,host,allowed",
    [
        (None, "127.0.0.1:8765", True),  # curl and other tools send no Origin
        ("chrome-extension://abcdef", "127.0.0.1:8765", True),
        ("moz-extension://0f3c-11aa", "127.0.0.1:8765", True),
        ("http://127.0.0.1:8765", "127.0.0.1:8765", True),
        ("https://translate.example.org", "translate.example.org", True),  # behind an HTTPS reverse proxy
        ("HTTP://LOCALHOST:8765", "localhost:8765", True),
        ("http://127.0.0.1:9999", "127.0.0.1:8765", False),  # another local app
        ("https://evil.example", "127.0.0.1:8765", False),
        ("http://127.0.0.1:8765.evil.example", "127.0.0.1:8765", False),
        ("null", "127.0.0.1:8765", False),  # sandboxed iframes and file: pages
        ("", "127.0.0.1:8765", False),
    ],
)
def test_origin_allowed(origin: str | None, host: str, allowed: bool) -> None:
    assert _origin_allowed(origin, host) is allowed


def test_state_changes_from_other_pages_are_rejected(app_env: ModuleType) -> None:
    # The kind of request a web page can send to 127.0.0.1 without a preflight
    for method, path in [
        ("POST", "/api/jobs/pause-all"),
        ("POST", "/api/jobs/resume-all"),
        ("DELETE", "/api/services/mock"),
        ("PUT", "/api/settings"),
    ]:
        resp = request(app_env, method, path, origin="https://evil.example")
        assert resp.status_code == 403, (method, path)
        assert resp.json()["detail"] == "不接受来自其他网页的请求"
    assert app_env.store.get("mock") is not None


def test_own_pages_extensions_and_tools_pass(app_env: ModuleType) -> None:
    for headers in [{"origin": "http://testserver"}, {"origin": "chrome-extension://abcdef"}, {}]:
        assert request(app_env, "POST", "/api/jobs/pause-all", **headers).status_code == 200, headers


def test_reads_are_not_checked(app_env: ModuleType) -> None:
    # Reads cannot change anything, and cross-origin pages cannot see the answer without CORS headers
    assert request(app_env, "GET", "/api/services", origin="https://evil.example").status_code == 200
