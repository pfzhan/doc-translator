"""Access from other machines needs the token in ACCESS_TOKEN (app/access.py)."""

from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from app import access

TOKEN = "a-token-of-sixteen-chars"
# Captured before conftest's local_test_client replaces it for each test
REAL_IS_LOCAL = access.is_local


@pytest.mark.parametrize(
    "client,host,forwarded,local",
    [
        ("127.0.0.1", "127.0.0.1:8765", False, True),
        ("127.0.0.1", "localhost:8765", False, True),
        ("127.0.0.1", "LOCALHOST", False, True),
        ("::1", "[::1]:8765", False, True),
        ("::ffff:127.0.0.1", "127.0.0.1:8765", False, True),
        ("127.0.0.2", "127.0.0.1:8765", False, True),  # all of 127/8 is this machine
        ("192.168.1.30", "127.0.0.1:8765", False, False),  # a client on the network can send any Host
        ("127.0.0.1", "192.168.1.20:8765", False, False),  # the LAN address, as a reverse proxy forwards it
        ("127.0.0.1", "127.0.0.1:8765", True, False),  # through a reverse proxy on this machine
        ("::ffff:192.168.1.30", "localhost", False, False),
        (None, "127.0.0.1:8765", False, False),
        ("testclient", "127.0.0.1:8765", False, False),
        ("127.0.0.1", "[::1", False, False),
        ("127.0.0.1", "", False, False),
    ],
)
def test_is_local(client: str | None, host: str, forwarded: bool, local: bool) -> None:
    assert REAL_IS_LOCAL(client, host, forwarded) is local


def test_a_short_token_is_refused_at_start() -> None:
    with pytest.raises(RuntimeError, match="at least 16"):
        access.load_token({"ACCESS_TOKEN": "short"})
    assert access.load_token({"ACCESS_TOKEN": f"  {TOKEN}\n"}) == TOKEN
    assert access.load_token({}) == ""


@pytest.fixture
def remote(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client on another machine: TestClient requests come from "testclient", which is not a loopback address."""
    monkeypatch.setattr(access, "is_local", REAL_IS_LOCAL)
    monkeypatch.setattr(access, "TOKEN", TOKEN)
    return TestClient(app_env.app, follow_redirects=False)


def test_without_a_token_on_the_server_other_machines_are_refused(
    remote: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(access, "TOKEN", "")
    for path in ("/api/jobs", "/", "/login.html"):
        resp = remote.get(path)
        assert resp.status_code == 403, path
        assert "ACCESS_TOKEN" in resp.text
    assert remote.post("/api/login", data={"token": ""}).status_code == 403
    assert remote.get("/api/jobs", headers={"Authorization": "Bearer "}).status_code == 403


def test_the_api_wants_the_token(remote: TestClient) -> None:
    assert remote.get("/api/jobs").status_code == 401
    assert remote.get("/api/jobs", headers={"Authorization": "Bearer wrong-token-of-length"}).status_code == 401
    assert remote.get("/api/jobs", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
    assert remote.post("/api/jobs/pause-all").status_code == 401
    resp = remote.get("/api/jobs", headers={"Authorization": f"bearer {TOKEN}"})
    assert resp.status_code == 200


def test_pages_send_the_browser_to_the_login_page(remote: TestClient) -> None:
    for path in ("/", "/index.html", "/app.js", "/epub.html"):
        resp = remote.get(path)
        assert resp.status_code == 303, path
        assert resp.headers["location"] == "/login.html"
    login = remote.get("/login.html")
    assert login.status_code == 200
    assert 'action="/api/login"' in login.text


def sign_in(client: TestClient, token: str, fragment: str = "") -> httpx.Response:
    resp: httpx.Response = client.post("/api/login", data={"token": token, "fragment": fragment})
    return resp


def test_signing_in_sets_a_session_cookie(remote: TestClient) -> None:
    resp = sign_in(remote, f" {TOKEN} ", "#read/abc")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/#read/abc"
    cookie = resp.headers["set-cookie"]
    assert cookie.startswith(f"{access.COOKIE}={access.session_value(TOKEN)};")
    assert TOKEN not in cookie
    for flag in ("HttpOnly", "SameSite=lax", "Max-Age=2592000", "Path=/"):
        assert flag in cookie
    assert "Secure" not in cookie  # plain http: a Secure cookie would never come back

    assert remote.get("/api/jobs").status_code == 200
    assert remote.get("/").status_code == 200


def test_a_wrong_token_goes_back_to_the_login_page(remote: TestClient) -> None:
    resp = sign_in(remote, "wrong-token-of-length", "#read/abc")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login.html?failed=1#read/abc"
    assert "set-cookie" not in resp.headers
    assert remote.get("/api/jobs").status_code == 401


@pytest.mark.parametrize("fragment", ["//evil.example", "https://evil.example", "/api/jobs"])
def test_signing_in_does_not_redirect_elsewhere(remote: TestClient, fragment: str) -> None:
    assert sign_in(remote, TOKEN, fragment).headers["location"] == "/"


def test_a_forged_cookie_is_refused(remote: TestClient) -> None:
    remote.cookies.set(access.COOKIE, access.session_value("another-token-of-length"))
    assert remote.get("/api/jobs").status_code == 401


def test_a_new_token_signs_browsers_out(remote: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    sign_in(remote, TOKEN)
    monkeypatch.setattr(access, "TOKEN", "the-next-token-of-length")
    assert remote.get("/api/jobs").status_code == 401


def test_this_machine_needs_no_token(app_env: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(access, "is_local", REAL_IS_LOCAL)
    monkeypatch.setattr(access, "TOKEN", TOKEN)
    local = TestClient(app_env.app, base_url="http://127.0.0.1:8765", client=("127.0.0.1", 50000))
    assert local.get("/api/jobs").status_code == 200
    assert local.post("/api/jobs/pause-all").status_code == 200
