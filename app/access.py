"""Access from other machines: a request that does not come from this machine needs the token in ACCESS_TOKEN.

The server listens on 127.0.0.1 by default, where only this machine reaches it. Started with --host 0.0.0.0 and
ALLOWED_HOSTS, anyone on the network could otherwise translate with the saved API keys, change the services, and
read or delete the records. Browsers sign in once on the login page and then send a session cookie; the extension
and scripts send the token in an Authorization: Bearer header.
"""

import hashlib
import hmac
import ipaddress
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response

TOKEN_ENV = "ACCESS_TOKEN"
# Signing in is not rate limited: a token this long, made with `secrets`, cannot be guessed
MIN_TOKEN_LENGTH = 16
COOKIE = "doc_translator_session"
COOKIE_MAX_AGE = 30 * 24 * 3600
LOGIN_PAGE = "/login.html"
LOGIN_PATH = "/api/login"
LOCAL_HOST_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
# Headers a reverse proxy adds. A request that went through one is not treated as local, even when the proxy runs on
# this machine and the request arrives from 127.0.0.1
FORWARDING_HEADERS = ("forwarded", "x-forwarded-for")

NO_TOKEN = "服务器没有设置访问令牌（环境变量 ACCESS_TOKEN），只能从本机访问"
NEED_TOKEN = "需要访问令牌"


def load_token(environ: Mapping[str, str] = os.environ) -> str:
    """The token from the environment, or "" when access from other machines is off."""
    token = environ.get(TOKEN_ENV, "").strip()
    if token and len(token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(f"{TOKEN_ENV} must be at least {MIN_TOKEN_LENGTH} characters long")
    return token


TOKEN = load_token()


def _host_name(host: str) -> str:
    """The name in a Host header, lower case and without the port or the brackets of an IPv6 address."""
    try:
        return urlsplit("//" + host).hostname or ""
    except ValueError:  # an unclosed [ of an IPv6 address
        return ""


def _is_loopback(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped.is_loopback
    return ip.is_loopback


def is_local(client: str | None, host: str, forwarded: bool) -> bool:
    """A request sent on this machine to one of its local names. Neither check is enough alone: a client on the
    network can send Host: 127.0.0.1, and a reverse proxy on this machine sends every request from 127.0.0.1."""
    return not forwarded and client is not None and _is_loopback(client) and _host_name(host) in LOCAL_HOST_NAMES


def session_value(token: str) -> str:
    """The cookie of a signed-in browser. Derived from the token, so the cookie does not hold the token itself, and a
    new token signs every browser out."""
    return hmac.new(token.encode(), b"doc-translator session", hashlib.sha256).hexdigest()


def _same(given: str, expected: str) -> bool:
    return hmac.compare_digest(given.encode(), expected.encode())


def _authorized(request: Request, token: str) -> bool:
    scheme, _, given = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer" and given.strip():
        return _same(given.strip(), token)
    cookie = request.cookies.get(COOKIE)
    return cookie is not None and _same(cookie, session_value(token))


def _refuse(request: Request, status: int, detail: str) -> Response:
    if request.url.path.startswith("/api/"):
        return JSONResponse({"detail": detail}, status_code=status)
    return PlainTextResponse(detail, status_code=status)


async def check_access(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    client = request.client.host if request.client else None
    forwarded = any(name in request.headers for name in FORWARDING_HEADERS)
    if is_local(client, request.headers.get("host", ""), forwarded):
        return await call_next(request)
    if not TOKEN:
        return _refuse(request, 403, NO_TOKEN)
    if request.url.path in (LOGIN_PAGE, LOGIN_PATH) or _authorized(request, TOKEN):
        return await call_next(request)
    if request.method in ("GET", "HEAD") and not request.url.path.startswith("/api/"):
        return RedirectResponse(LOGIN_PAGE, status_code=303)
    return _refuse(request, 401, NEED_TOKEN)


router = APIRouter()


@router.post(LOGIN_PATH)
def login(
    request: Request, token: Annotated[str, Form()] = "", fragment: Annotated[str, Form()] = ""
) -> RedirectResponse:
    """Sign a browser in from the login page. fragment is the part after # of the page the browser was sent to the
    login page from (a document in the reader, say); it is opened again after signing in."""
    # Only a fragment: anything else would let a link to the login page send the user to another site
    back = fragment if fragment.startswith("#") else ""
    if not TOKEN or not _same(token.strip(), TOKEN):
        return RedirectResponse(f"{LOGIN_PAGE}?failed=1{back}", status_code=303)
    response = RedirectResponse("/" + back, status_code=303)
    response.set_cookie(
        COOKIE,
        session_value(TOKEN),
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        # Lax still sends the cookie when a link to the server is opened from elsewhere; requests from other sites
        # that change something are refused by the Origin check in main
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return response
