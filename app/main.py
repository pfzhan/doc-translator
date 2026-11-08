"""Web server: upload a document -> translate in the background -> preview during translation -> download the
result. Translation records are persisted in data/jobs/."""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import access
from .job_control import _cleanup_quietly
from .routes import glossaries, jobs, models, services, settings, web

ROOT = Path(__file__).resolve().parent.parent
CLEANUP_INTERVAL = 3600


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Clean up expired records once at startup, then check every hour."""

    async def loop() -> None:
        while True:
            # Disk scans run in threads inside _cleanup, so the event loop is not blocked. A failed round is logged
            # and tried again next time; the background task must not end
            await _cleanup_quietly()
            await asyncio.sleep(CLEANUP_INTERVAL)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="Doc Translator", lifespan=lifespan)

# Uvicorn configures only its own loggers. Give the app's a handler with a time stamp, unless the process set up
# logging itself (a root handler)
_log = logging.getLogger("app")
if not _log.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)

# Accept only local addresses. Otherwise a malicious web page could resolve its own domain to 127.0.0.1 (DNS
# rebinding) and call these APIs as the same origin, for example to test an API address of its choice with a saved
# Key. When accessing by IP on a LAN, add the address to the environment variable ALLOWED_HOSTS, comma-separated;
# such requests also need the token in ACCESS_TOKEN (see access)
EXTRA_HOSTS = [h.strip() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()]
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "[::1]", *EXTRA_HOSTS]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# Requests that change state are accepted only from this server's own pages, from browser extensions, and from
# clients that send no Origin (curl and other tools). TrustedHostMiddleware does not cover this: any web page the
# user visits can POST to 127.0.0.1 across origins. With a form, or fetch in mode "no-cors" and a simple content
# type, there is no preflight, so the request runs even though the page cannot read the answer: it could pause all
# jobs, delete records or spend the translation quota. Browsers always send Origin on such requests.
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
EXTENSION_ORIGINS = ("chrome-extension://", "moz-extension://")


def _origin_allowed(origin: str | None, host: str) -> bool:
    if origin is None or origin.startswith(EXTENSION_ORIGINS):
        return True
    # Host and port only: behind an HTTPS reverse proxy the page's scheme differs from the one the server sees
    parsed = urlsplit(origin)
    return bool(parsed.netloc) and parsed.netloc.lower() == host.lower()


@app.middleware("http")
async def check_origin(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    if request.method not in SAFE_METHODS and not _origin_allowed(
        request.headers.get("origin"), request.headers.get("host", "")
    ):
        return JSONResponse({"detail": "不接受来自其他网页的请求"}, status_code=403)
    return await call_next(request)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Reject an upload above the limit before the request body is read. Otherwise Starlette first writes the whole
    file to a temporary file before the check in create_job runs."""
    if request.method == "POST" and request.url.path == "/api/jobs":
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        # Leave 1 MB for multipart boundaries and form fields
        if length > jobs.MAX_UPLOAD + 1024 * 1024:
            return JSONResponse({"detail": "文件太大（上限 200MB）"}, status_code=413)
    elif request.method == "POST" and request.url.path == "/api/web/translate":
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > web.MAX_WEB_REQUEST:
            return JSONResponse({"detail": "请求太大"}, status_code=413)
    return await call_next(request)


# Added last, so it runs first: a request from another machine without the token gets no further
app.middleware("http")(access.check_access)
if EXTRA_HOSTS and not access.TOKEN:
    _log.warning("ALLOWED_HOSTS is set but ACCESS_TOKEN is not: requests from other machines will be refused")


class NoCacheStaticFiles(StaticFiles):
    """Keep the browser from using cached frontend files directly: it asks the server with an ETag each time and
    gets 304 when unchanged, or the new file when changed.
    Otherwise, after app.js changes, the browser may keep using the old version (this happened: after a commit it
    still jumped into the reader)."""

    # Passed through to StaticFiles unchanged, so its own signature applies
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


# The model routes take /api/{kind}-model, so they come after the others; the static files answer everything else
for module in (access, services, glossaries, web, jobs, settings, models):
    app.include_router(module.router)
app.mount("/", NoCacheStaticFiles(directory=ROOT / "static", html=True), name="static")
