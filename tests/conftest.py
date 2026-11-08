import os
import shutil
import tempfile

import pytest

# TestClient requests come from testserver, and main accepts only local addresses. This must be set before main is
# imported
os.environ["ALLOWED_HOSTS"] = "testserver"
# The stores in app.state load at import time. Point them at an empty directory before anything imports the app, so
# the tests never read or rewrite the records of a server running from this checkout
_DATA_DIR = tempfile.mkdtemp(prefix="doc-translator-tests-")
os.environ["DOC_TRANSLATOR_DATA"] = _DATA_DIR

from app import ccswitch  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_DATA_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def local_test_client(monkeypatch):
    """TestClient requests come from "testclient" to "testserver". Treat them as requests from this machine, as a
    browser on it would send; tests/test_access.py puts the real check back."""
    from app import access

    monkeypatch.setattr(access, "is_local", lambda client, host, forwarded: True)


@pytest.fixture(autouse=True)
def isolate_ccswitch(tmp_path, monkeypatch):
    """By default tests do not read the real CC Switch database of this machine."""
    monkeypatch.setattr(ccswitch, "DB_PATH", tmp_path / "no-cc-switch.db")


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """The translation cache also uses a temporary database instead of the real data/cache.sqlite3. Otherwise
    translations left by an earlier test run would break the assertions about retranslation, and fake test
    translations would end up in the real cache."""
    from app import runner

    monkeypatch.setattr(runner, "_cache", runner.Cache(tmp_path / "cache.sqlite3"))


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Move the web service's translation services, records and settings to a temporary directory and add a mock
    translation service. The model directory is replaced too: the formula detection model is on by default, and a
    model downloaded on this machine must not make tests run the real model."""
    from app import layout_model, main, quantize, state
    from app.glossaries import GlossaryStore
    from app.history import JobStore
    from app.services import ServiceStore
    from app.settings import Settings

    store = ServiceStore(tmp_path / "services.json")
    store.services.append({"id": "mock", "provider": "mock", "builtin": True, "name": "Mock", "enabled": True})
    monkeypatch.setattr(state, "store", store)
    monkeypatch.setattr(state, "glossaries", GlossaryStore(tmp_path / "glossaries.json"))
    monkeypatch.setattr(state, "jobs", JobStore(tmp_path / "jobs"))
    monkeypatch.setattr(state, "settings", Settings(tmp_path / "settings.json"))
    monkeypatch.setattr(layout_model, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(state, "running", {})
    monkeypatch.setattr(state, "waiting", {})
    monkeypatch.setattr(state, "variant_tasks", {})
    monkeypatch.setattr(state, "cleaning", set())
    monkeypatch.setattr(state, "pause_requested", set())
    monkeypatch.setattr(state, "quantizers", {kind: quantize.Quantizer() for kind in state.quantizers})
    return main
