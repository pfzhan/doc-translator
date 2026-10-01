import pytest

from app import ccswitch


@pytest.fixture(autouse=True)
def isolate_ccswitch(tmp_path, monkeypatch):
    """测试默认不读本机真实的 CC Switch 数据库。"""
    monkeypatch.setattr(ccswitch, "DB_PATH", tmp_path / "no-cc-switch.db")


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """翻译缓存也用临时库：不读写真实的 data/cache.sqlite3。
    否则上次跑测试留下的译文会让“是否重新翻译”的断言失效，测试的假译文也会混进真实缓存。"""
    from app import runner

    monkeypatch.setattr(runner, "_cache", runner.Cache(tmp_path / "cache.sqlite3"))


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """把 Web 服务的翻译服务、翻译记录、设置都换到临时目录，并加一个 mock 翻译服务。"""
    from app import main
    from app.history import JobStore
    from app.services import ServiceStore
    from app.settings import Settings

    store = ServiceStore(tmp_path / "services.json")
    store.services.append({"id": "mock", "provider": "mock", "builtin": True, "name": "Mock", "enabled": True})
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "jobs", JobStore(tmp_path / "jobs"))
    monkeypatch.setattr(main, "settings", Settings(tmp_path / "settings.json"))
    monkeypatch.setattr(main, "running", {})
    return main
