"""The HTTP routes of translation services (app/routes/services.py)."""

from types import ModuleType

import httpx
import pytest
from fastapi.testclient import TestClient

from app.json_types import JSONDict
from app.routes import services as routes
from app.translators import MockTranslator, Translator


@pytest.fixture
def client(app_env: ModuleType) -> TestClient:
    return TestClient(app_env.app)


def detail(resp: httpx.Response) -> str:
    message: str = resp.json()["detail"]
    return message


def create(client: TestClient) -> str:
    resp = client.post("/api/services", json={"provider": "openai", "name": "Mine", "api_key": "sk-abcdefghijk"})
    assert resp.status_code == 200
    sid: str = resp.json()["id"]
    return sid


def ids(client: TestClient) -> list[str]:
    return [s["id"] for s in client.get("/api/services").json()["services"]]


def test_a_service_is_renamed(client: TestClient) -> None:
    sid = create(client)
    resp = client.put(f"/api/services/{sid}", json={"name": "Renamed"})
    assert resp.status_code == 200 and resp.json()["name"] == "Renamed"
    assert "api_key" not in resp.json()


def test_changing_or_removing_an_unknown_service_answers_400(client: TestClient) -> None:
    for method, path in [
        ("PUT", "/api/services/nope"),
        ("DELETE", "/api/services/nope"),
        ("POST", "/api/services/nope/default"),
        ("POST", "/api/services/nope/clone"),
    ]:
        resp = client.request(method, path, json={} if method == "PUT" else None)
        assert resp.status_code == 400 and detail(resp) == "服务不存在", (method, path)


def test_a_service_is_deleted_but_a_builtin_one_is_not(client: TestClient) -> None:
    sid = create(client)
    assert client.delete(f"/api/services/{sid}").json() == {"ok": True}
    assert sid not in ids(client)

    resp = client.delete("/api/services/mock")
    assert resp.status_code == 400 and detail(resp) == "内置服务不能删除"
    assert "mock" in ids(client)


def test_the_default_service_cannot_be_switched_off(client: TestClient) -> None:
    sid = create(client)
    assert client.post(f"/api/services/{sid}/default").json()["default"] == sid
    resp = client.put(f"/api/services/{sid}", json={"enabled": False})
    assert resp.status_code == 400 and "默认服务不能关闭" in detail(resp)


def test_only_cc_switch_services_are_cloned(client: TestClient) -> None:
    resp = client.post(f"/api/services/{create(client)}/clone")
    assert resp.status_code == 400 and detail(resp) == "本地服务无需复制，可直接编辑"


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Services become the mock engine, so testing one sends nothing upstream."""

    def mock(svc: JSONDict, target_lang: str, source_lang: str = "auto") -> Translator:
        return MockTranslator(target_lang, source_lang)

    monkeypatch.setattr(routes, "create_translator", mock)


@pytest.mark.usefixtures("offline")
def test_testing_a_service_translates_a_sample(client: TestClient) -> None:
    sid = create(client)
    resp = client.post("/api/services/test", json={"id": sid, "target_lang": "zh-CN"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "Hello, world! This is a translation test."
    assert body["result"] == "[zh-CN] Hello, world! This is a translation test."
    assert isinstance(body["ms"], int)

    # Into English, the sample is Chinese, so that it is not English translated into English
    english = client.post("/api/services/test", json={"id": sid, "target_lang": "en"}).json()
    assert english["source"] == "你好，世界！这是一条翻译测试。"
    # A text of the user's own
    own = client.post("/api/services/test", json={"id": sid, "text": "Good morning"}).json()
    assert own["result"] == "[zh-CN] Good morning"


@pytest.mark.usefixtures("offline")
def test_testing_a_service_with_a_bad_language_answers_400(client: TestClient) -> None:
    resp = client.post("/api/services/test", json={"id": create(client), "target_lang": "xx"})
    assert resp.status_code == 400 and "不支持的目标语言" in detail(resp)


@pytest.mark.usefixtures("offline")
def test_only_llm_services_list_models(client: TestClient) -> None:
    resp = client.post("/api/services/models", json={"id": create(client)})
    assert resp.status_code == 400 and detail(resp) == "该服务不提供模型列表"
