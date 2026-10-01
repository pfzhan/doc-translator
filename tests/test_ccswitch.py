import asyncio
import json
import sqlite3

import httpx
import pytest

from app import ccswitch
from app.services import ServiceError, ServiceStore
from app.translators import ClaudeTranslator, GeminiTranslator, OpenAITranslator, create_translator


def run(coro):
    return asyncio.run(coro)


CODEX_TOML = """model_provider = "custom"
model = "gpt-test"

[model_providers.custom]
name = "relay"
base_url = "https://relay.example.com/v1"
wire_api = "responses"
"""

GROK_TOML = """[models]
default = "grok-x"

[model]
[model."grok-x"]
model = "grok-x"
base_url = "https://grok.example.com"
api_key = "xai-test-key-123456"
api_backend = "responses"
"""


def make_db(path, rows, common=None):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE providers (id TEXT, app_type TEXT, name TEXT, settings_config TEXT,
                   meta TEXT DEFAULT '{}', is_current BOOLEAN DEFAULT 0)""")
    con.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)")
    for app, name, cfg, meta, current in rows:
        con.execute("INSERT INTO providers VALUES (?, ?, ?, ?, ?, ?)",
                    (f"{app}-{name}", app, name, json.dumps(cfg), json.dumps(meta), current))
    for k, v in (common or {}).items():
        con.execute("INSERT INTO settings VALUES (?, ?)", (f"common_config_{k}", v))
    con.commit()
    con.close()
    return path


@pytest.fixture
def cc_db(tmp_path):
    return make_db(tmp_path / "cc.db", [
        ("claude", "relay-claude", {"env": {"ANTHROPIC_AUTH_TOKEN": "sk-claude-token-123456",
                                            "ANTHROPIC_BASE_URL": "https://claude.example.com",
                                            "ANTHROPIC_MODEL": "claude-test[1M]"}}, {}, 1),
        ("claude", "other", {"env": {"ANTHROPIC_AUTH_TOKEN": "nope"}}, {}, 0),
        ("codex", "relay-gpt", {"auth": {"OPENAI_API_KEY": "sk-openai-key-123456"}, "config": CODEX_TOML}, {}, 1),
        ("gemini", "relay-gemini", {"env": {"GOOGLE_GEMINI_BASE_URL": "https://gemini.example.com",
                                            "GEMINI_API_KEY": "sk-gemini-key-123456"}},
         {"commonConfigEnabled": True}, 1),
        ("grokbuild", "relay-grok", {"config": GROK_TOML}, {}, 1),
    ], common={"gemini": json.dumps({"GEMINI_MODEL": "gemini-test"})})


def test_read_current(cc_db):
    items = {i["app"]: i for i in ccswitch.read_current(cc_db)}
    assert set(items) == {"claude", "codex", "gemini", "grokbuild"}

    c = items["claude"]
    assert (c["base_url"], c["model"], c["auth_style"], c["api_key"]) == (
        "https://claude.example.com/v1", "claude-test", "bearer", "sk-claude-token-123456")

    o = items["codex"]
    assert (o["provider"], o["base_url"], o["model"], o["api_format"]) == (
        "openai", "https://relay.example.com/v1", "gpt-test", "responses")

    g = items["gemini"]
    assert (g["base_url"], g["model"]) == ("https://gemini.example.com/v1beta", "gemini-test")  # 模型来自通用配置

    x = items["grokbuild"]
    assert (x["provider"], x["base_url"], x["model"], x["api_key"], x["api_format"]) == (
        "grok", "https://grok.example.com/v1", "grok-x", "xai-test-key-123456", "responses")


def test_official_login_is_unavailable(tmp_path):
    db = make_db(tmp_path / "cc.db", [
        ("grokbuild", "Grok Official", {"config": "[cli]\ninstaller = \"internal\"\n"}, {}, 1),
        ("claude", "Claude Official", {"env": {}}, {}, 1),
    ])
    items = {i["app"]: i for i in ccswitch.read_current(db)}
    assert not items["grokbuild"]["available"] and "OAuth" in items["grokbuild"]["reason"]
    assert not items["claude"]["available"]


def test_missing_or_broken_db(tmp_path):
    assert ccswitch.read_current(tmp_path / "missing.db") == []
    (tmp_path / "bad.db").write_text("not sqlite")
    assert ccswitch.read_current(tmp_path / "bad.db") == []


def test_store_merges_ccswitch_services(tmp_path, cc_db):
    store = ServiceStore(tmp_path / "s.json", ccswitch_db=cc_db)
    listed = store.list()
    ids = [s["id"] for s in listed["services"]]
    assert ids == ["google", "ccswitch-claude", "ccswitch-codex", "ccswitch-gemini", "ccswitch-grok"]
    assert all("api_key" not in s for s in listed["services"])
    claude = next(s for s in listed["services"] if s["id"] == "ccswitch-claude")
    assert claude["name"] == "Claude · relay-claude" and claude["api_key_hint"] == "sk-…3456"

    # 只能改本地设置，地址和 Key 改不了
    store.update("ccswitch-claude", {"model": "claude-other", "concurrency": 2,
                                     "base_url": "https://evil.example.com", "api_key": "x"})
    svc = store.get("ccswitch-claude")
    assert (svc["model"], svc["concurrency"], svc["base_url"], svc["api_key"]) == (
        "claude-other", 2, "https://claude.example.com/v1", "sk-claude-token-123456")
    # 模型改回和 CC Switch 一致时不单独保存
    store.update("ccswitch-claude", {"model": "claude-test"})
    assert store.ccswitch_settings["ccswitch-claude"]["model"] == ""

    with pytest.raises(ServiceError, match="不能删除"):
        store.delete("ccswitch-claude")

    store.set_default("ccswitch-codex")
    assert ServiceStore(tmp_path / "s.json", ccswitch_db=cc_db).list()["default"] == "ccswitch-codex"
    # CC Switch 读不到时默认服务退回谷歌翻译
    assert ServiceStore(tmp_path / "s.json", ccswitch_db=tmp_path / "gone.db").list()["default"] == "google"


def test_unavailable_service_cannot_be_enabled(tmp_path):
    db = make_db(tmp_path / "cc.db", [("grokbuild", "Grok Official", {"config": ""}, {}, 1)])
    store = ServiceStore(tmp_path / "s.json", ccswitch_db=db)
    assert store.get("ccswitch-grok")["enabled"] is False
    with pytest.raises(ServiceError, match="无法启用"):
        store.update("ccswitch-grok", {"enabled": True})
    with pytest.raises(ServiceError):
        store.set_default("ccswitch-grok")


def capture(reply):
    log = []

    def handler(request):
        log.append(request)
        return httpx.Response(200, json=reply)

    return log, httpx.MockTransport(handler)


def test_translators_from_ccswitch(tmp_path, cc_db):
    store = ServiceStore(tmp_path / "s.json", ccswitch_db=cc_db)

    log, t = capture({"content": [{"type": "text", "text": "你好"}]})
    tr = create_translator(store.get("ccswitch-claude"), "zh-CN", transport=t)
    assert isinstance(tr, ClaudeTranslator) and run(tr.translate_batch(["hi"])) == ["你好"]
    assert str(log[0].url) == "https://claude.example.com/v1/messages"
    assert log[0].headers["authorization"] == "Bearer sk-claude-token-123456"
    assert "x-api-key" not in log[0].headers
    assert json.loads(log[0].content)["model"] == "claude-test"

    log, t = capture({"output": [{"type": "message", "content": [{"type": "output_text", "text": "你好"}]}]})
    tr = create_translator(store.get("ccswitch-codex"), "zh-CN", transport=t)
    assert isinstance(tr, OpenAITranslator) and run(tr.translate_batch(["hi"])) == ["你好"]
    assert str(log[0].url) == "https://relay.example.com/v1/responses"
    body = json.loads(log[0].content)
    assert body["instructions"].startswith("你是专业的简体中文母语译者") and body["input"].endswith("hi")

    log, t = capture({"output_text": "你好"})
    tr = create_translator(store.get("ccswitch-grok"), "zh-CN", transport=t)
    assert run(tr.translate_batch(["hi"])) == ["你好"]
    assert str(log[0].url) == "https://grok.example.com/v1/responses"

    log, t = capture({"candidates": [{"content": {"parts": [{"text": "你好"}]}}]})
    tr = create_translator(store.get("ccswitch-gemini"), "zh-CN", transport=t)
    assert isinstance(tr, GeminiTranslator) and run(tr.translate_batch(["hi"])) == ["你好"]
    assert str(log[0].url) == "https://gemini.example.com/v1beta/models/gemini-test:generateContent"


def test_resolve_ignores_connection_overrides(tmp_path, cc_db):
    store = ServiceStore(tmp_path / "s.json", ccswitch_db=cc_db)
    svc = store.resolve({"id": "ccswitch-gemini", "base_url": "https://evil.example.com", "model": ""})
    assert svc["base_url"] == "https://gemini.example.com/v1beta" and svc["model"] == "gemini-test"
