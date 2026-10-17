import asyncio
import hashlib

import httpx
from fastapi.testclient import TestClient

from app import layout_model
from app.settings import Settings

PAYLOAD = b"fake onnx model bytes" * 1000


def test_settings_keep_the_layout_model_choice(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path)
    assert settings.layout_model == {"enabled": False, "path": ""}
    settings.update({"layout_model": {"enabled": True}})
    settings.update({"layout_model": {"path": " models/x.onnx "}})
    assert Settings(path).layout_model == {"enabled": True, "path": "models/x.onnx"}
    assert Settings(path).retention_days == 30


def test_empty_and_relative_paths_resolve_inside_the_project():
    assert layout_model.resolve("") == layout_model.default_path()
    assert layout_model.resolve("data/models/a.onnx") == layout_model.ROOT / "data/models/a.onnx"


def _download(tmp_path, monkeypatch, body: bytes, status: int = 200):
    monkeypatch.setattr(layout_model, "MODEL_SHA3", hashlib.sha3_256(PAYLOAD).hexdigest())
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=body))
    dest = tmp_path / "models" / "m.onnx"

    async def run():
        loader = layout_model.Downloader(transport)
        loader.start("hf-mirror", dest)
        await loader.task
        return loader.state

    return dest, asyncio.run(run())


def test_download_keeps_the_file_only_when_the_checksum_matches(tmp_path, monkeypatch):
    dest, state = _download(tmp_path, monkeypatch, PAYLOAD)
    assert state["error"] == "" and not state["running"]
    assert state["done"] == len(PAYLOAD) and dest.read_bytes() == PAYLOAD
    broken, state = _download(tmp_path / "b", monkeypatch, PAYLOAD[:-10])
    assert "校验" in state["error"]
    assert not broken.exists() and not broken.with_name("m.onnx.part").exists()
    missing, state = _download(tmp_path / "c", monkeypatch, b"", status=404)
    assert "404" in state["error"] and not missing.exists()


def test_layout_model_page_api(app_env, monkeypatch, tmp_path):
    main = app_env
    monkeypatch.setattr(layout_model, "runtime_version", lambda: "")
    with TestClient(main.app) as client:
        status = client.get("/api/layout-model").json()
        assert status["enabled"] is False and status["runtime"] == ""
        assert status["install_hint"] == layout_model.INSTALL_HINT
        assert {source["id"] for source in status["sources"]} == set(layout_model.SOURCES)
        assert main._layout_model_path() == ""

        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        saved = client.put("/api/layout-model", json={"enabled": True, "path": str(model)}).json()
        assert saved["enabled"] is True and saved["file"] == {"exists": True, "size": 1}
        # 开了模型，翻译时把文件路径交给 PDF 流程
        assert main._layout_model_path() == str(model)

        tested = client.post("/api/layout-model/test", json={"path": str(model)})
        assert tested.status_code == 400 and "onnxruntime" in tested.json()["detail"]
        bad = client.post("/api/layout-model/download", json={"source": "nowhere"})
        assert bad.status_code == 400
