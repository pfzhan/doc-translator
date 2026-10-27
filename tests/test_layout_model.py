import asyncio
import dataclasses
import hashlib

import httpx
import pytest
from fastapi.testclient import TestClient

from app import layout_model
from app.formats.pdf import _plan_key
from app.formats.pdf_vision import FormulaModel
from app.settings import Settings, SettingsError

PAYLOAD = b"fake onnx model bytes" * 1000


def test_settings_keep_the_layout_model_choice(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path)
    assert settings.layout_model == {"enabled": False, "path": "", "precision": "int8"}
    settings.update({"layout_model": {"enabled": True}})
    settings.update({"layout_model": {"path": " models/x.onnx "}})
    assert Settings(path).layout_model == {"enabled": True, "path": "models/x.onnx", "precision": "int8"}
    assert Settings(path).retention_days == 30


def test_the_formula_model_is_on_by_default(tmp_path):
    """公式检测模型默认开着，关掉以后记住。两个模型的设置互不影响"""
    path = tmp_path / "settings.json"
    settings = Settings(path)
    assert settings.formula_model == {"enabled": True, "path": "", "size": 1024, "precision": "int8", "workers": 0, "threads": 0}
    settings.update({"formula_model": {"enabled": False}})
    settings.update({"formula_model": {"size": 768}})
    assert Settings(path).formula_model == {"enabled": False, "path": "", "size": 768, "precision": "int8", "workers": 0, "threads": 0}
    assert Settings(path).layout_model == {"enabled": False, "path": "", "precision": "int8"}
    with pytest.raises(SettingsError):
        settings.update({"formula_model": {"size": 640}})
    assert Settings(path).formula_model["size"] == 768


def test_formula_parallel_settings_take_small_whole_numbers(tmp_path):
    """同时推理的页数、每页推理线程数存成整数，留空是 0（自动）。别的设置不受影响"""
    path = tmp_path / "settings.json"
    settings = Settings(path)
    settings.update({"formula_model": {"workers": 3, "threads": 2}})
    assert Settings(path).formula_model == {"enabled": True, "path": "", "size": 1024, "precision": "int8", "workers": 3, "threads": 2}
    settings.update({"formula_model": {"workers": "", "threads": None}})
    assert Settings(path).formula_model["workers"] == 0 and Settings(path).formula_model["threads"] == 0
    for bad in ({"workers": -1}, {"workers": 9}, {"threads": 65}, {"threads": 1.5}, {"workers": "2"},
                {"threads": True}):
        with pytest.raises(SettingsError):
            settings.update({"formula_model": bad})
    assert Settings(path).formula_model == {"enabled": True, "path": "", "size": 1024, "precision": "int8", "workers": 0, "threads": 0}


def test_empty_and_relative_paths_resolve_inside_the_project():
    for spec in layout_model.MODELS.values():
        assert spec.resolve("") == spec.default_path()
        assert spec.resolve("data/models/a.onnx") == layout_model.ROOT / "data/models/a.onnx"
    assert layout_model.LAYOUT.default_path() != layout_model.FORMULA.default_path()


def _download(tmp_path, monkeypatch, body: bytes, status: int = 200):
    spec = dataclasses.replace(layout_model.LAYOUT, sha3=hashlib.sha3_256(PAYLOAD).hexdigest())
    transport = httpx.MockTransport(lambda request: httpx.Response(status, content=body))
    dest = tmp_path / "models" / "m.onnx"

    async def run():
        loader = layout_model.Downloader(spec, transport)
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
        assert {source["id"] for source in status["sources"]} == set(layout_model.LAYOUT.sources)
        assert main._model_path("layout") == ""

        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        saved = client.put("/api/layout-model", json={"enabled": True, "path": str(model)}).json()
        assert saved["enabled"] is True and saved["file"] == {"exists": True, "size": 1}
        # 开了模型，翻译时把文件路径交给 PDF 流程
        assert main._model_path("layout") == str(model)

        tested = client.post("/api/layout-model/test", json={"path": str(model)})
        assert tested.status_code == 400 and "onnxruntime" in tested.json()["detail"]
        bad = client.post("/api/layout-model/download", json={"source": "nowhere"})
        assert bad.status_code == 400


def test_formula_model_page_api(app_env, monkeypatch, tmp_path):
    """公式检测模型的接口和版面模型一样，默认开着，设置分开存"""
    main = app_env
    monkeypatch.setattr(layout_model, "runtime_version", lambda: "")
    with TestClient(main.app) as client:
        status = client.get("/api/formula-model").json()
        # 本机 data/models 里下好的模型不算：测试的模型目录是临时的
        assert status["enabled"] is True and status["file"]["exists"] is False
        assert status["default_path"] == str(layout_model.FORMULA.default_path())
        assert {source["id"] for source in status["sources"]} == set(layout_model.FORMULA.sources)
        assert main._model_path("formula") == str(layout_model.FORMULA.default_path())

        model = tmp_path / "mfd.onnx"
        model.write_bytes(b"x")
        saved = client.put("/api/formula-model", json={"path": str(model)}).json()
        assert saved["enabled"] is True and saved["file"] == {"exists": True, "size": 1}
        assert main._model_path("formula") == str(model) and main._model_path("layout") == ""
        assert main._formula_model() == FormulaModel(str(model), 1024)
        assert client.put("/api/formula-model", json={"size": 768}).json()["size"] == 768
        assert main._formula_model() == FormulaModel(str(model), 768)
        assert client.put("/api/formula-model", json={"size": "big"}).status_code == 400
        assert status["auto"]["workers"] >= 1 and status["auto"]["threads"] >= 1
        saved = client.put("/api/formula-model", json={"workers": 1, "threads": 4}).json()
        assert (saved["workers"], saved["threads"]) == (1, 4)
        assert main._formula_model() == FormulaModel(str(model), 768, 1, 4)
        assert client.put("/api/formula-model", json={"threads": 100}).status_code == 400
        client.put("/api/formula-model", json={"enabled": False})
        assert main._model_path("formula") == "" and main._formula_model() is None

        tested = client.post("/api/formula-model/test", json={"path": str(model)})
        assert tested.status_code == 400 and "onnxruntime" in tested.json()["detail"]
        assert client.get("/api/other-model").status_code == 404


def test_saved_analysis_depends_on_the_formula_model_and_its_size(tmp_path):
    """开关公式模型、换输入尺寸，分析结果都不一样，存下的不能拿来用"""
    src = tmp_path / "a.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    model = tmp_path / "mfd.onnx"
    model.write_bytes(b"x")
    keys = {_plan_key(src, ""), _plan_key(src, "", FormulaModel(str(model), 1024)),
            _plan_key(src, "", FormulaModel(str(model), 768))}
    assert len(keys) == 3
    # 同时推理几页、用几个线程只影响快慢
    assert _plan_key(src, "", FormulaModel(str(model), 768, 2, 3)) == _plan_key(src, "", FormulaModel(str(model), 768))
