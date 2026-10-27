import asyncio

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import layout_model, quantize
from app.history import Job
from app.formats.pdf_vision import FormulaModel
from app.settings import Settings, SettingsError


def _pdf(path, pages):
    doc = pymupdf.open()
    for n in range(pages):
        doc.new_page(width=300, height=400).insert_text((40, 60), f"page {n}", fontsize=10)
    doc.save(path)
    return path


def test_calibration_pages_come_from_every_pdf(tmp_path):
    """各个 PDF 轮流抽，页数少的取完为止，一共不超过要的页数；打不开的文件跳过；同样的输入抽到同样的页"""
    small, medium, large = _pdf(tmp_path / "a.pdf", 2), _pdf(tmp_path / "b.pdf", 5), _pdf(tmp_path / "c.pdf", 40)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    picks = quantize.pick_pages([large, medium, small, broken], count=12)
    assert len(picks) == 12 and len(set(picks)) == 12
    by_file = {path.name: sum(1 for p, _ in picks if p == path) for path in (small, medium, large)}
    assert by_file == {"a.pdf": 2, "b.pdf": 5, "c.pdf": 5}
    assert quantize.pick_pages([large, medium, small, broken], count=12) == picks
    assert len(quantize.pick_pages([small, medium], count=64)) == 7
    assert quantize.pick_pages([broken]) == []


def test_calibration_images_fall_back_to_the_sample_page(tmp_path):
    """抽到的页按给的边长画成和翻译时一样的模型输入，存成 8 位；一页都没有时用样例页"""
    picks = [(_pdf(tmp_path / "a.pdf", 2), 1)]
    [image] = quantize.calibration_images(picks, layout_model.sample_page, 1024)
    assert image.dtype == np.uint8 and image.shape == (1, 3, 1024, 768)
    [small] = quantize.calibration_images(picks, layout_model.sample_page, 768)
    assert small.shape == (1, 3, 768, 576)
    [sample] = quantize.calibration_images([], layout_model.sample_page, 1024)
    assert sample.shape[:2] == (1, 3) and sample.max() == 255


def test_the_detection_head_decode_stays_in_float():
    """最后一个块里，cv2/cv3 分支前两层照常量化；最后一层卷积和分支外的解码（拼框、Sigmoid、DFL）不量化"""
    names = [
        "/model.0/conv/Conv", "/model.22/cv2/conv/Conv",
        "/model.23/cv2.0/cv2.0.0/conv/Conv", "/model.23/cv3.0/cv3.0.1/cv3.0.1.1/act/Mul",
        "/model.23/cv2.0/cv2.0.2/Conv", "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/Concat", "/model.23/dfl/Softmax", "/model.23/Sigmoid", "/model.23/Concat_25",
    ]
    assert quantize.decode_nodes(names) == [
        "/model.23/cv2.0/cv2.0.2/Conv", "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/Concat", "/model.23/dfl/Softmax", "/model.23/Sigmoid", "/model.23/Concat_25",
    ]
    # 版面模型（YOLOv10）的分支名前面多 one2one_
    v10 = [
        "/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv", "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv", "/model.23/dfl/conv/Conv", "/model.23/TopK",
    ]
    assert quantize.decode_nodes(v10) == [
        "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv", "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
        "/model.23/dfl/conv/Conv", "/model.23/TopK",
    ]
    assert quantize.decode_nodes(["Conv_0", "Add_1"]) == []
    assert quantize.int8_path(layout_model.MODELS_DIR / "m.onnx").name == "m.int8.onnx"


def test_boxes_are_counted_by_each_model_output_layout():
    """生成后核对框数：公式检测模型每列一个候选框（框和各类的分），版面模型每行一个框（两角坐标、分、类别）"""
    formula = np.zeros((1, 6, 4), np.float32)
    formula[0, 4] = [0.9, 0.1, 0.3, 0.0]
    formula[0, 5] = [0.0, 0.2, 0.1, 0.6]
    assert quantize.SPECS["formula"].boxes(formula) == 3
    layout = np.array([[[0, 0, 9, 9, 0.9, 1], [0, 0, 9, 9, 0.3, 3], [0, 0, 9, 9, 0.5, 2]]], np.float32)
    assert quantize.SPECS["layout"].boxes(layout) == 2


def test_quantizer_reports_failures_and_runs_one_at_a_time(monkeypatch, tmp_path):
    """同一时间只生成一份；量化出错时记下原因，页面上显示"""
    def broken(self, src, dest, images, spec):
        raise quantize.QuantizeError("量化后的模型框出的框太少")

    monkeypatch.setattr(quantize.Quantizer, "_quantize", broken)
    quantizer = quantize.Quantizer()
    spec = quantize.SPECS["formula"]

    async def main():
        quantizer.start(tmp_path / "m.onnx", tmp_path / "m.int8.onnx", [np.zeros((1, 3, 32, 32), np.uint8)], spec)
        assert quantizer.state["running"] and quantizer.state["total"] == 1
        with pytest.raises(quantize.QuantizeError):
            quantizer.start(tmp_path / "m.onnx", tmp_path / "m.int8.onnx", [], spec)
        await quantizer.task

    asyncio.run(main())
    assert quantizer.state["running"] is False and "太少" in quantizer.state["error"]


def test_int8_is_the_default_precision(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path)
    assert settings.formula_model["precision"] == "int8" and settings.layout_model["precision"] == "int8"
    settings.update({"layout_model": {"precision": "fp32"}})
    assert Settings(path).layout_model["precision"] == "fp32"
    settings.update({"formula_model": {"precision": "fp32"}})
    assert Settings(path).formula_model["precision"] == "fp32"
    with pytest.raises(SettingsError):
        settings.update({"formula_model": {"precision": "int4"}})
    assert Settings(path).formula_model["precision"] == "fp32"


def test_translation_uses_the_int8_file_once_it_exists(app_env, monkeypatch):
    """选了 int8：int8 文件生成了就用它，没生成时用 fp32；选了 fp32 一直用 fp32。状态里报出用的是哪个"""
    main = app_env
    monkeypatch.setattr(layout_model, "runtime_version", lambda: "")
    fp32 = layout_model.FORMULA.default_path()
    fp32.parent.mkdir(parents=True)
    fp32.write_bytes(b"x")
    int8 = quantize.int8_path(fp32)
    with TestClient(main.app) as client:
        status = client.get("/api/formula-model").json()
        assert status["precision"] == "int8" and status["in_use"] == str(fp32)
        assert status["int8"]["file"]["exists"] is False and status["int8"]["path"] == str(int8)
        assert main._formula_model() == FormulaModel(str(fp32), 1024)
        int8.write_bytes(b"y")
        assert client.get("/api/formula-model").json()["in_use"] == str(int8)
        assert main._formula_model() == FormulaModel(str(int8), 1024)
        assert client.put("/api/formula-model", json={"precision": "fp32"}).json()["in_use"] == str(fp32)
        assert main._formula_model() == FormulaModel(str(fp32), 1024)
        assert client.put("/api/formula-model", json={"precision": "bf16"}).status_code == 400


def test_layout_model_uses_its_int8_file_too(app_env, monkeypatch):
    """版面模型同样：开着、选了 int8 且 int8 文件生成了时翻译用它"""
    main = app_env
    monkeypatch.setattr(layout_model, "runtime_version", lambda: "")
    fp32 = layout_model.LAYOUT.default_path()
    fp32.parent.mkdir(parents=True)
    fp32.write_bytes(b"x")
    int8 = quantize.int8_path(fp32)
    int8.write_bytes(b"y")
    with TestClient(main.app) as client:
        assert main._model_path("layout") == ""
        status = client.put("/api/layout-model", json={"enabled": True}).json()
        assert status["precision"] == "int8" and status["in_use"] == str(int8)
        assert status["int8"]["file"]["exists"] is True
        assert main._model_path("layout") == str(int8)
        assert client.put("/api/layout-model", json={"precision": "fp32"}).json()["in_use"] == str(fp32)
        assert main._model_path("layout") == str(fp32)


def test_generating_int8_needs_onnx_and_the_model(app_env, monkeypatch):
    """没装 onnx、模型没下时不生成；能生成时用翻译过的 PDF 抽页，写到 fp32 文件旁边"""
    main = app_env
    started = []
    for quantizer in main.quantizers.values():
        monkeypatch.setattr(quantizer, "start",
                            lambda src, dest, images, spec: started.append((src, dest, images, spec)))
    with TestClient(main.app) as client:
        monkeypatch.setattr(quantize, "available", lambda: False)
        r = client.post("/api/formula-model/quantize")
        assert r.status_code == 400 and quantize.INSTALL_HINT in r.json()["detail"]
        monkeypatch.setattr(quantize, "available", lambda: True)
        r = client.post("/api/formula-model/quantize")
        assert r.status_code == 400 and "下载" in r.json()["detail"]
        fp32 = layout_model.FORMULA.default_path()
        fp32.parent.mkdir(parents=True)
        fp32.write_bytes(b"x")
        # 还没有翻译过 PDF：用样例页
        assert client.post("/api/formula-model/quantize").status_code == 200
        src, dest, images, spec = started[-1]
        assert src == fp32 and dest == quantize.int8_path(fp32) and len(images) == 1
        assert spec == quantize.SPECS["formula"]
        assert images[0].shape == (1, 3, 1024, 736)
        for job in (Job(id="p1", filename="paper.pdf", status="done"), Job(id="m1", filename="notes.md")):
            main.jobs.dir(job.id).mkdir(parents=True)
            main.jobs.add(job)
        _pdf(main.jobs.source_path(main.jobs.get("p1")), 3)
        main.jobs.source_path(main.jobs.get("m1")).write_text("# notes")
        assert client.post("/api/formula-model/quantize").status_code == 200
        assert [image.shape for image in started[-1][2]] == [(1, 3, 1024, 768)] * 3


def test_only_one_model_is_quantized_at_a_time(app_env, monkeypatch):
    """版面模型也能生成 int8 版；一个模型在生成时，另一个不开始"""
    main = app_env
    started = []
    monkeypatch.setattr(quantize, "available", lambda: True)
    monkeypatch.setattr(main.quantizers["layout"], "start",
                        lambda src, dest, images, spec: started.append((src, dest, spec)))

    class Busy:
        def done(self):
            return False

    with TestClient(main.app) as client:
        r = client.post("/api/layout-model/quantize")
        assert r.status_code == 400 and "版面模型" in r.json()["detail"]
        fp32 = layout_model.LAYOUT.default_path()
        fp32.parent.mkdir(parents=True)
        fp32.write_bytes(b"x")
        assert client.post("/api/layout-model/quantize").status_code == 200
        assert started == [(fp32, quantize.int8_path(fp32), quantize.SPECS["layout"])]
        monkeypatch.setattr(main.quantizers["formula"], "task", Busy())
        r = client.post("/api/layout-model/quantize")
        assert r.status_code == 400 and "另一个模型" in r.json()["detail"]
        assert client.post("/api/model-model/quantize").status_code == 404
        assert len(started) == 1
