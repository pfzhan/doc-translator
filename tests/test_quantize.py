import asyncio

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import layout_model, quantize
from app.formats.pdf_vision import FormulaModel
from app.history import Job
from app.settings import Settings, SettingsError


def _pdf(path, pages):
    doc = pymupdf.open()
    for n in range(pages):
        doc.new_page(width=300, height=400).insert_text((40, 60), f"page {n}", fontsize=10)
    doc.save(path)
    return path


def test_calibration_pages_come_from_every_pdf(tmp_path):
    """PDFs are sampled in turn until those with few pages run out, never more than the requested count in total;
    files that cannot be opened are skipped; the same input samples the same pages"""
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
    """Sampled pages are rendered at the given side as the same model input as during translation, stored as 8 bits;
    without any page the sample pages are used"""
    picks = [(_pdf(tmp_path / "a.pdf", 2), 1)]
    [image] = quantize.calibration_images(picks, layout_model.sample_page, 1024)
    assert image.dtype == np.uint8 and image.shape == (1, 3, 1024, 768)
    [small] = quantize.calibration_images(picks, layout_model.sample_page, 768)
    assert small.shape == (1, 3, 768, 576)
    [sample] = quantize.calibration_images([], layout_model.sample_page, 1024)
    assert sample.shape[:2] == (1, 3) and sample.max() == 255


def test_the_detection_head_decode_stays_in_float():
    """In the last block, the first two layers of the cv2/cv3 branches are quantized as usual; the last convolution and
    the decoding outside the branches (concatenating boxes, Sigmoid, DFL) are not"""
    names = [
        "/model.0/conv/Conv",
        "/model.22/cv2/conv/Conv",
        "/model.23/cv2.0/cv2.0.0/conv/Conv",
        "/model.23/cv3.0/cv3.0.1/cv3.0.1.1/act/Mul",
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/Concat",
        "/model.23/dfl/Softmax",
        "/model.23/Sigmoid",
        "/model.23/Concat_25",
    ]
    assert quantize.decode_nodes(names) == [
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/Concat",
        "/model.23/dfl/Softmax",
        "/model.23/Sigmoid",
        "/model.23/Concat_25",
    ]
    # Branch names of the layout model (YOLOv10) are prefixed with one2one_
    v10 = [
        "/model.23/one2one_cv2.0/one2one_cv2.0.1/conv/Conv",
        "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.0/conv/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
        "/model.23/dfl/conv/Conv",
        "/model.23/TopK",
    ]
    assert quantize.decode_nodes(v10) == [
        "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
        "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
        "/model.23/dfl/conv/Conv",
        "/model.23/TopK",
    ]
    assert quantize.decode_nodes(["Conv_0", "Add_1"]) == []
    assert quantize.int8_path(layout_model.MODELS_DIR / "m.onnx").name == "m.int8.onnx"


def test_boxes_are_counted_by_each_model_output_layout():
    """Box count check after generation: formula detection has one candidate per column (box and class scores), the
    layout model one box per row (corner coordinates, score, class)"""
    formula = np.zeros((1, 6, 4), np.float32)
    formula[0, 4] = [0.9, 0.1, 0.3, 0.0]
    formula[0, 5] = [0.0, 0.2, 0.1, 0.6]
    assert quantize.SPECS["formula"].boxes(formula) == 3
    layout = np.array([[[0, 0, 9, 9, 0.9, 1], [0, 0, 9, 9, 0.3, 3], [0, 0, 9, 9, 0.5, 2]]], np.float32)
    assert quantize.SPECS["layout"].boxes(layout) == 2


def test_quantizer_reports_failures_and_runs_one_at_a_time(monkeypatch, tmp_path):
    """Only one generation at a time; a quantization error is recorded and shown on the page"""

    def broken(self, src, dest, images, spec):
        raise quantize.QuantizeError("量化后的模型识别出的区域过少")

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
    assert quantizer.state["running"] is False and "过少" in quantizer.state["error"]


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
    """With int8 chosen, the int8 file is used once generated and fp32 before that; with fp32 chosen, always fp32. The
    status reports which one is used"""
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
    """Same for the layout model: translation uses int8 when enabled, int8 is chosen and the int8 file exists"""
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
    """No generation without onnx or a downloaded model; when possible, pages are sampled from translated PDFs and
    the result is written next to the fp32 file"""
    main = app_env
    started = []
    for quantizer in main.quantizers.values():
        monkeypatch.setattr(
            quantizer, "start", lambda src, dest, images, spec: started.append((src, dest, images, spec))
        )
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
        # No PDF translated yet: use the sample pages
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
    """The layout model can also get an int8 version; while one model is generating, the other does not start"""
    main = app_env
    started = []
    monkeypatch.setattr(quantize, "available", lambda: True)
    monkeypatch.setattr(
        main.quantizers["layout"], "start", lambda src, dest, images, spec: started.append((src, dest, spec))
    )

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
