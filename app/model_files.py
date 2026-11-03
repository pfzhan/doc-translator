"""Which layout and formula model files PDF translation uses, from the settings."""

from pathlib import Path

from . import layout_model, quantize, state
from .formats.pdf_vision import FormulaModel


def _model_path(kind: str) -> str:
    """Model file used for PDF translation (chosen by precision, see _model_file). An empty string when disabled;
    only the rules are used then."""
    spec = layout_model.MODELS[kind]
    conf = state.settings.model(spec.key)
    return _model_file(str(spec.resolve(conf["path"])), conf["precision"]) if conf["enabled"] else ""


def _formula_model() -> FormulaModel | None:
    path = _model_path("formula")
    conf = state.settings.formula_model
    return FormulaModel(path, conf["size"], conf["workers"], conf["threads"]) if path else None


def _model_file(path: str, precision: str) -> str:
    """Use the int8 file when int8 is selected and the int8 file has been generated, otherwise the fp32 file."""
    int8 = quantize.int8_path(Path(path))
    return str(int8) if precision == "int8" and int8.is_file() else path
