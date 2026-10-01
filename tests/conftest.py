import pytest

from app import ccswitch


@pytest.fixture(autouse=True)
def isolate_ccswitch(tmp_path, monkeypatch):
    """测试默认不读本机真实的 CC Switch 数据库。"""
    monkeypatch.setattr(ccswitch, "DB_PATH", tmp_path / "no-cc-switch.db")
