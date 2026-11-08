import json
from pathlib import Path

import pytest

from scripts import build_firefox

CHROME_MANIFEST = json.loads((build_firefox.SOURCE / "manifest.json").read_text(encoding="utf-8"))


def test_chrome_manifest_has_no_firefox_keys() -> None:
    # Chrome warns about background.scripts in Manifest V3
    assert CHROME_MANIFEST["background"] == {"service_worker": "background.js"}
    assert "browser_specific_settings" not in CHROME_MANIFEST


def test_background_scripts_match_what_chrome_imports() -> None:
    background = (build_firefox.SOURCE / "background.js").read_text(encoding="utf-8")
    imported = ", ".join(f"'{name}'" for name in build_firefox.BACKGROUND_SCRIPTS[:-1])
    assert f"importScripts({imported})" in background
    assert build_firefox.BACKGROUND_SCRIPTS[-1] == CHROME_MANIFEST["background"]["service_worker"]


def test_manifest_icons_exist_at_their_sizes() -> None:
    icons = {**CHROME_MANIFEST["icons"], **CHROME_MANIFEST["action"]["default_icon"]}
    for size, name in icons.items():
        png = (build_firefox.SOURCE / name).read_bytes()
        # The PNG header holds width and height as 4-byte big-endian numbers after the IHDR tag
        assert png[12:16] == b"IHDR", name
        assert (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")) == (int(size), int(size)), name


def test_build_writes_firefox_manifest(tmp_path: Path) -> None:
    out = build_firefox.build(tmp_path / "firefox")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["background"] == {"scripts": ["settings.js", "sites.js", "background.js"]}
    assert manifest["browser_specific_settings"]["gecko"]["id"] == "doc-translator@pfzhan"
    # Everything else is the Chrome manifest
    rest = {k: v for k, v in manifest.items() if k not in ("background", "browser_specific_settings")}
    assert rest == {k: v for k, v in CHROME_MANIFEST.items() if k != "background"}
    for name in ["content.js", "content.css", "selection.js", "options.html", *build_firefox.BACKGROUND_SCRIPTS]:
        assert (out / name).is_file(), name


def test_build_replaces_old_output(tmp_path: Path) -> None:
    out = tmp_path / "firefox"
    out.mkdir()
    (out / "stale.js").write_text("old", encoding="utf-8")
    build_firefox.build(out)
    assert not (out / "stale.js").exists()


def test_refuses_output_inside_the_source() -> None:
    with pytest.raises(ValueError):
        build_firefox.build(build_firefox.SOURCE / "firefox")


def test_extension_holds_only_files_chrome_loads() -> None:
    # Chrome refuses the whole directory when a name starts with "_" (a __pycache__ left by Python, for one)
    reserved = [p.name for p in build_firefox.SOURCE.rglob("*") if p.name.startswith("_")]
    assert reserved == []
    assert not list(build_firefox.SOURCE.rglob("*.py"))
