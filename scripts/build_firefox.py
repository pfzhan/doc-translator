"""Build the Firefox version of the browser extension.

extension/ is the Chrome extension. Firefox needs a different manifest: it runs the background as an event page
listed in background.scripts (Chrome warns about that key and takes only service_worker) and needs an add-on id.
This script copies extension/ to dist/firefox/ and writes that manifest; load dist/firefox/manifest.json in
about:debugging. Run it again after changing the extension. It lives outside extension/ because Chrome refuses to
load a directory holding names that start with "_", such as the __pycache__ Python leaves next to a script.

    python3 scripts/build_firefox.py [output directory]
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "extension"
DEFAULT_OUT = ROOT / "dist" / "firefox"
# The files background.js loads with importScripts in Chrome, then background.js itself: an event page has no
# importScripts, so Firefox loads them in this order
BACKGROUND_SCRIPTS = ["settings.js", "sites.js", "background.js"]
GECKO = {"id": "doc-translator@pfzhan", "strict_min_version": "128.0"}


def firefox_manifest(chrome: dict[str, Any]) -> dict[str, Any]:
    """The Chrome manifest with Firefox's background and add-on id."""
    manifest = dict(chrome)
    manifest["background"] = {"scripts": BACKGROUND_SCRIPTS}
    manifest["browser_specific_settings"] = {"gecko": GECKO}
    return manifest


def build(out: Path = DEFAULT_OUT) -> Path:
    if out.resolve() == SOURCE or SOURCE in out.resolve().parents:
        raise ValueError(f"output directory {out} is inside {SOURCE}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SOURCE, out)
    chrome = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    (out / "manifest.json").write_text(
        json.dumps(firefox_manifest(chrome), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out


if __name__ == "__main__":
    target = build(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
    print(f"Firefox extension written to {target}; load {target / 'manifest.json'} in about:debugging")
