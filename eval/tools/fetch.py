"""Download the evaluation papers from arXiv as listed in papers.tsv. Usage: fetch.py <papers.tsv> <output dir>
Each paper goes into its own empty directory <output dir>/<id without version>/: paper.pdf, meta.txt (category and
versioned id), and only the .tex, .bbl, .sty and .cls files of the LaTeX source, extracted to src/. The source is
read as text only, never compiled. A downloaded PDF whose SHA-256 differs from the list is reported (arXiv
occasionally regenerates PDFs). Existing directories are skipped."""

import gzip
import hashlib
import io
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

tsv, out = Path(sys.argv[1]), Path(sys.argv[2])
UA = {"User-Agent": "doc-translator-eval/0.1 (layout regression set)"}


def get(url: str) -> bytes:
    # arXiv asks for at least 3 seconds between requests
    time.sleep(3.1)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        data: bytes = r.read()
        return data


def safe_extract(blob: bytes, dest: Path) -> None:
    dest.mkdir(parents=True)
    try:
        tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")  # noqa: SIM115 - closed by the with below
    except tarfile.TarError:
        # For a submission with a single tex file, the e-print is one gzip-compressed file or plain text
        try:
            (dest / "main.tex").write_bytes(gzip.decompress(blob))
        except OSError:
            (dest / "main.tex").write_bytes(blob)
        return
    with tar:
        for m in tar.getmembers():
            name = Path(m.name)
            if not m.isfile() or name.is_absolute() or ".." in name.parts:
                continue
            if name.suffix.lower() not in (".tex", ".bbl", ".sty", ".cls"):
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            content = tar.extractfile(m)
            assert content is not None  # m is a regular file
            target.write_bytes(content.read())


rows = [line.split("\t") for line in tsv.read_text().splitlines()[1:] if line.strip()]
for aid, cat, _pages, sha256, _title in rows:
    home = out / aid.rsplit("v", 1)[0]
    if home.exists():
        continue
    try:
        pdf = get(f"https://arxiv.org/pdf/{aid}")
        src = get(f"https://arxiv.org/e-print/{aid}")
    except Exception as error:  # noqa: BLE001 - network errors, arXiv rate limiting
        print(aid, "skip", error, flush=True)
        continue
    home.mkdir(parents=True)
    (home / "paper.pdf").write_bytes(pdf)
    safe_extract(src, home / "src")
    (home / "meta.txt").write_text(f"{cat}\n{aid}\n")
    same = hashlib.sha256(pdf).hexdigest() == sha256
    print(aid, "ok" if same else "PDF 和列表里的 SHA-256 不同", flush=True)
