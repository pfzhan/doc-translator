"""按 papers.tsv 从 arXiv 下载评测集的论文。用法: fetch.py <papers.tsv> <输出目录>
每篇放进自己的空目录 <输出目录>/<不带版本号的编号>/：paper.pdf、meta.txt（分类和带版本号的编号），
LaTeX 源码只解出 .tex、.bbl、.sty、.cls 到 src/。源码只当文本读，不编译。
下载的 PDF 和列表里的 SHA-256 对不上时报出来（arXiv 偶尔会重新生成 PDF）。已有的目录跳过。"""
import gzip, hashlib, io, sys, tarfile, time, urllib.request
from pathlib import Path

tsv, out = Path(sys.argv[1]), Path(sys.argv[2])
UA = {"User-Agent": "doc-translator-eval/0.1 (layout regression set)"}


def get(url: str) -> bytes:
    # arXiv 要求两次请求之间隔 3 秒以上
    time.sleep(3.1)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def safe_extract(blob: bytes, dest: Path) -> None:
    dest.mkdir(parents=True)
    try:
        tar = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
    except tarfile.TarError:
        # 只有一个 tex 文件的投稿，e-print 是 gzip 压缩的单个文件或者纯文本
        try:
            (dest / "main.tex").write_bytes(gzip.decompress(blob))
        except OSError:
            (dest / "main.tex").write_bytes(blob)
        return
    for m in tar.getmembers():
        name = Path(m.name)
        if not m.isfile() or name.is_absolute() or ".." in name.parts:
            continue
        if name.suffix.lower() not in (".tex", ".bbl", ".sty", ".cls"):
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(tar.extractfile(m).read())


rows = [line.split("\t") for line in tsv.read_text().splitlines()[1:] if line.strip()]
for aid, cat, _pages, sha256, _title in rows:
    home = out / aid.rsplit("v", 1)[0]
    if home.exists():
        continue
    try:
        pdf = get(f"https://arxiv.org/pdf/{aid}")
        src = get(f"https://arxiv.org/e-print/{aid}")
    except Exception as error:  # noqa: BLE001  网络错误、arXiv 限流
        print(aid, "skip", error, flush=True)
        continue
    home.mkdir(parents=True)
    (home / "paper.pdf").write_bytes(pdf)
    safe_extract(src, home / "src")
    (home / "meta.txt").write_text(f"{cat}\n{aid}\n")
    same = hashlib.sha256(pdf).hexdigest() == sha256
    print(aid, "ok" if same else "PDF 和列表里的 SHA-256 不同", flush=True)
