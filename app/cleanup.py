"""清理一条翻译记录留下的缓存痕迹，原文件和记录本身保留。

包括三处：
- 翻译缓存里这份文件的译文和术语表。按缓存登记的归属找（见 runner.Cache.owner），
  登记归属之前存下的，按上次版面分析的送翻原文和服务现在的配置反推键再找一遍；
- 记录目录里生成的文件：译文（out/）、预览快照、版面分析结果、补生成的临时目录；
- 版面模型框出的区域（boxes.json）。和译文无关，重新识别要几分钟，用户勾选了才清。

同一份文件可以传成几条记录，它们共用缓存。清理一条时照样删，界面上提示共用的条数；
删除记录时只在没有别的记录用这份文件时才删缓存，免得删掉一条重复的连累另一条。
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import glossary
from .formats import document_digest
from .formats.pdf import saved_texts
from .history import Job, JobStore
from .runner import Cache, noted_key

# 记录目录里翻译时生成的文件和目录。原文件、job.json 不在其中；out/ 只清里面的译文，目录留着写下一次的
_GENERATED = ("preview.json", "plan.pickle", ".variant")
_MODEL_BOXES = "boxes.json"


@dataclass
class Footprint:
    """一条记录留下的痕迹。keys 是缓存里还在的条目。"""
    doc: str
    keys: set[str] = field(default_factory=set)
    generated: list[Path] = field(default_factory=list)
    boxes: Path | None = None
    shared: int = 0  # 用同一份文件的其他记录数

    def summary(self, outputs: int) -> dict:
        return {
            "entries": len(self.keys),
            "outputs": outputs,
            "generated_bytes": sum(_size(path) for path in self.generated),
            "boxes_bytes": _size(self.boxes) if self.boxes else 0,
            "shared": self.shared,
        }


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def document_of(job: Job, jobs: JobStore) -> str:
    """记录对应文件的摘要。旧记录没存，原文件还在就算一次存下。"""
    if not job.document:
        source = jobs.source_path(job)
        if not source.is_file():
            return ""
        job.document = document_digest(source)
        jobs.save(job)
    return job.document


def _legacy_keys(job: Job, jobs: JobStore, cache: Cache, translator, doc: str) -> set[str]:
    """登记归属之前存下的条目：用上次版面分析的送翻原文、服务现在的配置算键。
    服务的配置后来改过（换模型、改提示词）就对不上，那些条目只能等缓存满了被挤掉。"""
    texts = saved_texts(jobs.plan_path(job.id))
    if translator is None or not texts:
        return set()
    translator.detected_source = job.detected_lang
    terms = translator.cache_key + ":terms"
    keys = Cache.keys(terms, [f"file:{doc}"])
    found = cache.get_many(terms, [f"file:{doc}"])
    if found and hasattr(translator, "doc_glossary"):
        # 术语表是缓存键的一部分，有术语表的文件要带上它才算得对
        translator.doc_glossary = glossary.parse(next(iter(found.values())))
    return keys | Cache.keys(translator.cache_key, [noted_key(text, note) for text, note in texts])


def footprint(job: Job, jobs: JobStore, cache: Cache, translator=None) -> Footprint:
    """找出记录留下的痕迹，不改动任何东西。translator 按记录的服务现在的配置创建，用来反推旧条目；
    服务已经删了就传 None。"""
    doc = document_of(job, jobs)
    found = Footprint(doc)
    if doc:
        found.keys = cache.owned(doc, _legacy_keys(job, jobs, cache, translator, doc))
        found.shared = sum(1 for other in jobs.list() if other.id != job.id and document_of(other, jobs) == doc)
    folder = jobs.dir(job.id)
    outputs = jobs.out_dir(job.id)
    found.generated = [*(outputs.iterdir() if outputs.is_dir() else ()),
                       *(path for name in _GENERATED if (path := folder / name).exists())]
    boxes = folder / _MODEL_BOXES
    found.boxes = boxes if boxes.is_file() else None
    return found


def purge(job: Job, jobs: JobStore, cache: Cache, translator=None, boxes: bool = False) -> int:
    """清理记录留下的痕迹，返回删掉的缓存条目数。调用方保证记录不在翻译、补生成或删除中。"""
    found = footprint(job, jobs, cache, translator)
    removed = cache.purge(found.doc, found.keys) if found.doc else 0
    for path in found.generated + ([found.boxes] if boxes and found.boxes else []):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return removed


def forget(job: Job, jobs: JobStore, cache: Cache, translator=None) -> int:
    """删除记录前清掉它的缓存。还有别的记录用同一份文件时不动，返回 0。"""
    found = footprint(job, jobs, cache, translator)
    if not found.doc or found.shared:
        return 0
    return cache.purge(found.doc, found.keys)
