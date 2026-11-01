"""Clear what a translation record left in caches, keeping the source file and the record itself.

Four places are covered:
- Translations and glossary of this file in the translation cache, found by the owner registered in the cache
  (see runner.Cache.owner). Entries saved before owners were registered are found again by recomputing their keys
  from the source text of the last layout analysis and the service's current configuration;
- Files generated in the record directory: translations (out/), the preview snapshot, temporary directories for
  regeneration;
- The book's layout analysis result (layout/<file digest>/plan.pickle);
- Regions boxed by the layout model (boxes.json in the same directory). Unrelated to translations and taking
  minutes to recompute, they are cleared only when the user ticks the option.

One file can be uploaded as several records, which share the translation cache and the layout analysis result.
Clearing one record deletes them anyway, and the interface shows how many records share them; deleting a record
removes them only when no other record uses the file, so deleting a duplicate does not affect the other one.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import glossary
from .formats.pdf import saved_texts
from .history import Job, JobStore
from .runner import Cache, noted_key

# Files and directories generated in the record directory during translation. The source file and job.json are not
# among them; only the translations inside out/ are cleared, and the directory stays for the next run
_GENERATED = ("preview.json", ".variant")


@dataclass
class Footprint:
    """What a record left behind. keys are the entries still in the cache."""

    doc: str
    keys: set[str] = field(default_factory=set)
    generated: list[Path] = field(default_factory=list)
    boxes: Path | None = None
    shared: int = 0  # number of other records using the same file

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


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _legacy_keys(job: Job, jobs: JobStore, cache: Cache, translator, doc: str) -> set[str]:
    """Entries saved before owners were registered: keys computed from the source text of the last layout analysis
    and the service's current configuration. When the service configuration changed since (another model, another
    prompt), the keys no longer match, and those entries are only evicted when the cache fills up."""
    plan = jobs.plan_path(job)
    texts = saved_texts(plan) if plan else []
    if translator is None or not texts:
        return set()
    translator.detected_source = job.detected_lang
    terms = translator.cache_key + ":terms"
    keys = Cache.keys(terms, [f"file:{doc}"])
    found = cache.get_many(terms, [f"file:{doc}"])
    if found and hasattr(translator, "doc_glossary"):
        # The glossary is part of the cache key, so files with a glossary need it for the key to be correct
        translator.doc_glossary = glossary.parse(next(iter(found.values())))
    return keys | Cache.keys(translator.cache_key, [noted_key(text, note) for text, note in texts])


def footprint(job: Job, jobs: JobStore, cache: Cache, translator=None) -> Footprint:
    """Find what a record left behind without deleting anything (old layout analysis results in the record directory
    are moved to the shared directory along the way, see JobStore.layout_dir). translator is created from the
    current configuration of the record's service and used to recompute old entries; None when the service has
    been deleted."""
    doc = jobs.document(job)
    found = Footprint(doc)
    outputs = jobs.out_dir(job.id)
    found.generated = [
        *(outputs.iterdir() if outputs.is_dir() else ()),
        *(path for name in _GENERATED if (path := jobs.dir(job.id) / name).exists()),
    ]
    if not doc:
        return found
    found.keys = cache.owned(doc, _legacy_keys(job, jobs, cache, translator, doc))
    found.shared = sum(1 for other in jobs.list() if other.id != job.id and jobs.document(other) == doc)
    layout = jobs.layout_dir(job)
    if (layout / "plan.pickle").is_file():
        found.generated.append(layout / "plan.pickle")
    if (layout / "boxes.json").is_file():
        found.boxes = layout / "boxes.json"
    return found


def purge(job: Job, jobs: JobStore, cache: Cache, translator=None, boxes: bool = False) -> int:
    """Clear what a record left behind and return the number of cache entries deleted. The caller ensures the record
    is not being translated, regenerated or deleted."""
    found = footprint(job, jobs, cache, translator)
    removed = cache.purge(found.doc, found.keys) if found.doc else 0
    for path in found.generated + ([found.boxes] if boxes and found.boxes else []):
        _remove(path)
    return removed


def forget(job: Job, jobs: JobStore, cache: Cache, translator=None) -> int:
    """Before a record is deleted, clear the book's cache and layout analysis result. Nothing is touched, and 0 is
    returned, when another record uses the same file."""
    found = footprint(job, jobs, cache, translator)
    if not found.doc or found.shared:
        return 0
    removed = cache.purge(found.doc, found.keys)
    _remove(jobs.layout_root / found.doc)
    return removed


def prune_layouts(jobs: JobStore) -> list[str]:
    """Delete layout analysis results of books no record uses any more (left over when records were deleted, or
    whose source file is gone) and return the deleted digests."""
    if not jobs.layout_root.is_dir():
        return []
    used = {jobs.document(job) for job in jobs.list()}
    stale = [folder for folder in jobs.layout_root.iterdir() if folder.is_dir() and folder.name not in used]
    for folder in stale:
        _remove(folder)
    return [folder.name for folder in stale]
