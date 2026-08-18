"""Ingesting a whole manifest in one process.

`kb ingest` loads BGE-M3 before it ingests one file. Doing that a hundred times would spend more
wall clock on loading the model than on embedding, so the pipeline is built once here and the
manifest is walked with it.

The loop is resumable for free: ingestion of an unchanged file is a no-op (the content hash matches
the recorded Document), so a run that stops halfway can be restarted from the top.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from customer_support_chatbot.demo.manifest import Manifest, ManifestEntry
from customer_support_chatbot.ingestion.models import IngestOutcome
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline


@dataclass
class IngestSummary:
    ingested: int = 0
    unchanged: int = 0
    quarantined: int = 0
    missing: int = 0
    failed: int = 0
    pages: int = 0
    chunks: int = 0
    failures: list[str] = field(default_factory=list[str])

    @property
    def attempted(self) -> int:
        return self.ingested + self.unchanged + self.quarantined + self.missing + self.failed


def ingest_manifest(
    manifest: Manifest,
    pdf_root: Path,
    pipeline: IngestionPipeline,
    *,
    limit: int | None = None,
    force: bool = False,
    report: TextIO | None = sys.stdout,
) -> IngestSummary:
    entries = manifest.entries[:limit] if limit is not None else manifest.entries
    summary = IngestSummary()

    for position, entry in enumerate(entries, start=1):
        prefix = f"[{position:>3}/{len(entries)}] {entry.document_key}"
        path = pdf_root / entry.filename
        if not path.is_file():
            summary.missing += 1
            _say(report, f"{prefix}: missing, run `kb-demo fetch` first")
            continue

        line = _ingest_one(entry, path, pipeline, summary, force=force)
        _say(report, f"{prefix}: {line}")

    return summary


def _ingest_one(
    entry: ManifestEntry,
    path: Path,
    pipeline: IngestionPipeline,
    summary: IngestSummary,
    *,
    force: bool,
) -> str:
    # One unreadable Document out of a hundred must not end the run, so every failure is caught
    # and reported at the end rather than raised.
    try:
        result = pipeline.ingest(entry.document_key, path, title=entry.title, force=force)
    except Exception as error:
        summary.failed += 1
        summary.failures.append(f"{entry.document_key}: {error}")
        return f"failed, {error}"

    document = result.document
    if result.outcome is IngestOutcome.UNCHANGED:
        summary.unchanged += 1
        return "unchanged"
    if result.outcome is IngestOutcome.QUARANTINED:
        summary.quarantined += 1
        return f"QUARANTINED, {document.quarantine_reason}"

    summary.ingested += 1
    summary.pages += document.page_count
    summary.chunks += document.chunk_count
    return (
        f"ingested v{document.ingestion_version}, "
        f"{document.page_count} pages, {document.chunk_count} chunks"
    )


def _say(report: TextIO | None, message: str) -> None:
    if report is not None:
        print(message, file=report, flush=True)
