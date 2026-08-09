"""Ingesting documents: versioning, change detection, and what gets kept on disk."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from customer_support_chatbot.ingestion.models import DocumentStatus, IngestOutcome
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.store import KnowledgeBase
from tests.builders import write_document


def test_ingesting_a_new_document_makes_its_chunks_searchable(
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
) -> None:
    path = write_document(
        tmp_path,
        "schedule.txt",
        ["Vaccination schedule for children under six years of age.", "Booster doses follow."],
    )

    result = pipeline.ingest("vaccination-schedule", path)

    assert result.outcome is IngestOutcome.INGESTED
    assert result.document.status is DocumentStatus.LIVE
    assert result.document.ingestion_version == 1
    assert result.document.page_count == 2
    assert result.document.chunk_count > 0
    assert knowledge_base.count_chunks("vaccination-schedule") == result.document.chunk_count


def test_reingesting_an_unchanged_file_does_nothing(
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "schedule.txt", ["Vaccination schedule for children."])
    pipeline.ingest("vaccination-schedule", path)

    result = pipeline.ingest("vaccination-schedule", path)

    assert result.outcome is IngestOutcome.UNCHANGED
    assert result.document.ingestion_version == 1


def test_force_reingests_an_unchanged_file(
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "schedule.txt", ["Vaccination schedule for children."])
    pipeline.ingest("vaccination-schedule", path)

    result = pipeline.ingest("vaccination-schedule", path, force=True)

    assert result.outcome is IngestOutcome.INGESTED
    assert result.document.ingestion_version == 2


def test_a_revised_document_leaves_no_trace_of_the_old_one(
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "screening.txt", ["The screening programme runs on Tuesdays."])
    pipeline.ingest("screening-programme", path)

    path.write_text("The screening programme runs on Thursdays.", encoding="utf-8")
    result = pipeline.ingest("screening-programme", path)

    assert result.outcome is IngestOutcome.INGESTED
    assert result.document.ingestion_version == 2
    texts = knowledge_base.list_chunk_texts("screening-programme")
    assert len(texts) == result.document.chunk_count
    assert all("Tuesdays" not in text for text in texts)
    assert any("Thursdays" in text for text in texts)


def test_the_raw_file_is_kept_for_every_ingested_version(
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "notice.txt", ["First version of the notice text."])
    pipeline.ingest("notice", path)
    path.write_text("Second version of the notice text.", encoding="utf-8")
    pipeline.ingest("notice", path)

    stored = sorted((tmp_path / "raw" / "notice").iterdir())

    assert [item.name for item in stored] == ["0001.txt", "0002.txt"]


def test_a_relative_raw_files_root_still_yields_an_absolute_source_uri(
    make_pipeline: Callable[..., IngestionPipeline],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_pipeline = make_pipeline(raw_files_root=Path("./data/raw"))
    path = write_document(tmp_path, "notice.txt", ["A notice long enough to be ingested."])

    result = relative_pipeline.ingest("notice", path)

    assert result.document.source_uri.startswith("file:///")


def test_ingesting_a_missing_file_fails_loudly(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.ingest("nope", tmp_path / "does-not-exist.txt")
