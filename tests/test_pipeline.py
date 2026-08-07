from __future__ import annotations

from pathlib import Path

import pytest

from customer_support_chatbot.ingestion.chunking import ChunkingSettings
from customer_support_chatbot.ingestion.models import DocumentStatus, IngestOutcome
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.raw_files import LocalRawFileStore
from customer_support_chatbot.ingestion.store import KnowledgeBase
from tests.conftest import PAGE_BREAK, HashingEmbedder, TextFileExtractor


@pytest.fixture
def pipeline(
    knowledge_base: KnowledgeBase,
    extractor: TextFileExtractor,
    embedder: HashingEmbedder,
    tmp_path: Path,
) -> IngestionPipeline:
    return IngestionPipeline(
        extractor=extractor,
        embedder=embedder,
        knowledge_base=knowledge_base,
        raw_files=LocalRawFileStore(tmp_path / "raw"),
        chunking=ChunkingSettings(max_tokens=60, overlap_tokens=10),
        # Fixtures are single sentences, so the scan threshold is lowered rather than padding
        # every document. The threshold itself is covered in test_extraction.py.
        min_chars_per_page=20,
    )


def write_document(tmp_path: Path, name: str, pages: list[str]) -> Path:
    path = tmp_path / name
    path.write_text(PAGE_BREAK.join(pages), encoding="utf-8")
    return path


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


def test_a_document_with_no_text_layer_is_quarantined(
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "scan.txt", ["", "", ""])

    result = pipeline.ingest("scanned-circular", path)

    assert result.outcome is IngestOutcome.QUARANTINED
    assert result.document.status is DocumentStatus.QUARANTINED
    assert result.document.quarantine_reason
    assert result.document.chunk_count == 0
    assert knowledge_base.count_chunks("scanned-circular") == 0


def test_replacing_a_live_document_with_a_scan_removes_the_live_chunks(
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "policy.txt", ["The policy text is long enough to survive."])
    pipeline.ingest("policy", path)
    assert knowledge_base.count_chunks("policy") > 0

    path.write_text("", encoding="utf-8")
    result = pipeline.ingest("policy", path)

    assert result.outcome is IngestOutcome.QUARANTINED
    assert knowledge_base.count_chunks("policy") == 0


def test_quarantined_documents_are_visible_to_the_knowledge_manager(
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
) -> None:
    pipeline.ingest("scan", write_document(tmp_path, "scan.txt", [""]))

    documents = knowledge_base.list_documents()

    assert [document.key for document in documents] == ["scan"]
    assert documents[0].status is DocumentStatus.QUARANTINED


def test_deleting_a_document_removes_it_and_its_chunks(
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "retired.txt", ["A retired screening programme description."])
    pipeline.ingest("retired-programme", path)

    assert pipeline.delete("retired-programme") is True
    assert knowledge_base.get_document("retired-programme") is None
    assert knowledge_base.count_chunks("retired-programme") == 0


def test_deleting_an_unknown_document_reports_failure(pipeline: IngestionPipeline) -> None:
    assert pipeline.delete("never-existed") is False


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
    knowledge_base: KnowledgeBase,
    extractor: TextFileExtractor,
    embedder: HashingEmbedder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    relative_pipeline = IngestionPipeline(
        extractor=extractor,
        embedder=embedder,
        knowledge_base=knowledge_base,
        raw_files=LocalRawFileStore(Path("./data/raw")),
        min_chars_per_page=20,
    )
    path = write_document(tmp_path, "notice.txt", ["A notice long enough to be ingested."])

    result = relative_pipeline.ingest("notice", path)

    assert result.document.source_uri.startswith("file:///")


def test_ingesting_a_missing_file_fails_loudly(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.ingest("nope", tmp_path / "does-not-exist.txt")


def test_search_finds_the_document_that_mentions_the_query_terms(
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> None:
    pipeline.ingest(
        "vaccination-schedule",
        write_document(tmp_path, "vax.txt", ["Vaccination schedule for children under six."]),
    )
    pipeline.ingest(
        "enrolment-rules",
        write_document(tmp_path, "enrol.txt", ["Enrolment rules for postgraduate students."]),
    )

    hits = pipeline.search("enrolment rules postgraduate", limit=1)

    assert hits
    assert hits[0].document_key == "enrolment-rules"
    assert hits[0].page_number == 1


def test_the_fusion_candidate_count_bounds_what_search_can_return(
    pipeline: IngestionPipeline,
    tmp_path: Path,
) -> None:
    """`candidates` is a retrieval knob in its own right: it caps each branch before fusion."""
    for index in range(4):
        pipeline.ingest(
            f"notice-{index}",
            write_document(tmp_path, f"notice-{index}.txt", [f"Notice number {index} about fees."]),
        )

    narrow = pipeline.search("notice about fees", limit=4, candidates=1)
    wide = pipeline.search("notice about fees", limit=4, candidates=20)

    assert len(narrow) <= 2
    assert len(wide) == 4
