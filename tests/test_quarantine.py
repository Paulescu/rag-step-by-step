"""What the pipeline does with documents that carry no usable text layer.

The threshold itself is covered in test_extraction.py; these cover the consequences for the
knowledge base.
"""

from __future__ import annotations

from pathlib import Path

from customer_support_chatbot.ingestion.models import DocumentStatus, IngestOutcome
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.store import KnowledgeBase
from tests.builders import write_document


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
