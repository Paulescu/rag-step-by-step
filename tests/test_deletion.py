"""Removing Documents from the Knowledge Base."""

from __future__ import annotations

from pathlib import Path

from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.store import KnowledgeBase
from tests.builders import write_document


def test_deleting_a_document_removes_it_and_its_chunks(
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
) -> None:
    path = write_document(tmp_path, "retired.txt", ["A retired screening programme description."])
    pipeline.ingest("retired-programme", path)

    assert knowledge_base.delete_document("retired-programme") is True
    assert knowledge_base.get_document("retired-programme") is None
    assert knowledge_base.count_chunks("retired-programme") == 0


def test_deleting_an_unknown_document_reports_failure(knowledge_base: KnowledgeBase) -> None:
    assert knowledge_base.delete_document("never-existed") is False
