"""Searching the knowledge base through the pipeline's retrieval path."""

from __future__ import annotations

from pathlib import Path

from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from tests.builders import write_document


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
