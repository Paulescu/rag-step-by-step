from __future__ import annotations

import argparse
from pathlib import Path

from customer_support_chatbot.demo import cli
from customer_support_chatbot.demo.fetch import FetchSettings
from customer_support_chatbot.demo.ingest import IngestSummary
from customer_support_chatbot.demo.manifest import Manifest
from customer_support_chatbot.ingestion.chunking import ChunkingSettings
from customer_support_chatbot.ingestion.embedding import DEFAULT_EMBEDDING_MODEL
from customer_support_chatbot.ingestion.store import DEFAULT_CHUNKS_COLLECTION
from tests.test_demo_manifest import entry


def parse(argv: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(argv)


def test_fetch_defaults_to_the_targets_the_ticket_asked_for() -> None:
    args = parse(["fetch"])

    assert args.documents == 100
    assert args.pages == 1000
    assert cli.build_fetch_settings(args) == FetchSettings()


def test_the_manifest_is_tracked_and_the_pdfs_are_not() -> None:
    """`data/` is gitignored, `demo/` is not, which is why these are two separate paths."""
    args = parse(["fetch"])

    assert args.manifest == Path("./demo/manifest.json")
    assert args.pdf_root == Path("./data/demo")


def test_the_targets_and_screens_are_overridable() -> None:
    args = parse(
        [
            "fetch",
            "--documents",
            "20",
            "--pages",
            "200",
            "--min-pdf-pages",
            "2",
            "--max-pdf-pages",
            "40",
            "--max-megabytes",
            "5",
            "--issued-from",
            "2020",
            "--issued-to",
            "2024",
        ]
    )

    assert cli.build_fetch_settings(args) == FetchSettings(
        target_documents=20,
        target_pages=200,
        min_pdf_pages=2,
        max_pdf_pages=40,
        max_bytes=5_000_000,
        issued_from=2020,
        issued_to=2024,
    )


def test_topics_accumulate_so_the_default_spread_can_be_replaced() -> None:
    args = parse(["fetch", "--topic", "malaria", "--topic", "cholera"])

    assert args.topics == ["malaria", "cholera"]
    assert parse(["fetch"]).topics is None


def test_fetch_needs_neither_qdrant_nor_a_model() -> None:
    args = parse(["fetch"])

    assert not hasattr(args, "qdrant_url")
    assert not hasattr(args, "embedding_model")


def test_demo_ingest_carries_the_same_knobs_as_kb_ingest() -> None:
    args = parse(["ingest", "--max-tokens", "250", "--overlap-tokens", "25"])

    assert args.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert args.chunks_collection == DEFAULT_CHUNKS_COLLECTION
    assert args.qdrant_url != ""
    assert args.max_tokens == 250
    assert args.overlap_tokens == 25
    assert args.limit is None
    assert args.force is False


def test_demo_ingest_defaults_to_the_shared_chunking_settings() -> None:
    from customer_support_chatbot.ingestion.cli import build_chunking_settings

    assert build_chunking_settings(parse(["ingest"])) == ChunkingSettings()


def test_describing_a_manifest_reports_the_two_numbers_and_the_spread() -> None:
    manifest = Manifest(source="WHO IRIS", entries=[entry("one", pages=12), entry("two", pages=30)])

    description = cli.describe(manifest)

    assert "2 Documents, 42 pages" in description
    assert "WHO IRIS" in description
    assert "2  malaria" in description


def test_describing_an_empty_manifest_says_so() -> None:
    assert cli.describe(Manifest(source="WHO IRIS", entries=[])) == "The manifest is empty."


def test_the_ingest_summary_names_every_document_that_failed() -> None:
    summary = IngestSummary(
        ingested=2,
        unchanged=1,
        quarantined=1,
        missing=1,
        failed=1,
        pages=40,
        chunks=90,
        failures=["who-iris-7: cannot read who-iris-7.pdf"],
    )

    text = cli.format_summary(summary)

    assert "6 Documents attempted" in text
    assert "40 pages and 90 Chunks" in text
    assert "failed: who-iris-7: cannot read who-iris-7.pdf" in text
