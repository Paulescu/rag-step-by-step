from __future__ import annotations

import io
from pathlib import Path

from customer_support_chatbot.demo.ingest import ingest_manifest
from customer_support_chatbot.demo.manifest import Manifest
from customer_support_chatbot.ingestion.models import Page
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.raw_files import LocalRawFileStore
from customer_support_chatbot.ingestion.store import KnowledgeBase
from tests.builders import write_document
from tests.fakes import HashingEmbedder
from tests.iris_fakes import page_of
from tests.test_demo_manifest import entry


def stage(pdf_root: Path, key: str, pages: int = 2) -> None:
    pdf_root.mkdir(parents=True, exist_ok=True)
    write_document(pdf_root, f"{key}.pdf", [page_of() for _ in range(pages)])


def test_every_document_in_the_manifest_reaches_the_knowledge_base(
    tmp_path: Path,
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
) -> None:
    pdf_root = tmp_path / "pdfs"
    for key in ("one", "two", "three"):
        stage(pdf_root, key)
    manifest = Manifest(source="WHO IRIS", entries=[entry(key) for key in ("one", "two", "three")])

    summary = ingest_manifest(manifest, pdf_root, pipeline, report=None)

    assert summary.ingested == 3
    assert summary.pages == 6
    assert summary.chunks > 0
    assert len(knowledge_base.list_documents()) == 3


def test_the_recorded_title_is_the_one_the_knowledge_base_stores(
    tmp_path: Path,
    pipeline: IngestionPipeline,
    knowledge_base: KnowledgeBase,
) -> None:
    pdf_root = tmp_path / "pdfs"
    stage(pdf_root, "one")

    ingest_manifest(
        Manifest(source="WHO IRIS", entries=[entry("one")]), pdf_root, pipeline, report=None
    )

    stored = knowledge_base.get_document("one")
    assert stored is not None
    assert stored.title == "Title of one"


def test_a_second_run_is_a_no_op_so_an_interrupted_ingestion_can_be_restarted(
    tmp_path: Path, pipeline: IngestionPipeline
) -> None:
    pdf_root = tmp_path / "pdfs"
    stage(pdf_root, "one")
    manifest = Manifest(source="WHO IRIS", entries=[entry("one")])
    ingest_manifest(manifest, pdf_root, pipeline, report=None)

    summary = ingest_manifest(manifest, pdf_root, pipeline, report=None)

    assert summary.unchanged == 1
    assert summary.ingested == 0


def test_a_missing_file_is_counted_rather_than_raising(
    tmp_path: Path, pipeline: IngestionPipeline
) -> None:
    pdf_root = tmp_path / "pdfs"
    stage(pdf_root, "one")
    manifest = Manifest(source="WHO IRIS", entries=[entry("one"), entry("absent")])

    report = io.StringIO()
    summary = ingest_manifest(manifest, pdf_root, pipeline, report=report)

    assert summary.ingested == 1
    assert summary.missing == 1
    assert "run `kb-demo fetch` first" in report.getvalue()


def test_one_unreadable_document_does_not_end_the_run(
    tmp_path: Path,
    knowledge_base: KnowledgeBase,
    embedder: HashingEmbedder,
) -> None:
    """A hundred Documents in, a crash on number fifty must not cost the other fifty."""
    pdf_root = tmp_path / "pdfs"
    for key in ("one", "two"):
        stage(pdf_root, key)

    class ExplodingExtractor:
        def extract(self, path: Path) -> list[Page]:
            raise ValueError(f"cannot read {path.name}")

    pipeline = IngestionPipeline(
        extractor=ExplodingExtractor(),
        embedder=embedder,
        knowledge_base=knowledge_base,
        raw_files=LocalRawFileStore(tmp_path / "raw"),
    )
    manifest = Manifest(source="WHO IRIS", entries=[entry("one"), entry("two")])

    summary = ingest_manifest(manifest, pdf_root, pipeline, report=None)

    assert summary.failed == 2
    assert summary.failures == ["one: cannot read one.pdf", "two: cannot read two.pdf"]


def test_a_quarantined_document_is_counted_separately_from_a_failure(
    tmp_path: Path, pipeline: IngestionPipeline
) -> None:
    pdf_root = tmp_path / "pdfs"
    pdf_root.mkdir(parents=True, exist_ok=True)
    write_document(pdf_root, "one.pdf", ["", ""])

    summary = ingest_manifest(
        Manifest(source="WHO IRIS", entries=[entry("one")]), pdf_root, pipeline, report=None
    )

    assert summary.quarantined == 1
    assert summary.failed == 0


def test_the_limit_ingests_a_prefix_of_the_manifest(
    tmp_path: Path, pipeline: IngestionPipeline
) -> None:
    """So that a chunking setting can be tried on five Documents before committing to a hundred."""
    pdf_root = tmp_path / "pdfs"
    for key in ("one", "two", "three"):
        stage(pdf_root, key)
    manifest = Manifest(source="WHO IRIS", entries=[entry(key) for key in ("one", "two", "three")])

    summary = ingest_manifest(manifest, pdf_root, pipeline, limit=2, report=None)

    assert summary.attempted == 2
    assert summary.ingested == 2
