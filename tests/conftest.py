from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from customer_support_chatbot.ingestion.chunking import ChunkingSettings
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.raw_files import LocalRawFileStore
from customer_support_chatbot.ingestion.store import KnowledgeBase
from tests.fakes import DENSE_SIZE, HashingEmbedder, TextFileExtractor


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def extractor() -> TextFileExtractor:
    return TextFileExtractor()


@pytest.fixture
def knowledge_base(embedder: HashingEmbedder) -> Iterator[KnowledgeBase]:
    client = QdrantClient(":memory:")
    base = KnowledgeBase(client, dense_size=DENSE_SIZE, embedder=embedder)
    base.ensure_collections()
    yield base
    client.close()


@pytest.fixture
def make_pipeline(
    knowledge_base: KnowledgeBase,
    extractor: TextFileExtractor,
    embedder: HashingEmbedder,
    tmp_path: Path,
) -> Callable[..., IngestionPipeline]:
    """Builds a pipeline wired to the in-memory knowledge base, with the raw file root swappable."""

    def build(raw_files_root: Path | None = None) -> IngestionPipeline:
        return IngestionPipeline(
            extractor=extractor,
            embedder=embedder,
            knowledge_base=knowledge_base,
            raw_files=LocalRawFileStore(raw_files_root or tmp_path / "raw"),
            chunking=ChunkingSettings(max_tokens=60, overlap_tokens=10),
            # Fixtures are single sentences, so the scan threshold is lowered rather than padding
            # every document. The threshold itself is covered in test_extraction.py.
            min_chars_per_page=20,
        )

    return build


@pytest.fixture
def pipeline(make_pipeline: Callable[..., IngestionPipeline]) -> IngestionPipeline:
    return make_pipeline()
