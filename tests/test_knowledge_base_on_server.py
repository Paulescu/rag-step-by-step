"""Checks that only a real Qdrant server can answer.

Local (in-memory) Qdrant ignores payload indexes and does not exercise the server's filtering or
fusion paths, so these run only when QDRANT_URL points at a real instance:

    docker run -p 6333:6333 qdrant/qdrant
    QDRANT_URL=http://localhost:6333 uv run pytest tests/test_knowledge_base_on_server.py
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from qdrant_client import QdrantClient

from customer_support_chatbot.ingestion.models import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
)
from customer_support_chatbot.ingestion.store import KnowledgeBase
from tests.fakes import DENSE_SIZE, HashingEmbedder

QDRANT_URL = os.environ.get("QDRANT_URL")

pytestmark = pytest.mark.skipif(
    QDRANT_URL is None,
    reason="set QDRANT_URL to run against a real Qdrant server",
)


@pytest.fixture
def server_knowledge_base() -> Iterator[KnowledgeBase]:
    client = QdrantClient(url=QDRANT_URL)
    suffix = uuid.uuid4().hex[:8]
    chunks_collection = f"test_chunks_{suffix}"
    documents_collection = f"test_documents_{suffix}"
    base = KnowledgeBase(
        client,
        dense_size=DENSE_SIZE,
        embedder=HashingEmbedder(),
        chunks_collection=chunks_collection,
        documents_collection=documents_collection,
    )
    base.ensure_collections()
    try:
        yield base
    finally:
        client.delete_collection(chunks_collection)
        client.delete_collection(documents_collection)
        client.close()


def make_record(key: str, version: int) -> DocumentRecord:
    return DocumentRecord(
        key=key,
        title=key,
        status=DocumentStatus.LIVE,
        content_hash="hash",
        page_count=1,
        chunk_count=1,
        ingestion_version=version,
        ingested_at=datetime.now(UTC),
        source_uri="file:///tmp/example.pdf",
    )


def embedded(key: str, index: int, text: str) -> EmbeddedChunk:
    embedding = HashingEmbedder().embed([text])[0]
    return EmbeddedChunk(
        chunk=Chunk(document_key=key, page_number=1, index=index, text=text),
        embedding=embedding,
    )


def test_a_payload_only_documents_collection_round_trips(
    server_knowledge_base: KnowledgeBase,
) -> None:
    record = make_record("vaccination-schedule", 1)

    server_knowledge_base.upsert_document(record)

    stored = server_knowledge_base.get_document("vaccination-schedule")
    assert stored is not None
    assert stored.key == record.key
    assert stored.status is DocumentStatus.LIVE


def test_replacing_chunks_deletes_only_older_versions(
    server_knowledge_base: KnowledgeBase,
) -> None:
    server_knowledge_base.replace_chunks("doc", 1, [embedded("doc", 0, "first version text")])
    server_knowledge_base.replace_chunks("other", 1, [embedded("other", 0, "untouched text")])

    server_knowledge_base.replace_chunks("doc", 2, [embedded("doc", 0, "second version text")])

    assert server_knowledge_base.list_chunk_texts("doc") == ["second version text"]
    assert server_knowledge_base.list_chunk_texts("other") == ["untouched text"]


def test_hybrid_search_fuses_both_branches(server_knowledge_base: KnowledgeBase) -> None:
    server_knowledge_base.replace_chunks(
        "vaccination",
        1,
        [embedded("vaccination", 0, "Vaccination schedule for children under six")],
    )
    server_knowledge_base.replace_chunks(
        "enrolment",
        1,
        [embedded("enrolment", 0, "Enrolment rules for postgraduate students")],
    )

    hits = server_knowledge_base.search("enrolment rules postgraduate", limit=2)

    assert hits
    assert hits[0].document_key == "enrolment"
