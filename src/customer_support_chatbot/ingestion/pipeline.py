"""The ingestion pipeline itself: file in, Knowledge Base updated.

Ingestion is a function of a Document Key and a file. Re-running it with the same file is a no-op,
re-running it with a different file replaces the Document's Chunks (ADR-0004).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from customer_support_chatbot.ingestion.chunking import (
    ChunkingSettings,
    TokenCounter,
    approximate_token_count,
    chunk_pages,
)
from customer_support_chatbot.ingestion.embedding import Embedder
from customer_support_chatbot.ingestion.extraction import (
    DEFAULT_MIN_CHARS_PER_PAGE,
    PageExtractor,
    quarantine_reason,
)
from customer_support_chatbot.ingestion.models import (
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
    IngestOutcome,
    IngestResult,
)
from customer_support_chatbot.ingestion.raw_files import RawFileStore
from customer_support_chatbot.ingestion.store import KnowledgeBase


def file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        *,
        extractor: PageExtractor,
        embedder: Embedder,
        knowledge_base: KnowledgeBase,
        raw_files: RawFileStore,
        chunking: ChunkingSettings | None = None,
        count_tokens: TokenCounter = approximate_token_count,
        min_chars_per_page: float = DEFAULT_MIN_CHARS_PER_PAGE,
    ) -> None:
        self._extractor = extractor
        self._embedder = embedder
        self._knowledge_base = knowledge_base
        self._raw_files = raw_files
        self._chunking = chunking or ChunkingSettings()
        self._count_tokens = count_tokens
        self._min_chars_per_page = min_chars_per_page

    def ingest(
        self,
        document_key: str,
        path: Path,
        *,
        title: str | None = None,
        force: bool = False,
    ) -> IngestResult:
        if not path.is_file():
            raise FileNotFoundError(path)

        content_hash = file_content_hash(path)
        existing = self._knowledge_base.get_document(document_key)
        if existing is not None and existing.content_hash == content_hash and not force:
            return IngestResult(outcome=IngestOutcome.UNCHANGED, document=existing)

        version = existing.ingestion_version + 1 if existing is not None else 1
        # Written before anything is indexed, so the raw file always outlives a failed ingestion.
        source_uri = self._raw_files.store(document_key, version, path)

        pages = self._extractor.extract(path)
        reason = quarantine_reason(pages, self._min_chars_per_page)
        if reason is not None:
            return self._quarantine(
                document_key=document_key,
                title=title or path.stem,
                content_hash=content_hash,
                page_count=len(pages),
                version=version,
                source_uri=source_uri,
                reason=reason,
            )

        chunks = chunk_pages(pages, document_key, self._chunking, self._count_tokens)
        embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        self._knowledge_base.replace_chunks(document_key, version, embedded_chunks)
        record = DocumentRecord(
            key=document_key,
            title=title or path.stem,
            status=DocumentStatus.LIVE,
            content_hash=content_hash,
            page_count=len(pages),
            chunk_count=len(chunks),
            ingestion_version=version,
            ingested_at=datetime.now(UTC),
            source_uri=source_uri,
        )
        self._knowledge_base.upsert_document(record)
        return IngestResult(outcome=IngestOutcome.INGESTED, document=record)

    def _quarantine(
        self,
        *,
        document_key: str,
        title: str,
        content_hash: str,
        page_count: int,
        version: int,
        source_uri: str,
        reason: str,
    ) -> IngestResult:
        """Record an unusable Document and make sure no stale Chunks survive it."""
        self._knowledge_base.replace_chunks(document_key, version, [])
        record = DocumentRecord(
            key=document_key,
            title=title,
            status=DocumentStatus.QUARANTINED,
            content_hash=content_hash,
            page_count=page_count,
            chunk_count=0,
            ingestion_version=version,
            ingested_at=datetime.now(UTC),
            source_uri=source_uri,
            quarantine_reason=reason,
        )
        self._knowledge_base.upsert_document(record)
        return IngestResult(outcome=IngestOutcome.QUARANTINED, document=record)
