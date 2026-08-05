"""Core types of the ingestion pipeline.

The vocabulary here follows CONTEXT.md: a Document is a piece of institutional documentation
identified by a Document Key, and a Chunk is the unit of retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Page:
    """One page of extracted text, numbered from 1 as a reader would count it."""

    number: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A bounded passage of a single page, small enough to embed."""

    document_key: str
    page_number: int
    index: int
    text: str


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A sparse lexical vector: token ids and their learned weights."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True, slots=True)
class Embedding:
    """The dense and sparse representations of one piece of text."""

    dense: list[float]
    sparse: SparseVector


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: Embedding


class DocumentStatus(StrEnum):
    LIVE = "live"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """What the Knowledge Base knows about a Document, independently of its Chunks."""

    key: str
    title: str
    status: DocumentStatus
    content_hash: str
    page_count: int
    chunk_count: int
    ingestion_version: int
    ingested_at: datetime
    source_uri: str
    quarantine_reason: str | None = None


class IngestOutcome(StrEnum):
    INGESTED = "ingested"
    UNCHANGED = "unchanged"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class IngestResult:
    outcome: IngestOutcome
    document: DocumentRecord


@dataclass(frozen=True, slots=True)
class SearchHit:
    document_key: str
    page_number: int
    text: str
    score: float
