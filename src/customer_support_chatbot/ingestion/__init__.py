"""Document ingestion: PDFs in, a searchable Knowledge Base out."""

from customer_support_chatbot.ingestion.chunking import ChunkingSettings, chunk_pages
from customer_support_chatbot.ingestion.embedding import BgeM3Embedder, Embedder
from customer_support_chatbot.ingestion.extraction import (
    PageExtractor,
    PdfPageExtractor,
    quarantine_reason,
)
from customer_support_chatbot.ingestion.models import (
    Chunk,
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
    Embedding,
    IngestOutcome,
    IngestResult,
    Page,
    SearchHit,
    SparseVector,
)
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.raw_files import LocalRawFileStore, RawFileStore
from customer_support_chatbot.ingestion.store import KnowledgeBase

__all__ = [
    "BgeM3Embedder",
    "Chunk",
    "ChunkingSettings",
    "DocumentRecord",
    "DocumentStatus",
    "EmbeddedChunk",
    "Embedder",
    "Embedding",
    "IngestOutcome",
    "IngestResult",
    "IngestionPipeline",
    "KnowledgeBase",
    "LocalRawFileStore",
    "Page",
    "PageExtractor",
    "PdfPageExtractor",
    "RawFileStore",
    "SearchHit",
    "SparseVector",
    "chunk_pages",
    "quarantine_reason",
]
