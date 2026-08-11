"""Command line entry point for the ingestion pipeline.

The pipeline is a library; this is a thin shell over it so that we can drive ingestion by hand
while retrieval quality is being proved. `search` exists so the pipeline can be evaluated on its
own, without waiting for the chatbot to be built.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from customer_support_chatbot.ingestion.chunking import ChunkingSettings
from customer_support_chatbot.ingestion.embedding import (
    BGE_M3_DENSE_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MAX_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
    BgeM3Embedder,
)
from customer_support_chatbot.ingestion.extraction import (
    DEFAULT_MIN_CHARS_PER_PAGE,
    PdfPageExtractor,
)
from customer_support_chatbot.ingestion.models import DocumentStatus, IngestOutcome
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.raw_files import LocalRawFileStore
from customer_support_chatbot.ingestion.store import (
    DEFAULT_CHUNKS_COLLECTION,
    DEFAULT_DOCUMENTS_COLLECTION,
    DEFAULT_SEARCH_CANDIDATES,
    KnowledgeBase,
)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_RAW_FILES_ROOT = "./data/raw"
DEFAULT_SEARCH_RESULTS = 5


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    """Where the two stores live: the vectors in Qdrant, the uploaded files on disk."""
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL),
        help="URL of the Qdrant server.",
    )
    parser.add_argument(
        "--raw-files-root",
        type=Path,
        default=Path(os.environ.get("RAW_FILES_ROOT", DEFAULT_RAW_FILES_ROOT)),
        help="Directory where uploaded files are kept.",
    )


def add_collection_arguments(parser: argparse.ArgumentParser) -> None:
    """Collection names are global rather than per-command: ingesting into one pair of collections
    and searching another would silently return nothing. Overriding both is how two chunking
    configurations can be indexed side by side and compared.
    """
    parser.add_argument(
        "--chunks-collection",
        default=os.environ.get("CHUNKS_COLLECTION", DEFAULT_CHUNKS_COLLECTION),
        help="Qdrant collection holding Chunks.",
    )
    parser.add_argument(
        "--documents-collection",
        default=os.environ.get("DOCUMENTS_COLLECTION", DEFAULT_DOCUMENTS_COLLECTION),
        help="Qdrant collection holding Documents.",
    )


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    """Global for the same reason as the collections: a query has to be embedded by the model
    that embedded the Chunks.
    """
    embedding = parser.add_argument_group("embedding")
    embedding.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Name of the embedding model to load.",
    )
    embedding.add_argument(
        "--embedding-dense-size",
        type=int,
        default=BGE_M3_DENSE_SIZE,
        help="Width of the dense vector the model produces. Must match the collection.",
    )
    embedding.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Texts embedded per forward pass.",
    )
    embedding.add_argument(
        "--embedding-max-length",
        type=int,
        default=DEFAULT_EMBEDDING_MAX_LENGTH,
        help="Token budget per text. Longer texts are truncated by the model.",
    )
    embedding.add_argument(
        "--embedding-fp16",
        action="store_true",
        help="Load the model in half precision.",
    )


def add_ingest_arguments(ingest: argparse.ArgumentParser) -> None:
    ingest.add_argument("--key", required=True, help="Stable Document Key.")
    ingest.add_argument("path", type=Path, help="Path to the PDF.")
    ingest.add_argument("--title", default=None, help="Human-readable title.")
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the file is unchanged.",
    )
    ingest.add_argument(
        "--max-tokens",
        type=int,
        default=ChunkingSettings().max_tokens,
        help="Chunk size: the token budget a single Chunk is packed up to.",
    )
    ingest.add_argument(
        "--overlap-tokens",
        type=int,
        default=ChunkingSettings().overlap_tokens,
        help="Tokens carried from the end of one Chunk into the next.",
    )
    ingest.add_argument(
        "--min-chars-per-page",
        type=float,
        default=DEFAULT_MIN_CHARS_PER_PAGE,
        help="Below this, the Document is quarantined as a probable scan.",
    )


def add_delete_arguments(delete: argparse.ArgumentParser) -> None:
    delete.add_argument("--key", required=True, help="Document Key to remove.")


def add_search_arguments(search: argparse.ArgumentParser) -> None:
    search.add_argument("query", help="The question to search for.")
    search.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_RESULTS,
        help="How many Chunks to return.",
    )
    search.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_SEARCH_CANDIDATES,
        help="Chunks each of the dense and sparse branches contributes before fusion.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb",
        description="Ingest PDFs into the Knowledge Base and search it.",
    )
    add_connection_arguments(parser)
    add_collection_arguments(parser)
    add_embedding_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    add_ingest_arguments(subparsers.add_parser("ingest", help="Ingest or re-ingest a PDF."))
    add_delete_arguments(
        subparsers.add_parser("delete", help="Hard-delete a Document and its Chunks.")
    )
    add_search_arguments(
        subparsers.add_parser("search", help="Hybrid search over the Knowledge Base.")
    )
    subparsers.add_parser("list", help="List Documents in the Knowledge Base.")

    return parser


def build_knowledge_base(args: argparse.Namespace) -> KnowledgeBase:
    client = QdrantClient(url=str(args.qdrant_url))
    knowledge_base = KnowledgeBase(
        client,
        dense_size=int(args.embedding_dense_size),
        chunks_collection=str(args.chunks_collection),
        documents_collection=str(args.documents_collection),
    )
    knowledge_base.ensure_collections()
    return knowledge_base


def build_embedder(args: argparse.Namespace) -> BgeM3Embedder:
    return BgeM3Embedder(
        str(args.embedding_model),
        use_fp16=bool(args.embedding_fp16),
        batch_size=int(args.embedding_batch_size),
        max_length=int(args.embedding_max_length),
        dense_size=int(args.embedding_dense_size),
    )


def build_chunking_settings(args: argparse.Namespace) -> ChunkingSettings:
    """Chunking flags only exist on `ingest`, so the defaults stand in for the other commands."""
    defaults = ChunkingSettings()
    return ChunkingSettings(
        max_tokens=int(getattr(args, "max_tokens", defaults.max_tokens)),
        overlap_tokens=int(getattr(args, "overlap_tokens", defaults.overlap_tokens)),
    )


def build_pipeline(args: argparse.Namespace, knowledge_base: KnowledgeBase) -> IngestionPipeline:
    return IngestionPipeline(
        extractor=PdfPageExtractor(),
        embedder=build_embedder(args),
        knowledge_base=knowledge_base,
        raw_files=LocalRawFileStore(Path(args.raw_files_root)),
        chunking=build_chunking_settings(args),
        min_chars_per_page=float(getattr(args, "min_chars_per_page", DEFAULT_MIN_CHARS_PER_PAGE)),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    knowledge_base = build_knowledge_base(args)

    if args.command == "list":
        documents = knowledge_base.list_documents()
        if not documents:
            print("Knowledge Base is empty.")
            return 0
        for document in documents:
            marker = "!" if document.status is DocumentStatus.QUARANTINED else " "
            print(
                f"{marker} {document.key:<40} v{document.ingestion_version:<4} "
                f"{document.page_count:>4}p {document.chunk_count:>5} chunks  {document.title}"
            )
            if document.quarantine_reason:
                print(f"    quarantined: {document.quarantine_reason}")
        return 0

    if args.command == "delete":
        if knowledge_base.delete_document(str(args.key)):
            print(f"Deleted {args.key}.")
            return 0
        print(f"No Document with key {args.key}.", file=sys.stderr)
        return 1

    pipeline = build_pipeline(args, knowledge_base)

    if args.command == "ingest":
        result = pipeline.ingest(
            str(args.key),
            Path(args.path),
            title=args.title,
            force=bool(args.force),
        )
        document = result.document
        if result.outcome is IngestOutcome.UNCHANGED:
            print(f"{document.key}: unchanged, nothing to do.")
            return 0
        if result.outcome is IngestOutcome.QUARANTINED:
            print(
                f"{document.key}: QUARANTINED, not searchable.\n  {document.quarantine_reason}",
                file=sys.stderr,
            )
            return 2
        print(
            f"{document.key}: ingested v{document.ingestion_version}, "
            f"{document.page_count} pages, {document.chunk_count} chunks."
        )
        return 0

    if args.command == "search":
        hits = pipeline.search(
            str(args.query),
            limit=int(args.limit),
            candidates=int(args.candidates),
        )
        if not hits:
            print("No results.")
            return 0
        for rank, hit in enumerate(hits, start=1):
            snippet = hit.text[:280].replace("\n", " ")
            print(f"{rank}. {hit.document_key} p.{hit.page_number}  (score {hit.score:.4f})")
            print(f"   {snippet}")
        return 0

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
