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
from customer_support_chatbot.ingestion.embedding import BgeM3Embedder
from customer_support_chatbot.ingestion.extraction import (
    DEFAULT_MIN_CHARS_PER_PAGE,
    PdfPageExtractor,
)
from customer_support_chatbot.ingestion.models import DocumentStatus, IngestOutcome
from customer_support_chatbot.ingestion.pipeline import IngestionPipeline
from customer_support_chatbot.ingestion.raw_files import LocalRawFileStore
from customer_support_chatbot.ingestion.store import KnowledgeBase

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_RAW_FILES_ROOT = "./data/raw"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb",
        description="Ingest PDFs into the Knowledge Base and search it.",
    )
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest or re-ingest a PDF.")
    ingest.add_argument("--key", required=True, help="Stable Document Key.")
    ingest.add_argument("path", type=Path, help="Path to the PDF.")
    ingest.add_argument("--title", default=None, help="Human-readable title.")
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest even if the file is unchanged.",
    )
    ingest.add_argument("--max-tokens", type=int, default=ChunkingSettings().max_tokens)
    ingest.add_argument("--overlap-tokens", type=int, default=ChunkingSettings().overlap_tokens)
    ingest.add_argument(
        "--min-chars-per-page",
        type=float,
        default=DEFAULT_MIN_CHARS_PER_PAGE,
        help="Below this, the Document is quarantined as a probable scan.",
    )

    delete = subparsers.add_parser("delete", help="Hard-delete a Document and its Chunks.")
    delete.add_argument("--key", required=True, help="Document Key to remove.")

    search = subparsers.add_parser("search", help="Hybrid search over the Knowledge Base.")
    search.add_argument("query", help="The question to search for.")
    search.add_argument("--limit", type=int, default=5)

    subparsers.add_parser("list", help="List Documents in the Knowledge Base.")

    return parser


def _knowledge_base(args: argparse.Namespace) -> KnowledgeBase:
    client = QdrantClient(url=str(args.qdrant_url))
    knowledge_base = KnowledgeBase(client, dense_size=1024)
    knowledge_base.ensure_collections()
    return knowledge_base


def _pipeline(args: argparse.Namespace, knowledge_base: KnowledgeBase) -> IngestionPipeline:
    return IngestionPipeline(
        extractor=PdfPageExtractor(),
        embedder=BgeM3Embedder(),
        knowledge_base=knowledge_base,
        raw_files=LocalRawFileStore(Path(args.raw_files_root)),
        chunking=ChunkingSettings(
            max_tokens=getattr(args, "max_tokens", ChunkingSettings().max_tokens),
            overlap_tokens=getattr(args, "overlap_tokens", ChunkingSettings().overlap_tokens),
        ),
        min_chars_per_page=getattr(args, "min_chars_per_page", DEFAULT_MIN_CHARS_PER_PAGE),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    knowledge_base = _knowledge_base(args)

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

    pipeline = _pipeline(args, knowledge_base)

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
        hits = pipeline.search(str(args.query), limit=int(args.limit))
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
