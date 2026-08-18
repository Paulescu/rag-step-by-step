"""`kb-demo`: build the demo Knowledge Base from WHO IRIS.

Two commands, deliberately separate. `fetch` talks to IRIS and touches no Qdrant; `ingest` talks
to Qdrant and touches no network beyond the model. Fetching a hundred PDFs takes minutes and
embedding them takes hours, and nobody should have to redo the first to retry the second.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from customer_support_chatbot.demo.fetch import (
    DEFAULT_TOPICS,
    FetchSettings,
    LibraryFetcher,
)
from customer_support_chatbot.demo.ingest import IngestSummary, ingest_manifest
from customer_support_chatbot.demo.manifest import Manifest, read_manifest, write_manifest
from customer_support_chatbot.demo.who_iris import (
    DEFAULT_REQUEST_DELAY_SECONDS,
    IRIS_BASE_URL,
    HttpxClient,
    IrisRepository,
)
from customer_support_chatbot.ingestion.cli import (
    add_chunking_arguments,
    add_collection_arguments,
    add_connection_arguments,
    add_embedding_arguments,
    build_embedder,
    build_knowledge_base,
    build_pipeline,
)

DEFAULT_MANIFEST = Path("./demo/manifest.json")
DEFAULT_PDF_ROOT = Path("./data/demo")


def add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    """Where the record lives, and where the files it records live.

    The manifest is committed and the PDFs are not, so these two are separate paths rather than one
    directory holding both.
    """
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the manifest describing the demo Knowledge Base.",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=DEFAULT_PDF_ROOT,
        help="Directory the downloaded PDFs live in.",
    )


def add_fetch_arguments(fetch: argparse.ArgumentParser) -> None:
    add_manifest_arguments(fetch)
    defaults = FetchSettings()
    fetch.add_argument(
        "--documents",
        type=int,
        default=defaults.target_documents,
        help="How many Documents to collect.",
    )
    fetch.add_argument(
        "--pages",
        type=int,
        default=defaults.target_pages,
        help="How many pages to collect in total. Both targets have to be met.",
    )
    fetch.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=None,
        help="Search topic, repeatable. Defaults to a spread of public health subjects.",
    )
    fetch.add_argument(
        "--min-pdf-pages",
        type=int,
        default=defaults.min_pdf_pages,
        help="Skip publications shorter than this.",
    )
    fetch.add_argument(
        "--max-pdf-pages",
        type=int,
        default=defaults.max_pdf_pages,
        help="Skip publications longer than this, which are slow to embed.",
    )
    fetch.add_argument(
        "--max-megabytes",
        type=float,
        default=defaults.max_bytes / 1e6,
        help="Skip publications whose PDF is larger than this.",
    )
    fetch.add_argument(
        "--issued-from",
        type=int,
        default=defaults.issued_from,
        help="Earliest publication year. Later years are more likely to be born-digital.",
    )
    fetch.add_argument(
        "--issued-to",
        type=int,
        default=defaults.issued_to,
        help="Latest publication year.",
    )
    fetch.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY_SECONDS,
        help="Seconds to wait between calls to IRIS.",
    )
    fetch.add_argument(
        "--base-url",
        default=IRIS_BASE_URL,
        help="Root of the IRIS REST API.",
    )
    fetch.add_argument(
        "--from-manifest",
        action="store_true",
        help="Re-download exactly what the manifest already lists instead of searching IRIS.",
    )


def add_demo_ingest_arguments(ingest: argparse.ArgumentParser) -> None:
    """Qdrant and embedding flags sit here rather than at the top level, because `fetch` needs
    neither a running Qdrant nor a downloaded model.
    """
    add_manifest_arguments(ingest)
    add_connection_arguments(ingest)
    add_collection_arguments(ingest)
    add_embedding_arguments(ingest)
    add_chunking_arguments(ingest)
    ingest.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N Documents of the manifest.",
    )
    ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest Documents whose file has not changed.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kb-demo",
        description="Build a demo Knowledge Base from WHO IRIS publications.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_fetch_arguments(
        subparsers.add_parser("fetch", help="Download health publications and write a manifest.")
    )
    add_demo_ingest_arguments(
        subparsers.add_parser("ingest", help="Ingest every Document in the manifest.")
    )
    add_manifest_arguments(
        subparsers.add_parser("show", help="Summarise the manifest without touching the network.")
    )
    return parser


def build_fetch_settings(args: argparse.Namespace) -> FetchSettings:
    return FetchSettings(
        target_documents=int(args.documents),
        target_pages=int(args.pages),
        min_pdf_pages=int(args.min_pdf_pages),
        max_pdf_pages=int(args.max_pdf_pages),
        max_bytes=int(float(args.max_megabytes) * 1e6),
        issued_from=args.issued_from,
        issued_to=args.issued_to,
    )


def describe(manifest: Manifest) -> str:
    """The two numbers the ticket asked for, plus enough detail to see the spread."""
    if not manifest.entries:
        return "The manifest is empty."

    by_topic = Counter(entry.topic for entry in manifest.entries)
    lines = [
        f"{manifest.document_count} Documents, {manifest.page_count} pages, "
        f"{manifest.size_bytes / 1e6:.0f} MB",
        f"Source: {manifest.source}",
        "Topics:",
    ]
    lines.extend(f"  {count:>3}  {topic}" for topic, count in by_topic.most_common())
    return "\n".join(lines)


def run_fetch(args: argparse.Namespace) -> int:
    settings = build_fetch_settings(args)
    http = HttpxClient(delay_seconds=float(args.delay))
    try:
        repository = IrisRepository(http, base_url=str(args.base_url))
        fetcher = LibraryFetcher(
            repository,
            Path(args.pdf_root),
            settings=settings,
            report=sys.stdout,
        )
        if bool(args.from_manifest):
            manifest = fetcher.redownload(read_manifest(Path(args.manifest)))
        else:
            topics = tuple(args.topics) if args.topics else DEFAULT_TOPICS
            manifest = fetcher.fetch(topics)
    finally:
        http.close()

    write_manifest(Path(args.manifest), manifest)
    print()
    print(describe(manifest))
    print(f"Manifest written to {args.manifest}")

    if manifest.document_count < settings.target_documents:
        print(
            f"Only {manifest.document_count} of {settings.target_documents} Documents were found. "
            "Add more --topic values, or lower --issued-from.",
            file=sys.stderr,
        )
        return 1
    return 0


def run_ingest(args: argparse.Namespace) -> int:
    manifest = read_manifest(Path(args.manifest))
    embedder = build_embedder(args)
    knowledge_base = build_knowledge_base(args, embedder)
    pipeline = build_pipeline(args, knowledge_base, embedder)

    summary = ingest_manifest(
        manifest,
        Path(args.pdf_root),
        pipeline,
        limit=args.limit,
        force=bool(args.force),
    )
    print()
    print(format_summary(summary))
    return 0 if summary.failed == 0 and summary.missing == 0 else 1


def format_summary(summary: IngestSummary) -> str:
    lines = [
        f"{summary.attempted} Documents attempted: {summary.ingested} ingested, "
        f"{summary.unchanged} unchanged, {summary.quarantined} quarantined, "
        f"{summary.missing} missing, {summary.failed} failed.",
        f"{summary.pages} pages and {summary.chunks} Chunks added to the Knowledge Base.",
    ]
    lines.extend(f"  failed: {failure}" for failure in summary.failures)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fetch":
        return run_fetch(args)
    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "show":
        print(describe(read_manifest(Path(args.manifest))))
        return 0

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
