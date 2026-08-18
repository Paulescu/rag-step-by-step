"""Choosing which IRIS publications become the demo Knowledge Base, and fetching them.

Selection sweeps a list of public health topics rather than taking the top 100 hits for one query,
because a Knowledge Base where every Document says the same thing cannot show whether retrieval
picks the right one.

Every candidate is put through the pipeline's own quarantine gate before it is accepted. A scan
that would be quarantined on ingestion is not interesting demo material; it is the same failure a
hundred times over. Quarantine is still worth showing, so it gets its own Document in Lesson 1
rather than a share of this collection.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

from customer_support_chatbot.demo.manifest import Manifest, ManifestEntry
from customer_support_chatbot.demo.who_iris import IrisRepository, Publication
from customer_support_chatbot.ingestion.extraction import (
    DEFAULT_MIN_CHARS_PER_PAGE,
    PageExtractor,
    PdfPageExtractor,
    quarantine_reason,
)

# Broad enough that no single topic dominates, and all of it is the kind of guidance a public
# health body publishes for the public rather than for researchers.
DEFAULT_TOPICS: tuple[str, ...] = (
    "immunization and vaccines",
    "maternal and newborn health",
    "child and adolescent health",
    "tuberculosis",
    "malaria",
    "HIV and sexually transmitted infections",
    "nutrition and food safety",
    "mental health",
    "noncommunicable diseases",
    "tobacco and alcohol",
    "water sanitation and hygiene",
    "antimicrobial resistance",
    "health emergencies and outbreaks",
    "health systems and universal health coverage",
)

DOCUMENT_KEY_PREFIX = "who-iris"

SOURCE_DESCRIPTION = "WHO IRIS (https://iris.who.int), English publications, CC BY-NC-SA 3.0 IGO"


@dataclass(frozen=True, slots=True)
class FetchSettings:
    """The targets the ticket asked for, and the screens a candidate has to pass to count."""

    target_documents: int = 100
    target_pages: int = 1000
    # A one-page flyer teaches nothing about chunking; a 400-page compendium costs an hour of
    # embedding on CPU for one Document.
    min_pdf_pages: int = 4
    max_pdf_pages: int = 250
    max_bytes: int = 30 * 1024 * 1024
    min_chars_per_page: float = DEFAULT_MIN_CHARS_PER_PAGE
    # Born-digital PDFs, so the collection is not mostly scans that the gate would reject anyway.
    issued_from: int | None = 2015
    issued_to: int | None = None
    search_page_size: int = 20
    max_search_pages: int = 10


@dataclass(frozen=True, slots=True)
class Screening:
    """What reading the downloaded file told us about it."""

    page_count: int
    chars_per_page: float
    rejection: str | None


def document_key(handle: str) -> str:
    """A Document Key derived from the IRIS handle, for example `10665/380063` to `who-iris-380063`.

    A Document Key is normally chosen by a Knowledge Manager. Nobody is going to hand-name a
    hundred of them, and the handle is already the stable identifier IRIS promises to keep, so it
    stands in. Re-running fetch produces the same keys and therefore re-ingests the same Documents.
    """
    suffix = handle.rsplit("/", 1)[-1].strip()
    if not suffix:
        raise ValueError(f"handle {handle!r} has no identifier to build a Document Key from")
    return f"{DOCUMENT_KEY_PREFIX}-{suffix}"


def screen(
    path: Path,
    settings: FetchSettings,
    extractor: PageExtractor | None = None,
) -> Screening:
    """Read the file and decide whether it belongs in the demo Knowledge Base.

    The text-layer test is `quarantine_reason`, the pipeline's own, so a Document that gets through
    here is a Document that ingests.
    """
    pages = (extractor or PdfPageExtractor()).extract(path)
    page_count = len(pages)
    total_chars = sum(len(page.text.strip()) for page in pages)
    chars_per_page = total_chars / page_count if page_count else 0.0

    reason = quarantine_reason(pages, settings.min_chars_per_page)
    if reason is not None:
        rejection = reason
    elif page_count < settings.min_pdf_pages:
        rejection = f"only {page_count} pages, minimum is {settings.min_pdf_pages}"
    elif page_count > settings.max_pdf_pages:
        rejection = f"{page_count} pages, maximum is {settings.max_pdf_pages}"
    else:
        rejection = None

    return Screening(page_count=page_count, chars_per_page=chars_per_page, rejection=rejection)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidates(
    repository: IrisRepository,
    topics: Sequence[str],
    settings: FetchSettings,
) -> Iterator[tuple[str, Publication]]:
    """Publications from every topic in turn: the best hit for each topic, then the second, and on.

    Interleaving one publication at a time rather than one search page at a time is what makes an
    early stop leave every topic represented. A page holds 20 hits and the target is 100
    Documents, so page-at-a-time interleaving would fill the collection from the first five topics.
    """
    for search_page in range(settings.max_search_pages):
        found = {
            topic: _search_or_nothing(repository, topic, search_page, settings) for topic in topics
        }
        deepest = max((len(publications) for publications in found.values()), default=0)
        if deepest == 0:
            return
        for position in range(deepest):
            for topic in topics:
                publications = found[topic]
                if position < len(publications):
                    yield topic, publications[position]


def _search_or_nothing(
    repository: IrisRepository,
    topic: str,
    search_page: int,
    settings: FetchSettings,
) -> list[Publication]:
    """A topic IRIS will not answer for costs that topic, not the ninety Documents already found."""
    try:
        return repository.search(
            topic,
            page=search_page,
            page_size=settings.search_page_size,
            issued_from=settings.issued_from,
            issued_to=settings.issued_to,
        )
    except Exception:
        return []


class LibraryFetcher:
    """Downloads and screens candidates until both targets are met."""

    def __init__(
        self,
        repository: IrisRepository,
        pdf_root: Path,
        *,
        settings: FetchSettings | None = None,
        extractor: PageExtractor | None = None,
        report: TextIO | None = None,
    ) -> None:
        self._repository = repository
        self._pdf_root = pdf_root
        self._settings = settings or FetchSettings()
        self._extractor = extractor or PdfPageExtractor()
        self._report = report

    def fetch(self, topics: Sequence[str] = DEFAULT_TOPICS) -> Manifest:
        self._pdf_root.mkdir(parents=True, exist_ok=True)
        entries: list[ManifestEntry] = []
        seen: set[str] = set()
        pages = 0

        for topic, publication in candidates(self._repository, topics, self._settings):
            if self._targets_met(len(entries), pages):
                break
            if publication.handle in seen:
                continue
            seen.add(publication.handle)

            entry = self._accept(topic, publication)
            if entry is None:
                continue
            entries.append(entry)
            pages += entry.page_count
            self._say(
                f"[{len(entries):>3}/{self._settings.target_documents}] "
                f"{entry.document_key}  {entry.page_count:>4}p  "
                f"{pages:>5} pages total  {entry.title[:60]}"
            )

        return Manifest(source=SOURCE_DESCRIPTION, entries=entries)

    def redownload(self, manifest: Manifest) -> Manifest:
        """Fetch exactly the Documents an existing manifest lists, and re-screen them.

        Page counts and hashes are recomputed rather than trusted: if IRIS has republished a
        Document, this is where that shows up.
        """
        self._pdf_root.mkdir(parents=True, exist_ok=True)
        entries: list[ManifestEntry] = []

        for position, recorded in enumerate(manifest.entries, start=1):
            destination = self._pdf_root / recorded.filename
            size_bytes = self._ensure_downloaded(recorded.pdf_url, destination)
            if size_bytes is None:
                self._say(f"  skipped {recorded.document_key}: download failed")
                continue

            screening = screen(destination, self._settings, self._extractor)
            if screening.rejection is not None:
                self._say(f"  skipped {recorded.document_key}: {screening.rejection}")
                continue

            digest = file_sha256(destination)
            if digest != recorded.sha256:
                self._say(f"  ! {recorded.document_key} has changed since the manifest was written")
            entries.append(
                replace(
                    recorded,
                    size_bytes=size_bytes,
                    page_count=screening.page_count,
                    chars_per_page=round(screening.chars_per_page, 1),
                    sha256=digest,
                )
            )
            self._say(
                f"[{position:>3}/{len(manifest.entries)}] {recorded.document_key}  "
                f"{screening.page_count:>4}p  {recorded.title[:60]}"
            )

        return Manifest(source=manifest.source, entries=entries)

    def _targets_met(self, document_count: int, page_count: int) -> bool:
        return (
            document_count >= self._settings.target_documents
            and page_count >= self._settings.target_pages
        )

    def _accept(self, topic: str, publication: Publication) -> ManifestEntry | None:
        key = document_key(publication.handle)
        try:
            url = self._repository.pdf_url(publication)
        except Exception as error:
            self._say(f"  skipped {key}: {error}")
            return None
        if url is None:
            return None

        destination = self._pdf_root / f"{key}.pdf"
        size_bytes = self._ensure_downloaded(url, destination)
        if size_bytes is None:
            return None
        if size_bytes > self._settings.max_bytes:
            self._say(f"  skipped {key}: {size_bytes / 1e6:.0f} MB is over the size limit")
            destination.unlink(missing_ok=True)
            return None

        screening = screen(destination, self._settings, self._extractor)
        if screening.rejection is not None:
            self._say(f"  skipped {key}: {screening.rejection}")
            destination.unlink(missing_ok=True)
            return None

        return ManifestEntry(
            document_key=key,
            title=publication.title,
            handle=publication.handle,
            uuid=publication.uuid,
            topic=topic,
            issued=publication.issued,
            publisher=publication.publisher,
            rights=publication.rights,
            landing_url=publication.landing_url,
            pdf_url=url,
            filename=destination.name,
            size_bytes=size_bytes,
            page_count=screening.page_count,
            chars_per_page=round(screening.chars_per_page, 1),
            sha256=file_sha256(destination),
        )

    def _ensure_downloaded(self, url: str, destination: Path) -> int | None:
        """Download unless the file is already there, so an interrupted run can be resumed."""
        if destination.exists() and destination.stat().st_size > 0:
            return destination.stat().st_size
        # A single 404 or timed-out download must not end a run of a hundred, so it is reported
        # and the candidate is dropped.
        try:
            return self._repository.download(url, destination)
        except Exception as error:
            destination.unlink(missing_ok=True)
            self._say(f"  skipped {destination.stem}: {error}")
            return None

    def _say(self, message: str) -> None:
        if self._report is not None:
            print(message, file=self._report, flush=True)
