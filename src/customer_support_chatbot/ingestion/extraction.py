"""Turning a file into pages of text, and deciding whether that text is usable.

Only PDFs are supported. Everything downstream depends on the `PageExtractor` protocol rather
than on any PDF library, so a second format is a new implementation and not a change here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

from customer_support_chatbot.ingestion.models import Page

DEFAULT_MIN_CHARS_PER_PAGE = 100


class PageExtractor(Protocol):
    def extract(self, path: Path) -> list[Page]: ...


class PdfPageExtractor(PageExtractor):
    """Reads the text layer of a PDF. Does not OCR: scans come back empty, by design."""

    def extract(self, path: Path) -> list[Page]:
        reader = PdfReader(path)
        return [
            Page(number=number, text=page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]


def quarantine_reason(
    pages: list[Page],
    min_chars_per_page: float = DEFAULT_MIN_CHARS_PER_PAGE,
) -> str | None:
    """Why this Document cannot be ingested, or None if it can.

    A Document whose text layer is too thin is almost always a scan. Ingesting it would put empty
    Chunks in the Knowledge Base and the failure would be invisible, so it is quarantined instead.
    """
    if not pages:
        return "no pages could be read from the file"

    total_chars = sum(len(page.text.strip()) for page in pages)
    chars_per_page = total_chars / len(pages)
    if chars_per_page < min_chars_per_page:
        return (
            f"text layer too thin: {chars_per_page:.0f} characters per page "
            f"across {len(pages)} pages, minimum is {min_chars_per_page:.0f}. "
            "The file is probably a scan and needs OCR."
        )
    return None
