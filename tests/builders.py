"""Builders for the test data the ingestion tests feed into the pipeline."""

from __future__ import annotations

from pathlib import Path

from tests.fakes import PAGE_BREAK


def write_document(tmp_path: Path, name: str, pages: list[str]) -> Path:
    """Write a document whose pages TextFileExtractor will read back one per entry."""
    path = tmp_path / name
    path.write_text(PAGE_BREAK.join(pages), encoding="utf-8")
    return path
