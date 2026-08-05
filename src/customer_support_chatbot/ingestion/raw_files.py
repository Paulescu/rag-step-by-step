"""Keeping the uploaded file itself.

The Knowledge Base holds no history (ADR-0004), so the raw file is what makes a deletion
reversible and an ingestion auditable. It is also what a full rebuild of the index reads from.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol


class RawFileStore(Protocol):
    def store(self, document_key: str, ingestion_version: int, path: Path) -> str: ...


class LocalRawFileStore(RawFileStore):
    """Stores raw files on the local filesystem, one directory per Document Key."""

    def __init__(self, root: Path) -> None:
        # Resolved on the way in: the stored URI has to be absolute, and callers pass relative
        # paths like ./data/raw.
        self._root = root.resolve()

    def store(self, document_key: str, ingestion_version: int, path: Path) -> str:
        destination_directory = self._root / document_key
        destination_directory.mkdir(parents=True, exist_ok=True)
        destination = destination_directory / f"{ingestion_version:04d}{path.suffix}"
        shutil.copyfile(path, destination)
        return destination.as_uri()
