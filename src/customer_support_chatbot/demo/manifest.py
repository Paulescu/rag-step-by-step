"""The record of which Documents the demo Knowledge Base is made of.

The PDFs are 300 MB and are not in the repository. The manifest is 100 KB and is, so the exact
demo Knowledge Base can be rebuilt from a checkout: `fetch --from-manifest` re-downloads precisely
the Documents recorded here, and the recorded SHA-256 says whether IRIS has since republished one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

MANIFEST_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One Document of the demo Knowledge Base, and the file it was built from."""

    document_key: str
    title: str
    handle: str
    uuid: str
    topic: str
    issued: str
    publisher: str
    rights: str
    landing_url: str
    pdf_url: str
    filename: str
    size_bytes: int
    page_count: int
    chars_per_page: float
    sha256: str


@dataclass(frozen=True, slots=True)
class Manifest:
    source: str
    entries: list[ManifestEntry]

    @property
    def document_count(self) -> int:
        return len(self.entries)

    @property
    def page_count(self) -> int:
        return sum(entry.page_count for entry in self.entries)

    @property
    def size_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MANIFEST_VERSION,
        "source": manifest.source,
        "document_count": manifest.document_count,
        "page_count": manifest.page_count,
        "documents": [asdict(entry) for entry in manifest.entries],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> Manifest:
    document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a manifest")
    payload = cast(dict[str, object], document)

    version = payload.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(f"{path} is manifest version {version}, expected {MANIFEST_VERSION}")

    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError(f"{path} has no list of documents")

    return Manifest(
        source=_text(payload, "source"),
        entries=[_to_entry(entry) for entry in cast(list[object], documents)],
    )


def _to_entry(entry: object) -> ManifestEntry:
    if not isinstance(entry, dict):
        raise ValueError("manifest entry is not an object")
    fields = cast(dict[str, object], entry)
    return ManifestEntry(
        document_key=_text(fields, "document_key"),
        title=_text(fields, "title"),
        handle=_text(fields, "handle"),
        uuid=_text(fields, "uuid"),
        topic=_text(fields, "topic"),
        issued=_text(fields, "issued"),
        publisher=_text(fields, "publisher"),
        rights=_text(fields, "rights"),
        landing_url=_text(fields, "landing_url"),
        pdf_url=_text(fields, "pdf_url"),
        filename=_text(fields, "filename"),
        size_bytes=_whole_number(fields, "size_bytes"),
        page_count=_whole_number(fields, "page_count"),
        chars_per_page=_real_number(fields, "chars_per_page"),
        sha256=_text(fields, "sha256"),
    )


def _text(fields: dict[str, object], key: str) -> str:
    value = fields.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"manifest field {key} is not a string")
    return value


def _whole_number(fields: dict[str, object], key: str) -> int:
    value = fields.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"manifest field {key} is not a whole number")
    return value


def _real_number(fields: dict[str, object], key: str) -> float:
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"manifest field {key} is not a number")
    return float(value)
