from __future__ import annotations

import json
from pathlib import Path

import pytest

from customer_support_chatbot.demo.manifest import (
    MANIFEST_VERSION,
    Manifest,
    ManifestEntry,
    read_manifest,
    write_manifest,
)


def entry(key: str, pages: int = 12, size: int = 1_000) -> ManifestEntry:
    return ManifestEntry(
        document_key=key,
        title=f"Title of {key}",
        handle=f"10665/{key}",
        uuid=f"uuid-{key}",
        topic="malaria",
        issued="2024-01-01",
        publisher="World Health Organization",
        rights="CC BY-NC-SA 3.0 IGO",
        landing_url=f"https://iris.who.int/handle/10665/{key}",
        pdf_url=f"https://iris.who.int/bitstreams/{key}/content",
        filename=f"{key}.pdf",
        size_bytes=size,
        page_count=pages,
        chars_per_page=1800.5,
        sha256="a" * 64,
    )


def test_a_manifest_survives_a_round_trip_to_disk(tmp_path: Path) -> None:
    manifest = Manifest(source="WHO IRIS", entries=[entry("one"), entry("two")])
    path = tmp_path / "nested" / "manifest.json"

    write_manifest(path, manifest)

    assert read_manifest(path) == manifest


def test_the_totals_are_what_the_ticket_asked_about() -> None:
    manifest = Manifest(
        source="WHO IRIS",
        entries=[entry("one", pages=12, size=1_000), entry("two", pages=30, size=2_500)],
    )

    assert manifest.document_count == 2
    assert manifest.page_count == 42
    assert manifest.size_bytes == 3_500


def test_the_totals_are_written_out_so_the_file_can_be_read_without_the_code(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, Manifest(source="WHO IRIS", entries=[entry("one", pages=12)]))

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["version"] == MANIFEST_VERSION
    assert payload["document_count"] == 1
    assert payload["page_count"] == 12


def test_a_manifest_from_a_future_version_is_refused_rather_than_half_read(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 99, "documents": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest version 99"):
        read_manifest(path)


def test_a_page_count_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_manifest(path, Manifest(source="WHO IRIS", entries=[entry("one")]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["documents"][0]["page_count"] = "twelve"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="page_count"):
        read_manifest(path)
