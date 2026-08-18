from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pytest

from customer_support_chatbot.demo.fetch import (
    FetchSettings,
    LibraryFetcher,
    document_key,
    screen,
)
from customer_support_chatbot.demo.manifest import Manifest
from customer_support_chatbot.demo.who_iris import IrisRepository
from tests.builders import write_document
from tests.fakes import TextFileExtractor
from tests.iris_fakes import BASE_URL, FakeIrisHttp, FakeItem, page_of

SETTINGS = FetchSettings(
    target_documents=2,
    target_pages=4,
    min_pdf_pages=2,
    max_pdf_pages=10,
    max_search_pages=2,
    search_page_size=20,
)


def build_fetcher(
    results: dict[str, list[FakeItem]],
    tmp_path: Path,
    settings: FetchSettings = SETTINGS,
) -> tuple[LibraryFetcher, FakeIrisHttp]:
    http = FakeIrisHttp(results, page_size=settings.search_page_size)
    fetcher = LibraryFetcher(
        IrisRepository(http, base_url=BASE_URL),
        tmp_path / "pdfs",
        settings=settings,
        extractor=TextFileExtractor(),
    )
    return fetcher, http


def item(handle: str, pages: int = 3, pdf_name: str | None = "publication.pdf") -> FakeItem:
    return FakeItem(
        handle=handle,
        title=f"Publication {handle}",
        pages=[page_of() for _ in range(pages)],
        pdf_name=pdf_name,
    )


def test_the_document_key_comes_from_the_iris_handle() -> None:
    assert document_key("10665/380063") == "who-iris-380063"


def test_a_handle_with_no_identifier_is_rejected_rather_than_producing_a_bare_prefix() -> None:
    with pytest.raises(ValueError, match="Document Key"):
        document_key("10665/")


def test_screening_accepts_a_document_with_a_usable_text_layer(tmp_path: Path) -> None:
    path = write_document(tmp_path, "good.pdf", [page_of(), page_of(), page_of()])

    result = screen(path, SETTINGS, TextFileExtractor())

    assert result.rejection is None
    assert result.page_count == 3
    assert result.chars_per_page > 100


def test_screening_rejects_what_the_pipeline_would_quarantine(tmp_path: Path) -> None:
    """The gate here is the pipeline's own, so nothing in the manifest lands as Quarantined."""
    path = write_document(tmp_path, "scan.pdf", ["", "", ""])

    result = screen(path, SETTINGS, TextFileExtractor())

    assert result.rejection is not None
    assert "probably a scan" in result.rejection


def test_screening_rejects_documents_outside_the_page_bounds(tmp_path: Path) -> None:
    short = write_document(tmp_path, "short.pdf", [page_of()])
    long = write_document(tmp_path, "long.pdf", [page_of() for _ in range(11)])

    assert "minimum is 2" in (screen(short, SETTINGS, TextFileExtractor()).rejection or "")
    assert "maximum is 10" in (screen(long, SETTINGS, TextFileExtractor()).rejection or "")


def test_fetching_stops_once_both_targets_are_met(tmp_path: Path) -> None:
    results = {
        "a": [item("10665/1"), item("10665/2"), item("10665/3")],
        "b": [item("10665/4")],
    }
    fetcher, _ = build_fetcher(results, tmp_path)

    manifest = fetcher.fetch(["a", "b"])

    assert manifest.document_count == 2
    assert manifest.page_count >= SETTINGS.target_pages


def test_the_document_target_alone_is_not_enough(tmp_path: Path) -> None:
    """Two Documents of one page each meet the count and miss the point, so the run goes on."""
    settings = FetchSettings(
        target_documents=2,
        target_pages=9,
        min_pdf_pages=1,
        max_pdf_pages=10,
        max_search_pages=1,
    )
    results = {"a": [item(f"10665/{n}", pages=3) for n in range(1, 5)]}
    fetcher, _ = build_fetcher(results, tmp_path, settings)

    manifest = fetcher.fetch(["a"])

    assert manifest.document_count == 3
    assert manifest.page_count == 9


def test_topics_are_interleaved_so_an_early_stop_still_covers_the_subject(tmp_path: Path) -> None:
    results = {
        "a": [item("10665/1"), item("10665/2")],
        "b": [item("10665/3"), item("10665/4")],
    }
    fetcher, _ = build_fetcher(results, tmp_path)

    manifest = fetcher.fetch(["a", "b"])

    assert [entry.topic for entry in manifest.entries] == ["a", "b"]


def test_a_publication_found_under_two_topics_becomes_one_document(tmp_path: Path) -> None:
    shared = item("10665/1")
    results = {"a": [shared], "b": [shared, item("10665/2")]}
    fetcher, _ = build_fetcher(results, tmp_path)

    manifest = fetcher.fetch(["a", "b"])

    assert [entry.document_key for entry in manifest.entries] == ["who-iris-1", "who-iris-2"]


def test_a_publication_without_a_pdf_is_skipped(tmp_path: Path) -> None:
    results = {
        "a": [item("10665/1", pdf_name=None), item("10665/2"), item("10665/3")],
    }
    fetcher, _ = build_fetcher(results, tmp_path)

    manifest = fetcher.fetch(["a"])

    assert [entry.document_key for entry in manifest.entries] == ["who-iris-2", "who-iris-3"]


def test_a_scan_is_skipped_and_its_file_is_not_left_behind(tmp_path: Path) -> None:
    scan = FakeItem(handle="10665/1", title="A scan", pages=["", "", ""])
    results = {"a": [scan, item("10665/2"), item("10665/3")]}
    fetcher, _ = build_fetcher(results, tmp_path)

    manifest = fetcher.fetch(["a"])

    assert [entry.document_key for entry in manifest.entries] == ["who-iris-2", "who-iris-3"]
    assert not (tmp_path / "pdfs" / "who-iris-1.pdf").exists()


def test_an_oversized_publication_is_skipped(tmp_path: Path) -> None:
    settings = FetchSettings(
        target_documents=1,
        target_pages=1,
        min_pdf_pages=1,
        max_pdf_pages=10,
        max_bytes=100,
        max_search_pages=1,
    )
    results = {"a": [item("10665/1", pages=3), FakeItem(handle="10665/2", title="Tiny", pages=[])]}
    fetcher, _ = build_fetcher(results, tmp_path, settings)

    manifest = fetcher.fetch(["a"])

    assert manifest.entries == []
    assert not (tmp_path / "pdfs" / "who-iris-1.pdf").exists()


def test_an_already_downloaded_file_is_not_fetched_again(tmp_path: Path) -> None:
    """An interrupted run of a hundred downloads should not start over."""
    results = {"a": [item("10665/1"), item("10665/2")]}
    fetcher, http = build_fetcher(results, tmp_path)
    fetcher.fetch(["a"])
    downloads = len(http.downloads)

    second, second_http = build_fetcher(results, tmp_path)
    second.fetch(["a"])

    assert downloads == 2
    assert second_http.downloads == []


def test_the_manifest_records_where_each_document_came_from(tmp_path: Path) -> None:
    fetcher, _ = build_fetcher({"a": [item("10665/1"), item("10665/2")]}, tmp_path)

    entry = fetcher.fetch(["a"]).entries[0]

    assert entry.handle == "10665/1"
    assert entry.landing_url == f"{BASE_URL}/handle/10665/1"
    assert entry.rights == "CC BY-NC-SA 3.0 IGO"
    assert entry.filename == "who-iris-1.pdf"
    assert entry.sha256 != ""


def test_a_topic_iris_will_not_answer_for_costs_only_that_topic(tmp_path: Path) -> None:
    class FlakyHttp(FakeIrisHttp):
        def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, object]:
            if (params or {}).get("query") == "broken":
                raise RuntimeError("gave up after 4 attempts")
            return super().get_json(url, params)

    http = FlakyHttp({"a": [item("10665/1"), item("10665/2")], "broken": []})
    fetcher = LibraryFetcher(
        IrisRepository(http, base_url=BASE_URL),
        tmp_path / "pdfs",
        settings=SETTINGS,
        extractor=TextFileExtractor(),
    )

    manifest = fetcher.fetch(["broken", "a"])

    assert [entry.document_key for entry in manifest.entries] == ["who-iris-1", "who-iris-2"]


def test_a_candidate_whose_bundles_cannot_be_read_is_dropped_not_fatal(tmp_path: Path) -> None:
    class FlakyHttp(FakeIrisHttp):
        def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, object]:
            if url.endswith("/bundles") and "uuid-1/" in url:
                raise RuntimeError("gave up after 4 attempts")
            return super().get_json(url, params)

    http = FlakyHttp({"a": [item("10665/1"), item("10665/2"), item("10665/3")]})
    report = io.StringIO()
    fetcher = LibraryFetcher(
        IrisRepository(http, base_url=BASE_URL),
        tmp_path / "pdfs",
        settings=SETTINGS,
        extractor=TextFileExtractor(),
        report=report,
    )

    manifest = fetcher.fetch(["a"])

    assert [entry.document_key for entry in manifest.entries] == ["who-iris-2", "who-iris-3"]
    assert "skipped who-iris-1" in report.getvalue()


def test_redownloading_a_manifest_takes_exactly_what_it_lists(tmp_path: Path) -> None:
    """`--from-manifest` is what makes a checkout reproduce someone else's demo Knowledge Base."""
    results = {"a": [item("10665/1"), item("10665/2"), item("10665/3")]}
    recorded = build_fetcher(results, tmp_path)[0].fetch(["a"])

    fetcher, http = build_fetcher(results, tmp_path / "elsewhere")
    replayed = fetcher.redownload(recorded)

    assert [entry.document_key for entry in replayed.entries] == [
        entry.document_key for entry in recorded.entries
    ]
    assert len(http.downloads) == recorded.document_count
    assert http.search_params == []


def test_redownloading_reports_a_document_iris_has_republished(tmp_path: Path) -> None:
    results = {"a": [item("10665/1"), item("10665/2")]}
    recorded = build_fetcher(results, tmp_path)[0].fetch(["a"])
    stale = Manifest(
        source=recorded.source,
        entries=[replace(recorded.entries[0], sha256="0" * 64)],
    )

    report = io.StringIO()
    http = FakeIrisHttp(results)
    fetcher = LibraryFetcher(
        IrisRepository(http, base_url=BASE_URL),
        tmp_path / "elsewhere",
        settings=SETTINGS,
        extractor=TextFileExtractor(),
        report=report,
    )
    rebuilt = fetcher.redownload(stale)

    assert "has changed since the manifest was written" in report.getvalue()
    assert rebuilt.entries[0].sha256 == recorded.entries[0].sha256
