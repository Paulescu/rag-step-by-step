from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from customer_support_chatbot.demo.who_iris import HttpxClient, IrisRepository
from tests.iris_fakes import BASE_URL, FakeIrisHttp, FakeItem


@pytest.fixture
def item() -> FakeItem:
    return FakeItem(handle="10665/380063", title="Operational handbook on tuberculosis")


@pytest.fixture
def repository(item: FakeItem) -> IrisRepository:
    return IrisRepository(FakeIrisHttp({"tuberculosis": [item]}), base_url=BASE_URL)


def test_search_reads_the_fields_the_manifest_needs(
    repository: IrisRepository, item: FakeItem
) -> None:
    found = repository.search("tuberculosis")

    assert [publication.handle for publication in found] == [item.handle]
    assert found[0].title == item.title
    assert found[0].issued == item.issued
    assert found[0].rights == "CC BY-NC-SA 3.0 IGO"
    assert found[0].landing_url == f"{BASE_URL}/handle/10665/380063"


def test_search_asks_only_for_english_publications() -> None:
    http = FakeIrisHttp({"malaria": []})

    IrisRepository(http, base_url=BASE_URL).search("malaria")

    assert http.search_params[0]["f.language"] == "en,equals"
    assert http.search_params[0]["f.itemtype"] == "Publications,equals"
    assert "f.dateIssued" not in http.search_params[0]


def test_a_year_range_becomes_a_dspace_range_filter() -> None:
    http = FakeIrisHttp({"malaria": []})
    repository = IrisRepository(http, base_url=BASE_URL)

    repository.search("malaria", issued_from=2015)
    repository.search("malaria", issued_from=2015, issued_to=2024)
    repository.search("malaria", issued_to=2024)

    assert [params["f.dateIssued"] for params in http.search_params] == [
        "[2015 TO *],equals",
        "[2015 TO 2024],equals",
        "[* TO 2024],equals",
    ]


def test_the_page_size_is_capped_at_what_dspace_will_return() -> None:
    http = FakeIrisHttp({"malaria": []})

    IrisRepository(http, base_url=BASE_URL).search("malaria", page_size=500)

    assert http.search_params[0]["size"] == "100"


def test_a_hit_missing_its_identifiers_is_dropped_rather_than_raising() -> None:
    """One malformed record out of 148,000 should cost one Document, not the whole run."""

    class BrokenHttp:
        def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, object]:
            return {"_embedded": {"searchResult": {"_embedded": {"objects": [{}, "nonsense"]}}}}

        def download(self, url: str, destination: Path) -> int:
            raise AssertionError("no download expected")

    found = IrisRepository(BrokenHttp(), base_url=BASE_URL).search("malaria")

    assert found == []


def test_the_pdf_comes_from_the_original_bundle_not_the_derived_text_one(
    repository: IrisRepository, item: FakeItem
) -> None:
    url = repository.pdf_url(item.as_publication())

    assert url == item.content_url


def test_an_item_with_no_pdf_has_no_url(item: FakeItem) -> None:
    without_pdf = FakeItem(handle=item.handle, title=item.title, pdf_name=None)
    repository = IrisRepository(FakeIrisHttp({"x": [without_pdf]}), base_url=BASE_URL)

    assert repository.pdf_url(without_pdf.as_publication()) is None


def build_client(handler: Callable[[httpx.Request], httpx.Response]) -> HttpxClient:
    return HttpxClient(
        delay_seconds=0,
        backoff_seconds=0,
        max_attempts=3,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_a_dropped_connection_is_retried_because_iris_drops_them_mid_run() -> None:
    """A hundred Documents is a long enough run that IRIS will close a connection during it."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(200, json={"ok": True})

    assert build_client(handler).get_json("https://iris.test/thing") == {"ok": True}
    assert attempts == 3


def test_a_missing_record_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        build_client(handler).get_json("https://iris.test/gone")
    assert attempts == 1


def test_giving_up_says_how_many_attempts_it_took() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(RuntimeError, match="gave up after 3 attempts"):
        build_client(handler).get_json("https://iris.test/down")
