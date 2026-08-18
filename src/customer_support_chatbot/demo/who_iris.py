"""Reading WHO IRIS, the repository the demo Knowledge Base is built from.

IRIS (https://iris.who.int) is the World Health Organization's institutional repository. It runs
DSpace 7, so it has a public REST API, and its publications are released under CC BY-NC-SA 3.0
IGO. That combination, open licence plus machine-readable listing plus born-digital PDFs, is why
it was chosen over the other candidates (ADR-0005).

Two calls are needed per publication: one search call returns a page of metadata, and one call per
item resolves the PDF in its ORIGINAL bundle. DSpace does not expose the bitstream URL on the
search hit.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import httpx

IRIS_BASE_URL = "https://iris.who.int"

# DSpace caps a page of search results at 100.
MAX_PAGE_SIZE = 100

# Courtesy pause between calls to a public repository nobody is paying to run.
DEFAULT_REQUEST_DELAY_SECONDS = 0.5

# IRIS closes connections part way through a long run often enough that a fetch of a hundred
# Documents will not finish without retrying.
DEFAULT_MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0

USER_AGENT = "customer-support-chatbot demo corpus fetcher (+https://iris.who.int)"

ORIGINAL_BUNDLE = "ORIGINAL"


@dataclass(frozen=True, slots=True)
class Publication:
    """One IRIS item, as much of it as the demo needs.

    `handle` is the repository's permanent identifier (`10665/380063`). It is what makes a Document
    Key derived from it stable across re-runs, which is the property the Knowledge Base cares about.
    """

    handle: str
    uuid: str
    title: str
    issued: str
    publisher: str
    rights: str
    landing_url: str


class HttpClient(Protocol):
    """The two things this module does over the network, so that tests can do neither."""

    def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, object]: ...

    def download(self, url: str, destination: Path) -> int: ...


class HttpxClient(HttpClient):
    """The real client. Follows redirects, because bitstream content URLs issue them."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 60.0,
        delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = RETRY_BACKOFF_SECONDS,
        client: httpx.Client | None = None,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )
        self._delay_seconds = delay_seconds
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, object]:
        def read() -> dict[str, object]:
            response = self._client.get(url, params=params, headers={"Accept": "application/json"})
            response.raise_for_status()
            # DSpace returns HAL-JSON whose shape varies per endpoint, so it arrives untyped and
            # the accessors below narrow it a level at a time.
            payload: object = response.json()
            if not isinstance(payload, dict):
                raise ValueError(f"expected a JSON object from {url}")
            return cast(dict[str, object], payload)

        return self._with_retries(read)

    def download(self, url: str, destination: Path) -> int:
        def write() -> int:
            written = 0
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for block in response.iter_bytes(chunk_size=1024 * 256):
                        handle.write(block)
                        written += len(block)
            return written

        return self._with_retries(write)

    def close(self) -> None:
        self._client.close()

    def _with_retries[T](self, call: Callable[[], T]) -> T:
        """Retry a transport failure, but not a 404: the first is IRIS, the second is the record."""
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            self._pause()
            try:
                return call()
            except httpx.HTTPStatusError as error:
                if error.response.status_code < 500:
                    raise
                last_error = error
            except httpx.HTTPError as error:
                last_error = error
            time.sleep(self._backoff_seconds * (attempt + 1))
        raise RuntimeError(f"gave up after {self._max_attempts} attempts") from last_error

    def _pause(self) -> None:
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)


def _as_mapping(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _as_sequence(value: object) -> list[object]:
    return cast(list[object], value) if isinstance(value, list) else []


def _metadata_value(metadata: dict[str, object], field: str) -> str:
    """First value of a Dublin Core field, or the empty string.

    IRIS is inconsistent about which fields it fills in, so a missing publisher or licence is a
    normal record and not an error.
    """
    entries = _as_sequence(metadata.get(field))
    for entry in entries:
        value = _as_mapping(entry).get("value")
        if isinstance(value, str) and value:
            return value
    return ""


class IrisRepository:
    """Queries IRIS for English publications and resolves their PDFs."""

    def __init__(
        self,
        http: HttpClient,
        *,
        base_url: str = IRIS_BASE_URL,
    ) -> None:
        self._http = http
        self._base_url = base_url.rstrip("/")

    def search(
        self,
        query: str,
        *,
        page: int = 0,
        page_size: int = 20,
        issued_from: int | None = None,
        issued_to: int | None = None,
    ) -> list[Publication]:
        """One page of English publications matching `query`, most relevant first.

        The item type filter keeps out governing-body papers and journal articles, which are
        meeting minutes and reprints rather than the guidance an institution would answer from.
        """
        params = {
            "query": query,
            "dsoType": "item",
            "f.language": "en,equals",
            "f.itemtype": "Publications,equals",
            "page": str(page),
            "size": str(min(page_size, MAX_PAGE_SIZE)),
        }
        if issued_from is not None or issued_to is not None:
            low = str(issued_from) if issued_from is not None else "*"
            high = str(issued_to) if issued_to is not None else "*"
            params["f.dateIssued"] = f"[{low} TO {high}],equals"

        payload = self._http.get_json(
            f"{self._base_url}/server/api/discover/search/objects",
            params,
        )
        search_result = _as_mapping(_as_mapping(payload.get("_embedded")).get("searchResult"))
        objects = _as_sequence(_as_mapping(search_result.get("_embedded")).get("objects"))
        return [
            publication for hit in objects if (publication := self._to_publication(hit)) is not None
        ]

    def pdf_url(self, publication: Publication) -> str | None:
        """The download URL of the item's PDF, or None if it has no PDF to offer.

        Only the ORIGINAL bundle is considered. DSpace also derives TEXT and THUMBNAIL bundles; the
        plain-text one would sidestep extraction entirely and defeat the point of the demo.
        """
        payload = self._http.get_json(
            f"{self._base_url}/server/api/core/items/{publication.uuid}/bundles",
            {"embed": "bitstreams"},
        )
        bundles = _as_sequence(_as_mapping(payload.get("_embedded")).get("bundles"))
        for bundle in bundles:
            bundle_map = _as_mapping(bundle)
            if bundle_map.get("name") != ORIGINAL_BUNDLE:
                continue
            for bitstream in _bitstreams_of(bundle_map):
                name = bitstream.get("name")
                if not isinstance(name, str) or not name.lower().endswith(".pdf"):
                    continue
                href = _as_mapping(_as_mapping(bitstream.get("_links")).get("content")).get("href")
                if isinstance(href, str) and href:
                    return href
        return None

    def download(self, url: str, destination: Path) -> int:
        return self._http.download(url, destination)

    def _to_publication(self, hit: object) -> Publication | None:
        item = _as_mapping(_as_mapping(_as_mapping(hit).get("_embedded")).get("indexableObject"))
        handle = item.get("handle")
        uuid = item.get("uuid")
        name = item.get("name")
        if not isinstance(handle, str) or not isinstance(uuid, str) or not isinstance(name, str):
            return None

        metadata = _as_mapping(item.get("metadata"))
        return Publication(
            handle=handle,
            uuid=uuid,
            title=name,
            issued=_metadata_value(metadata, "dc.date.issued"),
            publisher=_metadata_value(metadata, "dc.publisher"),
            rights=_metadata_value(metadata, "dc.rights"),
            landing_url=f"{self._base_url}/handle/{handle}",
        )


def _bitstreams_of(bundle: dict[str, object]) -> Iterator[dict[str, object]]:
    embedded = _as_mapping(_as_mapping(bundle.get("_embedded")).get("bitstreams"))
    for bitstream in _as_sequence(_as_mapping(embedded.get("_embedded")).get("bitstreams")):
        yield _as_mapping(bitstream)
