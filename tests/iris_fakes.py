"""A stand-in for WHO IRIS.

The demo fetcher is mostly a set of rules about which publications to keep, and those rules are
what the tests are about. Serving DSpace-shaped JSON from a dictionary keeps them off the network
and lets a "PDF" be a text file that `TextFileExtractor` can read back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from customer_support_chatbot.demo.who_iris import Publication

BASE_URL = "https://iris.test"


@dataclass(frozen=True, slots=True)
class FakeItem:
    """One publication IRIS will hand out, and the bytes behind it."""

    handle: str
    title: str
    pages: list[str] = field(default_factory=list[str])
    issued: str = "2024-01-01"
    publisher: str = "World Health Organization"
    rights: str = "CC BY-NC-SA 3.0 IGO"
    pdf_name: str | None = "publication.pdf"

    @property
    def uuid(self) -> str:
        return f"uuid-{self.handle.rsplit('/', 1)[-1]}"

    @property
    def content_url(self) -> str:
        return f"{BASE_URL}/server/api/core/bitstreams/{self.uuid}/content"

    def as_publication(self) -> Publication:
        return Publication(
            handle=self.handle,
            uuid=self.uuid,
            title=self.title,
            issued=self.issued,
            publisher=self.publisher,
            rights=self.rights,
            landing_url=f"{BASE_URL}/handle/{self.handle}",
        )


def search_payload(items: list[FakeItem]) -> dict[str, object]:
    """The slice of DSpace's discover response the repository client reads."""
    return {
        "_embedded": {
            "searchResult": {
                "_embedded": {
                    "objects": [_search_hit(item) for item in items],
                }
            }
        }
    }


def _search_hit(item: FakeItem) -> dict[str, object]:
    return {
        "_embedded": {
            "indexableObject": {
                "uuid": item.uuid,
                "handle": item.handle,
                "name": item.title,
                "metadata": {
                    "dc.date.issued": [{"value": item.issued}],
                    "dc.publisher": [{"value": item.publisher}],
                    "dc.rights": [{"value": item.rights}],
                },
            }
        }
    }


def bundles_payload(item: FakeItem) -> dict[str, object]:
    """ORIGINAL alongside the derived bundles, because picking the right one is the behaviour."""
    bundles: list[dict[str, object]] = [
        _bundle("THUMBNAIL", [("preview.jpg", f"{BASE_URL}/thumbnail")]),
        _bundle("TEXT", [("publication.pdf.txt", f"{BASE_URL}/text")]),
    ]
    original: list[tuple[str, str]] = []
    if item.pdf_name is not None:
        original.append((item.pdf_name, item.content_url))
    bundles.append(_bundle("ORIGINAL", original))
    return {"_embedded": {"bundles": bundles}}


def _bundle(name: str, bitstreams: list[tuple[str, str]]) -> dict[str, object]:
    return {
        "name": name,
        "_embedded": {
            "bitstreams": {
                "_embedded": {
                    "bitstreams": [
                        {"name": bitstream_name, "_links": {"content": {"href": href}}}
                        for bitstream_name, href in bitstreams
                    ]
                }
            }
        },
    }


class FakeIrisHttp:
    """Serves search results per topic and files per bitstream, and records what was asked for."""

    def __init__(self, results: dict[str, list[FakeItem]], *, page_size: int = 20) -> None:
        self._results = results
        self._page_size = page_size
        self._items = {item.uuid: item for item in _all(results)}
        self.search_params: list[dict[str, str]] = []
        self.downloads: list[str] = []

    def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, object]:
        if url.endswith("/discover/search/objects"):
            query = (params or {}).get("query", "")
            self.search_params.append(dict(params or {}))
            page = int((params or {}).get("page", "0"))
            found = self._results.get(query, [])
            start = page * self._page_size
            return search_payload(found[start : start + self._page_size])

        uuid = url.rstrip("/").split("/")[-2]
        item = self._items.get(uuid)
        if item is None:
            raise LookupError(f"no item {uuid}")
        return bundles_payload(item)

    def download(self, url: str, destination: Path) -> int:
        self.downloads.append(url)
        item = next((item for item in self._items.values() if item.content_url == url), None)
        if item is None:
            raise LookupError(f"no bitstream {url}")
        body = "\f".join(item.pages).encode("utf-8")
        destination.write_bytes(body)
        return len(body)


def _all(results: dict[str, list[FakeItem]]) -> list[FakeItem]:
    return [item for items in results.values() for item in items]


def page_of(characters: int = 400, word: str = "vaccination") -> str:
    """A page with a text layer thick enough to clear the quarantine gate."""
    return (word + " ") * (characters // (len(word) + 1) + 1)
