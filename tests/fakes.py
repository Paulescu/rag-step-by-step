"""Test doubles for the ports the ingestion pipeline depends on.

These live outside conftest.py so test modules can import them explicitly rather than reaching
into another module's fixture definitions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from customer_support_chatbot.ingestion.models import Embedding, Page, SparseVector

DENSE_SIZE = 32
SPARSE_SIZE = 4096

PAGE_BREAK = "\f"


class TextFileExtractor:
    """Stand-in for the PDF extractor: reads a text file, form feeds separate pages.

    Lets the pipeline be tested end to end without PDF fixtures, which is the whole reason
    PageExtractor is a Protocol.
    """

    def extract(self, path: Path) -> list[Page]:
        raw = path.read_text(encoding="utf-8")
        return [
            Page(number=number, text=text)
            for number, text in enumerate(raw.split(PAGE_BREAK), start=1)
        ]


def _token_bucket(word: str, buckets: int) -> int:
    return int.from_bytes(hashlib.sha1(word.encode("utf-8")).digest()[:4], "big") % buckets


class HashingEmbedder:
    """Deterministic bag-of-words embedder.

    Not semantic, but it is stable and it makes texts sharing words come out close together,
    which is enough to test that the search path wires up correctly.
    """

    @property
    def dense_size(self) -> int:
        return DENSE_SIZE

    def embed(self, texts: list[str]) -> list[Embedding]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> Embedding:
        words = [word.strip(".,:;!?").lower() for word in text.split() if word.strip(".,:;!?")]

        dense = [0.0] * DENSE_SIZE
        for word in words:
            dense[_token_bucket(word, DENSE_SIZE)] += 1.0
        magnitude = sum(value * value for value in dense) ** 0.5
        if magnitude > 0:
            dense = [value / magnitude for value in dense]
        else:
            dense[0] = 1.0

        weights: dict[int, float] = {}
        for word in words:
            index = _token_bucket(word, SPARSE_SIZE)
            weights[index] = weights.get(index, 0.0) + 1.0
        if not weights:
            weights[0] = 1.0

        return Embedding(
            dense=dense,
            sparse=SparseVector(
                indices=list(weights.keys()),
                values=list(weights.values()),
            ),
        )
