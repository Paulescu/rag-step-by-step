from __future__ import annotations

import itertools

import pytest

from customer_support_chatbot.ingestion.chunking import (
    ChunkingSettings,
    approximate_token_count,
    chunk_pages,
)
from customer_support_chatbot.ingestion.models import Page


def word_count(text: str) -> int:
    return max(1, len(text.split()))


def make_page(number: int, paragraphs: list[str]) -> Page:
    return Page(number=number, text="\n\n".join(paragraphs))


def test_chunks_never_cross_a_page_boundary() -> None:
    pages = [
        make_page(1, ["alpha " * 10, "beta " * 10]),
        make_page(2, ["gamma " * 10]),
    ]

    chunks = chunk_pages(pages, "doc", ChunkingSettings(max_tokens=1000, overlap_tokens=0))

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert "gamma" not in chunks[0].text
    assert "alpha" not in chunks[1].text


def test_chunk_indexes_run_consecutively_across_the_document() -> None:
    pages = [make_page(1, ["one " * 40]), make_page(2, ["two " * 40])]

    chunks = chunk_pages(
        pages,
        "doc",
        ChunkingSettings(max_tokens=20, overlap_tokens=0),
        count_tokens=word_count,
    )

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert len(chunks) > 2


def test_chunks_stay_within_the_token_budget() -> None:
    pages = [make_page(1, ["word " * 200])]

    chunks = chunk_pages(
        pages,
        "doc",
        ChunkingSettings(max_tokens=25, overlap_tokens=0),
        count_tokens=word_count,
    )

    assert chunks
    assert all(word_count(chunk.text) <= 25 for chunk in chunks)


def test_consecutive_chunks_overlap() -> None:
    paragraphs = [f"paragraph{index} " * 10 for index in range(6)]
    pages = [make_page(1, paragraphs)]

    chunks = chunk_pages(
        pages,
        "doc",
        ChunkingSettings(max_tokens=30, overlap_tokens=8),
        count_tokens=word_count,
    )

    assert len(chunks) > 1
    for previous, following in itertools.pairwise(chunks):
        previous_tail = previous.text.split()[-8:]
        assert any(word in following.text.split() for word in previous_tail)


def test_a_single_oversized_paragraph_is_split() -> None:
    pages = [make_page(1, ["giant " * 500])]

    chunks = chunk_pages(
        pages,
        "doc",
        ChunkingSettings(max_tokens=50, overlap_tokens=0),
        count_tokens=word_count,
    )

    assert len(chunks) > 1
    assert all(word_count(chunk.text) <= 50 for chunk in chunks)


def test_blank_pages_produce_no_chunks() -> None:
    chunks = chunk_pages([Page(number=1, text="   \n\n  ")], "doc")

    assert chunks == []


def test_overlap_must_be_smaller_than_the_budget() -> None:
    with pytest.raises(ValueError):
        ChunkingSettings(max_tokens=10, overlap_tokens=10)


def test_approximate_token_count_never_returns_zero() -> None:
    assert approximate_token_count("") == 1
    assert approximate_token_count("a" * 400) == 100
