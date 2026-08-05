"""Splitting pages into Chunks.

Chunks never cross a page boundary, so every Chunk can cite the page it came from. Within a page,
text is packed paragraph by paragraph up to a token budget, with a small overlap carried across
so that an answer sitting on a chunk boundary is still retrievable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from customer_support_chatbot.ingestion.models import Chunk, Page

type TokenCounter = Callable[[str], int]

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"\s+")


def approximate_token_count(text: str) -> int:
    """Rough token count used when no real tokenizer is available.

    Four characters per token is the usual approximation for subword tokenizers. It is only used
    to decide where to split, so being a little off costs chunk size, not correctness.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class ChunkingSettings:
    max_tokens: int = 500
    overlap_tokens: int = 60

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")


def chunk_pages(
    pages: list[Page],
    document_key: str,
    settings: ChunkingSettings | None = None,
    count_tokens: TokenCounter = approximate_token_count,
) -> list[Chunk]:
    """Split every page into Chunks, numbered consecutively across the whole Document."""
    settings = settings or ChunkingSettings()
    chunks: list[Chunk] = []
    for page in pages:
        for text in _chunk_page_text(page.text, settings, count_tokens):
            chunks.append(
                Chunk(
                    document_key=document_key,
                    page_number=page.number,
                    index=len(chunks),
                    text=text,
                )
            )
    return chunks


def _chunk_page_text(
    text: str,
    settings: ChunkingSettings,
    count_tokens: TokenCounter,
) -> list[str]:
    units = _split_into_units(text, settings.max_tokens, count_tokens)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > settings.max_tokens:
            chunks.append("\n\n".join(current))
            overlap = _tail_within_budget(current[-1], settings.overlap_tokens, count_tokens)
            current = [overlap] if overlap else []
            current_tokens = count_tokens(overlap) if overlap else 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _split_into_units(text: str, max_tokens: int, count_tokens: TokenCounter) -> list[str]:
    """Break a page into pieces that each fit the token budget on their own."""
    units: list[str] = []
    for raw_paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = _WHITESPACE.sub(" ", raw_paragraph).strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph, max_tokens, count_tokens))
    return units


def _split_long_paragraph(
    paragraph: str,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[str]:
    """Cut a paragraph that exceeds the budget on its own, at word boundaries."""
    pieces: list[str] = []
    words = paragraph.split(" ")
    current: list[str] = []
    for word in words:
        candidate = [*current, word]
        if current and count_tokens(" ".join(candidate)) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail_within_budget(text: str, budget_tokens: int, count_tokens: TokenCounter) -> str:
    """The longest trailing run of words that fits the overlap budget."""
    if budget_tokens <= 0:
        return ""
    words = text.split(" ")
    tail: list[str] = []
    for word in reversed(words):
        candidate = [word, *tail]
        if count_tokens(" ".join(candidate)) > budget_tokens:
            break
        tail = candidate
    return " ".join(tail)
