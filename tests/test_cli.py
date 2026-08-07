from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from customer_support_chatbot.ingestion import cli
from customer_support_chatbot.ingestion.chunking import ChunkingSettings
from customer_support_chatbot.ingestion.embedding import (
    BGE_M3_DENSE_SIZE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MAX_LENGTH,
    DEFAULT_EMBEDDING_MODEL,
)
from customer_support_chatbot.ingestion.extraction import DEFAULT_MIN_CHARS_PER_PAGE
from customer_support_chatbot.ingestion.store import (
    DEFAULT_CHUNKS_COLLECTION,
    DEFAULT_DOCUMENTS_COLLECTION,
    DEFAULT_SEARCH_CANDIDATES,
)


def parse(argv: list[str]) -> argparse.Namespace:
    return cli.build_parser().parse_args(argv)


def test_ingest_falls_back_to_the_documented_defaults() -> None:
    args = parse(["ingest", "--key", "schedule", "schedule.pdf"])

    assert args.max_tokens == ChunkingSettings().max_tokens
    assert args.overlap_tokens == ChunkingSettings().overlap_tokens
    assert args.min_chars_per_page == DEFAULT_MIN_CHARS_PER_PAGE
    assert args.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert args.embedding_dense_size == BGE_M3_DENSE_SIZE
    assert args.embedding_batch_size == DEFAULT_EMBEDDING_BATCH_SIZE
    assert args.embedding_max_length == DEFAULT_EMBEDDING_MAX_LENGTH
    assert args.embedding_fp16 is False
    assert args.chunks_collection == DEFAULT_CHUNKS_COLLECTION
    assert args.documents_collection == DEFAULT_DOCUMENTS_COLLECTION


def test_chunking_parameters_are_overridable_on_ingest() -> None:
    args = parse(
        [
            "ingest",
            "--key",
            "schedule",
            "schedule.pdf",
            "--max-tokens",
            "250",
            "--overlap-tokens",
            "25",
            "--min-chars-per-page",
            "40",
        ]
    )

    assert cli.build_chunking_settings(args) == ChunkingSettings(max_tokens=250, overlap_tokens=25)
    assert args.min_chars_per_page == pytest.approx(40)


def test_search_defaults_and_overrides_the_fusion_knobs() -> None:
    defaults = parse(["search", "which vaccines"])
    overridden = parse(["search", "which vaccines", "--limit", "3", "--candidates", "100"])

    assert defaults.limit == cli.DEFAULT_SEARCH_RESULTS
    assert defaults.candidates == DEFAULT_SEARCH_CANDIDATES
    assert overridden.limit == 3
    assert overridden.candidates == 100


def test_embedding_parameters_are_global_so_ingest_and_search_agree() -> None:
    shared = [
        "--embedding-model",
        "BAAI/bge-m3-unsupervised",
        "--embedding-dense-size",
        "768",
        "--embedding-batch-size",
        "32",
        "--embedding-max-length",
        "512",
        "--embedding-fp16",
    ]
    ingest = parse([*shared, "ingest", "--key", "schedule", "schedule.pdf"])
    search = parse([*shared, "search", "which vaccines"])

    for args in (ingest, search):
        assert args.embedding_model == "BAAI/bge-m3-unsupervised"
        assert args.embedding_dense_size == 768
        assert args.embedding_batch_size == 32
        assert args.embedding_max_length == 512
        assert args.embedding_fp16 is True


def test_collection_names_are_overridable_so_configurations_can_sit_side_by_side() -> None:
    args = parse(
        [
            "--chunks-collection",
            "chunks-250",
            "--documents-collection",
            "documents-250",
            "search",
            "which vaccines",
        ]
    )

    assert args.chunks_collection == "chunks-250"
    assert args.documents_collection == "documents-250"


def test_chunking_uses_the_defaults_for_commands_without_chunking_flags() -> None:
    args = parse(["search", "which vaccines"])

    assert cli.build_chunking_settings(args) == ChunkingSettings()


def test_the_knowledge_base_is_built_from_the_embedding_and_collection_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dense size the collection is created with has to follow the model, not a literal."""
    recorded: dict[str, object] = {}

    class FakeKnowledgeBase:
        def __init__(self, client: object, **kwargs: object) -> None:
            recorded.update(kwargs)

        def ensure_collections(self) -> None:
            recorded["ensured"] = True

    def fake_client(url: str) -> str:
        return url

    monkeypatch.setattr(cli, "QdrantClient", fake_client)
    monkeypatch.setattr(cli, "KnowledgeBase", FakeKnowledgeBase)

    cli.build_knowledge_base(
        parse(
            [
                "--embedding-dense-size",
                "768",
                "--chunks-collection",
                "chunks-768",
                "--documents-collection",
                "documents-768",
                "list",
            ]
        )
    )

    assert recorded == {
        "dense_size": 768,
        "chunks_collection": "chunks-768",
        "documents_collection": "documents-768",
        "ensured": True,
    }


def test_the_raw_files_root_is_still_read_from_the_top_level_flag() -> None:
    args = parse(["--raw-files-root", "/tmp/raw", "ingest", "--key", "k", "k.pdf"])

    assert args.raw_files_root == Path("/tmp/raw")
