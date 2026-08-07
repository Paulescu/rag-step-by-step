"""Producing dense and sparse embeddings.

BGE-M3 gives both from a single forward pass: a dense vector for semantic search and learned
lexical weights for keyword search. See ADR-0002 for why the model is self-hosted.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, cast

from customer_support_chatbot.ingestion.models import Embedding, SparseVector

BGE_M3_DENSE_SIZE = 1024

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_BATCH_SIZE = 8
# Chunks longer than this are truncated by the model, so it is a retrieval-quality knob and not
# only a throughput one: it has to be read together with the chunking token budget.
DEFAULT_EMBEDDING_MAX_LENGTH = 1024


class Embedder(Protocol):
    @property
    def dense_size(self) -> int: ...

    def embed(self, texts: list[str]) -> list[Embedding]: ...


class _BgeM3Model(Protocol):
    """The slice of BGEM3FlagModel this module uses.

    FlagEmbedding ships no type information, so its surface is described here and the real object
    is cast to this shape once, at construction.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = ...,
        max_length: int = ...,
        return_dense: bool = ...,
        return_sparse: bool = ...,
        return_colbert_vecs: bool = ...,
    ) -> Mapping[str, Any]:  # numpy arrays and weight dicts, narrowed by the casts below
        ...


class BgeM3Embedder(Embedder):
    """BGE-M3 loaded in-process.

    FlagEmbedding and torch are an optional dependency group, so the import is deferred to
    construction. Nothing that only chunks or stores needs a model on disk.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        use_fp16: bool = False,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH,
        dense_size: int = BGE_M3_DENSE_SIZE,
    ) -> None:
        # Imported by name rather than with a plain import: FlagEmbedding is an optional
        # dependency and ships no type information, so this keeps the untyped surface to one line.
        flag_embedding = importlib.import_module("FlagEmbedding")
        model_class = cast("Callable[..., _BgeM3Model]", flag_embedding.BGEM3FlagModel)

        self._model = model_class(model_name, use_fp16=use_fp16)
        self._batch_size = batch_size
        self._max_length = max_length
        # Follows the model: a different BGE-M3-compatible checkpoint may have a different width,
        # and the Knowledge Base collection has to be created with the same one.
        self._dense_size = dense_size

    @property
    def dense_size(self) -> int:
        return self._dense_size

    def embed(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []

        output = self._model.encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = cast("Sequence[Sequence[float]]", output["dense_vecs"])
        # Token id to learned weight, keyed by the id rendered as a string.
        lexical_weights = cast("Sequence[Mapping[str, float]]", output["lexical_weights"])

        return [
            Embedding(
                dense=[float(value) for value in dense_vectors[index]],
                sparse=_to_sparse_vector(lexical_weights[index]),
            )
            for index in range(len(texts))
        ]

    def embed_one(self, text: str) -> Embedding:
        return self.embed([text])[0]


def _to_sparse_vector(weights: Mapping[str, float]) -> SparseVector:
    return SparseVector(
        indices=[int(token_id) for token_id in weights],
        values=[float(weight) for weight in weights.values()],
    )
