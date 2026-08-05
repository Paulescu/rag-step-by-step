"""The Knowledge Base, backed entirely by Qdrant.

Two collections: `chunks` holds one point per Chunk with a dense and a sparse vector, `documents`
holds one payload-only point per Document. There is no relational store and no transactions, so a
Chunk set is replaced by writing the new version first and deleting older versions afterwards.
See ADR-0001.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Final

from qdrant_client import QdrantClient, models
from qdrant_client.conversions.common_types import PointId

from customer_support_chatbot.ingestion.models import (
    DocumentRecord,
    DocumentStatus,
    EmbeddedChunk,
    Embedding,
    SearchHit,
)

DENSE_VECTOR: Final = "dense"
SPARSE_VECTOR: Final = "sparse"

_POINT_NAMESPACE: Final = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def _chunk_point_id(document_key: str, ingestion_version: int, chunk_index: int) -> str:
    return str(
        uuid.uuid5(_POINT_NAMESPACE, f"chunk:{document_key}:{ingestion_version}:{chunk_index}")
    )


def _document_point_id(document_key: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"document:{document_key}"))


def _record_to_payload(
    record: DocumentRecord,
) -> dict[str, Any]:  # Qdrant payloads are untyped JSON
    return {
        "document_key": record.key,
        "title": record.title,
        "status": str(record.status),
        "content_hash": record.content_hash,
        "page_count": record.page_count,
        "chunk_count": record.chunk_count,
        "ingestion_version": record.ingestion_version,
        "ingested_at": record.ingested_at.isoformat(),
        "source_uri": record.source_uri,
        "quarantine_reason": record.quarantine_reason,
    }


def _payload_to_record(
    payload: dict[str, Any],
) -> DocumentRecord:  # Qdrant payloads are untyped JSON
    return DocumentRecord(
        key=str(payload["document_key"]),
        title=str(payload["title"]),
        status=DocumentStatus(str(payload["status"])),
        content_hash=str(payload["content_hash"]),
        page_count=int(payload["page_count"]),
        chunk_count=int(payload["chunk_count"]),
        ingestion_version=int(payload["ingestion_version"]),
        ingested_at=datetime.fromisoformat(str(payload["ingested_at"])),
        source_uri=str(payload["source_uri"]),
        quarantine_reason=(
            str(payload["quarantine_reason"]) if payload.get("quarantine_reason") else None
        ),
    )


class KnowledgeBase:
    """Everything the pipeline knows how to read from and write to Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        dense_size: int,
        chunks_collection: str = "chunks",
        documents_collection: str = "documents",
    ) -> None:
        self._client = client
        self._dense_size = dense_size
        self._chunks = chunks_collection
        self._documents = documents_collection

    def ensure_collections(self) -> None:
        """Create the collections and payload indexes if they are not there yet."""
        if not self._client.collection_exists(self._chunks):
            self._client.create_collection(
                collection_name=self._chunks,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=self._dense_size,
                        distance=models.Distance.COSINE,
                    )
                },
                # No IDF modifier: BGE-M3 learns its own term weights, unlike BM25.
                sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
            )
            self._client.create_payload_index(
                collection_name=self._chunks,
                field_name="document_key",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self._chunks,
                field_name="ingestion_version",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        if not self._client.collection_exists(self._documents):
            self._client.create_collection(
                collection_name=self._documents,
                vectors_config={},
            )
            self._client.create_payload_index(
                collection_name=self._documents,
                field_name="document_key",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def get_document(self, document_key: str) -> DocumentRecord | None:
        points = self._client.retrieve(
            collection_name=self._documents,
            ids=[_document_point_id(document_key)],
            with_payload=True,
        )
        if not points or points[0].payload is None:
            return None
        return _payload_to_record(points[0].payload)

    def list_documents(self) -> list[DocumentRecord]:
        records: list[DocumentRecord] = []
        offset: PointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._documents,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            records.extend(_payload_to_record(point.payload) for point in points if point.payload)
            if offset is None:
                break
        return sorted(records, key=lambda record: record.key)

    def upsert_document(self, record: DocumentRecord) -> None:
        self._client.upsert(
            collection_name=self._documents,
            points=[
                models.PointStruct(
                    id=_document_point_id(record.key),
                    vector={},
                    payload=_record_to_payload(record),
                )
            ],
        )

    def replace_chunks(
        self,
        document_key: str,
        ingestion_version: int,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        """Swap in a new Chunk set for a Document.

        Written new-first, deleted-old-second on purpose. Qdrant has no transactions, so a crash
        between the two steps leaves duplicate Chunks rather than a Document with none.
        """
        if embedded_chunks:
            self._client.upsert(
                collection_name=self._chunks,
                points=[
                    models.PointStruct(
                        id=_chunk_point_id(document_key, ingestion_version, embedded.chunk.index),
                        vector={
                            DENSE_VECTOR: embedded.embedding.dense,
                            SPARSE_VECTOR: models.SparseVector(
                                indices=embedded.embedding.sparse.indices,
                                values=embedded.embedding.sparse.values,
                            ),
                        },
                        payload={
                            "document_key": embedded.chunk.document_key,
                            "page_number": embedded.chunk.page_number,
                            "chunk_index": embedded.chunk.index,
                            "text": embedded.chunk.text,
                            "ingestion_version": ingestion_version,
                        },
                    )
                    for embedded in embedded_chunks
                ],
            )

        self._client.delete(
            collection_name=self._chunks,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_key",
                            match=models.MatchValue(value=document_key),
                        ),
                        models.FieldCondition(
                            key="ingestion_version",
                            range=models.Range(lt=ingestion_version),
                        ),
                    ]
                )
            ),
        )

    def delete_document(self, document_key: str) -> bool:
        """Hard-delete a Document and all its Chunks. Returns False if it was not there."""
        if self.get_document(document_key) is None:
            return False

        self._client.delete(
            collection_name=self._chunks,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_key",
                            match=models.MatchValue(value=document_key),
                        )
                    ]
                )
            ),
        )
        self._client.delete(
            collection_name=self._documents,
            points_selector=models.PointIdsList(points=[_document_point_id(document_key)]),
        )
        return True

    def count_chunks(self, document_key: str) -> int:
        result = self._client.count(
            collection_name=self._chunks,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_key",
                        match=models.MatchValue(value=document_key),
                    )
                ]
            ),
            exact=True,
        )
        return result.count

    def list_chunk_texts(self, document_key: str) -> list[str]:
        """The stored text of a Document's Chunks, in order. For inspection and debugging."""
        chunks: list[tuple[int, str]] = []
        offset: PointId | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._chunks,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_key",
                            match=models.MatchValue(value=document_key),
                        )
                    ]
                ),
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                if point.payload is None:
                    continue
                chunks.append((int(point.payload["chunk_index"]), str(point.payload["text"])))
            if offset is None:
                break
        return [text for _, text in sorted(chunks)]

    def search(self, query: Embedding, limit: int = 10, candidates: int = 20) -> list[SearchHit]:
        """Hybrid search: dense and sparse branches fused by reciprocal rank fusion in Qdrant."""
        response = self._client.query_points(
            collection_name=self._chunks,
            prefetch=[
                models.Prefetch(
                    query=query.dense,
                    using=DENSE_VECTOR,
                    limit=candidates,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=query.sparse.indices,
                        values=query.sparse.values,
                    ),
                    using=SPARSE_VECTOR,
                    limit=candidates,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload
            if payload is None:
                continue
            hits.append(
                SearchHit(
                    document_key=str(payload["document_key"]),
                    page_number=int(payload["page_number"]),
                    text=str(payload["text"]),
                    score=point.score,
                )
            )
        return hits
