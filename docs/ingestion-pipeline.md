# How the ingestion pipeline works

This document walks through one ingestion from the outside in: what happens to a file between the
moment it is handed to the pipeline and the moment a question can be answered from it, what is
written where, and why each step is ordered the way it is.

Vocabulary (Document, Document Key, Chunk, Knowledge Base, Quarantined) is defined in
[CONTEXT.md](../CONTEXT.md). The decisions this design rests on are in [docs/adr/](./adr/). The
README has the same picture at a glance; this document is the long version.

## The shape of it

```
                                   ┌──────────────────┐
file ──┬──────────────────────────►│  RawFileStore    │  data/raw/<key>/<version>.pdf
       │                           └──────────────────┘
       │
       ▼
  PdfPageExtractor ──► quarantine gate ──► chunk_pages ──► BgeM3Embedder ──┐
   list[Page]           usable?             list[Chunk]     dense+sparse   │
                            │                                             ▼
                            │                                   ┌──────────────────┐
                            └──────────────────────────────────► │  KnowledgeBase  │  Qdrant
                              status=quarantined, zero Chunks    └──────────────────┘
```

Every box is a module in `src/customer_support_chatbot/ingestion/`:

| Module | Responsibility |
| --- | --- |
| `pipeline.py` | Orchestration. The only place that knows the order of the steps. |
| `extraction.py` | File to pages of text, and the judgement of whether that text is usable. |
| `chunking.py` | Pages to Chunks. |
| `embedding.py` | Chunk text to dense and sparse vectors. |
| `store.py` | The Knowledge Base: everything read from and written to Qdrant. |
| `raw_files.py` | Keeping the uploaded file itself. |
| `models.py` | The types that pass between them. |
| `cli.py` | A thin shell so ingestion can be driven by hand. |

`IngestionPipeline` takes its collaborators by constructor injection
(`pipeline.py:49`), and the ones that touch the outside world are Protocols: `PageExtractor`,
`Embedder`, `RawFileStore`. That is what lets the test suite run with a fake extractor and a
deterministic hashing embedder, needing neither a PDF nor a 2 GB model download.

## Step by step

The whole of ingestion is `IngestionPipeline.ingest(document_key, path, title=None, force=False)`
in `pipeline.py:69`. It is a function of a Document Key and a file, and it returns an
`IngestResult` whose outcome is one of `ingested`, `unchanged` or `quarantined`.

### 1. Hash the file and decide whether there is work to do

```python
content_hash = file_content_hash(path)
existing = self._knowledge_base.get_document(document_key)
if existing is not None and existing.content_hash == content_hash and not force:
    return IngestResult(outcome=IngestOutcome.UNCHANGED, document=existing)
```

`file_content_hash` (`pipeline.py:41`) is a streaming SHA-256, read in 1 MB blocks so a large PDF is
never held in memory. The hash is stored on the Document record, so re-running ingestion with a
byte-identical file is a no-op: no extraction, no embedding, no writes. This is what makes ingestion
safe to retry and cheap to run over a directory of mostly unchanged files. `--force` overrides it,
which is how a re-ingestion under changed chunking parameters is triggered without editing the file.

Note what identity is anchored to: the Document Key, not the filename. Two uploads under the same
key are the same Document, however the file was named.

### 2. Allocate an ingestion version and keep the raw file

```python
version = existing.ingestion_version + 1 if existing is not None else 1
source_uri = self._raw_files.store(document_key, version, path)
```

The version is a monotonic counter per Document, starting at 1. It is not a history mechanism
(ADR-0004 says Documents have no version history); it is the mechanism that makes Chunk replacement
survivable, and step 6 explains how.

`LocalRawFileStore.store` (`raw_files.py:26`) copies the file to
`<root>/<document_key>/<version:04d><suffix>` and returns a `file://` URI. The root defaults to
`./data/raw` and is resolved to an absolute path on construction, because the stored URI has to be
absolute while callers pass relative paths.

**The ordering here is deliberate.** The raw file is written before anything is extracted, chunked
or indexed, so the file always outlives a failed ingestion. Since the Knowledge Base keeps no
history, the retained file is what makes a deletion reversible by re-upload, an ingestion auditable,
and a full rebuild of the index possible.

One practical consequence: the file you ingest must not already live inside the raw files root, or
the copy fails with `shutil.SameFileError`.

### 3. Extract pages

```python
pages = self._extractor.extract(path)
```

`PdfPageExtractor` (`extraction.py:23`) reads the PDF's text layer with `pypdf` and returns one
`Page` per page, numbered from 1 as a reader would count them. There is no OCR, by design: a scan
comes back with empty text rather than with a plausible-looking transcription.

Everything downstream depends on the `PageExtractor` Protocol rather than on `pypdf`, so supporting
a second format is a new implementation of that Protocol and not a change to the pipeline.

### 4. The quarantine gate

```python
reason = quarantine_reason(pages, self._min_chars_per_page)
if reason is not None:
    return self._quarantine(...)
```

`quarantine_reason` (`extraction.py:34`) rejects a Document in two cases: no pages could be read at
all, or the average is below `--min-chars-per-page` (default 100) characters per page. The second
case is almost always a scan.

The alternative would be to ingest it anyway, which puts empty or near-empty Chunks in the Knowledge
Base and makes the failure invisible. Quarantine makes it visible: the Knowledge Manager is the
person responsible for what is in the Knowledge Base, so an unusable upload has to reach them.

What `_quarantine` (`pipeline.py:136`) does is worth reading closely, because it is not just an
early return:

```python
self._knowledge_base.replace_chunks(document_key, version, [])
record = DocumentRecord(..., status=DocumentStatus.QUARANTINED, chunk_count=0, ...)
self._knowledge_base.upsert_document(record)
```

It replaces the Chunk set with an empty one at the new version. That matters when a good Document is
re-uploaded as a bad scan: without this call, the previous version's Chunks would stay searchable
under a Document the operator has been told is quarantined. A quarantined Document holds no Chunks
and is never searchable, and the CLI exits with status 2 and prints the reason to stderr.

### 5. Chunk the pages

```python
chunks = chunk_pages(pages, document_key, self._chunking, self._count_tokens)
```

`chunk_pages` (`chunking.py:43`) walks pages in order and produces `Chunk`s numbered consecutively
across the whole Document. Two rules govern it:

**Chunks never cross a page boundary.** Chunking runs per page, which is what lets every search hit
cite the page it came from. The cost is that a paragraph split across a page break becomes two
Chunks.

**Within a page, text is packed paragraph by paragraph up to a token budget.** `_split_into_units`
splits on blank lines, normalises whitespace, and cuts any paragraph that exceeds the budget on its
own at word boundaries (`_split_long_paragraph`). Units are then packed into Chunks up to
`--max-tokens` (default 500). When a Chunk closes, the longest trailing run of words that fits
`--overlap-tokens` (default 60) is carried into the next one (`_tail_within_budget`), so an answer
sitting on a chunk boundary is still retrievable from at least one side.

Token counting is injected as a `TokenCounter`. The default, `approximate_token_count`, is
`len(text) // 4`, the usual four-characters-per-token approximation. It only decides where to split,
so being a little off costs chunk size, not correctness.

### 6. Embed

```python
embeddings = self._embedder.embed([chunk.text for chunk in chunks])
embedded_chunks = [EmbeddedChunk(chunk=chunk, embedding=embedding)
                   for chunk, embedding in zip(chunks, embeddings, strict=True)]
```

`BgeM3Embedder` (`embedding.py:50`) runs BGE-M3 in-process and returns, per text, both
representations from a single forward pass:

- a **dense** vector of 1024 floats, for semantic similarity, and
- a **sparse** vector, the model's learned lexical weights, mapped from token id to weight.

Both are stored per Chunk. BGE-M3 was chosen because the real corpus will be Montenegrin and
changing the embedding model later means re-embedding the whole Knowledge Base and changing the
vector width in Qdrant (ADR-0002). Its sparse output is also what makes keyword search work without
the store knowing the language's morphology, which is why no IDF modifier is applied: these weights
are learned, unlike BM25's.

`FlagEmbedding` and torch are an optional dependency group and the import is deferred to
construction, so nothing that only chunks or stores needs a model on disk.

`zip(..., strict=True)` is the guard that a Chunk and its vector cannot drift apart.

Note that `--embedding-max-length` (default 1024 tokens) is where the model truncates. It has to be
read together with `--max-tokens`: a chunking budget above the embedding budget silently discards
the tail of every long Chunk.

### 7. Write the Chunks, then the Document record

```python
self._knowledge_base.replace_chunks(document_key, version, embedded_chunks)
record = DocumentRecord(..., status=DocumentStatus.LIVE, ...)
self._knowledge_base.upsert_document(record)
```

`replace_chunks` (`store.py:176`) upserts the new Chunk set at the new ingestion version **first**,
then deletes every Chunk of that Document at a lower version by filter.

There are no transactions in Qdrant (ADR-0001), so this order is the whole safety argument: a crash
between the two steps leaves duplicate Chunks, which degrades ranking, rather than a Document with
no Chunks at all, which silently loses an answer. New-first, delete-old-second picks the survivable
failure.

The Document record is written last, so a partially indexed Document is never advertised as live.

## What is stored, and where

### Qdrant, `chunks` collection

One point per Chunk. Created in `ensure_collections` (`store.py:103`) with two named vectors:

| Vector | Config |
| --- | --- |
| `dense` | `VectorParams(size=1024, distance=COSINE)` |
| `sparse` | `SparseVectorParams()`, no IDF modifier |

Payload per point:

```json
{
  "document_key": "vaccination-schedule",
  "page_number": 3,
  "chunk_index": 17,
  "text": "...",
  "ingestion_version": 2
}
```

Payload indexes on `document_key` (keyword) and `ingestion_version` (integer). These exist for the
filtered delete in `replace_chunks` and for `delete_document`; without them those filters are a full
scan. Note that local in-memory Qdrant ignores payload indexes entirely, which is why
`tests/test_knowledge_base_on_server.py` exists and is skipped unless `QDRANT_URL` is set.

Point IDs are deterministic: `uuid5(namespace, "chunk:<key>:<version>:<index>")`
(`store.py:41`). Deterministic IDs make an interrupted upsert idempotent on retry, since re-writing
the same Chunk overwrites its point rather than adding a second one.

### Qdrant, `documents` collection

One payload-only point per Document, created with `vectors_config={}`. This collection is never
searched by vector; it is the pipeline's record of what it knows about each Document, and ADR-0001
is the decision to keep it here rather than in a relational store.

Payload is the serialised `DocumentRecord` (`store.py:51`):

| Field | Purpose |
| --- | --- |
| `document_key` | The stable identity. |
| `title` | Human-readable, defaults to the file stem. |
| `status` | `live` or `quarantined`. |
| `content_hash` | SHA-256 of the uploaded file. Drives the unchanged short-circuit. |
| `page_count`, `chunk_count` | What the last ingestion produced. |
| `ingestion_version` | The counter that Chunk replacement is keyed on. |
| `ingested_at` | UTC timestamp. |
| `source_uri` | Where the raw file was kept. |
| `quarantine_reason` | Populated only when quarantined, and shown to the operator. |

The point ID is `uuid5(namespace, "document:<key>")`, so a Document Key maps to exactly one point
and `get_document` is a direct retrieve by ID rather than a search.

### The raw file store

Outside Qdrant, on the filesystem under `./data/raw/<document_key>/<version>.pdf` by default. It is
addressed through the `RawFileStore` Protocol, so the local implementation is a stand-in for object
storage, which is where ADR-0001 puts it in production.

Unlike the Knowledge Base, this store does accumulate versions. That is intentional: it is the
audit trail and the rebuild source that ADR-0004 relies on when it refuses to keep old Chunks
retrievable.

## Deletion

`delete_document` (`store.py:230`) returns `False` if the Document is not there, and otherwise
deletes every Chunk with that `document_key` by filter, then deletes the Document point.

It is a hard delete. Nothing is soft-deleted, no previous version stays retrievable, and the reason
is in ADR-0004: for a chatbot serving public institutions, a confidently cited obsolete document is
worse than no answer, and every soft-delete scheme works only if every query path remembers to
filter it out. The raw files are not touched, so a deletion is reversible by re-upload.

## The read path

`IngestionPipeline.search` (`pipeline.py:127`) closes the loop and is the reason the CLI has a
`search` command at all: retrieval quality can be measured before the chatbot exists.

```python
embedding = self._embedder.embed([query])[0]
return self._knowledge_base.search(embedding, limit=limit, candidates=candidates)
```

The query goes through the same model that embedded the Chunks, producing the same dense and sparse
pair. `KnowledgeBase.search` (`store.py:296`) then issues one `query_points` call with two prefetch
branches, dense and sparse, each contributing `--candidates` (default 20) Chunks, fused by
reciprocal rank fusion inside Qdrant and cut to `--limit`.

Fusion happens server-side, in one round trip. Each hit carries its `document_key` and
`page_number`, so an answer built on it can cite a page.

This is also why the embedding and collection flags sit before the subcommand in the CLI
(`cli.py:63`): a query must be embedded by the same model, at the same width, and read from the same
collections the Chunks were written to. Overriding both collection names is how two chunking
configurations are indexed side by side and compared.

## Failure modes and what they cost

| Failure | Result |
| --- | --- |
| File missing | `FileNotFoundError` before anything is written. |
| Crash after the raw file is stored | Nothing indexed, raw file retained, next run redoes the work. |
| Text layer too thin | Document recorded as `quarantined`, no Chunks, exit code 2, reason reported. |
| Crash between the Chunk upsert and the delete of older versions | Duplicate Chunks, not missing ones. Fixed by re-running ingestion with `--force`. |
| Crash between the Chunk write and the Document upsert | Chunks are searchable but the record still shows the previous version. Re-running reconciles it. |
| Embedding model changed | Requires re-ingesting everything: the vector width is fixed at collection creation. |

## What this pipeline does not do

No OCR, so scans stay quarantined. No format other than PDF, and `PageExtractor` is the seam where
another would go. No tenant identifier anywhere, because each institution gets its own deployment
(ADR-0003). No scheduled or watched-directory ingestion: it runs on demand, engineer-driven, and an
upload interface for the Knowledge Manager is still to be built.
