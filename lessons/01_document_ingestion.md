# Lesson 1: Ingest a Document and ask a question of it

This lesson takes one Document, the vaccination and screening schedule published by the Institute
of Public Health, puts it in the Knowledge Base, and then asks a question that can only be answered
from one sentence buried inside it:

> Cervical screening is offered to women aged twenty five to sixty four every three years.

The question we will ask shares almost no words with that sentence. That is the point: if a
keyword match were enough, none of the machinery in `src/customer_support_chatbot/ingestion/`
would need to exist.

Vocabulary used here (Document, Document Key, Chunk, Knowledge Base, Quarantined) is defined in
[CONTEXT.md](../CONTEXT.md). The decisions the design rests on are in
[docs/adr/](../docs/adr/), and this lesson points at them where they explain a behaviour you can
see.

## What you need

Qdrant running:

```bash
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

The embedding model installed. This pulls torch, so it is a large install and the first `kb`
command that touches the model downloads BGE-M3 (about 2 GB):

```bash
uv sync --extra embeddings
```

And the PDF itself at `data/vaccination-schedule.pdf`. `data/` is gitignored, so the file is not in
the repository; put your copy of the schedule there.

One thing to get right before you start: **the file you ingest must live outside
`RAW_FILES_ROOT`** (default `./data/raw`). Ingestion copies the uploaded file into that directory
as its own permanent copy, and if the source is already the destination the copy fails with
`shutil.SameFileError`. Keep incoming files somewhere else, `data/` itself is fine.

## Step 1: Ingest the Document

```bash
uv run kb ingest \
  --key vaccination-schedule \
  --title "Vaccination and screening schedule" \
  data/vaccination-schedule.pdf
```

```
vaccination-schedule: ingested v1, 1 pages, 1 chunks.
```

Seven things happened, in this order (`ingestion/pipeline.py`), and the next section takes each of
them apart:

1. **Hashing.** The file's bytes were hashed, and the Knowledge Base was asked whether it already
   had this Document with that hash.
2. **Versioning.** No previous version existed, so this became ingestion version 1.
3. **Keeping the file.** The PDF was copied into `data/raw/vaccination-schedule/0001.pdf` before
   anything else happened.
4. **Extraction.** `PdfPageExtractor` read the PDF's text layer, one `Page` per page. There is no
   OCR. A scan would come back empty here.
5. **The quarantine gate.** The extracted text averaged well over `--min-chars-per-page`
   (default 100), so the Document passed. Had it not, the Document would have been recorded as
   `quarantined`, would hold no Chunks, and would never be searchable. A visible failure rather
   than a silent one.
6. **Chunking.** The page was packed into Chunks up to `--max-tokens`. Chunks never cross a page
   boundary, so every hit can cite a page number.
7. **Embedding and storage.** Each Chunk was embedded by BGE-M3 into a dense vector and a sparse
   one, and written to Qdrant along with a payload-only point describing the Document.

`--key vaccination-schedule` is the Document Key, the identity of this Document. It is chosen by
you, not derived from the filename, and it is what makes re-uploading the same schedule next
January a replacement rather than a second copy.

Check what is there:

```bash
uv run kb list
```

```
  vaccination-schedule                     v1       1p     1 chunks  Vaccination and screening schedule
```

Run the same ingest command again and nothing happens:

```
vaccination-schedule: unchanged, nothing to do.
```

The pipeline hashes the file's content, so re-ingesting identical bytes is a no-op. Use `--force`
to override that, which is what you want when you have changed a chunking parameter rather than
the file.

## Inside one ingestion

The whole of ingestion is `IngestionPipeline.ingest(document_key, path, title=None, force=False)`
in `pipeline.py:69`. It is a function of a Document Key and a file, and it returns an
`IngestResult` whose outcome is one of `ingested`, `unchanged` or `quarantined`. Those three
outcomes are exactly the three messages you can get out of `kb ingest`.

Before walking the steps, it is worth seeing the shape of the thing:

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

`IngestionPipeline` takes its collaborators by constructor injection (`pipeline.py:49`), and the
ones that touch the outside world are Protocols: `PageExtractor`, `Embedder`, `RawFileStore`. That
is what lets the test suite run with a fake extractor and a deterministic hashing embedder, needing
neither a PDF nor a 2 GB model download.

### 1. Hash the file and decide whether there is work to do

```python
content_hash = file_content_hash(path)
existing = self._knowledge_base.get_document(document_key)
if existing is not None and existing.content_hash == content_hash and not force:
    return IngestResult(outcome=IngestOutcome.UNCHANGED, document=existing)
```

`file_content_hash` (`pipeline.py:41`) is a streaming SHA-256, read in 1 MB blocks so a large PDF is
never held in memory. The hash is stored on the Document record, which is what produced the
`unchanged, nothing to do.` you saw above: no extraction, no embedding, no writes. This is what
makes ingestion safe to retry and cheap to run over a directory of mostly unchanged files.

Note what identity is anchored to: the Document Key, not the filename. Two uploads under the same
key are the same Document, however the file was named.

### 2. Allocate an ingestion version and keep the raw file

```python
version = existing.ingestion_version + 1 if existing is not None else 1
source_uri = self._raw_files.store(document_key, version, path)
```

The version is a monotonic counter per Document, starting at 1, which is the `v1` in the `kb list`
output. It is not a history mechanism
([ADR-0004](../docs/adr/0004-documents-have-no-version-history.md) says Documents have no version
history); it is the mechanism that makes Chunk replacement survivable, and step 7 explains how.

`LocalRawFileStore.store` (`raw_files.py:26`) copies the file to
`<root>/<document_key>/<version:04d><suffix>` and returns a `file://` URI. The root defaults to
`./data/raw` and is resolved to an absolute path on construction, because the stored URI has to be
absolute while callers pass relative paths.

**The ordering here is deliberate.** The raw file is written before anything is extracted, chunked
or indexed, so the file always outlives a failed ingestion. Since the Knowledge Base keeps no
history, the retained file is what makes a deletion reversible by re-upload, an ingestion auditable,
and a full rebuild of the index possible.

This is also the step that fails with `shutil.SameFileError` if you ingest a file that already
lives inside the raw files root.

### 3. Extract pages

```python
pages = self._extractor.extract(path)
```

`PdfPageExtractor` (`extraction.py:23`) reads the PDF's text layer with `pypdf` and returns one
`Page` per page, numbered from 1 as a reader would count them. There is no OCR, by design: a scan
comes back with empty text rather than with a plausible-looking transcription.

Everything downstream depends on the `PageExtractor` Protocol rather than on `pypdf`, so supporting
a second format is a new implementation of that Protocol and not a change to the pipeline. It is
also where the broken words you will see further down would be fixed.

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

You can watch this happen at the end of the lesson.

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

This is the step Step 3 of this lesson tunes, and the reason our first attempt returned a single
Chunk.

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
vector width in Qdrant ([ADR-0002](../docs/adr/0002-bge-m3-self-hosted-embeddings.md)). Its sparse
output is also what makes keyword search work without the store knowing the language's morphology,
which is why no IDF modifier is applied: these weights are learned, unlike BM25's.

`FlagEmbedding` and torch are an optional dependency group and the import is deferred to
construction, so nothing that only chunks or stores needs a model on disk. That is why `uv sync
--extra embeddings` is only needed for the commands that touch the model.

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

There are no transactions in Qdrant
([ADR-0001](../docs/adr/0001-qdrant-as-the-only-store-for-ingestion.md)), so this order is the whole
safety argument: a crash between the two steps leaves duplicate Chunks, which degrades ranking,
rather than a Document with no Chunks at all, which silently loses an answer. New-first,
delete-old-second picks the survivable failure.

The Document record is written last, so a partially indexed Document is never advertised as live.

## What is now stored, and where

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
  "page_number": 1,
  "chunk_index": 0,
  "text": "...",
  "ingestion_version": 1
}
```

Payload indexes on `document_key` (keyword) and `ingestion_version` (integer). These exist for the
filtered delete in `replace_chunks` and for `delete_document`; without them those filters are a full
scan. Note that local in-memory Qdrant ignores payload indexes entirely, which is why
`tests/test_knowledge_base_on_server.py` exists and is skipped unless `QDRANT_URL` is set.

Point IDs are deterministic: `uuid5(namespace, "chunk:<key>:<version>:<index>")` (`store.py:41`).
Deterministic IDs make an interrupted upsert idempotent on retry, since re-writing the same Chunk
overwrites its point rather than adding a second one.

### Qdrant, `documents` collection

One payload-only point per Document, created with `vectors_config={}`. This collection is never
searched by vector; it is the pipeline's record of what it knows about each Document, and ADR-0001
is the decision to keep it here rather than in a relational database.

Payload is the serialised `DocumentRecord` (`store.py:51`), and `kb list` is a formatted view of it:

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

Outside Qdrant, on the filesystem under `./data/raw/<document_key>/<version>.pdf` by default. Our
ingestion put a copy at `data/raw/vaccination-schedule/0001.pdf`. It is addressed through the
`RawFileStore` Protocol, so the local implementation is a stand-in for object storage, which is
where ADR-0001 puts it in production.

Unlike the Knowledge Base, this store does accumulate versions. That is intentional: it is the audit
trail and the rebuild source that ADR-0004 relies on when it refuses to keep old Chunks retrievable.

## Step 2: Ask a question

There is no chatbot yet, so we ask the Knowledge Base directly. `kb search` is the retrieval half
of the eventual answer: it returns the Chunks that would be handed to a model as context.

```bash
uv run kb search "I am a woman of thirty. How often should I be invited for a smear test?"
```

```
1. vaccination-schedule p.1  (score 1.0000)
   VACCINATION SCHEDULE FOR CHILDREN  Institute of Public Health. Effective from January 2026.  Children must receive the BCG vaccine within the first month of life. The hexava lent vaccine is given at two, four and eleven months of age. Parents should bring the child health booklet
```

The right Document came back, and the answer *is* inside it, but the result is not useful. The
whole page is a single Chunk, so retrieval had exactly one thing it could return and told us
nothing about *where* in the page to look. The snippet shown is about the BCG vaccine. A model
given this context would have to find the screening sentence itself, and everything else on the
page competes for its attention.

The default `--max-tokens 500` is the problem. This page collapses to roughly 262 tokens of actual
text, so the whole thing fits in one Chunk.

### What the query did

`IngestionPipeline.search` (`pipeline.py:127`) closes the loop, and is the reason the CLI has a
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
`page_number`, which is what lets the output above cite `p.1`.

This is also why the embedding and collection flags sit before the subcommand in the CLI
(`cli.py:63`): a query must be embedded by the same model, at the same width, and read from the same
collections the Chunks were written to. That constraint is what the next step exploits.

## Step 3: Chunk smaller, and ask again

Rather than overwriting what we have, ingest into a second pair of collections so the two
configurations can be compared side by side:

```bash
uv run kb --chunks-collection chunks-120 --documents-collection documents-120 \
  ingest --key vaccination-schedule --title "Vaccination and screening schedule" \
  --max-tokens 120 --overlap-tokens 30 \
  data/vaccination-schedule.pdf
```

```
vaccination-schedule: ingested v1, 1 pages, 4 chunks.
```

Four Chunks now, which is what you would hope for: the page has four sections (the header, the
childhood vaccines, sanitary booklets, and screening programmes), and chunking packs paragraph by
paragraph, so the boundaries land close to the section boundaries.

The collection flags come *before* the subcommand because they apply to searching as much as to
ingesting. Search the collection you ingested into or you will get nothing back.

Now the same question:

```bash
uv run kb --chunks-collection chunks-120 --documents-collection documents-120 \
  search "I am a woman of thirty. How often should I be invited for a smear test?" --limit 3
```

```
1. vaccination-schedule p.1  (score 1.0000)
   SCREENING PROGRAMMES  Cervical screening is offered to women aged twenty five to sixty four every thre e years. Colorectal screening is offered to adults aged fifty to seventy four every two y ears. Invitations are sent by post. Screening appointments are free of charge.
2. vaccination-schedule p.1  (score 0.5333)
   VACCINATION SCHEDULE FOR CHILDREN  Institute of Public Health. Effective from January 2026.  Children must receive the BCG vaccine within the first month of life. The hexava lent vaccine is given at two, four and eleven months of age. Parents should bring the child health booklet
3. vaccination-schedule p.1  (score 0.5333)
   vaccine is given at two, four and eleven months of age. Parents should bring the child health booklet to every appointment.  Measles, mumps and rubella vaccination is given at twelve months, with a second dose before the child starts primary school. A child who has missed a dose
```

The screening Chunk is now first, and it carries the sentence we were after. A model handed this
context can answer: **every three years, because cervical screening is offered to women aged
twenty five to sixty four at that interval, and thirty falls in that range.** The Chunk also
supports the follow-up questions the user did not ask, that invitations arrive by post and that
the appointment is free.

Note that the question contained neither "cervical" nor "screening" nor "three years". "Smear
test" matched because the dense BGE-M3 vector places it near cervical screening in meaning; a
purely lexical search would have ranked the childhood vaccination Chunks above it. Hybrid search
runs both branches and fuses them with reciprocal rank fusion inside Qdrant, which is why the
scores look like `1.0000` and `0.5333` rather than cosine similarities: they are fusion scores
derived from rank, not distance, and they are only meaningful relative to each other within one
result set.

Look at chunk 3 as well. It starts with "vaccine is given at two, four and eleven months of age",
which is also the tail of chunk 2. That is `--overlap-tokens 30` at work, the `_tail_within_budget`
carry-over from step 5: a sentence that lands on a Chunk boundary is still retrievable from the
Chunk on either side.

## What the extraction actually gave us

"thre e years", "hexava lent", "two y ears". The PDF wraps lines in the middle of words and the
text layer preserves the break, so `pypdf` hands us broken tokens. The dense branch tolerates this,
which is why the query still worked, but the sparse branch cannot match a term that has been split
in half. If you were tuning this system for real, normalising extracted text would be a more
valuable change than most chunk size adjustments, and `PageExtractor` is the seam where it
belongs.

## When ingestion goes wrong

Try the quarantine gate by forcing it shut on a Document you know is fine:

```bash
uv run kb ingest --key vaccination-schedule --min-chars-per-page 100000 --force \
  data/vaccination-schedule.pdf
```

The Document comes back `QUARANTINED, not searchable.`, followed by the reason: the measured
characters per page, the threshold it failed, and the suggestion that the file is probably a scan
and needs OCR. That goes to stderr with exit code 2, and `kb list` now marks the Document with a
`!`. Its Chunks are gone, because `_quarantine` replaced them with an empty set at the new version.
Re-ingest with `--force` and the default threshold to get it back.

The other failure modes and what each one costs:

| Failure | Result |
| --- | --- |
| File missing | `FileNotFoundError` before anything is written. |
| Crash after the raw file is stored | Nothing indexed, raw file retained, next run redoes the work. |
| Text layer too thin | Document recorded as `quarantined`, no Chunks, exit code 2, reason reported. |
| Crash between the Chunk upsert and the delete of older versions | Duplicate Chunks, not missing ones. Fixed by re-running ingestion with `--force`. |
| Crash between the Chunk write and the Document upsert | Chunks are searchable but the record still shows the previous version. Re-running reconciles it. |
| Embedding model changed | Requires re-ingesting everything: the vector width is fixed at collection creation. |

## Clean up

```bash
uv run kb --chunks-collection chunks-120 --documents-collection documents-120 \
  delete --key vaccination-schedule
uv run kb delete --key vaccination-schedule
```

```
Deleted vaccination-schedule.
```

`delete_document` (`store.py:230`) returns `False` if the Document is not there, which is the
`No Document with key ...` message and exit code 1. Otherwise it deletes every Chunk with that
`document_key` by filter, then deletes the Document point.

Deletion is a hard delete: the Document and all of its Chunks go, and nothing retrievable is left
behind ([ADR-0004](../docs/adr/0004-documents-have-no-version-history.md)). For a chatbot serving
public institutions, a confidently cited obsolete document is worse than no answer, and every
soft-delete scheme works only if every query path remembers to filter it out. The raw file under
`data/raw/vaccination-schedule/` survives, which is what makes this reversible and what a rebuild
of the index would read from.

## What this lesson did not show

- **Generating the answer.** Everything above stops at retrieval. Turning those Chunks into a
  sentence for the end-user is the chatbot, and it is not built yet.
- **Measuring whether 120 is better than 500.** We compared two configurations by reading three
  results and liking one more. That is an anecdote, not an evaluation. Every parameter that
  affects retrieval quality is a flag precisely so that a golden set can eventually score them,
  see "Tuning parameters" in the [README](../README.md).
- **OCR**, so a real scan stays quarantined rather than being read.
- **Formats other than PDF.** `PageExtractor` is the seam where they would be added.
- **Multiple institutions.** There is no tenant identifier anywhere in the schema, because each
  institution gets its own deployment
  ([ADR-0003](../docs/adr/0003-one-deployment-per-institution.md)).
- **Unattended ingestion.** It runs on demand, driven by an engineer at a terminal. An upload
  interface for the Knowledge Manager is still to be built.
