# Customer support chatbot

Components of the system:

- A document ingestion pipeline, that runs on demand. It pushes new document pdfs into the knowledge database.
- A real-time chatbot that receives incoming messages from external end-users and tries to answer them faithfully based on the data in the knowledge database.

Domain vocabulary lives in [CONTEXT.md](./CONTEXT.md). The decisions behind the design, and the
alternatives that were rejected, are in [docs/adr/](./docs/adr/).

## Document ingestion pipeline

Lives in `src/customer_support_chatbot/ingestion/`. A PDF goes in, searchable Chunks come out.

```
PDF ──► PdfPageExtractor ──► quarantine gate ──► chunk_pages ──► BgeM3Embedder ──► Qdrant
        (text layer only)    (scans rejected)    (page-aware)    (dense + sparse)
```

Everything is stored in Qdrant and nothing else (ADR-0001):

- **`chunks`** — one point per Chunk, with a `dense` vector (BGE-M3, 1024 dimensions) and a
  `sparse` vector (BGE-M3 lexical weights). Search runs both branches and fuses them with
  reciprocal rank fusion inside Qdrant.
- **`documents`** — one payload-only point per Document: Document Key, title, status, content
  hash, version.

Raw uploaded files are kept outside Qdrant, so the index can always be rebuilt and a deletion is
reversible.

### Behaviour worth knowing

- **A Document is identified by its Document Key**, chosen by the Knowledge Manager. Re-uploading
  under the same key replaces that Document's Chunks. Identical content is a no-op.
- **There is no version history.** A replaced or deleted Document leaves nothing retrievable
  behind, because a confidently cited obsolete policy is the failure mode that matters (ADR-0004).
- **Scans are quarantined, not ingested.** There is no OCR. A PDF whose text layer is too thin is
  recorded with status `quarantined` and reported, and it holds no Chunks.
- **Chunks never cross a page boundary**, so every hit can cite a page.

### Usage

Start Qdrant:

```bash
docker run -p 6333:6333 -v "$(pwd)/data/qdrant:/qdrant/storage" qdrant/qdrant
```

Install with the embedding model (pulls torch):

```bash
uv sync --extra embeddings
```

Then:

```bash
uv run kb ingest --key vaccination-schedule path/to/schedule.pdf
uv run kb list
uv run kb search "which vaccines are required before school"
uv run kb delete --key vaccination-schedule
```

Configuration is via flags or environment: `QDRANT_URL` (default `http://localhost:6333`),
`RAW_FILES_ROOT` (default `./data/raw`), `CHUNKS_COLLECTION` (default `chunks`),
`DOCUMENTS_COLLECTION` (default `documents`) and `EMBEDDING_MODEL` (default `BAAI/bge-m3`).

### Tuning parameters

Every parameter that affects retrieval quality is a flag, so a configuration can be swept without
editing code. Defaults are unvalidated, see "Not built yet" below.

| Flag | Default | What it does |
| --- | --- | --- |
| `--max-tokens` | `500` | Chunk size: the token budget a Chunk is packed up to. |
| `--overlap-tokens` | `60` | Tokens carried from the end of one Chunk into the next. |
| `--min-chars-per-page` | `100` | Below this, the Document is quarantined as a probable scan. |
| `--limit` | `5` | How many Chunks a search returns (top-k). |
| `--candidates` | `20` | Chunks each of the dense and sparse branches contributes before fusion. |
| `--embedding-max-length` | `1024` | Token budget per text. Longer texts are truncated by the model. |
| `--embedding-model` | `BAAI/bge-m3` | Which model produces the embeddings. |
| `--embedding-dense-size` | `1024` | Width of the dense vector. Must match the collection. |
| `--embedding-batch-size` | `8` | Texts embedded per forward pass. Throughput only. |
| `--embedding-fp16` | off | Load the model in half precision. |

`--max-tokens`, `--overlap-tokens` and `--min-chars-per-page` are `ingest` flags; `--limit` and
`--candidates` are `search` flags. The embedding and collection flags come before the subcommand,
because a query has to be embedded by the same model that embedded the Chunks and read from the
same collections they were written to.

A collection pair is bound to one configuration, so two of them can be compared side by side:

```bash
uv run kb --chunks-collection chunks-250 --documents-collection documents-250 \
  ingest --key vaccination-schedule --max-tokens 250 --overlap-tokens 30 path/to/schedule.pdf
uv run kb --chunks-collection chunks-250 --documents-collection documents-250 \
  search "which vaccines are required before school" --limit 5 --candidates 50
```

### Development

```bash
uv run pytest          # unit and in-memory Qdrant tests, no model download needed
uv run ruff check .
uv run pyright
```

Tests use a fake extractor, a deterministic hashing embedder and a fake IRIS serving DSpace-shaped
JSON from a dictionary, so the suite needs no PDFs, no BGE-M3 and no network.

#### Running against a real Qdrant server

Three tests in `tests/test_knowledge_base_on_server.py` cover behaviour only a real server has:
payload indexes, server-side filtering, and reciprocal rank fusion. Local in-memory Qdrant ignores
payload indexes entirely, so these would pass vacuously there. They are skipped unless `QDRANT_URL`
is set, which keeps the default run hermetic.

Start a server, then point the suite at it:

```bash
docker run -p 6333:6333 qdrant/qdrant
QDRANT_URL=http://localhost:6333 uv run pytest
```

The server tests create their own collections named `test_chunks_<uuid>` and
`test_documents_<uuid>` and delete them on teardown, so they never touch the `chunks` and
`documents` collections your own data lives in. A throwaway container without the `-v` mount is
fine, no volume is needed.

| Command | Tests run |
| --- | --- |
| `uv run pytest` | 90 run, 3 skipped |
| `QDRANT_URL=http://localhost:6333 uv run pytest` | 93 run |

Note that `QDRANT_URL` is also the CLI's own configuration variable. If you have it exported in
your shell for `uv run kb`, a plain `uv run pytest` will pick it up and run the server tests too.
To force the hermetic run regardless of environment, use `env -u QDRANT_URL uv run pytest`.

### Not built yet

- **Retrieval evaluation.** Chunk size, overlap, top-k and the fusion candidate count are currently
  unvalidated defaults. They are all exposed as flags (see "Tuning parameters"), but nothing scores
  them yet. `kb search` exists so a golden set can be scored without waiting for the chatbot, and
  `kb-demo` now supplies a Knowledge Base big enough for the scores to mean something. What is
  still missing is the golden set itself: queries with known correct Documents and pages.
- **An upload interface for the Knowledge Manager.** Ingestion is CLI-driven and run by engineers.
- **OCR**, so scanned PDFs stay quarantined.
- **Formats other than PDF.** `PageExtractor` is the seam where they would be added.

## The demo Knowledge Base

One Document proves the pipeline runs. It cannot tell you whether a Chunk size is right, because
there is nothing else for a query to wrongly retrieve. `kb-demo` fills a Knowledge Base with real
institutional documentation, so that retrieval has something to get wrong.

The source is [WHO IRIS](https://iris.who.int), the World Health Organization's institutional
repository: English publications, CC BY-NC-SA 3.0 IGO, a DSpace REST API to select them with, and
born-digital PDFs with real text layers. Why this collection and not another is
[ADR-0005](./docs/adr/0005-who-iris-as-the-demo-corpus.md).

`demo/manifest.json` records the collection that was built. It is committed; the PDFs it points at
are not.

| | |
| --- | --- |
| Documents | 100 |
| Pages | 8,376 |
| Size | 337 MB |
| Topics | 14, between 6 and 8 Documents each |
| Licence | CC BY-NC-SA 3.0 IGO |

```bash
uv run kb-demo show                    # what the committed manifest holds
uv run kb-demo fetch --from-manifest   # download exactly that, about fifteen minutes
uv run kb-demo ingest --limit 5        # try a configuration on five Documents first
uv run kb-demo ingest                  # the whole collection, hours on CPU
```

`fetch` needs neither Qdrant nor the model; `ingest` needs both. Both are resumable: a file already
on disk is not downloaded again, and re-ingesting an unchanged Document is a no-op, so an
interrupted run of either is restarted by running it again.

To build a different collection instead of replaying the committed one, drop `--from-manifest`:

```bash
uv run kb-demo fetch --documents 40 --pages 2000 \
  --topic "cholera" --topic "dengue" --issued-from 2020
```

Selection takes one publication from each topic in turn, so stopping early still leaves every topic
represented. Every candidate is put through `quarantine_reason`, the pipeline's own scan gate,
before it is accepted, so nothing in the manifest lands in the Knowledge Base as Quarantined.
Documents outside `--min-pdf-pages` and `--max-pdf-pages` (4 and 250) are skipped: a one-page flyer
teaches nothing about chunking, and a 400-page compendium costs an hour of embedding on its own.

Document Keys are derived from the IRIS handle, so `10665/380063` becomes `who-iris-380063` and a
re-fetch re-ingests rather than duplicates.
