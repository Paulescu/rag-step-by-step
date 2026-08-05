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

Configuration is via flags or environment: `QDRANT_URL` (default `http://localhost:6333`) and
`RAW_FILES_ROOT` (default `./data/raw`).

### Development

```bash
uv run pytest          # unit and in-memory Qdrant tests, no model download needed
uv run ruff check .
uv run pyright
```

Tests use a fake extractor and a deterministic hashing embedder, so the suite needs neither PDFs
nor BGE-M3. Tests that only a real server can answer live in `tests/test_knowledge_base_on_server.py`
and are skipped unless `QDRANT_URL` is set:

```bash
QDRANT_URL=http://localhost:6333 uv run pytest
```

### Not built yet

- **Retrieval evaluation.** Chunk size, overlap, top-k and the fusion candidate count are currently
  unvalidated defaults. `kb search` exists so a golden set can be scored without waiting for the
  chatbot.
- **An upload interface for the Knowledge Manager.** Ingestion is CLI-driven and run by engineers.
- **OCR**, so scanned PDFs stay quarantined.
- **Formats other than PDF.** `PageExtractor` is the seam where they would be added.
