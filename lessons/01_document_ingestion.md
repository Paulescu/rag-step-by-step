# Lesson 1: Ingest a Document and ask a question of it

This lesson takes one Document, the vaccination and screening schedule published by the Institute
of Public Health, puts it in the Knowledge Base, and then asks a question that can only be answered
from one sentence buried inside it:

> Cervical screening is offered to women aged twenty five to sixty four every three years.

The question we will ask shares almost no words with that sentence. That is the point: if a
keyword match were enough, none of the machinery in `src/customer_support_chatbot/ingestion/`
would need to exist.

Vocabulary used here (Document, Document Key, Chunk, Knowledge Base, Quarantined) is defined in
[CONTEXT.md](../CONTEXT.md).

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

Four things happened, in this order (see `ingestion/pipeline.py`):

1. **Extraction.** `PdfPageExtractor` read the PDF's text layer, one `Page` per page. There is no
   OCR. A scan would come back empty here.
2. **The quarantine gate.** The extracted text averaged well over `--min-chars-per-page`
   (default 100), so the Document passed. Had it not, the Document would have been recorded as
   `quarantined`, would hold no Chunks, and would never be searchable. A visible failure rather
   than a silent one.
3. **Chunking.** The page was packed into Chunks up to `--max-tokens`. Chunks never cross a page
   boundary, so every hit can cite a page number.
4. **Embedding and storage.** Each Chunk was embedded by BGE-M3 into a dense vector and a sparse
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
which is also the tail of chunk 2. That is `--overlap-tokens 30` at work: a sentence that lands on
a Chunk boundary is still retrievable from the Chunk on either side.

## What the extraction actually gave us

"thre e years", "hexava lent", "two y ears". The PDF wraps lines in the middle of words and the
text layer preserves the break, so `pypdf` hands us broken tokens. The dense branch tolerates this,
which is why the query still worked, but the sparse branch cannot match a term that has been split
in half. If you were tuning this system for real, normalising extracted text would be a more
valuable change than most chunk size adjustments, and `PageExtractor` is the seam where it
belongs.

## Clean up

```bash
uv run kb --chunks-collection chunks-120 --documents-collection documents-120 \
  delete --key vaccination-schedule
uv run kb delete --key vaccination-schedule
```

```
Deleted vaccination-schedule.
```

Deletion is a hard delete: the Document and all of its Chunks go, and nothing retrievable is left
behind ([ADR-0004](../docs/adr/0004-documents-have-no-version-history.md)). The raw file under
`data/raw/vaccination-schedule/` survives, which is what makes this reversible and what a rebuild
of the index would read from.

## What this lesson did not show

- **Generating the answer.** Everything above stops at retrieval. Turning those Chunks into a
  sentence for the end-user is the chatbot, and it is not built yet.
- **Measuring whether 120 is better than 500.** We compared two configurations by reading three
  results and liking one more. That is an anecdote, not an evaluation. Every parameter that
  affects retrieval quality is a flag precisely so that a golden set can eventually score them,
  see "Tuning parameters" in the [README](../README.md).
- **Quarantine.** Try it by ingesting a scanned PDF with no text layer, or by forcing the gate
  shut with `--min-chars-per-page 100000`.
