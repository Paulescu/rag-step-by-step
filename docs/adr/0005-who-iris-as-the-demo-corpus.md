# WHO IRIS as the source for the demo Knowledge Base

The demo Knowledge Base is built from WHO IRIS (https://iris.who.int), the World Health
Organization's institutional repository, restricted to English publications. `kb-demo fetch`
selects 100 Documents across fourteen public health topics and records them in `demo/manifest.json`.

Retrieval questions cannot be answered on one Document. Chunk size, overlap, top-k and the fusion
candidate count are all unvalidated defaults, and none of them can be scored until there is enough
material for the wrong Chunk to be retrievable. A hundred Documents and several thousand pages is
the smallest collection where a top-5 result set is a real judgement rather than a formality.

IRIS wins on four counts, and the alternatives lose on at least one each:

- **The licence is open.** CC BY-NC-SA 3.0 IGO, so the PDFs can be redistributed and the manifest
  can be published. NICE guidance and most national health services are more restrictive.
- **The listing is machine-readable.** DSpace 7 exposes a REST API with facets for language, item
  type and year, so selection is a query rather than a scrape. MedlinePlus and most government
  health sites are HTML pages to be crawled.
- **The PDFs are born-digital.** Everything published since about 2015 has a real text layer, so
  the collection exercises retrieval rather than the quarantine gate. Older repository holdings,
  and archive collections generally, are largely scans that would need OCR we have not built.
- **The material is the right shape.** IRIS publications are guidance, handbooks and fact sheets
  that an institution publishes for the public. This is what the chatbot is meant to answer from.
  PubMed Central's open access subset is larger and equally open, but it is research papers, which
  is a different retrieval problem with different questions.

## Consequences

**Document Keys are derived, not chosen.** A Document Key is normally picked by a Knowledge Manager
(CONTEXT.md). Nobody names a hundred by hand, so the IRIS handle becomes the key: `10665/380063`
becomes `who-iris-380063`. The handle is permanent, so re-fetching produces the same keys and
therefore re-ingests the same Documents rather than duplicating them.

**The PDFs are not in the repository, the manifest is.** 100 Documents is roughly 330 MB. The
manifest records the handle, the download URL, the page count and the SHA-256 of every file, and
`kb-demo fetch --from-manifest` re-downloads exactly that list. A checkout reproduces the same
demo Knowledge Base; a Document IRIS has republished since is reported rather than silently swapped.

**The collection is not a golden set.** It is material to retrieve from, not questions with known
answers. Scoring still needs queries and judgements, which nothing here produces.

**Ingesting all of it is a long job.** Several thousand pages through BGE-M3 on CPU is hours, not
minutes. `kb-demo ingest` builds the model once and is resumable, and `--limit` exists so a
configuration can be tried on five Documents first.
