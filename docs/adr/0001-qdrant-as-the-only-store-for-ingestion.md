# Qdrant is the only datastore in the ingestion pipeline

The pipeline needs both semantic and keyword search over the Knowledge Base. Qdrant supports dense
and sparse vectors as named vectors in a single collection and performs RRF fusion server-side, so
one service covers both retrieval modes. Document records (Document Key, status, content hash) live
in a payload-only Qdrant collection rather than a relational database, and raw PDFs live in object
storage so the index can always be rebuilt.

## Considered Options

**Postgres with pgvector**, using `tsvector` for keyword search, was the starting assumption. It was
rejected once BGE-M3 was chosen as the embedding model: BGE-M3 emits a learned sparse lexical
representation, which Qdrant indexes natively, whereas pgvector's `sparsevec` support is thinner and
fusion would have to be hand-written in SQL. Note that no IDF modifier is applied to the sparse
vectors: BGE-M3 learns its own term weights, so IDF is for BM25-style sparse models, not this one.

**Postgres for records plus Qdrant for search** was argued for and dropped. The argument for it was
about the system as a whole, not this pipeline. Examined at pipeline scope, Postgres bought only
transactional chunk replacement and concurrency safety, both of which are low-risk here.

## Consequences

There are no transactions. Replacing a Document's Chunks is an upsert of the new Chunk set at a new
ingestion version followed by a delete of older versions by filter. A crash between the two leaves
duplicate Chunks rather than missing ones, which is the survivable failure.

The chatbot half of the system will almost certainly introduce a relational database for
conversation logging and the statistics the institutions were promised. At that point Document
records and conversation records live in different stores, and questions like "which Document
answered which question" become application-level joins. This was accepted knowingly.
