# Self-hosted BGE-M3 for embeddings

Embeddings are produced by BGE-M3 running on our own infrastructure, 1024-dimensional dense vectors
plus the model's sparse lexical output. Both are stored per Chunk.

The corpus is English for the pilot but the real customers are Montenegrin institutions, so the
embedding model has to be multilingual from day one: changing it later means re-embedding the entire
Knowledge Base and changing the vector dimension in Qdrant. BGE-M3 covers Bosnian/Croatian/Serbian
well, which most English-first embedders do not.

The sparse output is the second reason. Keyword search over Montenegrin text cannot rely on
stemming that Postgres or Qdrant do not ship for that language, so a learned sparse representation
gives a lexical signal that does not depend on the store knowing the morphology. These weights are
learned by the model, so the Qdrant sparse vectors carry no IDF modifier.

## Consequences

We operate a model-serving component. At the expected load, tens of questions per day and bulk
offline ingestion, CPU inference is sufficient and no GPU is required.

The choice of embedding model and the choice of Qdrant are coupled: BGE-M3's sparse output is what
made Qdrant's native sparse-vector support decisive. See ADR-0001.
