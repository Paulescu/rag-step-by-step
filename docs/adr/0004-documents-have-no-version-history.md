# Documents have no version history

A Document has exactly one live set of Chunks. Re-uploading under the same Document Key replaces
that set. Removing a Document hard-deletes its record and its Chunks. Nothing is soft-deleted and no
previous version stays retrievable.

For a support chatbot serving public institutions, a confidently cited obsolete document is worse
than no answer. A retired screening programme or a superseded vaccination schedule that remains
searchable is the failure mode with real consequences, and every mechanism that keeps old content
around, soft-delete flags or version fields, works only if every query path remembers to filter it
out. Making the wrong answer impossible beats making it unlikely.

Raw uploaded files are retained in object storage, so a deletion is reversible by re-upload and the
source material remains available for audit.

## Consequences

There is no way to reconstruct exactly what the Knowledge Base contained on a given date. If an
institution is ever asked what its chatbot was telling people in a particular month, the retained
raw files answer that approximately, not precisely. If that requirement ever becomes real, it should
be met with an append-only ingestion log rather than by making Chunks versioned.
