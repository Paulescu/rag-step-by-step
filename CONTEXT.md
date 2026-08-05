# Customer Support Chatbot

A self-service chatbot that answers end-user questions for public institutions, grounded in
documentation those institutions publish. This context covers both the ingestion of that
documentation and the retrieval that answers questions from it.

## Language

### Knowledge

**Knowledge Base**:
The complete searchable body of institutional documentation the chatbot is allowed to answer from.
_Avoid_: vector database, index, corpus

**Document**:
A single piece of institutional documentation, as the institution thinks of it (a vaccination
schedule, an enrolment regulation). Its current published content, not any particular file.
_Avoid_: file, PDF, source

**Document Key**:
The stable, human-chosen identifier for a Document, assigned by the Knowledge Manager. Survives
renames and re-uploads. Two uploads under the same Document Key are the same Document.
_Avoid_: filename, document ID, slug

**Chunk**:
The unit of retrieval: a bounded passage of a single page of a Document, small enough to embed and
large enough to answer from. A Chunk always knows which Document and which page it came from.
_Avoid_: passage, segment, fragment, node

**Knowledge Manager**:
The person at the institution responsible for what is in the Knowledge Base. They upload
documentation and are the recipient of ingestion problems.
_Avoid_: admin, uploader, editor

### Ingestion

**Ingestion**:
The on-demand process that turns an uploaded file into searchable content in the Knowledge Base.
_Avoid_: indexing, embedding pipeline, sync

**Quarantined**:
The state of a Document whose text could not be usefully extracted, for example a scan with no text
layer. A Quarantined Document is never searchable and is always reported to the Knowledge Manager.
It is a visible failure, never a silent one.
_Avoid_: failed, skipped, errored
