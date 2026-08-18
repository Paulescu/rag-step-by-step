"""Building a demo Knowledge Base from a public collection of health documentation.

Lesson 1 uses a single Document. Everything that matters about retrieval, whether a Chunk size is
right, whether hybrid search beats either branch alone, only shows up at a hundred Documents and a
thousand pages. This package fetches that much real documentation from WHO IRIS and ingests it.

Nothing here is part of the ingestion pipeline. It is a Knowledge Manager standing in for an
institution that has not uploaded anything yet.
"""
