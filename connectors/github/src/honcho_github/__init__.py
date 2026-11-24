"""Read GitHub repositories' history as Honcho ingest records.

`mapping` turns GitHub's GraphQL objects into ingest records, `client` reads
GitHub's GraphQL API, and `export` walks each repository's issues, pull
requests and default-branch commits and hands its records, oldest first, to a
sink: JSONL files, or the ingest API.
"""
