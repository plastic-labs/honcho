"""Read a Discord server's history as Honcho ingest records.

`mapping` turns Discord's REST objects into ingest records, `client` reads
Discord's REST API, and `export` walks a server's channels and threads oldest
message first and hands each record to a sink: JSONL files, or the ingest API.
"""
