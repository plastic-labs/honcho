"""Small direct pgvector semantic benchmark for Honcho embeddings.

Runs inside a Honcho container. It embeds each query with the currently configured
embedding client, searches local pgvector tables, and prints compact top-k snippets.
Useful for comparing cloud vs local embedding configs without touching API auth.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db import SessionLocal
from src.embedding_client import embedding_client

DEFAULT_QUERIES = [
    "LINE digest bilingual Chinese English speaker date time",
    "LINE 摘要 中英雙語 說話者 日期 時間",
    "Obsidian MOC no YAML Dataview voicenotes immutable",
    "Andy 偏好 Obsidian MOC 不要 YAML Dataview voice notes 不可修改",
    "Mattermost tax receipt newest current period do not backfill older invoices",
    "稅務 收據 Mattermost 只貼最新 當期 不要補舊發票",
]


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(x):.9g}" for x in vector) + "]"


def _snippet(content: str, width: int = 220) -> str:
    clean = " ".join(content.split())
    return clean[:width] + ("…" if len(clean) > width else "")


async def _search_table(table: str, query_vector: Sequence[float], top_k: int) -> list[dict]:
    if table == "documents":
        id_col = "id::text"
        extra_cols = "workspace_name, observer as peer, observed as session_or_observed, level"
    elif table == "message_embeddings":
        id_col = "id::text"
        extra_cols = "workspace_name, peer_name as peer, session_name as session_or_observed, null::text as level"
    else:
        raise ValueError(table)

    sql = text(
        f"""
        select {id_col} as id,
               content,
               {extra_cols},
               embedding <=> cast(:query_vector as vector) as distance
        from {table}
        where embedding is not null
        order by embedding <=> cast(:query_vector as vector)
        limit :top_k
        """
    )
    async with SessionLocal() as db:
        result = await db.execute(
            sql, {"query_vector": _vector_literal(query_vector), "top_k": top_k}
        )
        rows = []
        for row in result.mappings():
            rows.append(
                {
                    "table": table,
                    "id": row["id"],
                    "distance": round(float(row["distance"]), 6),
                    "workspace": row["workspace_name"],
                    "peer": row["peer"],
                    "session_or_observed": row["session_or_observed"],
                    "level": row["level"],
                    "snippet": _snippet(row["content"]),
                }
            )
        return rows


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="append", help="Query to run; can be repeated")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of readable text")
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES
    all_results = []
    for query in queries:
        vector = await embedding_client.embed(query)
        docs = await _search_table("documents", vector, args.top_k)
        msgs = await _search_table("message_embeddings", vector, args.top_k)
        result = {"query": query, "documents": docs, "message_embeddings": msgs}
        all_results.append(result)

    if args.json:
        print(json.dumps(all_results, ensure_ascii=False, indent=2))
        return

    for result in all_results:
        print(f"\n## {result['query']}")
        for section in ("documents", "message_embeddings"):
            print(f"\n{section}:")
            for row in result[section]:
                print(
                    f"- d={row['distance']:.4f} {row['workspace']} "
                    f"{row['peer']}->{row['session_or_observed']} {row['id']}: {row['snippet']}"
                )


if __name__ == "__main__":
    asyncio.run(main())
