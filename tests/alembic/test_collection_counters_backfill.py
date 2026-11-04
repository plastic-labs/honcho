"""Revision b7d2e4f19c03's backfill on hash-partitioned collections and documents.

It runs one UPDATE per collections partition, against the documents partition
with the same bound, or against all of documents when no partition matches.
"""

from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings

REVISION = "b7d2e4f19c03"
DOWN_REVISION = "0c67be41049b"
TENANTS = [f"tenant-{i}" for i in range(6)]
COLLECTION_COLUMNS = (
    "id text NOT NULL, workspace_name text, observer text, observed text,"
    + " internal_metadata jsonb NOT NULL DEFAULT '{}'::jsonb"
)
DOCUMENT_COLUMNS = (
    "id text NOT NULL, workspace_name text, observer text, observed text,"
    + " level text, created_at timestamptz"
)


def _partition(alembic_engine: Engine, document_modulus: int) -> None:
    schema = settings.DB.SCHEMA
    with alembic_engine.begin() as conn:
        for table, columns, modulus in (
            ("documents", DOCUMENT_COLUMNS, document_modulus),
            ("collections", COLLECTION_COLUMNS, 2),
        ):
            conn.execute(text(f'DROP TABLE "{schema}".{table} CASCADE'))
            conn.execute(
                text(
                    f'CREATE TABLE "{schema}".{table} (tenant_id text NOT NULL, {columns},'
                    + " PRIMARY KEY (tenant_id, id)) PARTITION BY HASH (tenant_id)"
                )
            )
            for remainder in range(modulus):
                conn.execute(
                    text(
                        f'CREATE TABLE "{schema}".{table}_p{remainder}'
                        + f' PARTITION OF "{schema}".{table}'
                        + f" FOR VALUES WITH (MODULUS {modulus}, REMAINDER {remainder})"
                    )
                )
        for tenant in TENANTS:
            for observed in ("a", "b"):
                conn.execute(
                    text(
                        f'INSERT INTO "{schema}".collections'
                        + " (tenant_id, id, workspace_name, observer, observed)"
                        + " VALUES (:tenant, :id, 'w', 'a', :observed)"
                    ),
                    {
                        "tenant": tenant,
                        "id": f"{tenant}-{observed}",
                        "observed": observed,
                    },
                )
            for i, level in enumerate(("explicit", "explicit", "deductive")):
                conn.execute(
                    text(
                        f'INSERT INTO "{schema}".documents'
                        + " (tenant_id, id, workspace_name, observer, observed, level,"
                        + " created_at)"
                        + " VALUES (:tenant, :id, 'w', 'a', 'a', :level,"
                        + " TIMESTAMPTZ '2026-01-01 00:00:00+00' + make_interval(days => :i))"
                    ),
                    {"tenant": tenant, "id": f"{tenant}-{i}", "level": level, "i": i},
                )


@pytest.mark.parametrize("document_modulus", [2, 4], ids=["paired", "unpaired"])
def test_backfill_runs_per_partition(
    alembic_cfg: Config, alembic_engine: Engine, document_modulus: int
) -> None:
    schema = settings.DB.SCHEMA
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _partition(alembic_engine, document_modulus)
    with alembic_engine.connect() as conn:
        assert all(
            conn.scalar(text(f'SELECT count(*) FROM "{schema}".collections_p{r}'))
            for r in range(2)
        )

    command.upgrade(alembic_cfg, REVISION)

    with alembic_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT tenant_id, observed, explicit_count,"
                + " to_char(newest_explicit_at AT TIME ZONE 'UTC', 'YYYY-MM-DD')"
                + f' FROM "{schema}".collections ORDER BY tenant_id, observed'
            )
        ).all()
    assert [tuple(row) for row in rows] == [
        row
        for tenant in TENANTS
        for row in ((tenant, "a", 2, "2026-01-02"), (tenant, "b", 0, None))
    ]
