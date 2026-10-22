"""Revision a40bd716a0d6 on a partitioned documents table: built per partition, attached, valid."""

from __future__ import annotations

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings
from tests.alembic.test_online_index_build import (
    _attached,  # pyright: ignore[reportPrivateUsage]
    _valid,  # pyright: ignore[reportPrivateUsage]
)

REVISION = "a40bd716a0d6"
DOWN_REVISION = "647b773f05ac"
NAME = "ix_documents_explicit_created_at"


def _partition_documents(alembic_engine: Engine) -> None:
    schema = settings.DB.SCHEMA
    with alembic_engine.begin() as conn:
        conn.execute(text(f'DROP TABLE "{schema}".documents CASCADE'))
        conn.execute(
            text(
                f'CREATE TABLE "{schema}".documents (tenant_id text NOT NULL,'
                + " id text NOT NULL, level text, created_at timestamptz,"
                + " PRIMARY KEY (tenant_id, id)) PARTITION BY HASH (tenant_id)"
            )
        )
        for remainder in range(2):
            conn.execute(
                text(
                    f'CREATE TABLE "{schema}".documents_p{remainder}'
                    + f' PARTITION OF "{schema}".documents'
                    + f" FOR VALUES WITH (MODULUS 2, REMAINDER {remainder})"
                )
            )


def test_builds_on_every_partition_and_attaches(
    alembic_cfg: Config, alembic_engine: Engine
) -> None:
    command.upgrade(alembic_cfg, DOWN_REVISION)
    _partition_documents(alembic_engine)

    command.upgrade(alembic_cfg, REVISION)

    assert _valid(alembic_engine, NAME) is True
    assert _attached(alembic_engine, NAME) == [
        ("documents_p0", True),
        ("documents_p1", True),
    ]
    with alembic_engine.connect() as conn:
        predicate = conn.scalar(
            text(
                "SELECT pg_get_expr(i.indpred, i.indrelid) FROM pg_index i"
                + " WHERE i.indexrelid = to_regclass(:name)"
            ),
            {"name": f'"{settings.DB.SCHEMA}".{NAME}'},
        )
    assert predicate == "(level = 'explicit'::text)"

    command.downgrade(alembic_cfg, DOWN_REVISION)

    assert _valid(alembic_engine, NAME) is None
    assert _valid(alembic_engine, f"{NAME}_p0") is None
