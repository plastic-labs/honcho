"""Tests for the configure_embeddings script."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from scripts.configure_embeddings import (
    _apply_pgvector_alter,  # pyright: ignore[reportPrivateUsage]
    _build_pgvector_plan,  # pyright: ignore[reportPrivateUsage]
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _restore_schema_to(db_engine: AsyncEngine, dim: int) -> AsyncGenerator[None]:
    """ALTER both embedding columns back to ``dim`` on exit so this test
    leaves the shared test DB in a consistent state for subsequent tests."""
    try:
        yield
    finally:
        async with db_engine.begin() as conn:
            for table in ("documents", "message_embeddings"):
                await conn.execute(
                    text(
                        f"ALTER TABLE {table} ALTER COLUMN embedding"
                        + f" TYPE vector({dim}) USING NULL"
                    )
                )


async def _current_dims(db_engine: AsyncEngine) -> dict[str, int]:
    async with db_engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT c.relname AS table_name, a.atttypmod AS typmod
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public'
                  AND c.relname = ANY(:tables)
                  AND a.attname = 'embedding'
                """
            ),
            {"tables": ["documents", "message_embeddings"]},
        )
        return {row.table_name: row.typmod for row in result}


async def _hnsw_indexes(db_engine: AsyncEngine) -> set[str]:
    async with db_engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                  AND tablename IN ('documents', 'message_embeddings')
                  AND indexdef ILIKE '%USING hnsw%'
                """
            )
        )
        return {row.indexname for row in result}


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_no_alter_needed_when_dims_already_match(
    db_engine: AsyncEngine,
) -> None:
    plan = await _build_pgvector_plan(db_engine, target_dim=1536, schema="public")
    assert plan.needs_alter is False
    assert plan.current_dims == {"documents": 1536, "message_embeddings": 1536}


@pytest.mark.asyncio
async def test_plan_needs_alter_when_target_differs(db_engine: AsyncEngine) -> None:
    plan = await _build_pgvector_plan(db_engine, target_dim=768, schema="public")
    assert plan.needs_alter is True


@pytest.mark.asyncio
async def test_plan_raises_on_missing_column(db_engine: AsyncEngine) -> None:
    with pytest.raises(SystemExit, match="required vector columns missing"):
        await _build_pgvector_plan(db_engine, target_dim=1536, schema="no_such_schema")


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_alters_dims_and_recreates_hnsw_indexes(
    db_engine: AsyncEngine,
) -> None:
    # 768 is the canonical "small" dim used in non-1536 deployments and is
    # well below pgvector's 2000-dim HNSW limit.
    target = 768
    async with _restore_schema_to(db_engine, dim=1536):
        before_indexes = await _hnsw_indexes(db_engine)
        assert before_indexes, "test fixture should have HNSW indexes pre-alter"

        plan = await _build_pgvector_plan(db_engine, target_dim=target, schema="public")
        assert plan.needs_alter is True
        await _apply_pgvector_alter(db_engine, plan)

        after_dims = await _current_dims(db_engine)
        assert after_dims == {"documents": target, "message_embeddings": target}

        after_indexes = await _hnsw_indexes(db_engine)
        assert (
            after_indexes == before_indexes
        ), "HNSW indexes should be recreated with the same names"


@pytest.mark.asyncio
async def test_apply_refuses_when_embeddings_populated(
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALTER ... USING NULL would silently wipe non-null embeddings, so the
    pre-check must abort the transaction before any destructive action.

    We patch the count helper to simulate populated tables rather than wire
    up the full FK chain of workspace/peer/collection/document just to land
    one vector row.
    """

    async def fake_count(_conn: object, _schema: str, table: str) -> int:
        return 7 if table == "documents" else 0

    monkeypatch.setattr(
        "scripts.configure_embeddings._count_non_null_embeddings",
        fake_count,
    )

    async with _restore_schema_to(db_engine, dim=1536):
        plan = await _build_pgvector_plan(db_engine, target_dim=768, schema="public")
        with pytest.raises(
            SystemExit, match="refusing to ALTER populated embedding tables"
        ):
            await _apply_pgvector_alter(db_engine, plan)

        # The SystemExit aborts the transaction; nothing should have changed.
        dims_after_refuse = await _current_dims(db_engine)
        assert dims_after_refuse == {
            "documents": 1536,
            "message_embeddings": 1536,
        }


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idempotent_apply_is_a_noop(db_engine: AsyncEngine) -> None:
    """Build plan twice with the matching dim — second call should still
    return needs_alter=False without raising or making any changes."""
    plan_a = await _build_pgvector_plan(db_engine, target_dim=1536, schema="public")
    plan_b = await _build_pgvector_plan(db_engine, target_dim=1536, schema="public")
    assert plan_a.needs_alter is False
    assert plan_b.needs_alter is False
    assert plan_a.current_dims == plan_b.current_dims
