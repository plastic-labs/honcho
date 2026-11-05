"""The cross-tenant jobs read what changed, not the fleet.

The dream poller once counted every tenant's documents on every API pod, and
every unit test passed: its cost grew with the fleet, and no test ran at fleet
size. These tests seed a database shaped like a shared pool (many tenants, large
tables, small changed sets), run each cross-tenant job against it, and EXPLAIN
every statement the job issued. A sequential scan of a large table fails the
test: on a real pool it would read every tenant's rows on every run.

Small tables (the pending-only queue_item_batches, active_queue_sessions,
tenants) may be scanned: they hold the changed set, or a row per tenant.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy_utils import (
    create_database,  # pyright: ignore[reportUnknownVariableType]
)

from src.config import settings
from src.crud import deriver as crud_deriver
from src.crud import document as crud_document
from src.db import ServiceSessionLocal
from src.db import connect_args as app_connect_args
from src.dreamer.dream_due import (
    _filter_dreams_enabled,  # pyright: ignore[reportPrivateUsage]
    list_due_dreams,
)
from src.reconciler import sync_vectors
from src.reconciler.backfill_document_sources import drain_document_sources_batch
from src.reconciler.queue_cleanup import cleanup_queue_items
from tests.conftest import (
    _drop_database,  # pyright: ignore[reportPrivateUsage]
    _get_test_db_url,  # pyright: ignore[reportPrivateUsage]
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "master")
_DB_URL: URL = _get_test_db_url(_WORKER, tag="plans")

LARGE_TABLES = frozenset(
    {
        "messages",
        "message_embeddings",
        "documents",
        "document_sources",
        "conclusion_source_messages",
        "queue",
        "collections",
        "sessions",
    }
)

# 200 tenants. Every large table holds 1,000-60,000 rows; the rows a job should
# touch are a handful per tenant.
_SEED = """
INSERT INTO tenants (tenant_id, tier) SELECT 't'||g, 'shared' FROM generate_series(1,200) g;
INSERT INTO workspaces (tenant_id, id, name)
  SELECT 't'||g, 'w'||lpad(g::text,20,'0'), 'ws' FROM generate_series(1,200) g;
INSERT INTO peers (tenant_id, id, name, workspace_name)
  SELECT 't'||t, 'p'||lpad((t*10+p)::text,20,'0'), 'peer'||p, 'ws'
  FROM generate_series(1,200) t, generate_series(1,5) p;
INSERT INTO sessions (tenant_id, id, name, workspace_name)
  SELECT 't'||t, 's'||lpad((t*100+s)::text,20,'0'), 'sess'||s, 'ws'
  FROM generate_series(1,200) t, generate_series(1,20) s;
INSERT INTO messages (tenant_id, public_id, content, peer_name, workspace_name,
                      session_name, seq_in_session, token_count, created_at)
  SELECT 't'||t, 'm'||lpad((t*1000+m)::text,20,'0'), 'message '||m, 'peer'||(m%5+1),
         'ws', 'sess'||(m%20+1), m, 20, now() - make_interval(hours => 300 - m)
  FROM generate_series(1,200) t, generate_series(1,300) m;
INSERT INTO message_embeddings (tenant_id, content, message_id, workspace_name,
                                session_name, peer_name, sync_state, last_sync_at)
  SELECT 't'||t, 'chunk', 'm'||lpad((t*1000+m)::text,20,'0'), 'ws', 'sess'||(m%20+1),
         'peer'||(m%5+1), CASE WHEN m <= 2 THEN 'pending' ELSE 'synced' END,
         now() - interval '1 day'
  FROM generate_series(1,200) t, generate_series(1,300) m;
INSERT INTO collections (tenant_id, id, workspace_name, observer, observed,
                         explicit_count, newest_explicit_at, internal_metadata)
  SELECT 't'||t, 'c'||lpad((t*10+c)::text,20,'0'), 'ws', 'peer'||c, 'peer'||c,
         CASE WHEN c = 2 AND t % 20 = 0 THEN 500 ELSE 60 END, now() - interval '3 hours',
         jsonb_build_object('dream', jsonb_build_object('last_dream_document_count', 60))
  FROM generate_series(1,200) t, generate_series(1,5) c;
INSERT INTO documents (tenant_id, id, content, workspace_name, observer, observed,
                       session_name, level, sync_state, last_sync_at, deleted_at,
                       source_ids, created_at)
  SELECT 't'||t, 'd'||lpad((t*1000+d)::text,20,'0'), 'fact '||d, 'ws', 'peer'||(d%5+1),
         'peer'||(d%5+1), 'sess'||(d%20+1), 'explicit', CASE WHEN d = 1 THEN 'pending' ELSE 'synced' END,
         now() - interval '1 day', CASE WHEN d = 2 THEN now() - interval '1 hour' END,
         CASE WHEN d = 3 THEN '["x"]'::jsonb END, now() - make_interval(hours => 300 - d)
  FROM generate_series(1,200) t, generate_series(1,300) d;
INSERT INTO document_sources (tenant_id, derived_id, source_id, workspace_name)
  SELECT 't'||t, 'd'||lpad((t*1000+d)::text,20,'0'), 'd'||lpad((t*1000+d+1)::text,20,'0'), 'ws'
  FROM generate_series(1,200) t, generate_series(4,299) d;
INSERT INTO conclusion_source_messages (tenant_id, conclusion_id, message_id, workspace_name)
  SELECT 't'||t, 'd'||lpad((t*1000+d)::text,20,'0'), 'm'||lpad((t*1000+d)::text,20,'0'), 'ws'
  FROM generate_series(1,200) t, generate_series(4,299) d;
-- A pending backlog: 148 pending and 2 processed rows per tenant.
INSERT INTO queue (tenant_id, payload, task_type, work_unit_key, processed,
                   workspace_name, created_at)
  SELECT 't'||t, '{"observer":"peer1","observed":"peer1","session_name":"sess1"}',
         'representation', 't'||t||':representation:ws:sess'||(q%20+1)||':peer1:peer1',
         q <= 2, 'ws', now() - make_interval(mins => q)
  FROM generate_series(1,200) t, generate_series(1,150) q;
INSERT INTO active_queue_sessions (id, tenant_id, work_unit_key, last_updated)
  SELECT 'a'||lpad(t::text,20,'0'), 't'||t, 't'||t||':representation:ws:sess1:peer1:peer1',
         now() - interval '2 hours'
  FROM generate_series(1,10) t;
ANALYZE;
"""


@pytest.fixture(scope="module")
def fleet_db() -> Generator[str]:
    """A database with the real schema, seeded like a shared pool."""
    _drop_database(_DB_URL)
    create_database(_DB_URL)
    url = _DB_URL.render_as_string(hide_password=False)
    previous_uri = settings.DB.CONNECTION_URI
    previous_owner_uri = settings.DB.OWNER_CONNECTION_URI
    settings.DB.CONNECTION_URI = url
    settings.DB.OWNER_CONNECTION_URI = None
    try:
        cfg = Config(str(_REPO_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(_REPO_ROOT / "migrations"))
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
    finally:
        settings.DB.CONNECTION_URI = previous_uri
        settings.DB.OWNER_CONNECTION_URI = previous_owner_uri
    engine: Engine = create_engine(url)
    try:
        # A raw cursor with no parameters, so psycopg leaves the seed's % alone.
        raw = engine.raw_connection()
        try:
            cursor = raw.cursor()
            cursor.execute(_SEED)
            cursor.close()
            raw.commit()
        finally:
            raw.close()
        yield url
    finally:
        engine.dispose()
        _drop_database(_DB_URL)


@pytest.fixture
async def fleet_engine(
    fleet_db: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncGenerator[AsyncEngine]:
    """An engine on the fleet database, with the service factories bound to it."""
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings.DREAM, "ENABLED", True)
    engine = create_async_engine(fleet_db, connect_args=app_connect_args)
    service_bind = ServiceSessionLocal.kw["bind"]
    ServiceSessionLocal.configure(bind=engine)
    try:
        yield engine
    finally:
        ServiceSessionLocal.configure(bind=service_bind)
        await engine.dispose()


def _scans(
    node: dict[str, Any], found: list[tuple[str, str, float]]
) -> list[tuple[str, str, float]]:
    relation = node.get("Relation Name")
    if relation and "Scan" in node["Node Type"]:
        found.append((node["Node Type"], relation, node.get("Plan Rows", 0)))
    for child in node.get("Plans", []):
        _scans(child, found)
    return found


async def _large_seq_scans(
    engine: AsyncEngine, job: Callable[[AsyncSession], Awaitable[Any]]
) -> list[str]:
    """Run the job, EXPLAIN what it issued, and list its large-table Seq Scans."""
    issued: list[tuple[str, Any]] = []

    def capture(
        _conn: Any, _cursor: Any, statement: str, parameters: Any, *_: Any
    ) -> None:
        if (
            statement.lstrip()
            .upper()
            .startswith(("SELECT", "WITH", "UPDATE", "DELETE"))
        ):
            issued.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        async with AsyncSession(engine, expire_on_commit=False) as db:
            await job(db)
            await db.rollback()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert issued, "the job issued no statements"

    problems: list[str] = []
    async with engine.connect() as conn:
        for statement, parameters in issued:
            plan = await conn.exec_driver_sql(
                "EXPLAIN (FORMAT JSON) " + statement, parameters
            )
            raw = plan.scalar_one()
            tree = (json.loads(raw) if isinstance(raw, str) else raw)[0]["Plan"]
            for node_type, relation, rows in _scans(tree, []):
                if node_type == "Seq Scan" and relation in LARGE_TABLES:
                    problems.append(
                        f"Seq Scan on {relation} (~{rows:.0f} rows): "
                        + " ".join(statement.split())[:120]
                    )
    return problems


async def _claim(db: AsyncSession) -> None:
    await db.execute(crud_deriver.claim_rows_query(10))


async def _metrics(db: AsyncSession) -> None:
    await crud_deriver.get_deriver_metrics(db)


async def _due_dreams(db: AsyncSession) -> None:
    await list_due_dreams(db)


async def _dream_settings(db: AsyncSession) -> None:
    due = await list_due_dreams(db)
    await _filter_dreams_enabled(db, due)


async def _sync_selection(db: AsyncSession) -> None:
    await sync_vectors._get_documents_needing_sync(db, 50)  # pyright: ignore[reportPrivateUsage]
    await sync_vectors._get_message_embeddings_needing_sync(db, 50)  # pyright: ignore[reportPrivateUsage]


async def _source_backfill(db: AsyncSession) -> None:
    await drain_document_sources_batch(db)


class _Store:
    async def get_vector_namespace(self, *_args: Any, **_kwargs: Any) -> str:
        return "ns"

    async def delete_many(self, *_args: Any, **_kwargs: Any) -> None:
        return None


async def _soft_delete_cleanup(db: AsyncSession) -> None:
    await crud_document.cleanup_soft_deleted_documents(db, _Store())  # pyright: ignore[reportArgumentType]


async def _stale_claims(db: AsyncSession) -> None:
    await crud_deriver.cleanup_stale_work_units(db)


async def _queue_cleanup(_db: AsyncSession) -> None:
    await cleanup_queue_items()


_JOBS: tuple[Callable[[AsyncSession], Awaitable[Any]], ...] = (
    _claim,
    _metrics,
    _due_dreams,
    _dream_settings,
    _sync_selection,
    _source_backfill,
    _soft_delete_cleanup,
    _stale_claims,
    _queue_cleanup,
)


@pytest.mark.asyncio
async def test_cross_tenant_jobs_scan_no_large_table(fleet_engine: AsyncEngine) -> None:
    # One test, not one per job: each xdist worker that ran a job would build and
    # seed its own copy of the fleet database.
    problems = {
        job.__name__.lstrip("_"): found
        for job in _JOBS
        if (found := await _large_seq_scans(fleet_engine, job))
    }
    assert problems == {}
    # Without due collections the dream-settings lookup would have issued nothing.
    async with AsyncSession(fleet_engine) as db:
        assert await list_due_dreams(db), "the seed has due collections"
