"""Fresh single-tenant deployment gate.

A self-hoster's first boot is: empty database → ``alembic upgrade head`` (what
``docker/entrypoint.sh`` runs through ``scripts/provision_db.py``) → API and deriver
start, each running the three startup validators → first message → first claim.
This module walks that path against a throwaway database with the tenant settings
exactly as shipped — nothing here touches ``MULTI_TENANT`` or any setting it governs —
and pins what "no tenant configuration at all" has to mean:

- the migration seeds exactly one tenant, ``"default"``, and nothing else;
- every startup validator passes against the migrated schema, and the
  tenant-isolation validator does so without opening a connection;
- the first write stamps ``tenant_id = "default"`` on every row with no caller
  naming a tenant;
- the first enqueue produces an un-namespaced work unit (``tenant_id`` NULL), the
  trigger-maintained aggregate row the claim reads, and the flag-off claim returns it.
"""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from nanoid import generate as generate_nanoid
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy_utils import (
    create_database,  # pyright: ignore[reportUnknownVariableType]
)

from src import crud, schemas
from src.config import settings
from src.crud.deriver import claim_rows_query
from src.db import tenant_context
from src.deriver.enqueue import enqueue
from src.models import DEFAULT_TENANT_ID
from src.startup import (
    validate_embedding_schema,
    validate_queue_item_batches,
    validate_tenant_isolation,
)
from src.utils.work_unit import parse_work_unit_key, tenant_id_for_work_unit_key
from tests.conftest import (
    _drop_database,  # pyright: ignore[reportPrivateUsage]
    _get_test_db_url,  # pyright: ignore[reportPrivateUsage]
    untouchable_engine,
)

# src.deriver re-exports the enqueue *function* under the module's own name, so the
# module has to be fetched explicitly to patch its session factories.
_ENQUEUE_MODULE = importlib.import_module("src.deriver.enqueue")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"
_MIGRATIONS_DIR = _REPO_ROOT / "migrations"

# The tenant-scoped tables the first message touches; every row must read "default".
_FIRST_WRITE_TABLES = ("workspaces", "peers", "sessions", "session_peers", "messages")


@pytest_asyncio.fixture(scope="module")
async def fresh_engine(worker_id: str) -> AsyncGenerator[AsyncEngine]:
    """An empty database taken to alembic head exactly the way a first boot does."""
    # Named like the suite database (run id + worker suffix + a tag), so a run
    # that dies before teardown leaves something conftest's stale sweep reclaims.
    url = _get_test_db_url(worker_id, tag="fresh")
    _drop_database(url)  # only ever present under a pinned HONCHO_TEST_RUN_ID
    create_database(url)

    # str(URL) masks the password as '***', which then fails auth at migrate.
    url_str = url.render_as_string(hide_password=False)
    sync_engine = create_engine(url_str)
    try:
        with sync_engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    finally:
        sync_engine.dispose()

    # env.py reads the migration target from settings; point it at the throwaway
    # database for the upgrade only, the same way conftest builds the worker DB.
    previous_uri = settings.DB.CONNECTION_URI
    settings.DB.CONNECTION_URI = url_str
    try:
        cfg = Config(str(_ALEMBIC_INI))
        cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
        cfg.set_main_option("sqlalchemy.url", url_str)
        await asyncio.to_thread(command.upgrade, cfg, "head")
    finally:
        settings.DB.CONNECTION_URI = previous_uri

    engine = create_async_engine(url_str)
    try:
        yield engine
    finally:
        await engine.dispose()
        _drop_database(url)


@pytest.fixture(autouse=True)
def _public_schema(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    # The validators introspect pg_catalog by schema name and the migration built
    # in `public` (Base.metadata.schema is pinned there for the suite). SCHEMA is
    # not a tenant setting — those stay exactly as shipped for this whole module.
    monkeypatch.setattr(settings.DB, "SCHEMA", "public")


@pytest.mark.asyncio
async def test_first_boot_seeds_one_tenant_and_every_validator_passes(
    fresh_engine: AsyncEngine,
) -> None:
    assert settings.MULTI_TENANT is False, (
        "this gate runs under the shipped single-tenant defaults"
    )

    async with fresh_engine.connect() as conn:
        tenants = (await conn.execute(text("SELECT tenant_id FROM tenants"))).scalars()
        assert list(tenants) == [DEFAULT_TENANT_ID]

    # The API lifespan and the deriver entrypoint run these three before serving.
    await validate_embedding_schema(fresh_engine)
    await validate_tenant_isolation(fresh_engine, instance_type="api")
    await validate_queue_item_batches(fresh_engine)

    # Flag off, the isolation validator must not so much as open a connection.
    await validate_tenant_isolation(untouchable_engine(), instance_type="api")


@pytest.mark.asyncio
async def test_first_write_and_first_claim_need_no_tenant(
    fresh_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert settings.MULTI_TENANT is False
    assert tenant_context.get() is None  # nobody has named a tenant, and nobody will

    sessions = async_sessionmaker(bind=fresh_engine, expire_on_commit=False)
    workspace_name, peer_name, session_name = (generate_nanoid() for _ in range(3))

    # First write, through the same crud the routes use.
    async with sessions() as db:
        await crud.get_or_create_workspace(
            db, schemas.WorkspaceCreate(name=workspace_name)
        )
        await crud.get_or_create_peers(
            db, workspace_name, [schemas.PeerSpec(name=peer_name)]
        )
        await crud.get_or_create_session(
            db,
            schemas.SessionCreate(
                name=session_name,
                peers={peer_name: schemas.SessionPeerConfig(observe_me=True)},
            ),
            workspace_name,
        )
        [message] = await crud.create_messages(
            db,
            [schemas.MessageCreate(content="hello, honcho", peer_id=peer_name)],
            workspace_name,
            session_name,
        )
        await db.commit()
        payload: list[dict[str, Any]] = [
            {
                "workspace_name": workspace_name,
                "session_name": session_name,
                "message_id": message.id,
                "content": message.content,
                "metadata": message.h_metadata,
                "peer_name": peer_name,
                "created_at": message.created_at,
                "seq_in_session": message.seq_in_session,
            }
        ]

    async with fresh_engine.connect() as conn:
        for table in _FIRST_WRITE_TABLES:
            stamped = await conn.execute(
                text(f"SELECT DISTINCT tenant_id FROM {table}")
            )
            assert list(stamped.scalars()) == [DEFAULT_TENANT_ID], table

    # First enqueue, the way the messages route fires it. enqueue() opens its own
    # sessions by name; route them at the fresh database (conftest points them at
    # the per-worker suite database).
    @asynccontextmanager
    async def fresh_tracked_db(
        _: str | None = None, *, read_only: bool = False, tenant_id: str | None = None
    ) -> AsyncGenerator[AsyncSession]:
        del read_only, tenant_id
        async with sessions() as db:
            yield db

    monkeypatch.setattr(_ENQUEUE_MODULE, "tracked_db", fresh_tracked_db)
    monkeypatch.setattr(_ENQUEUE_MODULE, "service_db", fresh_tracked_db)

    with caplog.at_level("ERROR"):
        await enqueue(payload)

    async with sessions() as db:
        queued = (
            await db.execute(text("SELECT work_unit_key, tenant_id FROM queue"))
        ).all()
        # enqueue() is fire-and-forget and logs instead of raising; surface the log.
        assert len(queued) == 1, (queued, caplog.text)
        work_unit_key, queue_tenant = queued[0]
        assert queue_tenant is None
        assert work_unit_key.startswith("representation:")
        assert tenant_id_for_work_unit_key(work_unit_key) is None
        assert parse_work_unit_key(work_unit_key).tenant_id is None

        # The insert trigger maintained the aggregate the claim reads from...
        batches = await db.execute(
            text("SELECT work_unit_key, tenant_id FROM queue_item_batches")
        )
        assert [tuple(row) for row in batches.all()] == [(work_unit_key, None)]

        # ...but a lone first message sits below the representation batch's token
        # target and inside its age window, so the claim deliberately leaves it to
        # accumulate (REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS /
        # REPRESENTATION_BATCH_MAX_AGE_SECONDS: 512 tokens or 30 minutes by default,
        # whichever comes first).
        assert settings.DERIVER.FLUSH_ENABLED is False
        assert settings.DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS > 0
        assert settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS > 0
        assert (await db.execute(claim_rows_query(limit=8))).all() == []
        await db.rollback()

    # Let the age window elapse. The update trigger recomputes the aggregate only
    # when a row's processed flag flips (created_at never moves in production), so
    # backdating queue.created_at alone leaves oldest_created_at stale; simulating
    # the clock means shifting both the rows and the aggregate that rolls them up.
    # The flag-off claim — every tenant_id NULL, one partition, i.e. plain
    # oldest-first — now returns the unit.
    flush_age = timedelta(
        seconds=settings.DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS + 1
    )
    async with fresh_engine.begin() as conn:
        await conn.execute(
            text("UPDATE queue SET created_at = created_at - :age"), {"age": flush_age}
        )
        await conn.execute(
            text(
                "UPDATE queue_item_batches"
                + " SET oldest_created_at = oldest_created_at - :age"
            ),
            {"age": flush_age},
        )
    async with sessions() as db:
        claimed = (await db.execute(claim_rows_query(limit=8))).all()
        assert [row.work_unit_key for row in claimed] == [work_unit_key]
        await db.rollback()
