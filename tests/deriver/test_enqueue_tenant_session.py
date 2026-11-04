"""Per-tenant enqueue paths run on the tenant-bound session, not the RLS bypass.

enqueue_dream and the scope-task enqueue match peers by (workspace, name), which is
unique only within a tenant. On service_db (no RLS) those reads and writes reach
every tenant's same-named peers; on tracked_db the policy scopes them to the bound
tenant, which is what this pins down.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src import schemas
from src.config import settings
from src.db import tenant_context
from src.deriver.enqueue import enqueue_dream, enqueue_scope_backfill


@pytest.mark.asyncio
async def test_dream_and_scope_enqueue_use_tenant_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    session = AsyncMock()
    session.scalar = AsyncMock(return_value=False)
    opened_for: list[str | None] = []

    @asynccontextmanager
    async def fake_tracked_db(*_args: Any, **_kwargs: Any) -> AsyncGenerator[AsyncMock]:
        opened_for.append(tenant_context.get())
        yield session

    def no_service_db(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("per-tenant enqueue must not use the RLS-bypass session")

    token = tenant_context.set("tenant-a")
    try:
        with (
            patch("src.deriver.enqueue.tracked_db", fake_tracked_db),
            patch("src.deriver.enqueue.service_db", no_service_db),
            patch(
                "src.deriver.enqueue.crud.reject_scope_observed",
                new_callable=AsyncMock,
            ),
            patch(
                "src.deriver.enqueue.crud.update_scope_backfill_status",
                new_callable=AsyncMock,
            ),
            patch(
                "src.deriver.enqueue.crud.invalidate_scope_peer_cache",
                new_callable=AsyncMock,
            ),
        ):
            await enqueue_dream(
                "ws",
                observer="alice",
                observed="bob",
                dream_type=schemas.DreamType.OMNI,
            )
            await enqueue_scope_backfill("ws", scope_peer="scope.x", session_name="s")
    finally:
        tenant_context.reset(token)

    assert opened_for == ["tenant-a", "tenant-a"]
