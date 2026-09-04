"""One worker draining a mixed-tenant queue never works across tenants.

The deriver claims work units cross-tenant (via the RLS-bypassing service session)
but *processes* each unit under the tenant carried by its key. Two guarantees make
that safe, and they are what these tests pin down:

1. Work-unit keys are tenant-namespaced when MULTI_TENANT is on, so two tenants
   that share a workspace/session/observed name still land in separate units — a
   drained batch is always tenant-homogeneous. Without the prefix they collide into
   one mixed-tenant unit (the bug the namespacing fixes).
2. Processing a unit binds the tenant parsed from its key into tenant_context, so
   every per-tenant session opened while working it is scoped to that tenant.

These are pure-mechanism checks; draining behavior itself is covered by the other
deriver queue tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config import settings
from src.db import tenant_context
from src.deriver.queue_manager import QueueManager
from src.utils.work_unit import construct_work_unit_key, parse_work_unit_key

_PAYLOAD: dict[str, Any] = {
    "task_type": "representation",
    "session_name": "shared-session",
    "observed": "shared-observed",
}


def _key_for_tenant(tenant: str | None) -> str:
    token = tenant_context.set(tenant)
    try:
        return construct_work_unit_key("shared-workspace", _PAYLOAD)
    finally:
        tenant_context.reset(token)


def test_namespacing_separates_colliding_tenants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)

    key1 = _key_for_tenant("tenant-1")
    key2 = _key_for_tenant("tenant-2")

    # Identical workspace/session/observed, different tenants -> distinct units.
    assert key1 != key2
    assert key1.startswith("tenant-1:")
    assert key2.startswith("tenant-2:")
    # Each key round-trips to its own tenant, so a drained unit is homogeneous.
    assert parse_work_unit_key(key1).tenant_id == "tenant-1"
    assert parse_work_unit_key(key2).tenant_id == "tenant-2"


def test_keys_collide_without_namespacing(monkeypatch: pytest.MonkeyPatch) -> None:
    # The bug the prefix fixes: flag off, two tenants sharing names produce the SAME
    # key -> one mixed-tenant unit whose batch can't be bound to a single tenant.
    monkeypatch.setattr(settings, "MULTI_TENANT", False)

    assert _key_for_tenant("tenant-1") == _key_for_tenant("tenant-2")
    assert parse_work_unit_key(_key_for_tenant("tenant-1")).tenant_id is None


class _CaptureOwnership:
    """Stands in for QueueManager.worker_ownership to observe the tenant binding.

    process_work_unit binds the unit's tenant, then looks up worker ownership before
    doing any work. Recording tenant_context at that first lookup captures the
    binding; returning None makes the worker stop immediately (treated as lost
    ownership), so no real draining or DB access happens. Only the first lookup is
    recorded — process_work_unit's teardown looks ownership up again after resetting
    the tenant, and that later None must not clobber the observed binding.
    """

    def __init__(self) -> None:
        self.tenant_seen: str | None = "UNSET"

    def get(self, _worker_id: str) -> None:
        if self.tenant_seen == "UNSET":
            self.tenant_seen = tenant_context.get()
        return None


@pytest.mark.asyncio
async def test_process_work_unit_binds_parsed_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The "work per-tenant" half: the tenant comes from the unit's key, not from an
    # ambient value — so a single worker draining a mixed queue scopes each unit to
    # its own tenant and never writes across.
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    key = _key_for_tenant("tenant-1")
    assert key.startswith("tenant-1:")  # sanity: namespaced

    manager = QueueManager()
    capture = _CaptureOwnership()
    monkeypatch.setattr(manager, "worker_ownership", capture)

    reset = tenant_context.set(None)  # worker starts with no ambient tenant
    try:
        await manager.process_work_unit(key, "worker-1")
    finally:
        tenant_context.reset(reset)

    assert capture.tenant_seen == "tenant-1"  # bound from the key
    assert tenant_context.get() is None  # reset once the unit is done
