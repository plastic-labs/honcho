"""The tenant_id column default fails closed under MULTI_TENANT."""

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.db import tenant_context
from src.models import (
    DEFAULT_TENANT_ID,
    _default_tenant_id,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize(
    ("tenant_id", "expected"), [(None, DEFAULT_TENANT_ID), ("dedicated", "dedicated")]
)
def test_flag_off_unbound_is_the_instance_tenant(
    monkeypatch: pytest.MonkeyPatch, tenant_id: str | None, expected: str
):
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings, "TENANT_ID", tenant_id)
    token = tenant_context.set(None)
    try:
        assert _default_tenant_id() == expected
    finally:
        tenant_context.reset(token)


def test_flag_on_bound_stamps_tenant(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set("tenant-a")
    try:
        assert _default_tenant_id() == "tenant-a"
    finally:
        tenant_context.reset(token)


def test_flag_on_unbound_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(None)
    try:
        with pytest.raises(ValueError, match="requires a bound tenant"):
            _default_tenant_id()
    finally:
        tenant_context.reset(token)


@pytest.mark.asyncio
async def test_flag_on_unbound_insert_fails_at_flush(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set(None)
    try:
        db_session.add(models.Workspace(name=str(generate_nanoid())))
        # SQLAlchemy wraps a column-default error in a StatementError.
        with pytest.raises(StatementError, match="requires a bound tenant"):
            await db_session.flush()
    finally:
        tenant_context.reset(token)
        await db_session.rollback()
