"""models.scoped_tenant_id and the reads that filter on it explicitly."""

from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from src import models
from src.config import settings
from src.crud.document import get_documents_with_filters
from src.db import tenant_context


def _sql(stmt: Any) -> str:
    return str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


@pytest.mark.parametrize(
    ("tenant_id", "expected"), [(None, "default"), ("dedicated", "dedicated")]
)
def test_flag_off_with_no_tenant_bound_is_the_instance_tenant(
    monkeypatch: pytest.MonkeyPatch, tenant_id: str | None, expected: str
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings, "TENANT_ID", tenant_id)
    assert models.scoped_tenant_id() == expected


@pytest.mark.parametrize("multi_tenant", [True, False])
def test_a_bound_tenant_is_used(
    monkeypatch: pytest.MonkeyPatch, multi_tenant: bool
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", multi_tenant)
    token = tenant_context.set("tenant-a")
    try:
        assert models.scoped_tenant_id() == "tenant-a"
    finally:
        tenant_context.reset(token)


def test_flag_on_with_no_tenant_bound_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    with pytest.raises(ValueError, match="no tenant bound"):
        models.scoped_tenant_id()


@pytest.mark.parametrize(
    ("tenant_id", "expected"), [(None, "default"), ("dedicated", "dedicated")]
)
def test_conclusions_list_filters_on_the_tenant(
    monkeypatch: pytest.MonkeyPatch, tenant_id: str | None, expected: str
) -> None:
    """The (tenant_id, ...) indexes give order only with tenant_id bound in the
    query, so flag-off reads filter on the instance tenant too. The token window
    reads are checked in tests/crud/test_message_token_window.py."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    monkeypatch.setattr(settings, "TENANT_ID", tenant_id)
    page = _sql(get_documents_with_filters("ws"))
    assert f"documents.tenant_id = '{expected}'" in page
