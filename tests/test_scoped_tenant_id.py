"""models.scoped_tenant_id and the reads that filter on it explicitly."""

from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from src import models
from src.config import settings
from src.crud.document import get_documents_with_filters
from src.crud.message import _apply_token_limit  # pyright: ignore[reportPrivateUsage]
from src.db import tenant_context


def _sql(stmt: Any) -> str:
    return str(
        stmt.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def test_flag_off_with_no_tenant_bound_is_the_default_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    assert models.scoped_tenant_id() == models.DEFAULT_TENANT_ID


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


def test_conclusions_list_and_token_window_filter_on_the_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The (tenant_id, ...) indexes give order only with tenant_id bound in the
    query, so flag-off reads filter on the default tenant too."""
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    page = _sql(get_documents_with_filters("ws"))
    assert "documents.tenant_id = 'default'" in page

    window = _sql(
        _apply_token_limit([models.Message.session_name == "s"], token_limit=100)
    )
    assert window.count("messages.tenant_id = 'default'") == 2
