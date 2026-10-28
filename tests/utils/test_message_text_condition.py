"""search.message_text_condition: inline predicate flag-off, key lookup flag-on."""

from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from src import models
from src.config import settings
from src.db import tenant_context
from src.utils.search import message_text_condition


def _compile(expr: Any) -> tuple[str, dict[str, Any]]:
    compiled = expr.compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


@pytest.mark.parametrize("use_fts", [True, False])
def test_flag_off_matches_content_inline(
    monkeypatch: pytest.MonkeyPatch, use_fts: bool
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    sql, params = _compile(message_text_condition("w", "hello", use_fts=use_fts))
    assert models.MESSAGE_TEXT_MATCHES not in sql
    assert "messages.content ILIKE" in sql
    assert ("@@" in sql) is use_fts
    assert "%hello%" in params.values()


def test_flag_off_escapes_the_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    sql, params = _compile(message_text_condition("w", "50%_off", use_fts=False))
    assert "ESCAPE '\\\\'" in sql
    assert r"%50\%\_off%" in params.values()


@pytest.mark.parametrize("use_fts", [True, False])
def test_flag_on_reads_the_matching_ids_from_the_function(
    monkeypatch: pytest.MonkeyPatch, use_fts: bool
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set("tenant-a")
    try:
        sql, params = _compile(message_text_condition("w", "hello", use_fts=use_fts))
    finally:
        tenant_context.reset(token)
    assert f"messages.id IN (SELECT {models.MESSAGE_TEXT_MATCHES}(" in sql
    assert "ILIKE" not in sql
    assert "@@" not in sql
    assert list(params.values()) == ["tenant-a", "w", "hello", "%hello%", use_fts]


def test_flag_on_without_a_bound_tenant_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    with pytest.raises(ValueError, match="no tenant bound"):
        message_text_condition("w", "hello", use_fts=True)


def test_flag_on_without_a_workspace_keeps_the_inline_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    token = tenant_context.set("tenant-a")
    try:
        sql, _ = _compile(message_text_condition(None, "hello", use_fts=True))
    finally:
        tenant_context.reset(token)
    assert models.MESSAGE_TEXT_MATCHES not in sql
    assert "ILIKE" in sql
