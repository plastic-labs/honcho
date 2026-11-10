import logging
from types import SimpleNamespace

import psycopg
import pytest
from fastapi import Request
from sqlalchemy.exc import DataError
from starlette.types import Scope

from src.main import data_exception_handler


@pytest.mark.parametrize(
    "orig",
    [
        psycopg.errors.NumericValueOutOfRange("private input"),
        psycopg.errors.UntranslatableCharacter("private input"),
        psycopg.errors.InvalidTextRepresentation("private input: surrogate"),
        psycopg.DataError("private input: NUL (0x00)"),
    ],
)
@pytest.mark.parametrize("has_route", [True, False])
async def test_input_data_error_logs_only_sqlstate_and_route(
    caplog: pytest.LogCaptureFixture, orig: Exception, has_route: bool
):
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/v3/workspaces/private-workspace/peers",
        "query_string": b"secret=private-query",
        "headers": [(b"authorization", b"private-key")],
    }
    if has_route:
        scope["route"] = SimpleNamespace(path="/v3/workspaces/{workspace_id}/peers")
    request = Request(scope)
    exc = DataError("private SQL statement", {"value": "private parameter"}, orig)
    with caplog.at_level(logging.WARNING, logger="src.main"):
        response = await data_exception_handler(request, exc)
    assert response.status_code == 422
    [record] = [r for r in caplog.records if r.name == "src.main"]
    route = "/v3/workspaces/{workspace_id}/peers" if has_route else "unknown"
    sqlstate = getattr(orig, "sqlstate", None) or "unknown"
    assert record.__dict__["sqlstate"] == sqlstate
    assert record.__dict__["route"] == route
    assert route in record.getMessage()
    assert sqlstate in record.getMessage()
    assert "private" not in str(record.__dict__)
    assert record.exc_info is None


@pytest.mark.parametrize(
    "orig",
    [
        psycopg.errors.InvalidTextRepresentation("bad server cast"),
        psycopg.errors.DivisionByZero("server bug"),
        psycopg.DataError("unexpected driver error"),
    ],
)
async def test_unexpected_data_error_stays_500(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    orig: Exception,
):
    monkeypatch.setattr("src.main.SENTRY_ENABLED", False)
    request = Request({"type": "http", "method": "POST", "path": "/v3/workspaces"})
    with caplog.at_level(logging.WARNING, logger="src.main"):
        response = await data_exception_handler(request, DataError(None, None, orig))
    assert response.status_code == 500
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
