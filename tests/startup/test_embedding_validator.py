"""Startup embedding-schema validator + VECTOR_STORE_DIMENSIONS deprecation."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine

from src.config import settings
from src.startup.embedding_validator import (
    StartupValidationError,
    _assert_pgvector_dims_match,  # pyright: ignore[reportPrivateUsage]
    _sample_external_namespaces,  # pyright: ignore[reportPrivateUsage]
    validate_embedding_schema,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# pgvector stores the declared dim directly in atttypmod (no VARHDRSZ offset).
def _typmod(dim: int) -> int:
    return dim


# ---------------------------------------------------------------------------
# Pure-function unit tests for the dim assertion
# ---------------------------------------------------------------------------


def test_assert_pgvector_dims_match_passes_when_all_dims_align() -> None:
    _assert_pgvector_dims_match(
        {"documents": _typmod(1536), "message_embeddings": _typmod(1536)},
        schema="public",
        target_dim=1536,
    )


def test_assert_pgvector_dims_match_raises_on_dim_mismatch() -> None:
    with pytest.raises(StartupValidationError, match="dim .* does not match"):
        _assert_pgvector_dims_match(
            {"documents": _typmod(1536), "message_embeddings": _typmod(768)},
            schema="public",
            target_dim=1536,
        )


def test_assert_pgvector_dims_match_lists_all_missing_columns() -> None:
    with pytest.raises(StartupValidationError) as excinfo:
        _assert_pgvector_dims_match(
            {"documents": _typmod(1536)},
            schema="public",
            target_dim=1536,
        )
    msg = str(excinfo.value)
    assert "message_embeddings" in msg
    assert "alembic upgrade head" in msg


def test_assert_pgvector_dims_match_raises_on_unbounded_typmod() -> None:
    with pytest.raises(StartupValidationError, match="unbounded typmod"):
        _assert_pgvector_dims_match(
            {"documents": -1, "message_embeddings": _typmod(1536)},
            schema="public",
            target_dim=1536,
        )


def test_assert_pgvector_dims_match_respects_non_public_schema() -> None:
    with pytest.raises(StartupValidationError, match="my_schema.documents"):
        _assert_pgvector_dims_match(
            {},
            schema="my_schema",
            target_dim=1536,
        )


# ---------------------------------------------------------------------------
# Fail-closed retry behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_fails_closed_when_introspection_keeps_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the retry budget exhausts, the validator crashes — uncertainty
    is not a green light to serve traffic."""

    call_count = 0

    async def always_raise(_engine: AsyncEngine, _schema: str) -> dict[str, int]:
        nonlocal call_count
        call_count += 1
        raise OperationalError("SELECT 1", {}, Exception("DB unreachable"))

    monkeypatch.setattr(
        "src.startup.embedding_validator._introspect_pgvector_dims_once",
        always_raise,
    )
    # Make backoff effectively instant for the test.
    monkeypatch.setattr("src.startup.embedding_validator._RETRY_BACKOFF_SECONDS", 0.0)

    with pytest.raises(StartupValidationError, match="could not validate"):
        await validate_embedding_schema(engine=AsyncMock())

    assert call_count == 3, "should exhaust the retry budget before failing"


# ---------------------------------------------------------------------------
# Integration: real test DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validator_passes_against_test_database(
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The test DB is provisioned at the default dim (1536); the validator
    should accept it without raising."""
    # conftest provisions the test tables in `public`; pin the validator to it
    # so a developer's local .env DB_SCHEMA can't point it at another schema.
    monkeypatch.setattr(settings.DB, "SCHEMA", "public")
    await validate_embedding_schema(db_engine)


@pytest.mark.asyncio
async def test_validator_raises_when_schema_dim_diverges_from_settings(
    db_engine: AsyncEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALTER one of the embedding columns to a non-1536 dim and confirm the
    validator raises with an actionable message."""
    monkeypatch.setattr(settings.DB, "SCHEMA", "public")  # see test above
    async with db_engine.begin() as conn:
        await conn.execute(
            text(
                "ALTER TABLE documents"
                + " ALTER COLUMN embedding TYPE vector(768) USING NULL"
            )
        )
    try:
        with pytest.raises(StartupValidationError, match="dim .* does not match"):
            await validate_embedding_schema(db_engine)
    finally:
        async with db_engine.begin() as conn:
            await conn.execute(
                text(
                    "ALTER TABLE documents"
                    + " ALTER COLUMN embedding TYPE vector(1536) USING NULL"
                )
            )


# ---------------------------------------------------------------------------
# VECTOR_STORE_DIMENSIONS deprecation + dim-vs-MIGRATED guard removal
# ---------------------------------------------------------------------------


def test_vector_store_dimensions_explicit_set_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting VECTOR_STORE_DIMENSIONS explicitly should trigger a deprecation
    warning. EMBEDDING_VECTOR_DIMENSIONS remains authoritative."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("VECTOR_STORE_DIMENSIONS", "1536")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        from src.config import AppSettings

        AppSettings()
    messages = [
        str(w.message) for w in captured if issubclass(w.category, DeprecationWarning)
    ]
    assert any("VECTOR_STORE_DIMENSIONS is deprecated" in m for m in messages), (
        f"expected deprecation warning, got {messages!r}"
    )


def test_non_1536_pgvector_without_migrated_no_longer_raises_at_config_time() -> None:
    """The dim-vs-MIGRATED guard has been removed. Constructing AppSettings
    with non-1536 + default pgvector + MIGRATED=false should now succeed
    (the runtime schema validator at startup is the safety net)."""
    # Minimal env, NOT a copy of os.environ: load_dotenv() in the app mutates
    # the parent pytest process's environ, so inheriting it would leak a
    # developer's local .env (DB_SCHEMA, VECTOR_STORE_MIGRATED, ...) into the
    # child despite PYTHON_DOTENV_DISABLED. The child must see pure defaults
    # plus exactly the overrides below.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHON_DOTENV_DISABLED": "1",
        "EMBEDDING_VECTOR_DIMENSIONS": "768",
    }
    # Use a subprocess so the global settings singleton in this test
    # process is not perturbed and is re-evaluated freshly in the child.
    snippet = (
        "from src.config import AppSettings\n"
        "s = AppSettings()\n"
        "print(s.EMBEDDING.VECTOR_DIMENSIONS, s.VECTOR_STORE.TYPE, s.VECTOR_STORE.MIGRATED)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        env=env,
        cwd=str(_PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line == "768 pgvector False"


# ---------------------------------------------------------------------------
# Sampling a tenant that has no vector namespace key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_sample_skips_tenants_without_a_namespace_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tenant with no key is dropped from the sample instead of resolved.

    Its namespace cannot be named, so there is nothing to probe. Passing the null
    onward would read as "no prefix given" and send the resolver looking for an
    ambient tenant, which startup does not have — a best-effort dim check would
    then take the process down over a provisioning gap.
    """
    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    monkeypatch.setattr(settings.VECTOR_STORE, "TYPE", "turbopuffer")

    probed: list[str] = []

    class _Store:
        async def get_vector_namespace(
            self,
            namespace_type: str,
            workspace_name: str,
            observer: str | None = None,
            observed: str | None = None,
            *,
            prefix: str | None = None,
        ) -> str:
            assert prefix, "an unkeyed tenant must never reach the resolver"
            return f"{prefix}.{namespace_type[:3]}.{workspace_name}"

    async def _workspaces(_engine: object, _limit: int) -> list[tuple[str, str | None]]:
        return [("keyed-ws", "hch-old-app"), ("unkeyed-ws", None)]

    async def _collections(
        _engine: object, _limit: int
    ) -> list[tuple[str, str, str, str | None]]:
        return [("keyed-ws", "alice", "bob", "hch-old-app"), ("u", "a", "b", None)]

    async def _probe(_store: object, namespace: str) -> int | None:
        probed.append(namespace)
        return None

    monkeypatch.setattr(
        "src.startup.embedding_validator._sample_workspace_names", _workspaces
    )
    monkeypatch.setattr(
        "src.startup.embedding_validator._sample_collection_keys", _collections
    )
    monkeypatch.setattr("src.startup.embedding_validator._probe_namespace_dim", _probe)
    monkeypatch.setattr("src.vector_store.get_external_vector_store", lambda: _Store())

    await _sample_external_namespaces(AsyncMock(), target_dim=1536)

    assert probed == ["hch-old-app.mes.keyed-ws", "hch-old-app.doc.keyed-ws"]
