"""Hooks for revision 60fc2270fab7 (make_tenant_vector_correlation_id_unique)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

UNIQUE_INDEX = "uq_tenants_vector_correlation_id"
PLAIN_INDEX = "ix_tenants_vector_correlation_id"
# Seeded before the upgrade: distinct keys and two NULLs must all survive it.
KEYED = (("t-vcid-a", "vec-a"), ("t-vcid-b", "vec-b"))
UNKEYED = ("t-vcid-null-1", "t-vcid-null-2")


def _revision_module() -> ModuleType:
    path = next(
        (Path(__file__).parents[3] / "migrations" / "versions").glob(
            "60fc2270fab7_*.py"
        )
    )
    spec = importlib.util.spec_from_file_location("rev_60fc2270fab7", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _insert(verifier: MigrationVerifier, tenant_id: str, key: str | None) -> None:
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}"."tenants" '
            + '("tenant_id", "tier", "vector_correlation_id") '
            + "VALUES (:tid, 'shared', :key)"
        ),
        {"tid": tenant_id, "key": key},
    )


@register_before_upgrade("60fc2270fab7")
def prepare_make_tenant_vector_correlation_id_unique(
    verifier: MigrationVerifier,
) -> None:
    verifier.assert_indexes_exist([("tenants", PLAIN_INDEX)])
    for tenant_id, key in KEYED:
        _insert(verifier, tenant_id, key)
    for tenant_id in UNKEYED:
        _insert(verifier, tenant_id, None)


@register_after_upgrade("60fc2270fab7")
def verify_make_tenant_vector_correlation_id_unique(
    verifier: MigrationVerifier,
) -> None:
    verifier.assert_indexes_exist([("tenants", UNIQUE_INDEX)])
    verifier.assert_indexes_not_exist([("tenants", PLAIN_INDEX)])

    # A held key is refused to a second tenant; another NULL is not.
    with pytest.raises(IntegrityError) as rejection, verifier.conn.begin_nested():
        _insert(verifier, "t-vcid-dup", "vec-a")
    assert UNIQUE_INDEX in str(rejection.value)
    with verifier.conn.begin_nested():
        _insert(verifier, "t-vcid-null-3", None)

    # The pre-check names every duplicated key and its tenants. Exercised
    # directly, inside a savepoint with the index dropped, since a duplicate
    # seeded before the upgrade would abort the whole revision chain.
    savepoint = verifier.conn.begin_nested()
    try:
        verifier.conn.execute(text(f'DROP INDEX "{verifier.schema}"."{UNIQUE_INDEX}"'))
        _insert(verifier, "t-vcid-dup", "vec-a")
        with pytest.raises(RuntimeError) as refusal:
            _revision_module().raise_on_duplicate_keys(verifier.conn, verifier.schema)
        message = str(refusal.value)
        assert "'vec-a'" in message
        assert "t-vcid-a" in message and "t-vcid-dup" in message
        assert "vec-b" not in message
    finally:
        savepoint.rollback()
