"""Integration test for the shared-tenant schema bootstrap."""

# region ai
# Runs ``scripts/bootstrap_shared_schema`` against a throwaway database and
# asserts the partitioned schema it produces: every tenant-scoped table is
# HASH(tenant_id)-partitioned with its full set of partitions, the primary key
# leads with ``tenant_id``, and the tenant-scoped composite keys enforce (two
# tenants can share a workspace name, a duplicate name within a tenant is
# rejected, and an unknown ``tenant_id`` is rejected by the FK to ``tenants``).
#
# Heavier than a metadata-only shape check on purpose: it exercises the exact
# DDL the migration track will run, against real Postgres + pgvector, and it
# does not go through the app's ``create_all`` fixture (which does not create
# partitions).
# endregion

import pytest
from sqlalchemy import Engine, create_engine, make_url, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from scripts.bootstrap_shared_schema import (
    PARTITION_COUNT,
    bootstrap_shared_schema,
    partitioned_tables,
)
from src.config import settings
from src.models import Tenant, Workspace


@pytest.fixture
def bootstrapped_engine(worker_id: str):
    """A throwaway database with the shared-tenant schema bootstrapped into it."""
    db_name = f"test_bootstrap_shared_{worker_id}"
    admin_url = make_url(settings.DB.CONNECTION_URI).set(database="postgres")

    def _drop_create(create: bool) -> None:
        # ai: Pass the URL object (not str) so the password is not masked to '***'.
        admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)'))
            if create:
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        admin.dispose()

    _drop_create(create=True)
    engine = create_engine(make_url(settings.DB.CONNECTION_URI).set(database=db_name))
    # ai: AUTOCOMMIT: the partition DDL accumulates too many locks for one transaction.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        bootstrap_shared_schema(conn)
    try:
        yield engine
    finally:
        engine.dispose()
        _drop_create(create=False)


def test_every_tenant_table_is_partitioned(bootstrapped_engine: Engine) -> None:
    """Every tenant-scoped table is HASH-partitioned, carrying all its partitions."""

    # region ai
    # Guards the drift trap where a partitioned parent with zero partitions
    # rejects every insert.
    # endregion
    expected = partitioned_tables()
    assert len(expected) == 9
    with bootstrapped_engine.connect() as conn:
        for name in expected:
            relkind = conn.execute(
                text(
                    "SELECT relkind FROM pg_class"
                    + " WHERE relname = :n AND relnamespace = 'public'::regnamespace"
                ),
                {"n": name},
            ).scalar()
            assert relkind == "p", f"{name} is not partitioned (relkind={relkind!r})"
            n_partitions = conn.execute(
                text(
                    "SELECT count(*) FROM pg_inherits i"
                    + " JOIN pg_class p ON p.oid = i.inhparent WHERE p.relname = :n"
                ),
                {"n": name},
            ).scalar()
            assert n_partitions == PARTITION_COUNT, (
                f"{name} has {n_partitions} partitions, expected {PARTITION_COUNT}"
            )


def test_primary_key_leads_with_tenant_id(bootstrapped_engine: Engine) -> None:
    with bootstrapped_engine.connect() as conn:
        pk_cols = (
            conn.execute(
                text(
                    "SELECT a.attname FROM pg_index i"
                    + " JOIN pg_attribute a"
                    + " ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)"
                    + " WHERE i.indrelid = 'public.workspaces'::regclass AND i.indisprimary"
                    + " ORDER BY array_position(i.indkey, a.attnum)"
                )
            )
            .scalars()
            .all()
        )
    assert pk_cols == ["tenant_id", "id"]


def test_tenant_scoped_uniqueness_and_fk(bootstrapped_engine: Engine) -> None:
    """The tenant-scoped composite keys enforce as designed."""
    with Session(bootstrapped_engine) as session:
        session.add_all([Tenant(tenant_id="tenant_a"), Tenant(tenant_id="tenant_b")])
        session.commit()

        # region ai
        # Two different tenants may each own a workspace named "default": this is
        # the whole point of UNIQUE(tenant_id, name) replacing UNIQUE(name).
        # endregion
        session.add_all(
            [
                Workspace(tenant_id="tenant_a", name="default"),
                Workspace(tenant_id="tenant_b", name="default"),
            ]
        )
        session.commit()

    # Same (tenant, name) twice is rejected.
    with Session(bootstrapped_engine) as session:
        session.add(Workspace(tenant_id="tenant_a", name="default"))
        with pytest.raises(IntegrityError):
            session.commit()

    # An unknown tenant_id is rejected by the FK to tenants.
    with Session(bootstrapped_engine) as session:
        session.add(Workspace(tenant_id="ghost_tenant", name="scratch"))
        with pytest.raises(IntegrityError):
            session.commit()
