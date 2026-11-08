"""Real-migration provisioning shared by lifecycle regression fixtures."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import settings
from src.startup.tenant_isolation_validator import apply_tenant_isolation
from tests.conftest import CONNECTION_URI


@contextmanager
def migrated_lifecycle_engine(worker_id: str) -> Generator[Engine]:
    """Provision and remove an owned migrated RLS database with shared natural keys.

    The migrated dedicated schema keeps a tenant-less compatibility index for
    the previous release's upsert. Shared storage uses the tenant-qualified key.
    Remove that index only in this throwaway fixture, never in the application.
    """
    database = "lifecycle_" + uuid4().hex + "_" + worker_id
    app_role = "lifecycle_app_" + uuid4().hex
    url = CONNECTION_URI.set(database=database)
    maintenance = create_engine(
        CONNECTION_URI.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    with maintenance.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{database}"'))
        conn.execute(text(f"CREATE ROLE {app_role} NOLOGIN NOSUPERUSER NOBYPASSRLS"))
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        root = Path(__file__).resolve().parents[2]
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                settings.DB, "CONNECTION_URI", url.render_as_string(hide_password=False)
            )
            patch.setattr(settings.DB, "OWNER_CONNECTION_URI", None)
            config = Config(str(root / "alembic.ini"))
            config.set_main_option("script_location", str(root / "migrations"))
            command.upgrade(config, "head")
        with engine.begin() as conn:
            apply_tenant_isolation(conn, schema="public", app_role=app_role)
            conn.execute(
                text("DROP INDEX public.uq_session_peers_legacy_ws_session_peer")
            )
        yield engine
    finally:
        engine.dispose()
        with maintenance.connect() as conn:
            conn.execute(text(f'DROP DATABASE "{database}" WITH (FORCE)'))
            conn.execute(text(f"DROP ROLE {app_role}"))
        maintenance.dispose()
