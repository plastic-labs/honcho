"""Test pipeline for running alembic migrations and corresponding test hooks in order."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from pytest_alembic.runner import MigrationContext
from sqlalchemy import Engine

from tests.alembic.registry import get_registered_hooks
from tests.alembic.verifier import MigrationVerifier


def _load_revision_sequence() -> tuple[str, ...]:
    """Read the Alembic script directory to produce the linear revision order."""

    config_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    script = ScriptDirectory.from_config(Config(str(config_path)))
    revisions = list(script.walk_revisions())  # newest -> oldest
    revisions.reverse()
    return tuple(revision.revision for revision in revisions)


REVISION_SEQUENCE: tuple[str, ...] = _load_revision_sequence()
REVISION_PARAMS = [
    pytest.param(revision, id=f"{index:02d}_{revision}")
    for index, revision in enumerate(REVISION_SEQUENCE, start=1)
]


def _test_single_revision(
    revision: str,
    alembic_runner: MigrationContext,
    alembic_engine: Engine,
) -> None:
    """
    Test a single migration revision upgrade.

    For each revision:
    - Migrate to the previous revision
    - Run the before_upgrade hook to seed and validate the state of the DB before the revision
    - Migrate to the current revision
    - Run the after_upgrade hook to validate the state of the DB after the revision
    - Update our previous revision for the next iteration
    """
    hooks_map = get_registered_hooks()
    revision_order = list(REVISION_SEQUENCE)

    # Find the previous revision in the chain
    revision_index = revision_order.index(revision)
    previous_revision = (
        revision_order[revision_index - 1] if revision_index > 0 else "base"
    )

    # Migrate up to the previous revision
    alembic_runner.migrate_up_to(previous_revision)  # pyright: ignore[reportUnknownMemberType]

    # Run before_upgrade hook if it exists
    hooks = hooks_map.get(revision)
    if hooks and hooks.before_upgrade:
        with alembic_engine.begin() as conn:
            verifier = MigrationVerifier(conn, revision)
            hooks.before_upgrade(verifier)

    # Migrate to the current revision
    alembic_runner.migrate_up_to(revision)  # pyright: ignore[reportUnknownMemberType]

    # Run after_upgrade hook if it exists
    if hooks and hooks.after_upgrade:
        with alembic_engine.begin() as conn:
            verifier = MigrationVerifier(conn, revision)
            hooks.after_upgrade(verifier)


@pytest.mark.parametrize("revision", REVISION_PARAMS)
def test_migration_revision(
    revision: str,
    alembic_runner: MigrationContext,
    alembic_engine: Engine,
) -> None:
    _test_single_revision(revision, alembic_runner, alembic_engine)
