"""Test pipeline for running alembic migrations and corresponding test hooks in order."""

from __future__ import annotations

from pytest_alembic.runner import MigrationContext
from sqlalchemy import Engine

from tests.alembic.registry import get_registered_hooks
from tests.alembic.verifier import MigrationVerifier


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
    revision_order = [
        rev for rev in alembic_runner.history.revisions if rev not in {"base", "heads"}
    ]

    # Find the previous revision in the chain
    previous_revision = (
        revision_order[revision_order.index(revision) - 1]
        if revision_order.index(revision) > 0
        else "base"
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


# ============================================================================
# Individual test functions for each revision (in migration order)
# ============================================================================


def test_01_a1b2c3d4e5f6(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("a1b2c3d4e5f6", alembic_runner, alembic_engine)


def test_02_c3828084f472(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("c3828084f472", alembic_runner, alembic_engine)


def test_03_b765d82110bd(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("b765d82110bd", alembic_runner, alembic_engine)


def test_04_556a16564f50(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("556a16564f50", alembic_runner, alembic_engine)


def test_05_20f89a421aff(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("20f89a421aff", alembic_runner, alembic_engine)


def test_06_66e63cf2cf77(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("66e63cf2cf77", alembic_runner, alembic_engine)


def test_07_d429de0e5338(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("d429de0e5338", alembic_runner, alembic_engine)


def test_08_917195d9b5e9(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("917195d9b5e9", alembic_runner, alembic_engine)


def test_09_05486ce795d5(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("05486ce795d5", alembic_runner, alembic_engine)


def test_10_88b0fb10906f(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("88b0fb10906f", alembic_runner, alembic_engine)


def test_11_564ba40505c5(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("564ba40505c5", alembic_runner, alembic_engine)


def test_12_08894082221a(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("08894082221a", alembic_runner, alembic_engine)


def test_13_76ffba56fe8c(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    _test_single_revision("76ffba56fe8c", alembic_runner, alembic_engine)
