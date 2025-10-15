"""End-to-end migration pipeline verification."""

from __future__ import annotations

from pytest_alembic.runner import MigrationContext
from sqlalchemy import Engine

from tests.alembic.registry import get_registered_hooks
from tests.alembic.verifier import MigrationVerifier


def _ordered_revisions(alembic_runner: MigrationContext) -> list[str]:
    revisions = [
        rev for rev in alembic_runner.history.revisions if rev not in {"base", "heads"}
    ]
    return revisions


def test_migration_pipeline(
    alembic_runner: MigrationContext, alembic_engine: Engine
) -> None:
    """Walk migrations in order, seeding and validating at each step."""

    hooks_map = get_registered_hooks()
    revision_order = _ordered_revisions(alembic_runner)

    alembic_runner.migrate_down_to("base")  # pyright: ignore[reportUnknownMemberType]

    for revision in revision_order:
        hooks = hooks_map.get(revision)

        if hooks and hooks.before_upgrade:
            with alembic_engine.begin() as conn:
                for prepare in hooks.before_upgrade:
                    prepare(conn)

        alembic_runner.migrate_up_to(revision)  # pyright: ignore[reportUnknownMemberType]

        hooks = hooks_map.get(revision)
        if hooks and hooks.after_upgrade:
            with alembic_engine.begin() as conn:
                verifier = MigrationVerifier(conn, revision)
                for verification_hook in hooks.after_upgrade:
                    verification_hook(verifier)
