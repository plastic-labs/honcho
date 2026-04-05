"""Hooks for revision 066e87ca5b07 (align_schema_with_declarative_models)."""

from __future__ import annotations

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("066e87ca5b07")
def prepare_align_schema_with_declarative_models(verifier: MigrationVerifier) -> None:
    """Seed state and assertions before upgrading to 066e87ca5b07."""
    # Assert columns exist but are nullable before migration
    verifier.assert_column_exists(
        "active_queue_sessions", "work_unit_key", exists=True, nullable=True
    )
    verifier.assert_column_exists("documents", "embedding", exists=True, nullable=True)

    # Assert FK constraint does not exist yet
    verifier.assert_constraint_exists(
        "queue", "fk_queue_session_id", "foreign_key", exists=False
    )


@register_after_upgrade("066e87ca5b07")
def verify_align_schema_with_declarative_models(verifier: MigrationVerifier) -> None:
    """Add assertions validating the effects of 066e87ca5b07."""
    # Assert columns are now non-nullable
    verifier.assert_column_exists(
        "peers", "workspace_name", exists=True, nullable=False
    )
    verifier.assert_column_exists(
        "sessions", "workspace_name", exists=True, nullable=False
    )
    verifier.assert_column_exists(
        "active_queue_sessions", "work_unit_key", exists=True, nullable=False
    )
    verifier.assert_column_exists("documents", "embedding", exists=True, nullable=False)

    # Assert FK constraint now exists
    verifier.assert_constraint_exists(
        "queue", "fk_queue_session_id", "foreign_key", exists=True
    )
