"""Hooks for revision 0e32bf9d5a22 (add_webhook_endpoint_event_types)."""

from __future__ import annotations

from nanoid import generate as generate_nanoid
from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

WORKSPACE = "ws_webhook_event_types"


def _insert_endpoint(verifier: MigrationVerifier, url: str) -> None:
    """Insert an endpoint the way code from before this revision does."""
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".webhook_endpoints '
            + "(id, workspace_name, url) VALUES (:id, :ws, :url)"
        ),
        {"id": generate_nanoid(), "ws": WORKSPACE, "url": url},
    )


@register_before_upgrade("0e32bf9d5a22")
def prepare_add_webhook_endpoint_event_types(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("webhook_endpoints", "event_types", exists=False)
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".workspaces (id, name) VALUES (:id, :ws)'
        ),
        {"id": generate_nanoid(), "ws": WORKSPACE},
    )
    _insert_endpoint(verifier, "https://example.com/before")


@register_after_upgrade("0e32bf9d5a22")
def verify_add_webhook_endpoint_event_types(verifier: MigrationVerifier) -> None:
    verifier.assert_column_exists("webhook_endpoints", "event_types", nullable=True)
    # A write from an older image, which never names the column, still works.
    _insert_endpoint(verifier, "https://example.com/after")
    rows = verifier.conn.execute(
        text(
            f'SELECT url, event_types FROM "{verifier.schema}".webhook_endpoints '
            + "WHERE workspace_name = :ws ORDER BY url"
        ),
        {"ws": WORKSPACE},
    ).all()
    # Both receive every event type, as every endpoint did before.
    assert rows == [
        ("https://example.com/after", None),
        ("https://example.com/before", None),
    ]
