"""Merge progress and pending-work migration branches without changing DDL.

Revision ID: c6d9e2f74a10
Revises: a5f8da8f6d23, a7c2e4f819d0

Keep both published revision identities so a database at either parent still
runs the other parent's DDL before reaching the common head.
"""

from collections.abc import Sequence

revision: str = "c6d9e2f74a10"
down_revision: tuple[str, str] = ("a5f8da8f6d23", "a7c2e4f819d0")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
