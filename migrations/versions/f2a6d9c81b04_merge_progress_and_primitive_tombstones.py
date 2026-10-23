"""Merge progress and inactive primitive tombstone migration branches.

Revision ID: f2a6d9c81b04
Revises: c6d9e2f74a10, d8c4f6a2b901

Keep both published revision identities so databases on either branch still
apply the other branch's DDL before reaching the common head.
"""

from collections.abc import Sequence

revision: str = "f2a6d9c81b04"
down_revision: tuple[str, str] = ("c6d9e2f74a10", "d8c4f6a2b901")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
