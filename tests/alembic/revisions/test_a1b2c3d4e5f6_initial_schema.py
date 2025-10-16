"""Hooks for initial schema revision a1b2c3d4e5f6."""

from __future__ import annotations

from nanoid import generate as generate_nanoid

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier

APP_ID = generate_nanoid()
APP_NAME = "seed-app"
USER_ID = generate_nanoid()
USER_NAME = "seed-user"
SESSION_ID = generate_nanoid()
SESSION_NAME = "seed-session"
MESSAGE_PUBLIC_ID = generate_nanoid()
MESSAGE_CONTENT = "seed message"
METAMESSAGE_PUBLIC_ID = generate_nanoid()
METAMESSAGE_CONTENT = "seed metamessage"
COLLECTION_ID = generate_nanoid()
COLLECTION_NAME = "seed-collection"
DOCUMENT_ID = generate_nanoid()
DOCUMENT_CONTENT = "Seed document"


@register_before_upgrade("a1b2c3d4e5f6")
def prepare_initial(_verifier: MigrationVerifier) -> None:
    pass


@register_after_upgrade("a1b2c3d4e5f6")
def verify_initial_schema(_verifier: MigrationVerifier) -> None:
    pass
