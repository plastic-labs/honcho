"""Shared identifiers and literals for migration verification."""

from nanoid import generate as generate_nanoid

APP_ID = generate_nanoid()
APP_NAME = "seed-workspace"

WORKSPACE_NAME = APP_NAME

USER_ID = generate_nanoid()
USER_NAME = "seed-user"

SESSION_PUBLIC_ID = generate_nanoid()

MESSAGE_PUBLIC_ID = generate_nanoid()
METAMESSAGE_PUBLIC_ID = generate_nanoid()

COLLECTION_PUBLIC_ID = generate_nanoid()
COLLECTION_NAME = "seed-collection"

DOCUMENT_PUBLIC_ID = generate_nanoid()

WEBHOOK_ID = generate_nanoid()

AGENT_SESSION_NAME = "__global_observations__"

LEGACY_PEER_ID = generate_nanoid()
LEGACY_PEER_NAME = "legacy-peer"
LEGACY_MESSAGE_PUBLIC_ID = generate_nanoid()
