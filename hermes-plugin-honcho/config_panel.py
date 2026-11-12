"""The Honcho config panel in Hermes Desktop and the dashboard (``storage=STORAGE_PROVIDER``).

Hermes owns the panel's rendering, value coercion, write-only secrets and the ``.env`` copy of
``apiKey``; this module owns where each field lives: the active profile's host block in its
honcho.json, the only place the plugin reads.
"""

from __future__ import annotations

from typing import Any

from .client import resolve_active_host, resolve_config_path
from .config_schema import CONFIG_SCHEMA
from .host_config import apply_patch, effective_block, locked_config, migrated_block
from .oauth import ACCESS_TOKEN_PREFIX


def read_fields() -> dict[str, Any]:
    """``{"values": ..., "placeholders": ...}`` for the active host. A file that does not parse raises
    ``ValueError`` (the panel shows a 400 rather than an empty form that a save would write back)."""
    host = resolve_active_host()
    block = effective_block(resolve_config_path(), host)
    values: dict[str, Any] = {}
    for field in CONFIG_SCHEMA.fields:
        stored = next((block[k] for k in (field.key, *field.aliases) if block.get(k) is not None), None)
        if stored is not None:
            values[field.key] = stored
    # Blank fields show the resolved host Honcho will actually use.
    return {"values": values, "placeholders": {"workspace": host, "aiPeer": host}}


def write_fields(values: dict[str, Any]) -> None:
    """Apply the submitted fields to the active host's block, under the token refresh locks (an OAuth
    rotation is single-use; an unlocked read-modify-write can revoke the grant). ``None`` clears a key
    and its aliases. A new ``baseUrl`` without a new key drops the key the block held for its old server.
    A file that does not parse raises ``ValueError`` and nothing is written."""
    by_key = {field.key: field for field in CONFIG_SCHEMA.fields}
    host, path = resolve_active_host(), resolve_config_path()
    with locked_config(path) as cfg:
        block = migrated_block(cfg, host, path)
        patch: dict[str, Any] = {}
        for key, value in values.items():
            field = by_key.get(key)
            if field is None:
                continue
            if field.is_secret:
                # A grant beats a key: while signed in, the access token owns the apiKey slot.
                stored = block.get(key)
                if value and not (isinstance(stored, str) and stored.startswith(ACCESS_TOKEN_PREFIX)):
                    patch[key] = value
            elif value is None:
                patch.update(dict.fromkeys((key, *field.aliases)))
            else:
                patch[key] = value
        apply_patch(block, patch)
