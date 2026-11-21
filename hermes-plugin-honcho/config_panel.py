"""The Honcho config panel in Hermes Desktop and the dashboard (``storage=STORAGE_PROVIDER``).

Hermes owns the panel's rendering, value coercion, write-only secrets and the ``.env`` copy of
``apiKey``; this module owns where each field lives in honcho.json. Fields with ``scope="host"`` sit
in the active profile's host block (falling back to the root on read), ``scope="root"`` at the root.
"""

from __future__ import annotations

from typing import Any

from hermes_constants import mkdir_under_hermes_home
from utils import atomic_json_write

from .client import _host_block, resolve_active_host, resolve_config_path
from .config_schema import CONFIG_SCHEMA
from .oauth import ACCESS_TOKEN_PREFIX, _config_refresh_lock, _read_config_strict, _refresh_lock


def read_fields() -> dict[str, Any]:
    """``{"values": ..., "placeholders": ...}`` for the active host. A file that does not parse raises
    ``ValueError`` (the panel shows a 400 rather than an empty form that a save would write back)."""
    host, path = resolve_active_host(), resolve_config_path()
    raw = _read_config_strict(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    block = _host_block(raw, host)
    values: dict[str, Any] = {}
    for field in CONFIG_SCHEMA.fields:
        for source in ((block, raw) if field.scope == "host" else (raw,)):
            stored = next((source[k] for k in (field.key, *field.aliases) if source.get(k) is not None), None)
            if stored is not None:
                values[field.key] = stored
                break
    # Blank fields show the resolved host Honcho will actually use.
    return {"values": values, "placeholders": {"workspace": host, "aiPeer": host}}


def write_fields(values: dict[str, Any]) -> None:
    """Apply the submitted fields to the file reads resolve, under the token refresh locks (an OAuth
    rotation is single-use; an unlocked read-modify-write can revoke the grant). ``None`` clears a key
    and its aliases. A file that does not parse raises ``ValueError`` and nothing is written."""
    by_key = {field.key: field for field in CONFIG_SCHEMA.fields}
    host, path = resolve_active_host(), resolve_config_path()
    with _refresh_lock, _config_refresh_lock(path):
        cfg = _read_config_strict(path)
        if not isinstance(cfg, dict):
            raise ValueError(f"{path} does not hold a JSON object")
        hosts = cfg.get("hosts")
        cfg["hosts"] = hosts = hosts if isinstance(hosts, dict) else {}
        # Update the block reads resolve (legacy dot-form included), never shadow it.
        existing = _host_block(cfg, host)
        host_key = next((k for k, v in hosts.items() if v is existing), host) if existing else host
        block = hosts.setdefault(host_key, existing)

        for key, value in values.items():
            field = by_key.get(key)
            if field is None:
                continue
            if field.is_secret:
                # Persist where the client reads first, unless an OAuth token owns that slot.
                stored = block.get(key)
                if value and not (isinstance(stored, str) and stored.startswith(ACCESS_TOKEN_PREFIX)):
                    block[key] = value
                continue
            target = block if field.scope == "host" else cfg
            if value is None:
                for alias in (key, *field.aliases):
                    target.pop(alias, None)
            else:
                target[key] = value
        mkdir_under_hermes_home(path.parent)
        atomic_json_write(path, cfg, mode=0o600)
