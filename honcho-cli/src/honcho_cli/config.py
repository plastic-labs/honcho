"""Configuration management for Honcho CLI.

Config stored at ``~/.honcho/config.json`` with env var overrides.

The CLI owns exactly two top-level keys in that file:

    apiKey          -- Honcho admin JWT
    environmentUrl  -- Honcho API URL (full URL, e.g. https://api.honcho.dev)

All other top-level keys (``hosts``, ``sessions``, ``saveMessages``,
``sessionStrategy``, …) are written by sibling Honcho tools and are
preserved untouched on save.

Workspace / peer / session scoping is intentionally *not* persisted here —
pass ``-w`` / ``-p`` / ``-s`` flags or set ``HONCHO_WORKSPACE_ID`` /
``HONCHO_PEER_ID`` / ``HONCHO_SESSION_ID`` per command instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

CONFIG_DIR = Path.home() / ".honcho"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_BASE_URL = "https://api.honcho.dev"

# Env var mapping for runtime overrides.
#
# NOTE: ``api_key`` and ``base_url`` are intentionally NOT here. Both must
# live in ``~/.honcho/config.json`` so there's a single, inspectable source
# of truth for where you're connecting and with what credentials.
# (``honcho init`` still accepts ``--api-key`` / ``HONCHO_API_KEY`` and
# ``--base-url`` / ``HONCHO_BASE_URL`` as one-time pre-fills for the
# write-to-file prompts.)
ENV_MAP: dict[str, str] = {
    "workspace_id": "HONCHO_WORKSPACE_ID",
    "peer_id": "HONCHO_PEER_ID",
    "session_id": "HONCHO_SESSION_ID",
}


@dataclass
class CLIConfig:
    """CLI configuration with layered resolution: flag > env > file > default.

    ``workspace_id`` / ``peer_id`` / ``session_id`` exist on this dataclass so
    flag/env overrides flow through ``get_client_kwargs()``, but they are
    never read from or written to the config file — they're per-command.
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    workspace_id: str = ""
    peer_id: str = ""
    session_id: str = ""

    @classmethod
    def load(cls) -> CLIConfig:
        """Load config from file, then overlay env vars."""
        config = cls()

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}

            if isinstance(data, dict):
                url = data.get("environmentUrl")
                if isinstance(url, str) and url:
                    config.base_url = url
                key = data.get("apiKey")
                if isinstance(key, str):
                    config.api_key = key

        for fld_name, env_var in ENV_MAP.items():
            val = os.environ.get(env_var)
            if val:
                setattr(config, fld_name, val)
            elif val == "":
                # SDK reads these env vars directly and crashes on empty
                # strings with a Pydantic ValidationError. Drop them so the
                # SDK falls back to kwargs / defaults.
                os.environ.pop(env_var, None)

        return config

    def save(self) -> None:
        """Write ``apiKey`` + ``environmentUrl`` to config.json.

        Preserves unrelated top-level keys (``hosts``, ``sessions``,
        ``saveMessages``, ``sessionStrategy``, …) that other tools write.
        """
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        data: dict = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
            except (json.JSONDecodeError, OSError):
                data = {}

        data["environmentUrl"] = self.base_url
        if self.api_key:
            data["apiKey"] = self.api_key
        else:
            data.pop("apiKey", None)

        CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")

    def redacted(self) -> dict[str, str]:
        """Return config dict with api_key redacted."""
        d: dict[str, str] = {}
        for fld in fields(self):
            val = getattr(self, fld.name)
            if fld.name == "api_key" and val:
                d[fld.name] = val[:8] + "..." + val[-4:] if len(val) > 16 else "***"
            else:
                d[fld.name] = val
        return d


def get_client_kwargs(config: CLIConfig) -> dict:
    """Build kwargs for Honcho client from config."""
    kwargs: dict = {}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.workspace_id:
        kwargs["workspace_id"] = config.workspace_id
    return kwargs
