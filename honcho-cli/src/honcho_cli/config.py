"""Configuration management for Honcho CLI.

Config stored at ~/.honcho/config.toml with env var overrides.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]


CONFIG_DIR = Path.home() / ".honcho"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Env var mapping: field_name -> env var
ENV_MAP: dict[str, str] = {
    "base_url": "HONCHO_BASE_URL",
    "api_key": "HONCHO_API_KEY",
    "workspace_id": "HONCHO_WORKSPACE_ID",
    "peer_id": "HONCHO_PEER_ID",
    "session_id": "HONCHO_SESSION_ID",
}


@dataclass
class CLIConfig:
    """CLI configuration with layered resolution: flag > env > file > default."""

    base_url: str = "https://api.honcho.dev"
    api_key: str = ""
    workspace_id: str = ""
    peer_id: str = ""
    session_id: str = ""

    @classmethod
    def load(cls) -> CLIConfig:
        """Load config from file, then overlay env vars."""
        config = cls()

        # Layer 1: config file
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "rb") as f:
                data = tomllib.load(f)
            for fld in fields(cls):
                if fld.name in data:
                    setattr(config, fld.name, data[fld.name])

        # Layer 2: env vars
        for fld_name, env_var in ENV_MAP.items():
            val = os.environ.get(env_var)
            if val:
                setattr(config, fld_name, val)

        return config

    def save(self) -> None:
        """Write current config to ~/.honcho/config.toml."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        lines = []
        for fld in fields(self):
            val = getattr(self, fld.name)
            lines.append(f'{fld.name} = "{val}"')
        CONFIG_FILE.write_text("\n".join(lines) + "\n")

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
