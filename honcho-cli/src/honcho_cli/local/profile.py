"""Named local-stack profiles under ``$HONCHO_CONFIG_DIR/profiles``.

A profile is a Compose project directory, not an auth identity.
Resolution: ``--profile`` > ``HONCHO_PROFILE`` > ``local``.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import suppress
from dataclasses import dataclass, replace

from honcho_cli.local import (
    DEFAULT_API_PORT,
    DEFAULT_DB_PORT,
    DEFAULT_IMAGE,
    DEFAULT_PROFILE,
    DEFAULT_REDIS_PORT,
)
from honcho_cli.output import print_error

_PROFILE_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


def profiles_dir():
    from honcho_cli import config as cfg

    return cfg.CONFIG_DIR / "profiles"


def validate_profile_name(name: str) -> str:
    if name and _PROFILE_NAME.match(name):
        return name
    print_error(
        "INVALID_PROFILE",
        "Profile name must be lowercase alphanumeric, starting with a letter "
        "(hyphens and underscores allowed).",
        {"profile": name},
    )
    raise SystemExit(1)


def resolve_profile_name(flag: str | None) -> str:
    raw = (
        (flag or "").strip()
        or (os.environ.get("HONCHO_PROFILE") or "").strip()
        or DEFAULT_PROFILE
    )
    return validate_profile_name(raw)


def list_profile_names() -> list[str]:
    """Profile directories that already have a Compose file."""
    root = profiles_dir()
    if not root.is_dir():
        return []
    names: list[str] = []
    for path in sorted(root.iterdir()):
        if (
            path.is_dir()
            and _PROFILE_NAME.match(path.name)
            and (path / "docker-compose.yml").exists()
        ):
            names.append(path.name)
    return names


@dataclass
class LocalProfile:
    """Ports and image for one local stack."""

    name: str
    api_port: int = DEFAULT_API_PORT
    db_port: int = DEFAULT_DB_PORT
    redis_port: int = DEFAULT_REDIS_PORT
    image: str = DEFAULT_IMAGE

    @property
    def project_name(self) -> str:
        return f"honcho-{self.name}"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def dir(self):
        return profiles_dir() / self.name

    def compose_file(self):
        return self.dir() / "docker-compose.yml"

    def env_file(self):
        return self.dir() / ".env"

    def profile_file(self):
        return self.dir() / "profile.json"

    def config_file(self):
        return self.dir() / "config.toml"

    def endpoints(self) -> dict[str, str]:
        return {
            "api": self.base_url,
            "docs": f"{self.base_url}/docs",
            "postgres": f"postgresql://postgres:postgres@127.0.0.1:{self.db_port}/postgres",
            "redis": f"redis://127.0.0.1:{self.redis_port}/0",
        }

    def overlay(self, **fields) -> LocalProfile:
        return replace(self, **{k: v for k, v in fields.items() if v is not None})


def load_profile(name: str) -> LocalProfile:
    profile = LocalProfile(name=validate_profile_name(name))
    path = profile.profile_file()
    if not path.exists():
        return profile
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return profile
    if not isinstance(data, dict):
        return profile
    image = data.get("image")
    return replace(
        profile,
        api_port=_port(data.get("apiPort"), profile.api_port),
        db_port=_port(data.get("dbPort"), profile.db_port),
        redis_port=_port(data.get("redisPort"), profile.redis_port),
        image=image if isinstance(image, str) and image else profile.image,
    )


def save_profile(profile: LocalProfile) -> None:
    directory = profile.dir()
    directory.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        os.chmod(directory, 0o700)
    payload = {
        "apiPort": profile.api_port,
        "dbPort": profile.db_port,
        "redisPort": profile.redis_port,
        "image": profile.image,
    }
    profile.profile_file().write_text(json.dumps(payload, indent=2) + "\n")


def _port(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return parsed if 1 <= parsed <= 65535 else default
