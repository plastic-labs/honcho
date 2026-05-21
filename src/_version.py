"""Single source of truth for the Honcho service version.

Reads pyproject.toml directly so the value never drifts from the authoritative
source. Falls back to installed package metadata for wheel-only deploys where
pyproject.toml may not be shipped.

Used by:
- src/main.py for the FastAPI app `version=...`
- src/telemetry/emitter.py and src/telemetry/events/base.py for event tagging
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import tomllib


def _read_version() -> str:
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        try:
            return _pkg_version("honcho")
        except PackageNotFoundError:
            return "unknown"


HONCHO_VERSION: str = _read_version()
