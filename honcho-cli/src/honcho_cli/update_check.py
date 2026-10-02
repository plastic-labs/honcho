"""Once-a-day stderr notice when a newer honcho-cli is on PyPI.

Fail-open: any error is swallowed. Cache is ``update-check.json`` beside
config, not ``config.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

from honcho_cli import __version__
from honcho_cli.branding import ICON_RUN
from honcho_cli.config import _config_dir
from honcho_cli.output import console, use_json

_INTERVAL_S = 24 * 60 * 60
_PYPI_URL = "https://pypi.org/pypi/honcho-cli/json"


def maybe_print_update_nag() -> None:
    if use_json() or "--json" in sys.argv:
        return
    if os.environ.get("HONCHO_NO_UPDATE_CHECK", "").lower() in ("1", "true"):
        return
    try:
        path = _config_dir() / "update-check.json"
        now = time.time()
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            cache = {}
        if isinstance(cache, dict) and now - float(cache.get("t") or 0) < _INTERVAL_S:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"t": now}), encoding="utf-8")
        latest = httpx.get(_PYPI_URL, timeout=1.0).json()["info"]["version"]
        if not isinstance(latest, str) or not _is_newer(latest, __version__):
            return
        console.print(f"  {ICON_RUN}  honcho-cli {latest} is available (you have {__version__})")
        console.print("     [dim]uv tool upgrade honcho-cli[/dim]")
    except Exception:
        return


def _is_newer(latest: str, current: str) -> bool:
    def parts(version: str) -> tuple[int, ...]:
        out: list[int] = []
        for segment in version.lstrip("v").split("."):
            num = ""
            for ch in segment:
                if ch.isdigit():
                    num += ch
                else:
                    break
            if not num:
                break
            out.append(int(num))
        return tuple(out) or (0,)

    a, b = parts(latest), parts(current)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) > b + (0,) * (n - len(b))
