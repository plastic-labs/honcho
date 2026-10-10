"""Checks run with the target interpreter, using only installed dependencies.

Keep this file compatible with Python 3.8 (the SDK's declared minimum).
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path


def installed_module(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__file__ is not None
    location = Path(module.__file__).resolve()
    assert Path(sys.prefix).resolve() in location.parents, (name, location)
    if hasattr(module, "__path__"):
        for item in pkgutil.walk_packages(module.__path__, name + "."):
            if not item.name.endswith(".__main__"):
                importlib.import_module(item.name)


def sdk_smoke() -> None:
    import httpx
    from honcho import Honcho, MessageCreateParams
    from honcho.api_types import WorkspaceConfiguration
    from honcho.http import AsyncHonchoHTTPClient, HonchoHTTPClient

    installed_module("honcho")
    assert MessageCreateParams(peer_id="alice", content="hello").content == "hello"
    config = WorkspaceConfiguration.model_validate({"reasoning": {"enabled": False}})
    assert config.reasoning is not None and config.reasoning.enabled is False

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v3/workspaces"
        assert request.headers["authorization"] == "Bearer install-smoke"
        return httpx.Response(
            200, json={"items": [], "page": 1, "size": 50, "total": 0}
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as http:
        client = HonchoHTTPClient(
            base_url="http://install.invalid", api_key="install-smoke", http_client=http
        )
        assert client.request("GET", "/v3/workspaces")["items"] == []

    async def check_async() -> None:
        async with httpx.AsyncClient(transport=transport) as http:
            client = AsyncHonchoHTTPClient(
                base_url="http://install.invalid",
                api_key="install-smoke",
                http_client=http,
            )
            assert (await client.request("GET", "/v3/workspaces"))["items"] == []

    asyncio.run(check_async())
    with httpx.Client(transport=transport) as http:
        client = Honcho(
            workspace_id="install-smoke",
            api_key="install-smoke",
            base_url="http://install.invalid",
            http_client=http,
        )
        assert client.workspace_id == "install-smoke"


def cli_smoke() -> None:
    from honcho_cli.local.env import render_stack
    from honcho_cli.local.profile import LocalProfile

    installed_module("honcho_cli")
    executable = Path(sys.executable).parent / (
        "honcho.exe" if os.name == "nt" else "honcho"
    )
    for args in (
        ("--version",),
        ("--help",),
        ("workspace", "--help"),
        ("start", "--help"),
        ("status", "--help"),
    ):
        result = subprocess.run(
            [str(executable), *args], capture_output=True, text=True, timeout=30
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)

    # Rendering exercises packaged Compose/SQL resources without requiring Docker.
    profile = LocalProfile(name="install-smoke")
    render_stack(profile, {})
    assert "services:" in profile.compose_file().read_text(encoding="utf-8")
    assert "vector" in (profile.dir() / "init.sql").read_text(encoding="utf-8")


def main() -> None:
    package, expected_version = sys.argv[1:]
    distribution = "honcho-cli" if package == "cli" else "honcho-ai"
    assert importlib.metadata.version(distribution) == expected_version
    # Base installations must not acquire the server/ORM/provider graph.
    for name in (
        "src",
        "honcho_runtime",
        "sqlalchemy",
        "psycopg",
        "fastapi",
        "openai",
        "honcho_mock_provider",
    ):
        assert importlib.util.find_spec(name) is None, (
            "Unexpected server dependency: " + name
        )
    if package == "sdk":
        assert importlib.util.find_spec("honcho_cli") is None
    sdk_smoke()
    if package == "cli":
        cli_smoke()
    print(
        json.dumps(
            {
                "distribution": distribution,
                "version": expected_version,
                "python": sys.version,
                "prefix": sys.prefix,
                "result": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
