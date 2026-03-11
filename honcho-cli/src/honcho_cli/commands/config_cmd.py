"""Config management commands: init, set, show."""

from __future__ import annotations

from dataclasses import fields
from typing import Optional

import typer

from honcho_cli.config import CLIConfig
from honcho_cli.output import print_error, print_result, status

app = typer.Typer(help="Manage CLI configuration.")


@app.command()
def init(
    base_url: str = typer.Option("https://api.honcho.dev", prompt="Base URL"),
    api_key: str = typer.Option("", prompt="API key (admin JWT)"),
    workspace_id: str = typer.Option("", prompt="Default workspace ID"),
) -> None:
    """Interactive setup: set base_url, api_key, default workspace."""
    config = CLIConfig(
        base_url=base_url,
        api_key=api_key,
        workspace_id=workspace_id,
    )
    config.save()
    status(f"Config saved to {config.save.__func__}")
    print_result(config.redacted())


@app.command("set")
def set_value(
    key: str = typer.Argument(help="Config key (base_url, api_key, workspace_id, peer_id, session_id)"),
    value: str = typer.Argument(help="Config value"),
) -> None:
    """Set a config value."""
    valid_keys = {f.name for f in fields(CLIConfig)}
    if key not in valid_keys:
        print_error("INVALID_KEY", f"Unknown config key: {key}", {"valid_keys": sorted(valid_keys)})
        raise typer.Exit(1)

    config = CLIConfig.load()
    setattr(config, key, value)
    config.save()
    status(f"Set {key}")


@app.command()
def show() -> None:
    """Show current config (redacted keys)."""
    config = CLIConfig.load()
    print_result(config.redacted())
