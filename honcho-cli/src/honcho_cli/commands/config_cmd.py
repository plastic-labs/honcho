"""Config inspection command: ``honcho config show``.

Writing to ``~/.honcho/config.json`` is done only via ``honcho init``, which
manages the two CLI-owned keys (``apiKey`` + ``environmentUrl``).
Workspace / peer / session scoping is per-command via flags / env vars, not
persisted defaults.
"""

from __future__ import annotations

import typer

from honcho_cli.config import CLIConfig
from honcho_cli.output import print_result

app = typer.Typer(help="Inspect CLI configuration.")


@app.command()
def show() -> None:
    """Show current config (api key redacted)."""
    config = CLIConfig.load()
    print_result(config.redacted())
