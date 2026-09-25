"""Config inspection command: ``honcho config``.

Writing to ``~/.honcho/config.json`` is done only via ``honcho init``, which
manages the two CLI-owned keys (``apiKey`` + ``environmentUrl``).
Workspace / peer / session scoping is per-command via flags / env vars, not
persisted defaults.
"""

from __future__ import annotations

import typer

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.common import handle_cmd_flags
from honcho_cli.config import CLIConfig
from honcho_cli.output import print_result

app = typer.Typer(cls=HonchoTyperGroup, help="Inspect CLI configuration.", invoke_without_command=True)


@app.callback(invoke_without_command=True)
def config(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Show current config (api key redacted)."""
    if ctx.invoked_subcommand is not None:
        return
    handle_cmd_flags(json_output=json_output)
    cfg = CLIConfig.load()
    print_result(cfg.redacted())
