"""Experimental commands: listed with a label, run only when opted in.

Ingest into auto workspaces is under active development, and its commands may
change or stop working without notice. They show in help marked
"Experimental:", and run only when ``HONCHO_EXPERIMENTAL=1`` is set; without
it they say how to opt in and exit 1.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import click
import typer

from honcho_cli.output import print_error, set_json_mode

ENV_VAR = "HONCHO_EXPERIMENTAL"
LABEL = "Experimental:"

P = ParamSpec("P")
R = TypeVar("R")


def enabled() -> bool:
    """Whether experimental commands are turned on."""
    return os.environ.get(ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def require(what: str | None = None) -> None:
    """Exit 1 with how to opt in, unless experimental commands are turned on.

    Args:
        what: What is experimental, for the message. Defaults to the command
            being run.
    """
    if enabled():
        return
    if what is None:
        ctx = click.get_current_context(silent=True)
        what = f"`{ctx.command_path}`" if ctx is not None else "This command"
    print_error(
        "EXPERIMENTAL",
        f"{what} is experimental and off by default: it may change or stop"
        + f" working without notice. Set {ENV_VAR}=1 to use it.",
        {"enable": f"{ENV_VAR}=1"},
    )
    raise typer.Exit(1)


def experimental(command: Callable[P, R]) -> Callable[P, R]:
    """Mark a command experimental: label its help, and run it only when opted in.

    Goes under ``@app.command(...)``, so the command keeps its parameters.
    """

    @functools.wraps(command)
    def gated(*args: P.args, **kwargs: P.kwargs) -> R:
        # Before the command's own flag handling runs, so the refusal honors
        # its `--json` as well as the group's.
        if kwargs.get("json_output"):
            set_json_mode(True)
        require()
        return command(*args, **kwargs)

    gated.__doc__ = f"{LABEL} {command.__doc__ or ''}".rstrip()
    return gated
