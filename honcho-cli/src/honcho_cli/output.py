"""Output formatting: JSON, tables, and structured errors.

Detects TTY to auto-switch between human-readable and machine-parseable output.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console(stderr=True)
stdout_console = Console()


def is_tty() -> bool:
    """Check if stdout is a TTY."""
    return sys.stdout.isatty()


# Global state for --json flag
_force_json = False


def set_json_mode(enabled: bool) -> None:
    global _force_json
    _force_json = enabled



def use_json() -> bool:
    """Should we output JSON?"""
    return _force_json or os.environ.get("HONCHO_JSON", "").lower() in ("1", "true") or not is_tty()


def print_json(data: Any) -> None:
    """Print a single JSON value to stdout."""
    print(json.dumps(data, indent=2, default=str))


def print_table(columns: list[str], rows: list[list[str]], title: str | None = None) -> None:
    """Print a rich table to stdout."""
    table = Table(title=title, show_header=True, header_style="bold")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    stdout_console.print(table)


def print_result(data: Any, columns: list[str] | None = None, title: str | None = None) -> None:
    """Print data as JSON or table depending on mode.

    For lists, uses JSON arrays in JSON mode or tables in TTY mode.
    For dicts, uses JSON or key-value display.
    """
    if use_json():
        print_json(data)
    else:
        if isinstance(data, list) and columns:
            rows = []
            for item in data:
                row = [str(item.get(col, "")) if isinstance(item, dict) else str(item) for col in columns]
                rows.append(row)
            print_table(columns, rows, title=title)
        elif isinstance(data, dict):
            table = Table(show_header=False)
            table.add_column("Field", style="bold")
            table.add_column("Value")
            for k, v in data.items():
                val = json.dumps(v, default=str) if isinstance(v, (dict, list)) else str(v)
                table.add_row(k, val)
            stdout_console.print(table)
        else:
            stdout_console.print(data)


def print_error(code: str, message: str, details: dict | None = None) -> None:
    """Print structured error."""
    err = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details:
        err["error"]["details"] = details

    if use_json():
        print(json.dumps(err, default=str), file=sys.stderr)
    else:
        console.print(f"[red]Error[/red] ({code}): {message}")
        if details:
            for k, v in details.items():
                console.print(f"  {k}: {v}")


def status(msg: str) -> None:
    """Print a status message to stderr."""
    console.print(f"[dim]{msg}[/dim]")
