"""Output formatting: JSON, tables, and structured errors.

Detects TTY to auto-switch between human-readable and machine-parseable output.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from honcho_cli.branding import ICON_FAIL, ICON_OK, ICON_RUN

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


def step(msg: str) -> None:
    """Print a progress step. No-op in JSON mode."""
    if not use_json():
        console.print(f"  {ICON_RUN}  {msg}")


def ok(msg: str) -> None:
    """Print a success line. No-op in JSON mode."""
    if not use_json():
        console.print(f"  {ICON_OK}  {msg}")


def fail(msg: str) -> None:
    """Print a failure line. No-op in JSON mode."""
    if not use_json():
        console.print(f"  {ICON_FAIL}  {msg}")


# Stable peer-color palette for transcript rendering. Brand blue first so the
# primary peer lands on brand when there's only one speaker.
_PEER_COLORS = (
    "#B6DAFD",  # brand
    "#9ccfd8",  # foam
    "#c4a7e7",  # iris
    "#ebbcba",  # rose
    "#f6c177",  # gold
    "#a3be8c",  # pine-ish green
    "#ea9a97",  # love
)


#: Rendered width of :func:`_format_timestamp` output.
TIMESTAMP_WIDTH = len("2026-01-01T00:00:00.000Z")


def _format_timestamp(value: Any) -> str:
    """Compact UTC timestamp: ``YYYY-MM-DDTHH:MM:SS.mmmZ``.

    Offsets are converted to UTC; naive values are assumed UTC. Unparseable
    values pass through verbatim.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip())
        except ValueError:
            return str(value).strip()
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return f"{parsed:%Y-%m-%dT%H:%M:%S}.{parsed.microsecond // 1000:03d}Z"


def print_transcript(
    messages: list[dict[str, Any]],
    *,
    session_id: str,
    total: int | None = None,
    page: int | None = None,
    pages: int | None = None,
    show_ids: bool = False,
    next_page_hint: str | None = None,
) -> None:
    """Render a session transcript as a row-delimited table, or JSON.

    Each message dict must have ``peer_id``, ``content``, ``created_at``;
    ``id`` is optional and only shown when ``show_ids`` is set.
    ``next_page_hint`` is printed below the table when given.
    """
    if use_json():
        print_json(messages)
        return

    shown = len(messages)
    parts = [f"session {session_id}"]
    if page is not None and pages is not None:
        parts.append(f"page {page}/{pages}")
    elif page is not None:
        parts.append(f"page {page}")
    if total is not None and shown != total:
        parts.append(f"showing {shown} of {total}")
    else:
        parts.append(f"{shown} message{'s' if shown != 1 else ''}")
    title = " · ".join(parts)

    if not messages:
        stdout_console.print(f"[dim]── {title} ──[/dim]")
        stdout_console.print("[dim]  (empty)[/dim]")
        return

    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        show_lines=True,  # delimiters between rows
        expand=True,
        pad_edge=False,
    )
    # time is fixed-width ISO-UTC; ids and peers wrap rather than truncate;
    # content takes the rest.
    table.add_column("time", style="dim", no_wrap=True, width=TIMESTAMP_WIDTH)
    if show_ids:
        table.add_column("id", style="dim", no_wrap=True)
    table.add_column("peer", overflow="fold", max_width=24)
    table.add_column("content", overflow="fold", ratio=1, min_width=40)

    peer_color: dict[str, str] = {}
    for msg in messages:
        peer = str(msg.get("peer_id") or "?")
        if peer not in peer_color:
            peer_color[peer] = _PEER_COLORS[len(peer_color) % len(_PEER_COLORS)]

        # Text, not Markdown or console markup: content renders verbatim.
        row: list[Any] = [_format_timestamp(msg.get("created_at"))]
        if show_ids:
            row.append(Text(str(msg.get("id") or "")))
        row.append(Text(peer, style=f"bold {peer_color[peer]}"))
        row.append(Text(str(msg.get("content") or "")))
        table.add_row(*row)

    stdout_console.print(table)
    if next_page_hint:
        status(f"more: {next_page_hint}")
