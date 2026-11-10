"""Dream commands: list, inspect.

Representation History: every dream records what it changed in a peer's
representation — the conclusions it created (with the conclusions each was
derived from), the conclusions it removed (with their text as it read when
removed), and the peer card before and after. Runs are kept for the server's
retention window (30 days by default). Needs a workspace or admin key.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import typer
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.commands.workspace import _handle_error
from honcho_cli.common import add_common_options, get_client, handle_cmd_flags
from honcho_cli.output import (
    _format_timestamp,
    print_error,
    print_json,
    status,
    stdout_console,
    use_json,
)
from honcho_cli.validation import validate_resource_id

app = typer.Typer(
    cls=HonchoTyperGroup,
    help="List dreams and inspect what each one changed in a peer's representation.",
)
add_common_options(app)


_STATUS_STYLE = {
    "completed": "green",
    "running": "cyan",
    "failed": "red",
    "interrupted": "yellow",
}

def _short_time(value: datetime | None) -> str:
    """``YYYY-MM-DD HH:MM`` UTC — a dream takes minutes, not milliseconds."""
    return _format_timestamp(value)[:16].replace("T", " ") if value else ""


def _changes(run: Any) -> Text:
    text = Text()
    text.append(f"+{run.created_count}", style="green" if run.created_count else "dim")
    text.append(" ")
    text.append(f"-{run.removed_count}", style="red" if run.removed_count else "dim")
    if run.peer_card_updated:
        text.append("  card", style="cyan")
    return text


def _print_runs(runs: list, title: str) -> None:
    """Dream runs as a table. Run IDs never truncate: they feed `dream inspect`."""
    table = Table(title=title, show_header=True, header_style="bold")
    # min_width holds the id whole in a narrow terminal; the rest wrap instead.
    table.add_column("id", no_wrap=True, min_width=max(len(run.id) for run in runs))
    table.add_column("started (UTC)", style="dim")
    table.add_column("status")
    table.add_column("observer → observed", overflow="fold")
    table.add_column("type", style="dim", overflow="fold")
    table.add_column("changes")
    for run in runs:
        table.add_row(
            run.id,
            _short_time(run.started_at),
            Text(run.status, style=_STATUS_STYLE.get(run.status, "")),
            f"{run.observer_id} → {run.observed_id}",
            " · ".join(part for part in (run.dream_type, run.trigger_reason) if part),
            _changes(run),
        )
    stdout_console.print(table)


@app.command("list")
def list_dreams(
    observer: Optional[str] = typer.Option(
        None,
        "--observer",
        help="Only dreams from this observer's perspective (with -p: one representation)",
    ),
    limit: int = typer.Option(20, "--limit", help="Dreams per page"),
    page: int = typer.Option(1, "--page", help="Page number, newest dreams first"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(
        None, "--peer", "-p", help="Only dreams about this peer (the observed peer)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List dreams in the workspace, newest first.

    Pass -p to see the dreams about one peer, and --observer to narrow to one
    observer's model of that peer. `honcho dream inspect <id>` shows what a
    dream changed.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer)
    if limit < 1 or page < 1:
        print_error("INVALID_ARGUMENT", "--limit and --page must be positive")
        raise typer.Exit(1)
    if observer:
        validate_resource_id(observer, "peer")
    client, config = get_client()

    try:
        # An unset peer is "" in config; "" would filter to no peer at all.
        result = client.dreams(
            observed=config.peer_id or None, observer=observer or None, page=page, size=limit
        )
        if use_json():
            print_json([run.model_dump(mode="json") for run in result.items])
            return
        title = "Dreams"
        if config.peer_id:
            title += f" about {config.peer_id}"
        if result.pages:
            title += f" · page {result.page}/{result.pages}"
        if not result.items:
            status("No dreams recorded in the retention window.")
            return
        _print_runs(result.items, title)
        if result.pages and page < result.pages:
            status(f"more: --page {page + 1}")
    except Exception as e:
        _handle_error(e, "dream", "list")


def _duration(started: datetime, finished: datetime | None) -> str:
    if finished is None:
        return ""
    seconds = int((finished - started).total_seconds())
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def _removed_note(removed_in_run: str | None, run_id: str) -> str:
    if removed_in_run == run_id:
        return "removed in this dream"
    return f"removed by dream {removed_in_run}" if removed_in_run else "removed"


def _source_line(source: Any, run_id: str) -> Text:
    """One source conclusion, marked by what has happened to it since."""
    line = Text("  ")
    if source.status == "current":
        line.append("● ", style="green")
        line.append(source.content or "")
    elif source.status == "removed":
        line.append("✕ ", style="red")
        line.append(source.content or "", style="strike dim")
        line.append(f"  {_removed_note(source.removed_in_run, run_id)}", style="dim red")
    else:
        line.append("? ", style="dim")
        line.append(
            f"{source.id} (unavailable: deleted, or removed longer ago than history is kept)",
            style="dim",
        )
    return line


def _header(detail: Any) -> Panel:
    status_text = Text(detail.status, style=f"bold {_STATUS_STYLE.get(detail.status, '')}")
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("representation", f"{detail.observer_id} → {detail.observed_id}")
    grid.add_row(
        "type",
        " · ".join(part for part in (detail.dream_type, detail.trigger_reason) if part),
    )
    grid.add_row("status", status_text)
    grid.add_row("started", _format_timestamp(detail.started_at))
    if detail.finished_at:
        grid.add_row(
            "finished",
            f"{_format_timestamp(detail.finished_at)}  "
            f"({_duration(detail.started_at, detail.finished_at)})",
        )
    if detail.pruned_at:
        grid.add_row(
            "pruned",
            Text(
                f"{_format_timestamp(detail.pruned_at)}  past retention: only "
                "created conclusions that still exist are shown",
                style="dim",
            ),
        )
    grid.add_row(
        "changes",
        Text.assemble(
            (f"+{detail.created_count} created", "green"),
            "  ",
            (f"-{detail.removed_count} removed", "red"),
            "  ",
            ("peer card updated" if detail.peer_card_updated else "peer card unchanged", "dim"),
        ),
    )
    return Panel(grid, title=f"dream {detail.id}", title_align="left")


def _created_table(detail: Any) -> Table:
    table = Table(
        title=f"Created ({len(detail.created)})",
        title_justify="left",
        show_header=True,
        header_style="bold",
        show_lines=True,
        expand=True,
    )
    table.add_column("specialist", style="dim", no_wrap=True)
    table.add_column("level", no_wrap=True)
    table.add_column("conclusion, and what it was derived from", ratio=1)
    for created in detail.created:
        conclusion = created.conclusion
        body = Text()
        body.append("+ ", style="bold green")
        if conclusion.status == "current":
            body.append(conclusion.content or "")
        elif conclusion.status == "removed":
            body.append(conclusion.content or "", style="strike")
            body.append(
                f"  ({_removed_note(conclusion.removed_in_run, detail.id)})", style="dim red"
            )
        else:
            body.append(f"{conclusion.id} (unavailable)", style="dim")
        lines = [body] + [_source_line(source, detail.id) for source in created.sources]
        table.add_row(created.specialist or "", conclusion.level or "", Group(*lines))
    return table


def _removed_table(detail: Any) -> Table:
    replacements = {c.conclusion.id: c.conclusion.content for c in detail.created}
    table = Table(
        title=f"Removed ({len(detail.removed)})",
        title_justify="left",
        show_header=True,
        header_style="bold",
        show_lines=True,
        expand=True,
    )
    table.add_column("specialist", style="dim", no_wrap=True)
    table.add_column("level", no_wrap=True)
    table.add_column("conclusion, as it read when removed", ratio=1)
    for removed in detail.removed:
        body = Text()
        body.append("- ", style="bold red")
        body.append(removed.content)
        lines = [body]
        if removed.replaced_by:
            replacement = replacements.get(removed.replaced_by) or removed.replaced_by
            lines.append(Text(f"  replaced by: {replacement}", style="dim"))
        table.add_row(removed.specialist or "", removed.level or "", Group(*lines))
    return table


def _peer_card_panel(detail: Any) -> Panel:
    before = list(detail.peer_card.before or [])
    after = list(detail.peer_card.after)
    lines: list[Text] = []
    for entry in before:
        if entry not in after:
            lines.append(Text(f"- {entry}", style="red"))
    for entry in after:
        if entry in before:
            lines.append(Text(f"  {entry}", style="dim"))
        else:
            lines.append(Text(f"+ {entry}", style="green"))
    if not lines:
        lines.append(Text("(empty)", style="dim"))
    return Panel(Group(*lines), title="Peer card", title_align="left")


@app.command()
def inspect(
    run_id: str = typer.Argument(help="Dream run ID, from `honcho dream list`"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Show what one dream changed.

    Lists the conclusions it created, each with the conclusions it was derived
    from; the conclusions it removed, as they read when removed; and the peer
    card before and after. Each source shows where it stands now: current,
    removed (and by which dream), or unavailable once deleted or removed longer
    ago than history is kept.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    validate_resource_id(run_id, "dream")
    client, _config = get_client()

    try:
        detail = client.get_dream(run_id)
    except Exception as e:
        _handle_error(e, "dream", run_id)
        return

    if use_json():
        print_json(detail.model_dump(mode="json"))
        return

    stdout_console.print(_header(detail))
    if detail.created:
        stdout_console.print(_created_table(detail))
    if detail.removed:
        stdout_console.print(_removed_table(detail))
    if detail.peer_card is not None:
        stdout_console.print(_peer_card_panel(detail))
    if not (detail.created or detail.removed or detail.peer_card):
        status("This dream committed no changes.")
