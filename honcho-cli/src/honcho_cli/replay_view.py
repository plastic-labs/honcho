"""What `honcho workspace replay` shows a person: progress, the plan, the summary.

The command builds one report dict per run. In JSON mode that dict is printed
as is; otherwise the functions here render the same dict for reading. Progress
always goes to stderr, so stdout stays a single JSON value or a single report.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from honcho_cli.branding import ICON_FAIL, ICON_OK
from honcho_cli.output import console, stdout_console

# A dry run of a large workspace lists this many sessions; --json has them all.
MAX_SESSION_ROWS = 40
# Without a terminal, a long step logs a progress line this often.
LOG_INTERVAL_SECONDS = 15.0
# Below this width the session table would wrap every cell; use a list instead.
MIN_TABLE_WIDTH = 100

_ACTION_ORDER = {"conflict": 0, "resume": 1, "up_to_date": 2, "create": 3}
_ACTION_STYLE = {
    "conflict": "red",
    "resume": "yellow",
    "up_to_date": "dim",
    "create": "green",
}


def plural(count: int, noun: str) -> str:
    """``1 session`` / ``2 sessions``."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def format_duration(seconds: float) -> str:
    """Seconds as ``14s``, ``3m 05s`` or ``1h 02m``."""
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _format_time(value: str | None) -> str:
    """A timestamp to the minute, in UTC."""
    if not value:
        return "-"
    return f"{datetime.fromisoformat(value).astimezone(UTC):%Y-%m-%d %H:%M}"


class Reporter:
    """Progress on stderr: one live status line in a terminal, plain lines elsewhere.

    ``line`` prints something that stays. ``tick`` reports progress inside a
    long step: it updates the status line in a terminal, and otherwise logs at
    most one line every ``LOG_INTERVAL_SECONDS`` so a log file shows the run is
    alive without one line per request.
    """

    def __init__(self) -> None:
        """Use a live status line only when stderr is a terminal."""
        self._status = (
            console.status("", spinner="dots") if console.is_terminal else None
        )
        self._phase = ""
        self._last_log = time.monotonic()

    def __enter__(self) -> Reporter:
        """Start the status line."""
        if self._status is not None:
            self._status.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Stop the status line, also when the command exits with an error."""
        if self._status is not None:
            self._status.stop()

    def phase(self, text: str) -> None:
        """Name the step now running."""
        self._phase = text
        if self._status is not None:
            self._status.update(f"[dim]{escape(text)}[/dim]")
        else:
            self.line(text)

    def line(self, text: str) -> None:
        """Print a progress line that stays on screen and in logs."""
        # soft_wrap: one progress line stays one line in a log file.
        console.print(f"[dim]{escape(text)}[/dim]", soft_wrap=True)
        self._last_log = time.monotonic()
        if self._status is not None:
            self._status.update(f"[dim]{escape(self._phase)}[/dim]")

    def tick(self, label: str, done: int, total: int | None) -> None:
        """Report progress inside the current step."""
        text = f"{label}: {done}/{total}" if total else f"{label}: {done} messages sent"
        if self._status is not None:
            self._status.update(f"[dim]{escape(text)}[/dim]")
        elif time.monotonic() - self._last_log >= LOG_INTERVAL_SECONDS:
            self.line(text)


def _facts(rows: list[tuple[str, str | Text]]) -> Table:
    """A borderless label/value block, indented under the headline."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column(overflow="fold")
    for label, value in rows:
        grid.add_row(f"  {label}", value)
    return grid


def _endpoint(ref: dict[str, Any], note: str = "") -> Text:
    """A workspace and the instance it lives on, with an optional second line."""
    text = Text(str(ref["workspace_id"]), style="bold")
    text.append(f" at {ref['url']}", style="not bold")
    if note:
        text.append(f"\n{note}", style="not bold")
    return text


def _not_copied(report: dict[str, Any]) -> tuple[str, str]:
    """The report row for memory a replay leaves behind in the source."""
    count = report["not_copied"]["conclusions_in_source"]
    if count is None:
        what = "Conclusions and peer cards."
    elif count:
        what = f"{plural(count, 'conclusion')} and any peer cards."
    else:
        what = "Peer cards. The source has no conclusions."
    return (
        "Not copied",
        f"{what} The destination derives its own from the messages. "
        "Conclusions and peer cards written through the API are not recreated.",
    )


def _join(parts: list[str]) -> str:
    """Join the non-empty parts of one report row."""
    return " · ".join(part for part in parts if part)


def _session_action(session: dict[str, Any]) -> tuple[str, str]:
    """``(what the destination has, what a replay will do)`` for one session."""
    action, present = session["action"], session["messages_in_destination"]
    if action == "conflict":
        return f"differs at message {session['conflict_at']}", "conflict, skipped"
    if action == "resume":
        return f"has {present}", f"resume, send {session['messages_to_send']}"
    if action == "up_to_date":
        return f"has {present}", "up to date"
    return "-", "create"


def _session_table(sessions: list[dict[str, Any]]) -> Table:
    """One row per session, for a terminal wide enough to hold it."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("session", overflow="fold")
    table.add_column("messages", justify="right", no_wrap=True)
    table.add_column("newest message (UTC)", no_wrap=True)
    table.add_column("destination", no_wrap=True)
    table.add_column("action", no_wrap=True)
    for session in sessions:
        in_destination, label = _session_action(session)
        table.add_row(
            Text(str(session["id"])),
            str(session["message_count"] or 0),
            _format_time(session["last_message_at"]),
            in_destination,
            Text(label, style=_ACTION_STYLE[session["action"]]),
        )
    return table


def _session_list(sessions: list[dict[str, Any]]) -> Table:
    """The session table for a narrow terminal: a few short lines per session."""
    grid = Table.grid()
    grid.add_column(overflow="fold")
    for session in sessions:
        in_destination, label = _session_action(session)
        head = Text(f"  {session['id']}  ", style="bold")
        head.append(label, style=f"not bold {_ACTION_STYLE[session['action']]}")
        grid.add_row(head)
        grid.add_row(
            Text(
                f"      {plural(session['message_count'] or 0, 'message')}, newest "
                f"{_format_time(session['last_message_at'])} UTC",
                style="dim",
            )
        )
        if session["action"] != "create":
            grid.add_row(Text(f"      destination {in_destination}", style="dim"))
    return grid


def print_plan(report: dict[str, Any]) -> None:
    """Render a dry-run report."""
    out = stdout_console
    actions = report["actions"]
    dest_sessions = report["destination"]["sessions"]
    conflicting = sum(
        s["message_count"] or 0 for s in report["sessions"] if s["action"] == "conflict"
    )
    already = report["messages_in_sessions"] - report["messages_to_send"] - conflicting

    out.print("[bold]Dry run.[/bold] Nothing was written.\n")
    out.print(
        _facts(
            [
                ("From", _endpoint(report["source"])),
                (
                    "To",
                    _endpoint(
                        report["destination"],
                        f"It has {plural(dest_sessions, 'session')}."
                        if dest_sessions
                        else "It has no sessions yet.",
                    ),
                ),
                (
                    "Sessions",
                    _join(
                        [
                            f"{report['sessions_to_replay']} of "
                            f"{report['sessions_in_source']} in the source",
                            f"{actions['create']} to create"
                            if actions["create"]
                            else "",
                            f"{actions['resume']} to resume"
                            if actions["resume"]
                            else "",
                            f"{actions['up_to_date']} up to date"
                            if actions["up_to_date"]
                            else "",
                            f"{actions['conflict']} in conflict"
                            if actions["conflict"]
                            else "",
                        ]
                    ),
                ),
                (
                    "Messages",
                    _join(
                        [
                            f"{report['messages_to_send']} to send",
                            f"{already} already in the destination" if already else "",
                            f"{conflicting} in conflicting sessions"
                            if conflicting
                            else "",
                        ]
                    ),
                ),
                (
                    "Peers",
                    _join(
                        [
                            f"{report['peers_to_copy']} to copy",
                            f"{report['peers_in_destination']} already in the "
                            "destination"
                            if report["peers_in_destination"]
                            else "",
                        ]
                    ),
                ),
                _not_copied(report),
            ]
        )
    )

    sessions = report["sessions"]
    shown = sessions
    if len(sessions) > MAX_SESSION_ROWS:
        # Stable sort: the sessions that need attention first, then source order.
        shown = sorted(sessions, key=lambda s: _ACTION_ORDER[s["action"]])
        shown = shown[:MAX_SESSION_ROWS]
    if shown:
        out.print()
        wide = out.width >= MIN_TABLE_WIDTH
        out.print(_session_table(shown) if wide else _session_list(shown))
    if len(shown) < len(sessions):
        out.print(
            f"[dim]Showing {len(shown)} of {len(sessions)} sessions. "
            "Pass --json for all of them.[/dim]"
        )

    out.print()
    if actions["conflict"]:
        out.print(
            f"[red]{plural(actions['conflict'], 'session')} in conflict.[/red] "
            "The destination already has different messages under the same "
            "session ID. A replay skips these sessions and exits 1. To copy "
            "them, pass a different --to workspace."
        )
    if dest_sessions:
        out.print(
            "The destination already has sessions, so it keeps its own "
            "workspace metadata and configuration."
        )
    if report["peers_in_destination"]:
        out.print(
            "Peers already in the destination take the source's metadata "
            "and configuration."
        )
    out.print("Run the same command without --dry-run to replay.")


def print_summary(report: dict[str, Any]) -> None:
    """Render the report of a finished replay."""
    out = stdout_console
    actions = report["actions"]
    replayed, failed = report["sessions_replayed"], report["failed"]
    elapsed = format_duration(report["elapsed_seconds"])

    if failed:
        out.print(
            f"{ICON_FAIL} Replayed {replayed} of "
            f"{plural(replayed + len(failed), 'session')} in {elapsed}. "
            f"[red]{len(failed)} failed.[/red]\n"
        )
    elif replayed and actions["up_to_date"] == replayed:
        out.print(f"{ICON_OK} Already up to date. Nothing was sent ({elapsed}).\n")
    else:
        out.print(f"{ICON_OK} Replayed {plural(replayed, 'session')} in {elapsed}.\n")

    rows: list[tuple[str, str | Text]] = [
        ("From", _endpoint(report["source"])),
        ("To", _endpoint(report["destination"])),
        (
            "Sessions",
            _join(
                [
                    f"{actions['created']} created" if actions["created"] else "",
                    f"{actions['resumed']} resumed" if actions["resumed"] else "",
                    f"{actions['up_to_date']} already up to date"
                    if actions["up_to_date"]
                    else "",
                    f"{len(failed)} failed" if failed else "",
                ]
            )
            or "none",
        ),
        (
            "Messages",
            _join(
                [
                    f"{report['messages_created']} sent",
                    f"{report['messages_skipped']} already present"
                    if report["messages_skipped"]
                    else "",
                ]
            ),
        ),
        (
            "Peers",
            _join(
                [
                    f"{report['peers_created']} created",
                    f"{report['peers_existing']} already existed and took "
                    "the source's settings"
                    if report["peers_existing"]
                    else "",
                ]
            ),
        ),
    ]
    rows.append(_not_copied(report))
    if not report["workspace_settings_copied"]:
        rows.append(
            (
                "Settings",
                "The destination already had sessions, so it kept its own "
                "workspace metadata and configuration.",
            )
        )
    out.print(_facts(rows))

    if failed:
        out.print("\n[bold]Failed sessions[/bold]")
        table = Table.grid(padding=(0, 2))
        table.add_column(overflow="fold")
        table.add_column(style="red", no_wrap=True)
        table.add_column(overflow="fold")
        for entry in failed:
            table.add_row(
                Text(f"  {entry['session_id']}"), entry["reason"], Text(entry["error"])
            )
        out.print(table)
        reasons = {entry["reason"] for entry in failed}
        if "error" in reasons:
            out.print(
                "Rerun the same command to retry. Messages already sent are skipped."
            )
        if "conflict" in reasons:
            out.print(
                "A conflict does not clear on a rerun. Pass a different --to "
                "workspace to copy those sessions."
            )

    if report["check_derivation"]:
        out.print(
            "\nThe destination derives conclusions in the background. To watch the queue:"
        )
        # soft_wrap: the command must stay on one line so it can be copied.
        out.print(f"  {report['check_derivation']}", markup=False, soft_wrap=True)
