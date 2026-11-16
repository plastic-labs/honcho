"""Ingest commands: import, status, derive.

Load a connector's export into an auto workspace without deriving it, review
what it became, exclude what shouldn't be modeled, then derive:

    honcho workspace create acme --auto
    honcho ingest import records.jsonl -w acme
    honcho ingest status -w acme
    honcho peer exclude 3 -w acme
    honcho ingest derive -w acme --watch

Changes made before `ingest derive` apply to what is derived: configuration is
read when records are released. Needs a workspace or admin key.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any, NoReturn, Optional, TypeVar

import typer
from honcho import ConflictError, Honcho, NotFoundError, Peer
from honcho.api_types import (
    IngestRecord,
    IngestStatusResponse,
    PeerConfig,
    ReasoningConfiguration,
    SessionConfiguration,
)
from pydantic import ValidationError
from rich.table import Table
from rich.text import Text

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.commands.workspace import _handle_error
from honcho_cli.common import add_common_options, get_client, handle_cmd_flags
from honcho_cli.ingest_refs import resolve, save_listing
from honcho_cli.replay import read_only
from honcho_cli.output import print_error, print_json, status, stdout_console, use_json

app = typer.Typer(
    cls=HonchoTyperGroup,
    help="Import source records into an auto workspace, review them, and derive.",
)
add_common_options(app)

CHUNK = 1000
"""Records read from the file per `ingest` call; the SDK sends 100 per request."""

T = TypeVar("T")


def _credential(client: Honcho) -> str:
    """The client's API key, which keys the saved numbering by tenant."""
    return getattr(client._http, "api_key", None) or ""  # pyright: ignore[reportPrivateUsage]


def _ingest_error(e: Exception, workspace_id: str) -> NoReturn:
    if isinstance(e, NotFoundError):
        print_error(
            "WORKSPACE_NOT_FOUND",
            f"Workspace '{workspace_id}' not found. Create it with"
            + f" `honcho workspace create {workspace_id} --auto`.",
            {"workspace": workspace_id},
        )
        raise typer.Exit(1)
    if isinstance(e, ConflictError):
        print_error(
            "NOT_AUTO_WORKSPACE",
            f"Workspace '{workspace_id}' isn't an auto workspace. Ingest writes only"
            + " to auto workspaces: create one with `honcho workspace create NAME --auto`.",
            {"workspace": workspace_id},
        )
        raise typer.Exit(1)
    _handle_error(e, "workspace", workspace_id)
    raise typer.Exit(1)


def _lines(path: str) -> Iterator[tuple[int, str]]:
    if path == "-":
        yield from enumerate(sys.stdin, start=1)
        return
    try:
        with open(path, encoding="utf-8") as handle:
            yield from enumerate(handle, start=1)
    except OSError as e:
        print_error("FILE_ERROR", f"Can't read {path}: {e.strerror}", {"file": path})
        raise typer.Exit(1) from None


def _records(path: str) -> Iterator[IngestRecord]:
    """Each non-blank line as a validated record; a bad line stops the import."""
    for number, line in _lines(path):
        if not line.strip():
            continue
        try:
            yield IngestRecord.model_validate(json.loads(line))
        except (ValueError, ValidationError) as e:
            detail = (
                e.errors(include_url=False, include_input=False)[0]["msg"]
                if isinstance(e, ValidationError)
                else str(e)
            )
            print_error(
                "INVALID_RECORD",
                f"Line {number} isn't a valid ingest record: {detail}",
                {"line": number},
            )
            raise typer.Exit(1) from None


def _chunks(items: Iterable[T], size: int) -> Iterator[list[T]]:
    chunk: list[T] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


@app.command("import")
def import_records(
    file: str = typer.Argument(
        help="JSONL file of ingest records, one per line (`-` reads stdin)"
    ),
    derive: str = typer.Option(
        "later",
        "--derive",
        help="later (default): store without deriving until `honcho ingest derive`;"
        + " now: derive as records arrive",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Import source records, such as a connector's `records.jsonl`.

    Each record becomes a message; its container picks the session and its
    author and participants the peers. Records already imported come back as
    duplicates, so running the same import again is safe.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    if derive not in ("now", "later"):
        print_error("INVALID_ARGUMENT", "--derive must be 'now' or 'later'")
        raise typer.Exit(1)
    client, config = get_client()
    wid = config.workspace_id

    outcomes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    sessions: set[str] = set()
    authors: set[str] = set()
    sent = 0
    try:
        for chunk in _chunks(_records(file), CHUNK):
            result = client.ingest(chunk, derive="now" if derive == "now" else "later")
            for record in result.records:
                outcomes[record.outcome] += 1
                if record.reason:
                    reasons[record.reason] += 1
                if record.session_id:
                    sessions.add(record.session_id)
                if record.peer_id:
                    authors.add(record.peer_id)
            sent += len(chunk)
            if not use_json():
                status(f"sent {sent:,} records")
    except typer.Exit:
        if sent:
            status(f"{sent:,} records were stored before this; importing again is safe.")
        raise
    except Exception as e:
        if sent:
            status(f"{sent:,} records were stored before this; importing again is safe.")
        _ingest_error(e, wid)

    summary: dict[str, Any] = {
        "workspace": wid,
        "records": sent,
        "created": outcomes["created"],
        "duplicate": outcomes["duplicate"],
        "rejected": outcomes["rejected"],
        "rejected_reasons": dict(reasons),
        "sessions": len(sessions),
        "authors": len(authors),
        "derive": derive,
    }
    if use_json():
        print_json(summary)
        return
    stdout_console.print(
        f"{sent:,} records: {outcomes['created']:,} created · "
        + f"{outcomes['duplicate']:,} duplicate · {outcomes['rejected']:,} rejected"
    )
    for reason, count in reasons.most_common():
        stdout_console.print(f"  rejected, {reason}: {count:,}")
    stdout_console.print(f"{len(sessions):,} sessions, {len(authors):,} authors")
    if derive == "later":
        status(
            "Nothing is derived yet. Review with `honcho ingest status`, exclude what"
            + " shouldn't be modeled, then run `honcho ingest derive`."
        )


def _first(items: Iterable[T], limit: int) -> list[T]:
    taken: list[T] = []
    for item in items:
        if len(taken) == limit:
            break
        taken.append(item)
    return taken


def _modeled(modeled: bool) -> Text:
    return Text("yes", style="green") if modeled else Text("no", style="yellow")


def _print_status(
    st: IngestStatusResponse, people: list[Any], sessions: list[Any]
) -> None:
    r = st.records
    stdout_console.print(
        f"Records: {r.deferred:,} deferred · {r.pending:,} pending · {r.queued:,} queued"
    )
    if r.deferred:
        stdout_console.print(f"{st.deferred_tokens:,} tokens waiting to be derived")

    people_table = Table(
        title=f"People ({st.people:,}, {st.people_not_modeled:,} not modeled)",
        show_header=True,
        header_style="bold",
    )
    people_table.add_column("#", justify="right", style="dim")
    people_table.add_column("name", overflow="fold")
    people_table.add_column("peer", no_wrap=True)
    people_table.add_column("accounts", style="dim", overflow="fold")
    people_table.add_column("messages", justify="right")
    people_table.add_column("modeled")
    for number, p in enumerate(people, start=1):
        accounts = ", ".join(
            sorted({f"{a.source}" for a in p.accounts})
        ) + (f" ({len(p.accounts)})" if len(p.accounts) > 1 else "")
        people_table.add_row(
            str(number), p.display_name or "", p.peer_id, accounts,
            f"{p.messages:,}", _modeled(p.modeled),
        )
    stdout_console.print(people_table)
    if len(people) < st.people:
        status(f"showing {len(people)} of {st.people:,} people; --limit for more")

    sessions_table = Table(
        title=f"Sessions ({st.sessions:,}, {st.sessions_not_modeled:,} not modeled)",
        show_header=True,
        header_style="bold",
    )
    sessions_table.add_column("#", justify="right", style="dim")
    sessions_table.add_column("name", overflow="fold")
    sessions_table.add_column("session", no_wrap=True)
    sessions_table.add_column("source", style="dim")
    sessions_table.add_column("messages", justify="right")
    sessions_table.add_column("modeled")
    for number, s in enumerate(sessions, start=1):
        sessions_table.add_row(
            str(number), s.display_name or "", s.session_id, s.source,
            f"{s.messages:,}", _modeled(s.modeled),
        )
    stdout_console.print(sessions_table)
    if len(sessions) < st.sessions:
        status(f"showing {len(sessions)} of {st.sessions:,} sessions; --limit for more")
    if r.deferred:
        status(
            "Exclude with `honcho peer exclude N` or `honcho session exclude N`,"
            + " then run `honcho ingest derive`."
        )


@app.command("status")
def ingest_status(
    limit: int = typer.Option(50, "--limit", help="People and sessions to list, most records first"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """What the workspace's imports became: records, people and sessions.

    People and sessions are numbered, most records first; `honcho peer exclude
    N` and `honcho session exclude N` take those numbers.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    if limit < 1:
        print_error("INVALID_ARGUMENT", "--limit must be positive")
        raise typer.Exit(1)
    client, config = get_client()
    wid = config.workspace_id
    size = min(limit, 100)
    try:
        st = client.ingest_status()
        people = _first(client.ingest_people(size=size), limit)
        sessions = _first(client.ingest_sessions(size=size), limit)
    except Exception as e:
        _ingest_error(e, wid)
    save_listing(
        str(client.base_url),
        _credential(client),
        wid,
        [p.peer_id for p in people],
        [s.session_id for s in sessions],
    )
    if use_json():
        print_json(
            {
                "status": st.model_dump(mode="json"),
                "people": [p.model_dump(mode="json") for p in people],
                "sessions": [s.model_dump(mode="json") for s in sessions],
            }
        )
        return
    _print_status(st, people, sessions)


STALL_SECONDS = 90
"""How long `derive --watch` waits without progress before explaining why."""


@app.command("derive")
def derive_records(
    watch: bool = typer.Option(False, "--watch", help="Follow the queue until derivation finishes"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Derive the records imported with `--derive later`.

    Releases every deferred record for derivation. Exclusions made before this
    apply; changes made after it only affect records released later.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()
    wid = config.workspace_id
    try:
        released = client.ingest_derive()
    except Exception as e:
        _ingest_error(e, wid)
    if not watch:
        if use_json():
            print_json(released.model_dump())
            return
        stdout_console.print(f"Released {released.pending:,} records for derivation.")
        if released.remaining:
            status(f"{released.remaining:,} are still deferred (records still arriving).")
        status("Follow progress with `honcho ingest derive --watch` or `honcho workspace queue-status`.")
        return
    _watch(client, released.pending)


def _watch(client: Honcho, released: int) -> None:
    if not use_json():
        stdout_console.print(f"Released {released:,} records for derivation.")
    last_left: int | None = None
    last_change = time.monotonic()
    explained = False
    try:
        while True:
            # Released records are queued right away, or by a background sweep
            # when their session was busy; until then they aren't in the queue.
            # Read them first: a sweep between the two reads then shows up in
            # the queue rather than being missed.
            pending = client.ingest_status().records.pending
            queue = client.queue_status()
            left = queue.pending_work_units + queue.in_progress_work_units
            if left == 0 and pending == 0:
                break
            if left != last_left:
                last_left, last_change = left, time.monotonic()
                if not use_json():
                    status(f"{queue.completed_work_units:,} done · {left:,} to go")
            elif not explained and time.monotonic() - last_change > STALL_SECONDS:
                explained = True
                status(
                    "The rest may be waiting to batch: the server derives a small"
                    + " group of messages once it has waited a while (30 minutes by"
                    + " default)."
                )
            time.sleep(2)
    except KeyboardInterrupt:
        status("Stopped watching; derivation continues on the server.")
        raise typer.Exit(130) from None
    if use_json():
        print_json({"released": released, "finished": True})
    else:
        stdout_console.print("Derivation finished.")


# ---------------------------------------------------------------------------
# Exclusions, registered as `honcho peer exclude/include` and
# `honcho session exclude/include`.
# ---------------------------------------------------------------------------


def _note_derived(client: Honcho) -> None:
    """Say that an exclusion doesn't undo what was already derived."""
    try:
        queued = client.ingest_status().records.queued
    except Exception:
        return
    if queued:
        status(
            "Applies to records not yet released; conclusions already derived"
            + " stay as they are."
        )


def set_peer_modeled(ref: str, modeled: bool) -> None:
    """Turn derivation about an imported person off or on (`observe_me`)."""
    client, config = get_client()
    wid = config.workspace_id
    try:
        people = [(p.peer_id, p.display_name) for p in client.ingest_people(size=100)]
    except Exception as e:
        _ingest_error(e, wid)
    peer_id = resolve(
        ref, kind="people", base_url=str(client.base_url),
        credential=_credential(client), workspace_id=wid, candidates=people,
    )
    try:
        # `observe_me` is a peer's only configuration, so nothing else to keep.
        # The listing just found the workspace, so the update mustn't create it.
        with read_only(client):
            Peer(peer_id, client).set_configuration(PeerConfig(observe_me=modeled))
    except Exception as e:
        _handle_error(e, "peer", peer_id)
    name = dict(people).get(peer_id)
    if use_json():
        print_json({"peer_id": peer_id, "display_name": name, "modeled": modeled})
        return
    label = f"{name} ({peer_id})" if name else peer_id
    stdout_console.print(f"{label}: {'modeled' if modeled else 'not modeled'}")
    _note_derived(client)


def set_session_modeled(ref: str, modeled: bool) -> None:
    """Turn reasoning for an imported session off or on; summaries continue."""
    client, config = get_client()
    wid = config.workspace_id
    try:
        listed = [
            (s.session_id, s.display_name) for s in client.ingest_sessions(size=100)
        ]
    except Exception as e:
        _ingest_error(e, wid)
    session_id = resolve(
        ref, kind="sessions", base_url=str(client.base_url),
        credential=_credential(client), workspace_id=wid, candidates=listed,
    )
    try:
        # Read through the list, which never creates; the update replaces the
        # whole configuration, so everything else in it is sent back as is.
        # The listing just found the workspace, so neither the read nor the
        # update may create it.
        with read_only(client):
            session = next(iter(client.sessions(filters={"id": session_id})), None)
            if session is None:
                print_error(
                    "SESSION_NOT_FOUND",
                    f"Session '{session_id}' no longer exists.",
                    {"session": session_id},
                )
                raise typer.Exit(1)
            current = session.configuration or SessionConfiguration()
            reasoning = (current.reasoning or ReasoningConfiguration()).model_copy(
                update={"enabled": modeled}
            )
            session.set_configuration(
                current.model_copy(update={"reasoning": reasoning})
            )
    except typer.Exit:
        raise
    except Exception as e:
        _handle_error(e, "session", session_id)
    name = dict(listed).get(session_id)
    if use_json():
        print_json({"session_id": session_id, "display_name": name, "modeled": modeled})
        return
    label = f"{name} ({session_id})" if name else session_id
    stdout_console.print(f"{label}: {'modeled' if modeled else 'not modeled'}")
    _note_derived(client)
