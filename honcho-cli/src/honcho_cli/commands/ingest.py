"""Ingest commands: import, from, status, suggestions, derive, reset, link, unlink.

Experimental, and off unless `HONCHO_EXPERIMENTAL=1` is set (see
`honcho_cli.experimental`).

Load a connector's export into an auto workspace without deriving it, review
what it became, exclude what shouldn't be modeled, then derive:

    honcho workspace create acme --auto
    honcho ingest import records.jsonl -w acme
    honcho ingest status -w acme
    honcho peer exclude 3 -w acme
    honcho ingest derive -w acme --watch

Two accounts of one person (a dry run shows them apart) are linked, then the
import runs again: `honcho ingest reset`, `honcho ingest link 1 2`,
`honcho ingest import`. A link changes who an account maps to; it never moves
what was already imported.

Changes made before `ingest derive` apply to what is derived: configuration is
read when records are released. Needs a workspace or admin key.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, NoReturn, Optional, TypeVar

import typer
from honcho import ConflictError, Honcho, NotFoundError, Peer, filter_bot_records
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
from honcho_cli import ingest_sources
from honcho_cli.experimental import experimental
from honcho_cli.experimental import require as require_experimental
from honcho_cli.ingest_refs import (
    resolve,
    save_listing,
    save_suggestions,
    saved_suggestion,
)
from honcho_cli.output import print_error, print_json, status, stdout_console, use_json
from honcho_cli.replay import read_only

app = typer.Typer(
    cls=HonchoTyperGroup,
    help="Experimental: import source records into an auto workspace, review them,"
    + " and derive. Off by default: set HONCHO_EXPERIMENTAL=1.",
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


_DERIVE_HELP = (
    "later (default): store without deriving until `honcho ingest derive`;"
    + " now: derive as records arrive"
)
_BOTS_HELP = (
    "Bot records to import. interacted (default): only those a person replied to"
    + " or mentioned the bot about; all; none"
)


def _check_choices(derive: str, bots: str) -> None:
    if derive not in ("now", "later"):
        print_error("INVALID_ARGUMENT", "--derive must be 'now' or 'later'")
        raise typer.Exit(1)
    if bots not in ("interacted", "all", "none"):
        print_error("INVALID_ARGUMENT", "--bots must be 'interacted', 'all' or 'none'")
        raise typer.Exit(1)


def _hold_back(
    records: Iterable[IngestRecord], bots: str
) -> tuple[Iterable[IngestRecord], int]:
    """The records to send, and how many bot records `bots` holds back.

    With `all` the records stream through; otherwise they're read whole, since
    a reply can come after what it answers.
    """
    if bots == "all":
        return records, 0
    return filter_bot_records(records, keep="interacted" if bots == "interacted" else "none")


class _Sender:
    """Sends records in batches and tallies what ingest made of them."""

    def __init__(self, client: Honcho, derive: str) -> None:
        self.client = client
        self.derive = derive
        self.outcomes: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.sessions: set[str] = set()
        self.authors: set[str] = set()
        self.sent = 0
        self.held_back = 0

    def send(self, records: Iterable[IngestRecord]) -> None:
        for chunk in _chunks(records, CHUNK):
            result = self.client.ingest(
                chunk, derive="now" if self.derive == "now" else "later"
            )
            for record in result.records:
                self.outcomes[record.outcome] += 1
                if record.reason:
                    self.reasons[record.reason] += 1
                if record.session_id:
                    self.sessions.add(record.session_id)
                if record.peer_id:
                    self.authors.add(record.peer_id)
            self.sent += len(chunk)
            if not use_json():
                status(f"sent {self.sent:,} records")

    def stored_before(self) -> None:
        """Say what was stored before a failure; a second import is safe."""
        if self.sent:
            status(f"{self.sent:,} records were stored before this; importing again is safe.")

    def summary(self, workspace: str) -> dict[str, Any]:
        return {
            "workspace": workspace,
            "records": self.sent,
            "created": self.outcomes["created"],
            "duplicate": self.outcomes["duplicate"],
            "rejected": self.outcomes["rejected"],
            "rejected_reasons": dict(self.reasons),
            "bot_records_held_back": self.held_back,
            "sessions": len(self.sessions),
            "authors": len(self.authors),
            "derive": self.derive,
        }


def _print_import_summary(summary: dict[str, Any], bots: str) -> None:
    if use_json():
        print_json(summary)
        return
    sent = summary["records"]
    stdout_console.print(
        f"{sent:,} records: {summary['created']:,} created · "
        + f"{summary['duplicate']:,} duplicate · {summary['rejected']:,} rejected"
    )
    for reason, count in Counter(summary["rejected_reasons"]).most_common():
        stdout_console.print(f"  rejected, {reason}: {count:,}")
    stdout_console.print(f"{summary['sessions']:,} sessions, {summary['authors']:,} authors")
    held_back = summary["bot_records_held_back"]
    if held_back:
        stdout_console.print(
            f"{held_back:,} bot records held back"
            + (" (no one interacted with them)" if bots == "interacted" else "")
            + "; `--bots all` imports them"
        )
    if summary["derive"] == "later":
        status(
            "Nothing is derived yet. Review with `honcho ingest status`, exclude what"
            + " shouldn't be modeled, then run `honcho ingest derive`."
        )


CHECK_ERRORS_SHOWN = 20
"""Invalid lines `import --check` lists before it stops listing them."""

DEFAULT_MAX_CONTENT = 25_000
"""The server's default MAX_MESSAGE_SIZE: the most characters a record's content
can have. A server can set another; `--check` checks against the default."""


def _check_file(path: str, bots: str) -> None:
    """Validate every line of an import without sending anything."""
    valid: list[IngestRecord] = []
    errors: list[dict[str, Any]] = []
    invalid = 0
    for number, line in _lines(path):
        if not line.strip():
            continue
        try:
            record = IngestRecord.model_validate(json.loads(line))
            if len(record.content) > DEFAULT_MAX_CONTENT:
                raise ValueError(
                    f"content: {len(record.content):,} characters, over the server's"
                    + f" default limit of {DEFAULT_MAX_CONTENT:,} (MAX_MESSAGE_SIZE)"
                )
            valid.append(record)
        except (ValueError, ValidationError) as e:
            invalid += 1
            if len(errors) < CHECK_ERRORS_SHOWN:
                errors.append({"line": number, "error": _validation_message(e)})
    _, held_back = _hold_back(valid, bots)
    sources = Counter(f"{r.source}:{r.source_instance}" for r in valid)
    summary: dict[str, Any] = {
        "valid": invalid == 0,
        "records": len(valid) + invalid,
        "invalid": invalid,
        "errors": errors,
        "sources": dict(sources),
        "containers": len({(r.source, r.source_instance, r.container.id) for r in valid if r.container}),
        "authors": len({(r.source, r.source_instance, r.author.id or r.author.name) for r in valid if r.author}),
        "bot_records": sum(1 for r in valid if r.author is not None and r.author.bot),
        "bot_records_held_back": held_back,
        "would_send": len(valid) - held_back,
    }
    if use_json():
        print_json(summary)
    else:
        for error in errors:
            stdout_console.print(f"line {error['line']}: {error['error']}")
        if invalid > len(errors):
            stdout_console.print(f"… and {invalid - len(errors):,} more invalid lines")
        stdout_console.print(
            f"{summary['records']:,} records: {len(valid):,} valid · {invalid:,} invalid"
        )
        for source, count in sources.most_common():
            stdout_console.print(f"  {source}: {count:,}")
        stdout_console.print(
            f"{summary['containers']:,} containers, {summary['authors']:,} authors,"
            + f" {summary['bot_records']:,} bot records ({held_back:,} held back"
            + f" with --bots {bots})"
        )
        if invalid == 0:
            status("Valid. Nothing was sent: run it again without --check to import.")
    if invalid:
        raise typer.Exit(1)


def _validation_message(e: Exception) -> str:
    if isinstance(e, ValidationError):
        error = e.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in error["loc"])
        return f"{where}: {error['msg']}" if where else error["msg"]
    return str(e)


@app.command("import")
@experimental
def import_records(
    file: str = typer.Argument(
        help="JSONL file of ingest records, one per line (`-` reads stdin)"
    ),
    derive: str = typer.Option("later", "--derive", help=_DERIVE_HELP),
    bots: str = typer.Option("interacted", "--bots", help=_BOTS_HELP),
    check: bool = typer.Option(
        False,
        "--check",
        help="Validate every line and report what would be sent; send nothing",
    ),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Import source records, such as a connector's `records.jsonl`.

    Each record becomes a message; its container picks the session and its
    author and participants the peers. Records already imported come back as
    duplicates, so running the same import again is safe.

    Bot records no person interacted with are held back by default: nothing is
    derived about a bot, but its text is read as context around the people, so
    unanswered bot posts cost tokens. The file is read whole first, since a
    reply can come after what it answers. `--bots all` imports them too.

    `--check` validates every line against the ingest record format without a
    workspace or credentials, and sends nothing.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    _check_choices(derive, bots)
    if check:
        _check_file(file, bots)
        return
    client, config = get_client()
    wid = config.workspace_id

    records, held_back = _hold_back(_records(file), bots)
    sender = _Sender(client, derive)
    sender.held_back = held_back
    try:
        sender.send(records)
    except typer.Exit:
        sender.stored_before()
        raise
    except Exception as e:
        sender.stored_before()
        _ingest_error(e, wid)

    _print_import_summary(sender.summary(wid), bots)


class _SourceSink:
    """Takes a source's records a container at a time, and sends each container's
    records once it's complete, so bot records are judged against the replies
    and mentions around them."""

    def __init__(self, sender: _Sender, bots: str) -> None:
        self.sender = sender
        self.bots = bots
        self.containers = 0
        self._name = "(before any container)"
        self._buffer: list[IngestRecord] = []

    def container(self, name: str) -> None:
        self.flush()
        self.containers += 1
        self._name = name

    def record(self, record: Mapping[str, Any]) -> None:
        try:
            self._buffer.append(IngestRecord.model_validate(record))
        except ValidationError as e:
            raise _SourceRecordError(self._name, _validation_message(e)) from None

    def flush(self) -> None:
        if not self._buffer:
            return
        kept, held_back = _hold_back(self._buffer, self.bots)
        self._buffer = []
        self.sender.held_back += held_back
        self.sender.send(kept)


class _SourceRecordError(Exception):
    def __init__(self, container: str, message: str) -> None:
        super().__init__(f"{container}: {message}")
        self.container = container


def _list_sources(*, as_text: bool = False) -> None:
    rows: list[dict[str, str]] = []
    for name, point in ingest_sources.installed().items():
        try:
            help_text = ingest_sources.load(point).help
        except ingest_sources.SourceError as e:
            help_text = f"(unavailable: {e})"
        rows.append({"source": name, "help": help_text, "entry_point": point.value})
    if use_json() and not as_text:
        print_json(rows)
        return
    if not rows:
        stdout_console.print(
            "No ingest sources are installed. Install a connector package, or pipe any"
            + " exporter's JSONL into `honcho ingest import -`."
        )
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Source")
    table.add_column("Reads")
    for row in rows:
        table.add_row(row["source"], row["help"])
    stdout_console.print(table)
    status("Run `honcho ingest from SOURCE --help` for a source's own options.")


@app.command(
    "from",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        # `--help` after the source name is the source's own help.
        "help_option_names": [],
    },
)
def import_from(
    ctx: typer.Context,
    source: Optional[str] = typer.Argument(
        None, help="An installed source; leave it out to list them"
    ),
    derive: str = typer.Option("later", "--derive", help=_DERIVE_HELP),
    bots: str = typer.Option("interacted", "--bots", help=_BOTS_HELP),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Experimental: import straight from an installed connector, such as `github`.

    The source's own options follow its name: `honcho ingest from github --help`
    shows them. Records are sent the way `ingest import` sends a file: deriving
    later, and holding back bot records no person interacted with, judged one
    container at a time. `--derive`, `--bots`, `-w` and `--json` are this
    command's own; everything else goes to the source.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    argv = list(ctx.args)
    # Help, this command's or a source's, needs no opt-in, workspace or
    # credentials, as `--help` on the other ingest commands doesn't.
    asked_for_help = (source in ("-h", "--help") and not argv) or (
        source is None and argv in (["-h"], ["--help"])
    )
    if asked_for_help:
        stdout_console.print(ctx.get_help())
        _list_sources(as_text=True)
        return
    named = source is not None and not source.startswith("-")
    if source is not None and named and any(arg in ("-h", "--help") for arg in argv):
        code = _run_source(_load_source(source), argv, None)
        raise typer.Exit(code)

    require_experimental()
    if source is None and not argv:
        _list_sources()
        return
    if source is None or not named:
        # A source option before the name: Click took it for the name.
        print_error(
            "INVALID_ARGUMENT",
            "Name the source first: `honcho ingest from SOURCE [its options]`."
            + " `honcho ingest from` lists the installed sources.",
        )
        raise typer.Exit(1)
    _check_choices(derive, bots)
    loaded = _load_source(source)

    client, config = get_client()
    wid = config.workspace_id
    # Refuse before a long read, rather than when the first batch is sent.
    try:
        client.ingest_status()
    except Exception as e:
        _ingest_error(e, wid)

    sender = _Sender(client, derive)
    sink = _SourceSink(sender, bots)
    try:
        code = _run_source(loaded, argv, sink)
        sink.flush()
    except _SourceRecordError as e:
        sender.stored_before()
        print_error(
            "INVALID_RECORD",
            f"'{source}' produced a record that isn't a valid ingest record, in {e}",
            {"source": source, "container": e.container},
        )
        raise typer.Exit(1) from None
    except typer.Exit:
        sender.stored_before()
        raise
    except Exception as e:
        sender.stored_before()
        _ingest_error(e, wid)

    summary = sender.summary(wid)
    summary["source"] = source
    summary["containers"] = sink.containers
    _print_import_summary(summary, bots)
    if code:
        raise typer.Exit(code)


def _load_source(source: str) -> ingest_sources.IngestSource:
    """The installed source by that name, or an error and exit 1."""
    point = ingest_sources.installed().get(source)
    if point is None:
        names = ", ".join(ingest_sources.installed()) or "none"
        print_error(
            "UNKNOWN_SOURCE",
            f"No ingest source named '{source}' is installed (installed: {names}).",
            {"source": source},
        )
        raise typer.Exit(1)
    try:
        return ingest_sources.load(point)
    except ingest_sources.SourceError as e:
        print_error("SOURCE_UNAVAILABLE", f"'{source}': {e}", {"source": source})
        raise typer.Exit(1) from None


def _run_source(
    loaded: ingest_sources.IngestSource,
    argv: list[str],
    sink: _SourceSink | None,
) -> int:
    """Run a source's `main`, turning its own exits (argparse's `--help` and
    usage errors, or a message) into an exit code."""
    try:
        return loaded.main(argv, sink if sink is not None else _NoSink())
    except SystemExit as exit_:
        if exit_.code is None:
            return 0
        if isinstance(exit_.code, int):
            return exit_.code
        print_error("SOURCE_ERROR", str(exit_.code))
        return 1


class _NoSink:
    """For a source run that only prints its help: records are an error."""

    def container(self, name: str) -> None:
        raise RuntimeError("a source run for --help must not read anything")

    def record(self, record: Mapping[str, Any]) -> None:
        raise RuntimeError("a source run for --help must not read anything")


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
    if st.sessions_being_deleted:
        stdout_console.print(
            f"{st.sessions_being_deleted:,} sessions still being deleted by a reset"
        )

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
            dict.fromkeys(a.source for a in p.accounts)
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
@experimental
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
@experimental
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


# ---------------------------------------------------------------------------
# Reset, link, unlink: fixing who is who, applied on the next import.
# ---------------------------------------------------------------------------

RESET_POLL_SECONDS = 2
RESET_WAIT_SECONDS = 600
"""How long `ingest reset` waits for the background deletion before giving up."""


def _refused(e: Exception, code: str, *, hint: str | None) -> NoReturn:
    """A 409 from link or unlink: say the server's reason, and how to proceed."""
    message = str(e)
    if hint and "reset" in message.lower():
        message += f" Run the same command with {hint} to reset the import first."
    print_error(code, message)
    raise typer.Exit(1)


def _reset(client: Honcho, wid: str, *, yes: bool, wait: bool = True) -> Any:
    """Reset the import, after a confirmation, and wait for the deletion.

    Returns the server's reset response. Raises ``typer.Exit`` when the
    confirmation is missing, the reset fails, or the wait times out.
    """
    if not yes:
        if use_json():
            print_error("CONFIRMATION_REQUIRED", "Pass --yes to reset without a prompt.")
            raise typer.Exit(1)
        typer.confirm(
            f"Delete every imported session in '{wid}', with its messages?", abort=True
        )
    try:
        reset = client.ingest_reset()
    except Exception as e:
        _ingest_error(e, wid)
    if not wait:
        return reset
    deadline = time.monotonic() + RESET_WAIT_SECONDS
    try:
        while client.ingest_status().sessions_being_deleted:
            if time.monotonic() > deadline:
                print_error(
                    "RESET_STILL_RUNNING",
                    "Still deleting in the background; check `honcho ingest status`.",
                )
                raise typer.Exit(1)
            time.sleep(RESET_POLL_SECONDS)
    except KeyboardInterrupt:
        status("Stopped waiting; the deletion continues on the server.")
        raise typer.Exit(130) from None
    return reset


def _people(client: Honcho) -> list[Any]:
    return list(client.ingest_people(size=100))


@app.command("reset")
@experimental
def reset_import(
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait until the deletion has finished"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Delete the imported sessions, to import again after a link or a fix.

    Deletes every session holding imported records, with their messages.
    People, their exclusions and links stay; session exclusions go with the
    sessions. Use it before anything is derived: the import runs again for free.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()
    wid = config.workspace_id
    reset = _reset(client, wid, yes=yes, wait=wait)
    if use_json():
        print_json({"sessions": len(reset.sessions), "waited": wait})
        return
    stdout_console.print(f"Reset {len(reset.sessions):,} imported sessions.")
    status("Link or fix what you need, then run `honcho ingest import` again.")


@app.command("link")
@experimental
def link_people(
    refs: list[str] = typer.Argument(
        help="People: number from `ingest status`, peer ID, or display name. Without"
        + " --keep, the first is kept"
    ),
    keep: Optional[str] = typer.Option(
        None,
        "--keep",
        help="Keep this peer ID, taken literally (any peer, including one no import"
        + " created); every person in REFS is linked into it",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Reset the import first (deletes the imported sessions and waits); the"
        + " people are found before the reset, so their numbers stay right",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before --reset"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Treat several imported accounts as one person, from the next import on.

    Every account of the other people maps to the first. Nothing already
    imported moves, so an import that already holds their records is reset
    first: `honcho ingest link 1 24 --reset`, then `honcho ingest import` again.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    if len(refs) + (1 if keep else 0) < 2:
        print_error(
            "INVALID_ARGUMENT",
            "Name at least two people (the first is kept), or --keep and one more.",
        )
        raise typer.Exit(1)
    client, config = get_client()
    wid = config.workspace_id
    try:
        people = _people(client)
    except Exception as e:
        _ingest_error(e, wid)
    candidates = [(p.peer_id, p.display_name) for p in people]

    def person(ref: str) -> str:
        return resolve(
            ref, kind="people", base_url=str(client.base_url),
            credential=_credential(client), workspace_id=wid, candidates=candidates,
        )

    # The kept person can be any peer, including one made by hand that no
    # import created; the server checks that it exists. --keep takes the ID
    # literally, for an ID that reads like a list number or a display name.
    if keep:
        peer_ids = [keep, *(person(ref) for ref in refs)]
    else:
        first = refs[0]
        listed = {i for i, _ in candidates} | {
            n.casefold() for _, n in candidates if n
        }
        kept = (
            person(first)
            if first.isdigit() or first in listed or first.casefold() in listed
            else first
        )
        peer_ids = [kept, *(person(ref) for ref in refs[1:])]
    if reset:
        _reset(client, wid, yes=yes)
    try:
        linked = client.ingest_link(peer_ids)
    except ConflictError as e:
        _refused(e, "LINK_REFUSED", hint=None if reset else "--reset")
    except Exception as e:
        _ingest_error(e, wid)
    if use_json():
        print_json(linked.model_dump(mode="json"))
        return
    name = dict(candidates).get(linked.peer_id) or linked.peer_id
    accounts = ", ".join(
        f"{a.source} {a.display_name or a.id}" for a in linked.accounts
    )
    stdout_console.print(f"{name} ({linked.peer_id}) now has: {accounts}")
    status("Run `honcho ingest import` again: their records go to one person.")


@app.command("unlink")
@experimental
def unlink_account(
    person: str = typer.Argument(help="The person the account was linked into"),
    account: str = typer.Argument(
        help="The account: its ID or display name in the source, or SOURCE:ID"
    ),
    reset: bool = typer.Option(
        False, "--reset", help="Reset the import first (deletes the imported sessions and waits)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before --reset"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Take a linked account back out of a person, from the next import on.

    As for a link, an import that already holds its records is reset first:
    `honcho ingest unlink 1 alexdev --reset`, then import again.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()
    wid = config.workspace_id
    try:
        people = _people(client)
    except Exception as e:
        _ingest_error(e, wid)
    peer_id = resolve(
        person, kind="people", base_url=str(client.base_url),
        credential=_credential(client), workspace_id=wid,
        candidates=[(p.peer_id, p.display_name) for p in people],
    )
    accounts = next(p.accounts for p in people if p.peer_id == peer_id)

    def named(a: Any, text: str) -> bool:
        return a.id == text or (a.display_name or "").casefold() == text.casefold()

    # The ID or name as given first (an ID can contain ":"), then SOURCE:ID.
    matches = [a for a in accounts if a.id is not None and named(a, account)]
    if not matches and ":" in account:
        source, _, rest = account.partition(":")
        matches = [
            a
            for a in accounts
            if a.id is not None and a.source == source and named(a, rest)
        ]
    if len(matches) != 1:
        print_error(
            "ACCOUNT_NOT_FOUND" if not matches else "AMBIGUOUS",
            f"{'No' if not matches else 'More than one'} account of this person"
            + f" matches '{account}'.",
            {"accounts": [f"{a.source}:{a.id} ({a.display_name})" for a in accounts]},
        )
        raise typer.Exit(1)
    chosen = matches[0]
    assert chosen.id is not None
    if reset:
        _reset(client, wid, yes=yes)
    try:
        unlinked = client.ingest_unlink(
            peer_id,
            source=chosen.source,
            source_instance=chosen.source_instance,
            account_id=chosen.id,
        )
    except ConflictError as e:
        _refused(e, "UNLINK_REFUSED", hint=None if reset else "--reset")
    except Exception as e:
        _ingest_error(e, wid)
    if use_json():
        print_json(unlinked.model_dump(mode="json"))
        return
    stdout_console.print(
        f"{chosen.source} {chosen.display_name or chosen.id} goes back to"
        + f" {unlinked.peer_id} from the next import on."
    )


# ---------------------------------------------------------------------------
# Suggested links
# ---------------------------------------------------------------------------

suggestions_app = typer.Typer(
    cls=HonchoTyperGroup,
    help="Experimental: accounts that look like one person, to accept or dismiss.",
)
app.add_typer(suggestions_app, name="suggestions")

REFRESH_POLL_SECONDS = 1
REFRESH_WAIT_SECONDS = 300

# A suggestion's label is the stage that made it and its confidence, such as
# `rules:high`. Labels name groups to list or accept.
SUGGESTION_STAGES = ("rules",)
SUGGESTION_CONFIDENCES = ("high", "low")
SUGGESTION_LABELS = tuple(
    f"{stage}:{confidence}"
    for stage in SUGGESTION_STAGES
    for confidence in SUGGESTION_CONFIDENCES
)


def _suggestion_label(suggestion: Any) -> str:
    return f"{suggestion.stage}:{suggestion.confidence}"


def _parse_label(value: str) -> tuple[str, str]:
    """A label's stage and confidence; exits on one that isn't a label."""
    if value not in SUGGESTION_LABELS:
        print_error(
            "INVALID_ARGUMENT",
            f"Unknown suggestion label {value!r}: use one of "
            + ", ".join(SUGGESTION_LABELS),
        )
        raise typer.Exit(1)
    stage, confidence = value.split(":")
    return stage, confidence


def _wait_for_refresh(client: Honcho) -> bool:
    """Wait until no refresh is waiting or running; False if it outlasted the wait."""
    deadline = time.monotonic() + REFRESH_WAIT_SECONDS
    try:
        while client.ingest_status().suggestions.refreshing:
            if time.monotonic() > deadline:
                return False
            time.sleep(REFRESH_POLL_SECONDS)
    except KeyboardInterrupt:
        status("Stopped waiting; the refresh continues on the server.")
        raise typer.Exit(130) from None
    return True


def _account_label(account: Any) -> str:
    name = account.display_name or account.id or "?"
    handle = f" ({account.handle})" if account.handle and account.handle != name else ""
    return f"{account.source} {name}{handle}"


def _print_suggestions(found: list[Any], total: int) -> None:
    table = Table(
        title=f"Suggested links ({total})", show_header=True, header_style="bold"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("label")
    table.add_column("accounts", overflow="fold")
    table.add_column("why", overflow="fold")
    for number, s in enumerate(found, start=1):
        label = Text(
            _suggestion_label(s), style="green" if s.confidence == "high" else "yellow"
        )
        accounts = " + ".join(_account_label(a) for a in s.accounts)
        if s.linked:
            accounts += " (already one person)"
        why = "; ".join(e.detail for e in s.evidence)
        table.add_row(str(number), label, accounts, why)
    stdout_console.print(table)
    if found:
        status(
            "Accept with `honcho ingest suggestions accept N --reset` (or a label,"
            + " such as `accept rules:high --reset`), dismiss with"
            + " `honcho ingest suggestions dismiss N`, then run `honcho ingest"
            + " import` again."
        )


@suggestions_app.callback(invoke_without_command=True)
@experimental
def list_suggestions(
    ctx: typer.Context,
    refresh: bool = typer.Option(
        True,
        "--refresh/--no-refresh",
        help="Refresh the suggestions first and wait for it (on by default)",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Only suggestions with this label: " + ", ".join(SUGGESTION_LABELS),
    ),
    state: str = typer.Option(
        "open", "--status", help="open (default), accepted, dismissed or all"
    ),
    limit: int = typer.Option(50, "--limit", help="Suggestions to list, strongest first"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Accounts from different sources that look like one person, with the evidence.

    Refreshes the suggestions from the imported records first (run it after
    importing, before a reset), then lists them numbered, strongest first. Each
    has a label: the stage that made it and its confidence. `rules:high` ones
    have evidence to accept at a glance; `rules:low` ones need a closer look.
    Nothing is linked until you accept one.
    """
    if ctx.invoked_subcommand is not None:
        return
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    stage, confidence = _parse_label(only) if only else (None, None)
    if state not in ("open", "accepted", "dismissed", "all"):
        print_error(
            "INVALID_ARGUMENT", "--status must be open, accepted, dismissed or all"
        )
        raise typer.Exit(1)
    client, config = get_client()
    wid = config.workspace_id
    try:
        if refresh:
            client.ingest_refresh_suggestions()
            if not _wait_for_refresh(client):
                status(
                    "Still refreshing (is the deriver running?); showing the"
                    + " suggestions as they are now."
                )
        page = client.ingest_suggestions(
            status=None if state == "all" else state,  # pyright: ignore[reportArgumentType]
            confidence=confidence,  # pyright: ignore[reportArgumentType]
            stage=stage,  # pyright: ignore[reportArgumentType]
            size=min(limit, 100),
        )
        found = _first(page, limit)
        total = page.total or len(found)
    except Exception as e:
        _ingest_error(e, wid)
    save_suggestions(str(client.base_url), _credential(client), wid, [s.id for s in found])
    if use_json():
        print_json(
            [{**s.model_dump(mode="json"), "label": _suggestion_label(s)} for s in found]
        )
        return
    _print_suggestions(found, total)


def _suggestion_ids(client: Honcho, wid: str, refs: list[str]) -> list[str]:
    return [
        saved_suggestion(str(client.base_url), _credential(client), wid, ref)
        for ref in refs
    ]


@suggestions_app.command("accept")
@experimental
def accept_suggestions(
    refs: list[str] | None = typer.Argument(
        None,
        help="Suggestions: numbers from `ingest suggestions`, IDs, or labels such as"
        + " `rules:high` for every open one with that label",
    ),
    keep: str | None = typer.Option(
        None,
        "--keep",
        help="The peer to keep (one suggestion only). By default an explicit peer,"
        + " else the account with more records",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Reset the import first (deletes the imported sessions and waits):"
        + " accepting links the people, which needs them without imported records",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask before --reset"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Accept suggestions: link each one's two accounts into one person.

    Name them by number, by ID, or by label: `rules:high` accepts every open
    suggestion with that label. Accepting is `honcho ingest link` for the
    suggestion's two people, so it applies from the next import on: reset first
    (`--reset`), accept, then run `honcho ingest import` again. The suggestions
    are found before the reset, so their numbers stay right.
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    if not refs:
        print_error(
            "INVALID_ARGUMENT",
            "Name suggestions to accept: numbers, IDs, or a label such as rules:high.",
        )
        raise typer.Exit(1)
    # Suggestion IDs never contain a colon, so one marks a label.
    labels = [_parse_label(ref) for ref in refs if ":" in ref]
    named = [ref for ref in refs if ":" not in ref]
    if keep and (labels or len(named) != 1):
        print_error("INVALID_ARGUMENT", "--keep goes with exactly one suggestion.")
        raise typer.Exit(1)
    client, config = get_client()
    wid = config.workspace_id
    ids = _suggestion_ids(client, wid, named)
    for stage, confidence in labels:
        try:
            ids += [
                s.id
                for s in client.ingest_suggestions(
                    status="open",
                    stage=stage,  # pyright: ignore[reportArgumentType]
                    confidence=confidence,  # pyright: ignore[reportArgumentType]
                    size=100,
                )
                if not s.linked
            ]
        except Exception as e:
            _ingest_error(e, wid)
    ids = list(dict.fromkeys(ids))
    if not ids:
        print_error(
            "NOTHING_TO_ACCEPT",
            "No open suggestions with "
            + " or ".join(f"{stage}:{confidence}" for stage, confidence in labels)
            + ".",
        )
        raise typer.Exit(1)
    if reset:
        _reset(client, wid, yes=yes)
    accepted: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    # Each accept is its own link: one refused leaves the others done, so keep
    # going and report both.
    for suggestion_id in ids:
        try:
            result = client.ingest_accept_suggestion(suggestion_id, keep=keep)
        except Exception as e:
            failed.append({"suggestion": suggestion_id, "reason": str(e)})
            continue
        accepted.append(result.model_dump(mode="json"))
        if not use_json():
            if result.link is None:
                stdout_console.print(
                    "Already one person: "
                    + " + ".join(_account_label(a) for a in result.suggestion.accounts)
                )
            else:
                accounts = ", ".join(
                    f"{a.source} {a.display_name or a.id}" for a in result.link.accounts
                )
                stdout_console.print(f"{result.link.peer_id} now has: {accounts}")
    if use_json():
        print_json({"accepted": accepted, "failed": failed})
    else:
        for failure in failed:
            print_error("ACCEPT_REFUSED", f"{failure['suggestion']}: {failure['reason']}")
        if accepted:
            status("Run `honcho ingest import` again: linked records go to one person.")
    if failed:
        raise typer.Exit(1)


@suggestions_app.command("dismiss")
@experimental
def dismiss_suggestions(
    refs: list[str] = typer.Argument(
        help="Suggestions: numbers from `ingest suggestions`, or IDs"
    ),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Dismiss suggestions: the accounts stay apart, and a refresh won't bring them back."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()
    wid = config.workspace_id
    dismissed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for suggestion_id in _suggestion_ids(client, wid, refs):
        try:
            result = client.ingest_dismiss_suggestion(suggestion_id)
        except Exception as e:
            failed.append({"suggestion": suggestion_id, "reason": str(e)})
            continue
        dismissed.append(result.model_dump(mode="json"))
        if not use_json():
            stdout_console.print(
                "Dismissed: " + " + ".join(_account_label(a) for a in result.accounts)
            )
    if use_json():
        print_json({"dismissed": dismissed, "failed": failed})
    else:
        for failure in failed:
            print_error("DISMISS_REFUSED", f"{failure['suggestion']}: {failure['reason']}")
    if failed:
        raise typer.Exit(1)
