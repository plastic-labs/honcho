"""Session commands: list, inspect, view, context, summaries, peers, search, representation, metadata."""

from __future__ import annotations

import json
import shlex
from typing import List, Optional

import typer

from honcho import HonchoError, Session

from honcho_cli.commands.workspace import _config_to_dict, _handle_error, _raw_list
from honcho_cli.output import print_error, print_result, print_transcript, status, use_json
from honcho_cli.validation import validate_resource_id

from honcho_cli._help import HonchoTyperGroup
from honcho_cli.common import (
    add_common_options,
    get_client,
    get_flag_overrides,
    get_resolved_config,
    handle_cmd_flags,
)

app = typer.Typer(cls=HonchoTyperGroup, help="List, inspect, view, create, delete, and manage conversation sessions and their peers.")
add_common_options(app)


def _get_session_id(session_id: str | None) -> str:

    config = get_resolved_config()
    sid = session_id or config.session_id
    if not sid:
        print_error("NO_SESSION", "No session ID provided. Pass --session/-s or set HONCHO_SESSION_ID.")
        raise typer.Exit(1)
    return validate_resource_id(sid, "session")


@app.command("list")
def list_sessions(
    peer_id: Optional[str] = typer.Option(None, "--peer", "-p", help="Filter by peer"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List sessions in the workspace."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    client, config = get_client()

    try:
        if peer_id:
            peer = client.peer(peer_id)
            raw_sessions = _raw_list(peer.sessions())
        else:
            raw_sessions = _raw_list(client.sessions())

        items = [
            {
                "id": s.id,
                "is_active": s.is_active,
                "metadata": s.metadata,
                "created_at": str(s.created_at),
            }
            for s in raw_sessions
        ]
        print_result(items, columns=["id", "is_active", "metadata", "created_at"], title="Sessions")
    except Exception as e:
        _handle_error(e, "session", "list")


@app.command("create")
def create_session(
    session_id: str = typer.Argument(help="Session ID to create or get"),
    peers: Optional[str] = typer.Option(None, "--peers", help="Comma-separated peer IDs to add to the session"),
    metadata: Optional[str] = typer.Option(None, "--metadata", help="JSON metadata to associate with the session"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Create or get a session."""
    handle_cmd_flags(json_output=json_output, workspace=workspace)
    sid = validate_resource_id(session_id, "session")
    client, config = get_client()

    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            print_error("INVALID_JSON", f"--metadata must be valid JSON: {e}", {})
            raise typer.Exit(1)

    peer_ids = [p.strip() for p in peers.split(",") if p.strip()] if peers else []
    for pid in peer_ids:
        validate_resource_id(pid, "peer")

    try:
        sess = client.session(sid, metadata=parsed_metadata)
        if peer_ids:
            sess.add_peers(peer_ids)
        result: dict[str, object] = {"session_id": sess.id}
        if parsed_metadata is not None:
            result["metadata"] = parsed_metadata
        if peer_ids:
            result["peers"] = peer_ids
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def inspect(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Inspect a session: peers, message count, summaries, config."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        peers = sess.peers()
        msg_page = sess.messages()
        summaries = sess.summaries()
        sess_config = sess.get_configuration()

        result = {
            "session_id": sid,
            "peers": [{"id": p.id} for p in peers],
            "message_count": msg_page.total,
            "summaries": {
                "short": summaries.short_summary if hasattr(summaries, "short_summary") else None,
                "long": summaries.long_summary if hasattr(summaries, "long_summary") else None,
            },
            "configuration": _config_to_dict(sess_config) if sess_config else None,
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


# Server-side ceiling on page size (fastapi-pagination's default ``Params``
# declares ``size`` as ``Query(50, ge=1, le=100)``).
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 50


def _fetch_recent_messages(sess, filters: dict | None, last: int) -> tuple[list, int | None]:
    """Fetch the ``last`` most recent messages, newest first.

    Walks as many newest-first server pages as it takes to fill the window.
    Returns the messages plus the session's total message count (if reported).
    """
    page = sess.messages(
        filters=filters,
        reverse=True,
        size=min(max(last, 1), MAX_PAGE_SIZE),
    )
    total = page.total
    msgs = list(page.items)
    while len(msgs) < last and page.has_next_page():
        page = page.get_next_page()
        if page is None:
            break
        msgs.extend(page.items)
    return msgs[:last], total


def _next_page_command(
    session_id: str,
    next_page: int,
    size: int,
    *,
    reverse: bool,
    show_ids: bool,
    workspace: str | None,
    peer: str | None,
) -> str:
    """Continuation command for the next page, carrying this invocation's scope.

    Scoping flags are echoed only when passed as flags; anything resolved from
    the environment or config file resolves the same way on the next run. IDs
    are shell-quoted — they may contain spaces and metacharacters, and this
    string is meant to be pasted into a shell.
    """
    parts = [
        "honcho",
        "session",
        "view",
        session_id,
        "--page",
        str(next_page),
        "--size",
        str(size),
    ]
    if reverse:
        parts.append("--reverse")
    if show_ids:
        parts.append("--ids")
    if workspace:
        parts += ["-w", workspace]
    if peer:
        parts += ["-p", peer]
    return shlex.join(parts)


def _fetch_all_messages(sess, filters: dict | None) -> tuple[list, int | None]:
    """Fetch every message in the session, oldest first."""
    page = sess.messages(filters=filters, reverse=False, size=MAX_PAGE_SIZE)
    total = page.total
    msgs = list(page.items)
    while page.has_next_page():
        page = page.get_next_page()
        if page is None:
            break
        msgs.extend(page.items)
    return msgs, total


@app.command()
def view(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    last: Optional[int] = typer.Option(
        None,
        "--last",
        help=f"Show only the N most recent messages (default when no --page/--all: {DEFAULT_PAGE_SIZE})",
    ),
    page_number: Optional[int] = typer.Option(
        None,
        "--page",
        help="1-indexed page of the full transcript. Use for page 2+.",
    ),
    size: Optional[int] = typer.Option(
        None,
        "--size",
        help=f"Messages per page; requires --page (1-{MAX_PAGE_SIZE}, default: {DEFAULT_PAGE_SIZE})",
    ),
    all_messages: bool = typer.Option(False, "--all", help="Show the full transcript (every page)"),
    reverse: bool = typer.Option(
        False,
        "--reverse",
        help="Newest first (default is chronological: oldest at top)",
    ),
    show_ids: bool = typer.Option(False, "--ids", help="Include message IDs in the transcript"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    peer: Optional[str] = typer.Option(None, "--peer", "-p", help="Filter by peer ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """View a session transcript as a chat log.

    Modes (pick one):

    - default / --last N: tail of the conversation (most recent N)
    - --page N [--size M]: page through the full transcript
    - --all: every message

    Paging follows the requested order: --page 1 starts at the oldest message,
    or the newest with --reverse.

    Human mode prints a row-delimited table. JSON mode emits the message list
    (same shape as message list).
    """
    handle_cmd_flags(json_output=json_output, workspace=workspace, peer=peer, session=session)
    sid = _get_session_id(session_id)

    # Validate every flag before touching the network.
    modes = sum([
        last is not None,
        page_number is not None,
        all_messages,
    ])
    if modes > 1:
        print_error(
            "INVALID_FLAGS",
            "--last, --page, and --all are mutually exclusive",
            {"last": last, "page": page_number, "all": all_messages},
        )
        raise typer.Exit(1)

    if page_number is not None and page_number < 1:
        print_error("INVALID_FLAGS", "--page must be >= 1", {"page": page_number})
        raise typer.Exit(1)
    if size is not None and page_number is None:
        print_error("INVALID_FLAGS", "--size only applies with --page", {"size": size})
        raise typer.Exit(1)
    if size is not None and not 1 <= size <= MAX_PAGE_SIZE:
        print_error(
            "INVALID_FLAGS",
            f"--size must be between 1 and {MAX_PAGE_SIZE}",
            {"size": size},
        )
        raise typer.Exit(1)
    if last is not None and last < 1:
        print_error("INVALID_FLAGS", "--last must be >= 1", {"last": last})
        raise typer.Exit(1)

    # Default: tail of conversation (most recent 50).
    mode = "page" if page_number is not None else ("all" if all_messages else "last")
    tail = last if last is not None else DEFAULT_PAGE_SIZE
    page_size = size if size is not None else DEFAULT_PAGE_SIZE

    client, config = get_client()
    # Read-only: client.session() is a get-or-create POST, so build the Session directly.
    sess = Session(sid, client)

    try:
        filters = {"peer_id": config.peer_id} if config.peer_id else None
        page_meta: int | None = None
        pages_meta: int | None = None

        if mode == "page":
            # Page in the order the caller asked for.
            result_page = sess.messages(
                filters=filters,
                page=page_number,
                size=page_size,
                reverse=reverse,
            )
            msgs = list(result_page.items)
            total = result_page.total
            page_meta = result_page.page if result_page.page is not None else page_number
            pages_meta = result_page.pages
        elif mode == "all":
            msgs, total = _fetch_all_messages(sess, filters)
            if reverse:
                msgs = list(reversed(msgs))
        else:
            # Tail window: fetched newest-first, flipped to chronological unless --reverse.
            msgs, total = _fetch_recent_messages(sess, filters, tail)
            if not reverse:
                msgs = list(reversed(msgs))

        items = [
            {
                "id": m.id,
                "peer_id": m.peer_id,
                "content": m.content,
                "token_count": m.token_count,
                "metadata": m.metadata,
                "created_at": str(m.created_at),
            }
            for m in msgs
        ]
    except Exception as e:
        _handle_error(e, "session", sid)
        raise  # unreachable: _handle_error always exits

    next_page_hint = None
    if page_meta is not None and pages_meta is not None and page_meta < pages_meta:
        # Effective overrides, not the command-level params: -w/-p also parse at
        # group and top level.
        overrides = get_flag_overrides()
        next_page_hint = _next_page_command(
            sid,
            page_meta + 1,
            page_size,
            reverse=reverse,
            show_ids=show_ids,
            workspace=overrides["workspace"],
            peer=overrides["peer"],
        )

    # Rendered outside the try: output failures aren't session API errors.
    print_transcript(
        items,
        session_id=sid,
        total=total,
        page=page_meta,
        pages=pages_meta,
        show_ids=show_ids,
        next_page_hint=next_page_hint,
    )


@app.command()
def context(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    tokens: Optional[int] = typer.Option(None, help="Token budget"),
    summary: bool = typer.Option(True, help="Include summary"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get session context (what an agent would see)."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        ctx = sess.context(tokens=tokens, summary=summary)
        result = ctx.__dict__ if hasattr(ctx, "__dict__") else ctx
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def summaries(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get session summaries (short + long)."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        s = sess.summaries()
        result = {
            "session_id": sid,
            "short_summary": s.short_summary if hasattr(s, "short_summary") else None,
            "long_summary": s.long_summary if hasattr(s, "long_summary") else None,
        }
        print_result(result)
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def delete(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Delete a session and all its data. Destructive — requires --yes or interactive confirm."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    if not yes:
        # Show a short preview so the user knows what's about to disappear.
        # Only in interactive/TTY mode — scripted (--json) callers already
        # know what they're deleting, and they still need to pass --yes.
        # Narrow the except to HonchoError so auth/network failures surface
        # before the user types 'y' on a destructive op.
        if not use_json():
            try:
                peers = sess.peers()
                msg_page = sess.messages()
                peer_ids = [p.id for p in peers]
                typer.echo(
                    f"  session:  {sid}\n"
                    f"  peers:    {', '.join(peer_ids) if peer_ids else '(none)'}\n"
                    f"  messages: {msg_page.total}"
                )
            except HonchoError as preview_err:
                status(f"preview unavailable: {preview_err}")
        typer.confirm(f"Delete session '{sid}' and all its messages, conclusions, and queue items?", abort=True)

    try:
        sess.delete()
        status(f"Session '{sid}' deleted")
        print_result({"deleted": sid})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("peers")
def session_peers(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """List peers in a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        peers = sess.peers()
        items = [{"id": p.id} for p in peers]
        print_result(items, columns=["id"], title=f"Session peers ({sid})")
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("add-peers")
def add_peers(
    session_id: str = typer.Argument(help="Session ID"),
    peer_ids: List[str] = typer.Argument(help="Peer IDs to add to the session"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Add peers to a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        sess.add_peers(peer_ids)
        print_result({"session_id": sid, "added_peers": peer_ids})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("remove-peers")
def remove_peers(
    session_id: str = typer.Argument(help="Session ID"),
    peer_ids: List[str] = typer.Argument(help="Peer IDs to remove from the session"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Remove peers from a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        sess.remove_peers(peer_ids)
        print_result({"session_id": sid, "removed_peers": peer_ids})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def search(
    query: str = typer.Argument(help="Search query"),
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    limit: int = typer.Option(10, help="Max results"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Search messages in a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        results = sess.search(query, limit=limit)
        items = [
            {
                "id": m.id,
                "peer_id": m.peer_id,
                "content": m.content if use_json() else m.content[:200],
                "created_at": str(m.created_at),
            }
            for m in results
        ]
        print_result(items, columns=["id", "peer_id", "content", "created_at"], title=f"Session search: {query}")
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command()
def representation(
    peer_id: str = typer.Argument(help="Peer ID to get representation for"),
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    target: Optional[str] = typer.Option(None, help="Target peer (what peer_id knows about target)"),
    search_query: Optional[str] = typer.Option(None, help="Semantic search query to filter conclusions"),
    max_conclusions: Optional[int] = typer.Option(None, help="Maximum number of conclusions to include"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get the representation of a peer within a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        result = sess.representation(
            peer_id,
            target=target,
            search_query=search_query,
            max_conclusions=max_conclusions,
        )
        print_result({"session_id": sid, "peer_id": peer_id, "target": target, "representation": result})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("get-metadata")
def get_metadata(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Get metadata for a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()
    sess = client.session(sid)

    try:
        result = sess.get_metadata()
        print_result({"session_id": sid, "metadata": result})
    except Exception as e:
        _handle_error(e, "session", sid)


@app.command("set-metadata")
def set_metadata(
    session_id: Optional[str] = typer.Argument(None, help="Session ID (uses default if omitted)"),
    metadata: str = typer.Option(..., "--data", "-d", help="JSON metadata to set (e.g. '{\"key\": \"value\"}')"),
    workspace: Optional[str] = typer.Option(None, "--workspace", "-w", help="Override workspace ID"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Override session ID"),
    json_output: bool = typer.Option(False, "--json", help="Force JSON output"),
) -> None:
    """Set metadata for a session."""

    handle_cmd_flags(json_output=json_output, workspace=workspace, session=session)
    sid = _get_session_id(session_id)
    client, config = get_client()

    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as e:
        print_error("INVALID_JSON", f"metadata must be valid JSON: {e}", {})
        raise typer.Exit(1)

    sess = client.session(sid)

    try:
        sess.set_metadata(parsed)
        print_result({"session_id": sid, "metadata": parsed})
    except Exception as e:
        _handle_error(e, "session", sid)
