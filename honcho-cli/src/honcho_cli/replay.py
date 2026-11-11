"""Replay a workspace's sessions into another workspace.

A replay copies structure (workspace, peer and session metadata and
configuration, session membership, scopes) and then re-sends every message in
its original order with its original ``created_at``. The destination derives
its own conclusions from those messages; nothing derived is copied.

Two things a replay cannot reproduce:

- Conclusions written through the API and peer cards set by hand. They have no
  source message, so the destination cannot derive them again.
  ``count_conclusions`` gives the reports a number to warn with.
- Membership over time. The API does not say when a peer joined or left a
  session, so the destination session has the source's current members from
  its first message. A peer that joined late observes the whole session.

The functions here take two SDK clients and know nothing about flags, config
files or output, so the command in ``commands/workspace.py`` stays thin.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from honcho import (
    AuthenticationError,
    Honcho,
    MessageCreateParams,
    NotFoundError,
    PermissionDeniedError,
    Session,
)
from honcho.api_types import SessionPeerConfig
from honcho.http import routes

# Server caps: 100 items per list page, 100 messages per create request.
PAGE_SIZE = 100
MESSAGE_BATCH_SIZE = 100

_WORKERS = 8

# Goes into both reports, next to the number of conclusions in the source.
NOT_COPIED_NOTE = (
    "Conclusions and peer cards are not copied. The destination derives its "
    "own from the replayed messages. Conclusions and peer cards written "
    "through the API have no source message, so they are not recreated."
)
_EPOCH = datetime.min.replace(tzinfo=UTC)

# Called as ``on_tick(label, done, total)`` while a long step makes progress.
Tick = Callable[[str, int, "int | None"], None]


def _no_tick(_label: str, _done: int, _total: int | None) -> None:
    """The default progress callback: report nothing."""
    pass


class ReplayMismatch(Exception):
    """The destination session holds messages that are not a prefix of the source."""

    def __init__(self, position: int) -> None:
        """Record the 1-based position of the first differing message."""
        self.position = position
        super().__init__(
            "the destination session already has different messages "
            f"(first difference at message {position})"
        )


class ReadOnlyUnsupported(Exception):
    """The installed SDK gives no known way to read without creating a workspace."""

    def __init__(self) -> None:
        """Say what is missing and what to do about it."""
        super().__init__(
            "This version of the Honcho SDK (honcho-ai) has no known way to read "
            "a workspace without creating it, so the destination cannot be "
            "checked safely. Upgrade honcho-cli."
        )


class NothingToReplay(Exception):
    """A scoped key found no sessions and no peers in the source workspace."""

    def __init__(self, workspace_id: str) -> None:
        """Say that the workspace is empty or missing; the key cannot tell which."""
        super().__init__(
            f"Workspace '{workspace_id}' has no sessions and no peers, or does "
            "not exist. This API key cannot list workspaces, so it cannot tell "
            "the two apart. Nothing was created."
        )


@dataclass
class SessionPlan:
    """One source session, plus what the probe and destination check learned."""

    session: Any
    message_count: int | None = None
    last_message_at: datetime | None = None
    # Set by ``Replayer.check_sessions``: what a replay would do with it.
    action: str | None = None  # create | resume | up_to_date | conflict
    messages_in_destination: int = 0  # leading messages the destination has
    conflict_at: int | None = None

    @property
    def messages_to_send(self) -> int:
        """How many messages a replay would send for this session."""
        if self.action == "conflict":
            return 0
        return max((self.message_count or 0) - self.messages_in_destination, 0)

    def to_dict(self) -> dict[str, Any]:
        """The plan as JSON-ready data, with the destination check when it ran."""
        data: dict[str, Any] = {
            "id": self.session.id,
            "message_count": self.message_count,
            "last_message_at": (
                self.last_message_at.isoformat() if self.last_message_at else None
            ),
        }
        if self.action is not None:
            data["action"] = self.action
            data["messages_in_destination"] = self.messages_in_destination
            data["messages_to_send"] = self.messages_to_send
        if self.conflict_at is not None:
            data["conflict_at"] = self.conflict_at
        return data


@dataclass
class SessionResult:
    session_id: str
    messages_created: int = 0
    messages_skipped: int = 0  # already in the destination from an earlier run

    @property
    def action(self) -> str:
        """What the replay did with the session: created, resumed or up_to_date."""
        if self.messages_skipped and not self.messages_created:
            return "up_to_date"
        return "resumed" if self.messages_skipped else "created"

    def to_dict(self) -> dict[str, Any]:
        """The result as JSON-ready data."""
        return {
            "id": self.session_id,
            "action": self.action,
            "messages_created": self.messages_created,
            "messages_skipped": self.messages_skipped,
        }


@dataclass
class ReplayResult:
    sessions: list[SessionResult] = field(default_factory=list)
    # Each entry: session_id, reason ("conflict" or "error"), error.
    failed: list[dict[str, str]] = field(default_factory=list)
    peers_created: int = 0
    peers_existing: int = 0  # already in the destination; given the source's settings
    workspace_copied: bool = False


@contextmanager
def read_only(client: Honcho) -> Iterator[None]:
    """Keep SDK reads on ``client`` from creating its workspace.

    Every workspace-scoped SDK call first runs a get-or-create on the
    workspace. A dry run must leave the destination untouched, so this marks
    the workspace as already ensured for the duration of the reads.

    ``_workspace_ensured`` is private to the SDK. If a later SDK drops it,
    setting it would do nothing and the reads would create the workspace, so
    this refuses to run instead. Two tests watch the flag itself:
    ``TestRealSDKReadOnly`` runs a dry run through the real client and fails on
    any write request, and ``tests/sdk/test_workspace_ensured.py`` checks the
    flag against a real server in the suite that CI runs.

    Raises:
        ReadOnlyUnsupported: the client has no ``_workspace_ensured``.
    """
    if not hasattr(client, "_workspace_ensured"):
        raise ReadOnlyUnsupported
    ensured = client._workspace_ensured
    client._workspace_ensured = True
    try:
        yield
    finally:
        client._workspace_ensured = ensured


def _dump(model: Any) -> dict[str, Any]:
    """A configuration object as a plain dict with unset fields dropped."""
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return dict(model)


def read_workspace(source: Honcho) -> tuple[dict[str, Any], Any]:
    """Return the source workspace's ``(metadata, configuration)``.

    Looks the workspace up in the workspace list first, because every other
    read in the SDK goes through a get-or-create: a mistyped ``--from`` would
    otherwise create an empty workspace on the source instance.

    Listing needs an admin key. A workspace-scoped key can read the settings
    only through the get-or-create, and has no way to ask whether the workspace
    exists. So with such a key this first looks for sessions and peers without
    creating anything. If there are some, the workspace exists and the
    get-or-create is a plain read. If there are none, there is nothing to
    replay and the workspace may be missing, so it stops there.

    Raises:
        NotFoundError: an admin key was used and the workspace does not exist.
        NothingToReplay: a scoped key was used and the workspace has no
            sessions and no peers, or does not exist.
    """
    try:
        page = source.workspaces(filters={"id": source.workspace_id}, size=PAGE_SIZE)
        while page is not None:
            for workspace in page._raw_items:
                if workspace.id == source.workspace_id:
                    return workspace.metadata or {}, workspace.configuration
            page = page.get_next_page()
    except (AuthenticationError, PermissionDeniedError):
        with read_only(source):
            try:
                has_content = bool(
                    source.sessions(size=1).items or source.peers(size=1).items
                )
            except NotFoundError:
                has_content = False
        if not has_content:
            raise NothingToReplay(source.workspace_id) from None
        configuration = source.get_configuration()
        return source.metadata or {}, configuration
    raise NotFoundError(f"Workspace '{source.workspace_id}' not found")


def count_conclusions(source: Honcho) -> int | None:
    """How many conclusions the source workspace holds, or None if unknown.

    A replay copies none of them. The ones derived from messages are derived
    again in the destination; the ones written through the API are not, and the
    API does not mark which is which. The reports use this number to say so.
    One request. A source that cannot answer must not stop a replay.
    """
    try:
        return source.conclusions.list(size=1).total
    except Exception:  # noqa: BLE001 - the count only feeds a warning
        return None


def list_sessions(source: Honcho) -> list[SessionPlan]:
    """Every active source session, oldest first, with its scope names."""
    return [
        SessionPlan(session)
        for session in source.sessions(size=PAGE_SIZE, include_scopes=True)
    ]


def _for_each(
    plans: list[SessionPlan],
    work: Callable[[SessionPlan], None],
    label: str,
    on_tick: Tick,
) -> None:
    """Run ``work`` on every plan, a few at a time, reporting progress."""
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        for done, _ in enumerate(pool.map(work, plans), start=1):
            on_tick(label, done, len(plans))


def probe_sessions(plans: list[SessionPlan], on_tick: Tick = _no_tick) -> None:
    """Fill in each session's message count and the time of its newest message.

    The API has no workspace-wide "sessions by latest message" listing, so this
    is one small request per session, run a few at a time.
    """

    def probe(plan: SessionPlan) -> None:
        """Read one session's newest message and its message count."""
        page = plan.session.messages(reverse=True, size=1)
        newest = page.items
        plan.message_count = page.total
        plan.last_message_at = newest[0].created_at if newest else None

    _for_each(plans, probe, "Reading source sessions", on_tick)


def select_most_recent(plans: list[SessionPlan], count: int) -> list[SessionPlan]:
    """Keep the ``count`` sessions with the newest messages, in their original order.

    A session with no messages has no recency of its own, so it ranks below
    every session that has one and is ordered by its creation time.
    """

    def recency(plan: SessionPlan) -> tuple[bool, datetime]:
        """Sort key: sessions with messages first, then by newest message."""
        if plan.last_message_at is not None:
            return True, plan.last_message_at
        return False, plan.session.created_at or _EPOCH

    keep = {id(plan) for plan in sorted(plans, key=recency, reverse=True)[:count]}
    return [plan for plan in plans if id(plan) in keep]


def _session_member_ids(source: Honcho, session: Any) -> list[str]:
    """The session's active peers.

    Pages the route directly: ``Session.peers()`` reads only the first page, and
    a member missed here would be re-added by its first message with default
    settings.
    """
    peer_ids: list[str] = []
    page = 1
    while True:
        data = source._http.get(
            routes.session_peers(source.workspace_id, session.id),
            query={"page": page, "size": PAGE_SIZE},
        )
        peer_ids.extend(item["id"] for item in data.get("items", []))
        if page >= (data.get("pages") or 1):
            break
        page += 1
    return peer_ids


def _session_members(
    source: Honcho, session: Any
) -> list[tuple[str, SessionPeerConfig]]:
    """The session's active peers and their per-session observer settings."""
    return [
        (peer_id, session.get_peer_configuration(peer_id))
        for peer_id in _session_member_ids(source, session)
    ]


def _same_message(existing: Any, message: Any) -> bool:
    """Whether a destination message is the replayed copy of a source message."""
    return (
        existing.peer_id == message.peer_id
        and existing.content == message.content
        and existing.created_at == message.created_at
    )


class Replayer:
    """Copies sessions from ``source`` into ``dest``, creating peers as needed."""

    def __init__(self, source: Honcho, dest: Honcho) -> None:
        """Hold both clients and the peers seen so far."""
        self.source = source
        self.dest = dest
        self._source_peers: dict[str, Any] = {}
        self._ensured_peers: set[str] = set()
        self._in_destination: dict[str, bool] = {}  # peer ID -> existed before us

    def load_source_peers(self) -> list[str]:
        """Read every regular source peer's metadata and configuration."""
        self._source_peers = {
            peer.id: peer for peer in self.source.peers(size=PAGE_SIZE)
        }
        return list(self._source_peers)

    def copy_workspace(self, metadata: dict[str, Any], configuration: Any) -> bool:
        """Copy workspace metadata and configuration into an empty destination.

        A destination that already has sessions keeps its own settings: this is
        either a resumed replay, which copied them on its first run, or a
        workspace someone else configured.
        """
        if self.destination_session_count() > 0:
            return False
        # A get-or-create: the setters below need the workspace to exist.
        self.dest.get_metadata()
        if _dump(configuration):
            self.dest.set_configuration(configuration)
        if metadata:
            self.dest.set_metadata(metadata)
        return True

    def copy_scope_metadata(self, scope_names: Iterable[str]) -> None:
        """Carry over the metadata of the scopes the replayed sessions belong to."""
        wanted = set(scope_names)
        if not wanted:
            return
        for scope in self.source.scopes(size=PAGE_SIZE):
            if scope.id in wanted and scope.metadata:
                self.dest.scope(scope.id, metadata=scope.metadata)

    def destination_session_count(self) -> int:
        """How many sessions the destination has now. Creates nothing."""
        with read_only(self.dest):
            try:
                page = self.dest.sessions(size=1)
            except NotFoundError:
                return 0
        return page.total if page.total is not None else len(page.items)

    def peers_in_destination(self, peer_ids: Iterable[str]) -> set[str]:
        """Which of these peers the destination had before this replay. Creates nothing."""
        wanted = list(dict.fromkeys(peer_ids))
        unknown = [peer_id for peer_id in wanted if peer_id not in self._in_destination]
        with read_only(self.dest):
            for start in range(0, len(unknown), PAGE_SIZE):
                chunk = unknown[start : start + PAGE_SIZE]
                try:
                    found = {
                        peer.id
                        for peer in self.dest.peers(
                            filters={"id": {"in": chunk}}, size=PAGE_SIZE
                        )
                    }
                except NotFoundError:
                    found = set()
                for peer_id in chunk:
                    self._in_destination[peer_id] = peer_id in found
        return {peer_id for peer_id in wanted if self._in_destination[peer_id]}

    def ensure_peers(self, peer_ids: Iterable[str]) -> None:
        """Create each peer in the destination with the source's settings, once.

        Must run before a peer's first message is posted: message creation
        auto-creates an unknown author with default settings, and the peer's
        ``observe_me`` decides whether that message is derived at all.
        """
        todo = [
            peer_id
            for peer_id in dict.fromkeys(peer_ids)
            if peer_id not in self._ensured_peers
        ]
        if not todo:
            return
        self.peers_in_destination(todo)  # record which ones we are about to create
        for peer_id in todo:
            peer = self._source_peers.get(peer_id)
            kwargs: dict[str, Any] = {}
            if peer is not None:
                if peer.metadata:
                    kwargs["metadata"] = peer.metadata
                if _dump(peer.configuration):
                    kwargs["configuration"] = peer.configuration
            self.dest.peer(peer_id, **kwargs)
            self._ensured_peers.add(peer_id)

    @property
    def peers_created(self) -> int:
        """Peers this replay created in the destination."""
        return sum(1 for p in self._ensured_peers if not self._in_destination[p])

    @property
    def peers_existing(self) -> int:
        """Peers the destination already had, which took the source's settings."""
        return len(self._ensured_peers) - self.peers_created

    def peers_for(self, plans: list[SessionPlan], *, all_peers: bool) -> list[str]:
        """The peers a replay of ``plans`` copies, as far as a dry run can tell.

        A partial replay copies the selected sessions' members. Authors who
        have since left a session are also copied, but finding them means
        reading every message, so a dry run does not count them.
        """
        if all_peers:
            return self.load_source_peers()
        members: dict[str, None] = {}

        def collect(plan: SessionPlan) -> None:
            """Add one session's members to the set."""
            for peer_id in _session_member_ids(self.source, plan.session):
                members[peer_id] = None

        _for_each(plans, collect, "Reading session members", _no_tick)
        return list(members)

    def _existing_messages(self, session_id: str) -> Iterator[Any]:
        """The destination session's messages, oldest first; none if it is new.

        Reads through a plain ``Session``: ``client.session()`` is a
        get-or-create, and nothing may be written before the comparison passes.
        """
        try:
            return iter(Session(session_id, self.dest).messages(size=PAGE_SIZE))
        except NotFoundError:
            return iter(())

    def _source_messages(self, session_id: str) -> Iterator[Any]:
        """The source session's messages, oldest first, fetched on first use."""
        yield from Session(session_id, self.source).messages(size=PAGE_SIZE)

    def _count_matching(self, session_id: str, source_messages: Iterator[Any]) -> int:
        """Walk the destination's messages against the source's. Read-only.

        Returns how many leading source messages the destination already has,
        and leaves ``source_messages`` at the first one it does not have.

        Raises:
            ReplayMismatch: a destination message differs from the source's.
        """
        matched = 0
        for existing in self._existing_messages(session_id):
            message = next(source_messages, None)
            if message is None:
                break  # the destination has everything, and more
            if not _same_message(existing, message):
                raise ReplayMismatch(matched + 1)
            matched += 1
        return matched

    def check_sessions(
        self, plans: list[SessionPlan], on_tick: Tick = _no_tick
    ) -> None:
        """Set each plan's ``action``: what a replay would do. Writes nothing.

        Needs ``probe_sessions`` to have run. An empty destination costs one
        request in total; otherwise it is one request per session, plus a read
        of the messages both sides share.
        """
        if self.destination_session_count() == 0:
            for plan in plans:
                plan.action = "create"
            return

        def check(plan: SessionPlan) -> None:
            """Compare one session with the destination and set its action."""
            try:
                matched = self._count_matching(
                    plan.session.id, self._source_messages(plan.session.id)
                )
            except ReplayMismatch as mismatch:
                plan.action, plan.conflict_at = "conflict", mismatch.position
                plan.messages_in_destination = mismatch.position - 1
                return
            plan.messages_in_destination = matched
            if matched == 0:
                plan.action = "create"
            elif matched >= (plan.message_count or 0):
                plan.action = "up_to_date"
            else:
                plan.action = "resume"

        with read_only(self.dest):
            _for_each(plans, check, "Checking destination sessions", on_tick)

    def replay_session(self, session: Any, on_tick: Tick = _no_tick) -> SessionResult:
        """Recreate one session in the destination and re-send its messages.

        Safe to rerun: messages the destination already holds are compared with
        the source one by one and skipped, and only the remainder is posted.

        Raises:
            ReplayMismatch: the destination session's messages differ from the
                source's, so appending would interleave two conversations.
                Raised before anything is written to that session.
        """
        result = SessionResult(session.id)
        members = _session_members(self.source, session)
        member_ids = {peer_id for peer_id, _ in members}
        departed: set[str] = set()

        def track(message: Any) -> Any:
            """Note an author who is no longer a member of the source session."""
            if message.peer_id not in member_ids:
                departed.add(message.peer_id)
            return message

        source_messages = map(track, self._source_messages(session.id))

        # Read-only pass: nothing is written until the comparison passes.
        result.messages_skipped = self._count_matching(session.id, source_messages)

        self.ensure_peers(peer_id for peer_id, _ in members)
        kwargs: dict[str, Any] = {}
        if session.metadata:
            kwargs["metadata"] = session.metadata
        if _dump(session.configuration):
            kwargs["configuration"] = session.configuration
        if members:
            kwargs["peers"] = members
        if session.scopes:
            kwargs["scopes"] = list(session.scopes)
        dest_session = self.dest.session(session.id, **kwargs)

        batch: list[MessageCreateParams] = []

        def flush() -> None:
            """Send the pending batch, creating its authors first."""
            if not batch:
                return
            self.ensure_peers(message.peer_id for message in batch)
            dest_session.add_messages(batch)
            result.messages_created += len(batch)
            batch.clear()
            on_tick(session.id, result.messages_created, None)

        def remove_departed() -> None:
            """Take authors who have left the source session back out.

            Such an author is re-added by its own messages. This covers authors
            of skipped messages too, in case an earlier run stopped before this
            step; removing a peer that is not a member is a no-op.
            """
            leaving: list[Any] = sorted(departed)
            if leaving:
                dest_session.remove_peers(leaving)

        try:
            for message in source_messages:
                batch.append(
                    MessageCreateParams(
                        content=message.content,
                        peer_id=message.peer_id,
                        metadata=message.metadata or None,
                        created_at=message.created_at,
                    )
                )
                if len(batch) >= MESSAGE_BATCH_SIZE:
                    flush()
            flush()
        except Exception:
            # A batch that was sent may have re-added a departed author. Do not
            # leave it a member until someone reruns; the sending error is the
            # one to report, so a failure here is dropped.
            with suppress(Exception):
                remove_departed()
            raise

        remove_departed()
        return result


def replay(
    replayer: Replayer,
    plans: list[SessionPlan],
    *,
    workspace_metadata: dict[str, Any],
    workspace_configuration: Any,
    all_peers: bool,
    on_progress: Callable[[str], None] = lambda _msg: None,
    on_tick: Tick = _no_tick,
) -> ReplayResult:
    """Run a whole replay. Auth errors propagate; other per-session errors are collected.

    Args:
        all_peers: copy every source peer up front (a full replay). When False
            (a partial replay) only peers that take part in the selected
            sessions are created.
    """
    result = ReplayResult()
    result.workspace_copied = replayer.copy_workspace(
        workspace_metadata, workspace_configuration
    )

    peer_ids = replayer.load_source_peers()
    if all_peers:
        replayer.ensure_peers(peer_ids)

    # A source server that predates scopes has none to copy.
    with suppress(NotFoundError):
        replayer.copy_scope_metadata(
            name for plan in plans for name in (plan.session.scopes or [])
        )

    for index, plan in enumerate(plans, start=1):
        session_id = plan.session.id
        prefix = f"[{index}/{len(plans)}] {session_id}: "
        try:
            session_result = replayer.replay_session(plan.session, on_tick)
        except (AuthenticationError, PermissionDeniedError):
            raise
        except ReplayMismatch as e:
            result.failed.append(
                {"session_id": session_id, "reason": "conflict", "error": str(e)}
            )
            on_progress(f"{prefix}not replayed, {e}")
            continue
        except Exception as e:  # noqa: BLE001 - one bad session must not stop the rest
            result.failed.append(
                {"session_id": session_id, "reason": "error", "error": str(e)}
            )
            on_progress(f"{prefix}failed, {e}")
            continue
        result.sessions.append(session_result)
        sent, present = session_result.messages_created, session_result.messages_skipped
        if session_result.action == "up_to_date":
            on_progress(f"{prefix}up to date, {present} already present")
        elif present:
            on_progress(f"{prefix}{sent} sent, {present} already present")
        else:
            on_progress(f"{prefix}{sent} sent")

    result.peers_created = replayer.peers_created
    result.peers_existing = replayer.peers_existing
    return result
