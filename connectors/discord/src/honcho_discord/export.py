"""Walk a Discord server's history and hand each record to a sink.

Containers are visited channel by channel, each followed by its threads, and
each container's messages oldest first: the ingest API keeps arrival order
within a session, so a container's history has to arrive in order.
"""

import contextlib
import datetime
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast, final

import httpx

from honcho_discord.client import DiscordClient, DiscordError
from honcho_discord.mapping import (
    EXPORTABLE_TYPES,
    MESSAGE_CHANNEL_TYPES,
    THREAD_PARENT_TYPES,
    THREAD_TYPES,
    Container,
    Json,
    account,
    is_bot,
    items,
    obj,
    to_record,
)

logger = logging.getLogger(__name__)

INGEST_BATCH = 100


class Sink(Protocol):
    """Where exported records go."""

    def container(self, container: Container) -> None:
        """A container is about to be read. Records for it follow."""
        ...

    def record(self, record: dict[str, Any]) -> None: ...

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        """The export finished: flush, and keep the accounts and summary."""
        ...


@dataclass
class Selection:
    """The channels to read and the threads under them."""

    channels: list[Container]
    threads: dict[str, list[Container]]
    categories: dict[str, str]
    # Channels whose threads couldn't all be listed, with why.
    incomplete: dict[str, str] = field(default_factory=dict)

    def containers(self) -> list[Container]:
        """Each channel followed by its threads, threads oldest first."""
        ordered: list[Container] = []
        for channel in self.channels:
            if channel.type in MESSAGE_CHANNEL_TYPES:
                ordered.append(channel)
            ordered.extend(
                sorted(self.threads.get(channel.id, []), key=lambda t: int(t.id))
            )
        return ordered


class SelectionError(ValueError):
    """The requested channels can't be exported as asked."""


def select(
    client: DiscordClient, guild_id: str, channel_ids: Collection[str] | None
) -> Selection:
    """The guild's channels to export (all, or `channel_ids`) and their threads.

    Raises:
        SelectionError: `channel_ids` is empty, or names channels the guild
            doesn't have or that hold no messages (categories, for example).
    """
    if channel_ids is not None and not channel_ids:
        raise SelectionError("No channels selected")
    all_channels = client.channels(guild_id)
    categories = {
        str(c["id"]): str(c.get("name") or c["id"])
        for c in all_channels
        if c.get("type") == 4
    }
    exportable = {
        str(c["id"]) for c in all_channels if c.get("type") in EXPORTABLE_TYPES
    }
    if channel_ids is not None:
        unknown = sorted(set(channel_ids) - exportable)
        if unknown:
            raise SelectionError(
                "Not exportable channels in this guild: " + ", ".join(unknown)
            )
    wanted = [
        Container.from_channel(c)
        for c in sorted(
            all_channels, key=lambda c: (c.get("position", 0), int(c["id"]))
        )
        if str(c["id"]) in exportable
        and (channel_ids is None or str(c["id"]) in channel_ids)
    ]
    wanted_ids = {c.id for c in wanted}
    threads: dict[str, list[Container]] = {}
    seen: set[str] = set()
    incomplete: dict[str, str] = {}

    def keep(thread: Json) -> None:
        if thread.get("type") not in THREAD_TYPES or str(thread["id"]) in seen:
            return
        parent = str(thread.get("parent_id"))
        if parent in wanted_ids:
            seen.add(str(thread["id"]))
            threads.setdefault(parent, []).append(Container.from_channel(thread))

    for thread in client.active_threads(guild_id):
        keep(thread)
    for channel in wanted:
        if channel.type not in THREAD_PARENT_TYPES:
            continue
        try:
            for thread in client.archived_threads(channel.id):
                keep(thread)
        except DiscordError as error:
            logger.warning(
                "Couldn't list archived threads of %s: %s", channel.name, error
            )
            incomplete[channel.id] = str(error)
    # Again: a thread unarchived while the archives were read is in neither list
    # the first time. The export is still not an atomic snapshot.
    for thread in client.active_threads(guild_id):
        keep(thread)
    return Selection(
        channels=wanted, threads=threads, categories=categories, incomplete=incomplete
    )


@dataclass
class Stats:
    records: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    containers: int = 0
    unreadable: list[str] = field(default_factory=list)
    # Channels whose threads couldn't all be listed.
    incomplete: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unreadable and not self.incomplete


def export(
    client: DiscordClient,
    guild_id: str,
    make_sink: Callable[[], Sink],
    *,
    channel_ids: Collection[str] | None = None,
    limit_per_container: int | None = None,
    progress: Callable[[str], None] = logger.info,
) -> tuple[Stats, Sink]:
    """Read the selected history into a sink, oldest first per container.

    The sink is made only once the selection is known to be valid, so a bad
    selection never creates a workspace or an output file.
    """
    selection = select(client, guild_id, channel_ids)
    sink = make_sink()
    stats = Stats(incomplete=sorted(selection.incomplete))
    members: dict[str, Json] = {}
    try:
        for member in client.members(guild_id):
            members[str(member["user"]["id"])] = member
    except DiscordError as error:
        # Without the members intent, names fall back to global display names.
        logger.warning("Couldn't list members (%s); nicknames won't be used", error)

    accounts: dict[str, dict[str, Any]] = {}
    for container in selection.containers():
        sink.container(container)
        stats.containers += 1
        count = 0
        try:
            for message in client.messages(container.id):
                author = message["author"]
                author_id = str(author["id"])
                if author_id not in accounts:
                    accounts[author_id] = account(
                        author, members.get(author_id), bot=is_bot(message)
                    )
                record = to_record(message, container, guild_id, members)
                if record is None:
                    stats.skipped["system_or_empty"] += 1
                    continue
                for person in record["participants"]:
                    if person["id"] not in accounts:
                        user = _mentioned_user(message, person["id"])
                        if user is not None:
                            accounts[person["id"]] = account(
                                user,
                                members.get(person["id"]),
                                bot=bool(person.get("bot")),
                            )
                sink.record(record)
                stats.records += 1
                count += 1
                if limit_per_container is not None and count >= limit_per_container:
                    break
        except DiscordError as error:
            logger.warning("Couldn't read %s: %s", container.name, error)
            stats.unreadable.append(container.id)
        progress(f"{container.name}: {count} records")

    summary = {
        "source": "discord",
        "source_instance": guild_id,
        "exported_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "records": stats.records,
        "containers": stats.containers,
        "skipped": dict(stats.skipped),
        "unreadable_containers": stats.unreadable,
        "incomplete_thread_lists": selection.incomplete,
        "complete": stats.complete,
        "categories": selection.categories,
    }
    sink.close(accounts.values(), summary)
    return stats, sink


def _mentioned_user(message: Json, user_id: str) -> Json | None:
    mentions = list(items(message.get("mentions")))
    for snapshot in items(message.get("message_snapshots")):
        mentions.extend(items(obj(snapshot.get("message")).get("mentions")))
    for user in mentions:
        if str(user["id"]) == user_id:
            return user
    author = obj(obj(message.get("referenced_message")).get("author"))
    if str(author.get("id")) == user_id:
        return author
    return None


def inside_git_worktree(path: Path) -> bool:
    """Whether `path` is inside a git checkout, where exported data could get committed."""
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / ".git").exists():
            return True
    return False


@final
class JsonlSink:
    """Writes `records.jsonl`, `accounts.jsonl`, `containers.jsonl` and `export.json`.

    Refuses a directory inside a git checkout unless `allow_git_dir`, so exported
    messages don't end up committed. The directory is opened once, refusing a
    symlink, and every file is created inside that open directory without
    following symlinks, owner-only, so nothing redirects the writes afterwards.
    """

    def __init__(self, directory: Path, *, allow_git_dir: bool = False):
        if inside_git_worktree(directory) and not allow_git_dir:
            raise SelectionError(
                f"{directory} is inside a git checkout, where exported messages"
                + " could be committed. Choose another directory."
            )
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory: Path = directory
        self._dir: int = os.open(
            directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        self._records: Any = self._create("records.jsonl")
        self._containers: Any = self._create("containers.jsonl")

    def _create(self, name: str) -> Any:
        # A file left by an earlier export is unlinked, not truncated: it could
        # be a hard link to a file elsewhere, which truncating would overwrite.
        # The new file is created exclusively, so nothing can be swapped in
        # between.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name, dir_fd=self._dir)
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._dir,
        )
        return os.fdopen(fd, "w", encoding="utf-8")

    def container(self, container: Container) -> None:
        self._containers.write(
            json.dumps({**container.as_record(), "type": container.type}) + "\n"
        )

    def record(self, record: dict[str, Any]) -> None:
        self._records.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        self._records.close()
        self._containers.close()
        with self._create("accounts.jsonl") as f:
            for item in sorted(accounts, key=lambda a: a["id"]):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        with self._create("export.json") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        os.close(self._dir)


def _ingest_problem(response: httpx.Response) -> str:
    """Why ingest refused a request, without echoing the records it was sent.

    Validation errors name the field and the error type only: their `input`
    would print message text.
    """
    try:
        body = response.json()
    except ValueError:
        return str(response.status_code)
    detail: Any = obj(body).get("detail")
    if isinstance(detail, list):
        problems: list[str] = []
        for entry in cast(list[Any], detail)[:5]:
            error = obj(entry)
            where = ".".join(
                str(part) for part in cast(list[Any], error.get("loc") or [])
            )
            problems.append(f"{where}: {error.get('type', 'invalid')}")
        return f"{response.status_code} " + "; ".join(problems)
    # Honcho's own messages for a missing or conflicting workspace are useful
    # and carry no records; other statuses' text could.
    if isinstance(detail, str) and response.status_code in (404, 409):
        return f"{response.status_code} {detail[:200]}"
    return str(response.status_code)


class IngestError(Exception):
    """The ingest API refused a batch."""


@final
class IngestSink:
    """Sends records to Honcho's ingest API, a container at a time, deriving later.

    The workspace is created as an auto workspace if it doesn't exist; ingest
    writes only to auto workspaces. Records are sent with `derive: "later"`, so
    nothing is derived until someone calls `POST /ingest/derive`.
    """

    def __init__(
        self,
        base_url: str,
        workspace: str,
        *,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url, headers=headers, transport=transport, timeout=120
        )
        self.workspace = workspace
        self.outcomes: Counter[str] = Counter()
        self.rejections: Counter[str] = Counter()
        self._batch: list[dict[str, Any]] = []
        response = self._post("/v3/workspaces", {"id": workspace, "auto": True})
        if response.status_code not in (200, 201):
            raise IngestError(
                f"Couldn't use workspace {workspace}: {_ingest_problem(response)}"
            )

    def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._http.post(path, json=body)
        except httpx.TransportError as error:
            # The exception's text can include request headers.
            raise IngestError(
                f"Couldn't reach Honcho: {type(error).__name__}"
            ) from None

    def container(self, container: Container) -> None:
        # Finish the previous container's history before starting the next.
        del container
        self._flush()

    def record(self, record: dict[str, Any]) -> None:
        self._batch.append(record)
        if len(self._batch) >= INGEST_BATCH:
            self._flush()

    def _flush(self) -> None:
        if not self._batch:
            return
        # A 503 means contention outlasted the server's retries and nothing was
        # written; resending the same batch is safe.
        body = {"derive": "later", "records": self._batch}
        path = f"/v3/workspaces/{self.workspace}/ingest"
        response = self._post(path, body)
        for _ in range(4):
            if response.status_code != 503:
                break
            self._sleep(float(response.headers.get("Retry-After", 1)))
            response = self._post(path, body)
        if response.status_code != 200:
            raise IngestError(f"Ingest refused a batch: {_ingest_problem(response)}")
        for result in response.json()["records"]:
            self.outcomes[result["outcome"]] += 1
            if result.get("reason"):
                self.rejections[result["reason"]] += 1
        self._batch = []

    def close(
        self, accounts: Iterable[dict[str, Any]], summary: dict[str, Any]
    ) -> None:
        # The server keeps what ingest built; accounts and the summary are for
        # file exports.
        del accounts, summary
        self._flush()
        self._http.close()
