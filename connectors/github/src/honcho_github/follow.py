"""`honcho ingest from github --follow`: poll repositories for what changed.

Every `--interval` (five minutes by default) a poll reads, for each selected
repository, the issues and pull requests updated since its watermark (each in
full: comments, reviews and review comments, edited ones with their edit time)
and the default branch's commits since the head it last saw, and sends what it
hasn't sent already. The position it saves in Honcho, per repository, is the
watermark and the head:

* **Watermark:** a poll notes when it starts; once everything it sent for a
  repository is settled (accepted by ingest, or a bot record nobody answered),
  the repository's watermark becomes that start time less a minute, for clock
  skew. Until then it stays where it was, so a restart reads it all again.
* **Head:** the commits read are those the default branch's head has that the
  head saved last time hasn't (GitHub's compare, as `git log old..new`), so a
  commit dated earlier but pushed or merged later is still read. At most the
  newest thousand per poll; if the saved head is gone (history rewritten), the
  newest thousand.

What it doesn't see: deletions (polling can't), and an edit to a comment or a
review on an issue or pull request nothing else changed (GitHub doesn't count
it as an update). Both need webhooks. Experimental, like the rest of ingest.
"""

from __future__ import annotations

import datetime
import logging
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from honcho_github.client import GitHubClient, GitHubError
from honcho_github.export import (
    Fetched,
    SelectionError,
    commits_between,
    fetch,
    head_commit,
    read_commits,
    records_for,
    select,
)
from honcho_github.mapping import Json, Repo, instance_for, obj

logger = logging.getLogger(__name__)

INTERVAL = 300.0
OVERLAP = datetime.timedelta(minutes=1)
MAX_COMMITS = 1_000
"""Commits read back to the saved head in one poll, at most."""
FLUSH_RECORDS = 100
MAX_BACKOFF = 600.0
MAX_SEEN = 200_000
"""Records remembered as sent; past this the memory starts afresh (ingest
takes a record sent again as a duplicate)."""


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _stamp(moment: datetime.datetime) -> str:
    """GitHub's timestamp format, UTC, as `--since` and `updatedAt` use it."""
    return moment.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Flushed(Protocol):
    settled: set[str]
    held: set[str]
    error: str | None


class FollowSink(Protocol):
    """What `honcho ingest from github --follow` hands over: `honcho_cli.follow`."""

    stop: threading.Event

    def container(self, name: str) -> None: ...

    def record(self, record: Mapping[str, Any]) -> None: ...

    def flush(self, *, live: bool = True) -> Flushed: ...

    def position(self, instance: str) -> dict[str, Any] | None: ...

    def save_position(
        self,
        instance: str,
        position: dict[str, Any],
        *,
        last_record_at: datetime.datetime | None = None,
        error: str | None = None,
    ) -> bool: ...


FOLLOW_METHODS = ("flush", "position", "save_position")


@dataclass
class _RepoState:
    saved: dict[str, Any]
    """The watermark and head as saved: what the next poll reads from."""
    next: dict[str, Any] | None = None
    """What to save once everything sent for the repository is settled."""
    unsettled: set[str] = field(default_factory=set)


@dataclass
class FollowStats:
    records: int = 0
    polls: int = 0
    skipped: Counter[str] = field(default_factory=Counter)


class GitHubFollower:
    """Poll the selected repositories until the sink's `stop` is set."""

    def __init__(
        self,
        client: GitHubClient,
        repos: Sequence[str],
        owners: Sequence[str],
        sink: FollowSink,
        *,
        interval: float = INTERVAL,
        since: str | None = None,
        now: Callable[[], datetime.datetime] = _utcnow,
    ) -> None:
        self.client: GitHubClient = client
        self.repo_names: Sequence[str] = repos
        self.owners: Sequence[str] = owners
        self.sink: FollowSink = sink
        self.interval: float = interval
        self.since: str | None = since
        self.now: Callable[[], datetime.datetime] = now
        self.instance: str = instance_for(client.host)
        self.stats: FollowStats = FollowStats()
        self.repos: dict[str, _RepoState] = {}
        self._loaded: dict[str, Any] = {}
        # What was handed over this run, by source ID: its revision (edit time,
        # else creation time), so a poll that reads it again doesn't resend it.
        self._seen: dict[str, str] = {}
        self._created: dict[str, str] = {}
        self._newest: str | None = None
        self._accounts: dict[str, dict[str, Any]] = {}
        self._problems: dict[str, str] = {}
        self._handed: int = 0
        self._failures: int = 0
        self._saved: tuple[dict[str, Any], str | None] | None = None

    def run(self) -> FollowStats:
        """Poll until stopped; send and save after every poll and at the end.

        Raises SystemExit when the repositories can't be selected at the start.
        """
        self._loaded = dict(obj((self.sink.position(self.instance) or {}).get("repos")))
        first = True
        while not self.sink.stop.is_set():
            if self._failures and not self._flush_and_save():
                self._wait(
                    min(MAX_BACKOFF, self.interval * 2 ** min(self._failures, 8))
                )
                continue
            started = self.now()
            try:
                selected = select(self.client, self.repo_names, self.owners)
            except (SelectionError, GitHubError) as error:
                if first:
                    raise SystemExit(str(error)) from None
                logger.warning(
                    "Couldn't list the repositories (%s); trying again", error
                )
                self._problems["(selection)"] = str(error)
                self._wait(self.interval)
                continue
            self._problems.pop("(selection)", None)
            first = False
            for listed in selected:
                if self.sink.stop.is_set():
                    break
                self._poll(listed.repo, started)
            self._flush_and_save()
            self.stats.polls += 1
            self._wait(self.interval)
        self._flush_and_save()
        return self.stats

    def _wait(self, seconds: float) -> None:
        self.sink.stop.wait(seconds)

    def _poll(self, repo: Repo, started: datetime.datetime) -> None:
        name = repo.name_with_owner
        state = self.repos.get(name)
        if state is None:
            state = _RepoState(saved=dict(obj(self._loaded.get(name))))
            self.repos[name] = state
        since = state.saved.get("updated_since") or self.since
        head = state.saved.get("head")
        try:
            # The head first: commits pushed while the rest is read come with
            # the history, or with the next poll.
            new_head = head_commit(self.client, repo)
            fetched = fetch(self.client, repo, since, with_commits=False)
            commits, cut = self._commits(repo, head, new_head)
            if cut:
                logger.warning(
                    "%s: more than %d commits since the last poll; read the newest %d",
                    name,
                    MAX_COMMITS,
                    MAX_COMMITS,
                )
            records, skipped = records_for(
                self.client,
                repo,
                Fetched(threads=fetched.threads, commits=commits),
                self.instance,
                self._accounts,
            )
        except GitHubError as error:
            logger.warning("Couldn't read %s: %s", name, error)
            self._problems[name] = str(error)
            return
        self._problems.pop(name, None)
        self.stats.skipped.update(skipped)
        self.sink.container(name)
        sent = 0
        for record in records:
            key = str(record["source_id"])
            revision = str(record.get("revised_at") or record["created_at"])
            if self._seen.get(key) == revision:
                continue
            if len(self._seen) >= MAX_SEEN:
                self._seen.clear()
            self._seen[key] = revision
            self._created[key] = str(record["created_at"])
            self.sink.record(record)
            state.unsettled.add(key)
            sent += 1
            self._handed += 1
            # While Honcho is failing, the run loop retries with backoff.
            if self._handed >= FLUSH_RECORDS and not self._failures:
                self._flush_and_save()
        state.next = {
            "updated_since": _stamp(started - OVERLAP),
            "head": new_head or head,
        }
        self.stats.records += sent
        if sent:
            logger.info("%s: %d new or changed records", name, sent)

    def _commits(
        self, repo: Repo, head: str | None, new_head: str | None
    ) -> tuple[list[Json], bool]:
        """The default branch's commits since `head` (every one the first time)."""
        if head is None:
            return read_commits(self.client, repo, since=self.since)
        if new_head is None or new_head == head:
            return [], False
        found = commits_between(self.client, repo, head, new_head, limit=MAX_COMMITS)
        if found is not None:
            return found
        logger.warning(
            "%s: the head it last saw is gone (history rewritten); reading the"
            + " newest %d commits",
            repo.name_with_owner,
            MAX_COMMITS,
        )
        return read_commits(self.client, repo, limit=MAX_COMMITS)

    def _flush_and_save(self) -> bool:
        """Send what's pending, move the positions it settled, and save them.

        False when Honcho failed in a way that may pass; what wasn't sent stays
        pending.
        """
        flushed = self.sink.flush(live=True)
        settled = flushed.settled - flushed.held
        for key in settled:
            created = self._created.pop(key, None)
            if created is not None and (self._newest is None or created > self._newest):
                self._newest = created
        for state in self.repos.values():
            state.unsettled -= settled
            if state.next is not None and not state.unsettled:
                state.saved, state.next = state.next, None
        if flushed.error is not None:
            self._failures += 1
            logger.warning("Couldn't send to Honcho (%s); trying again", flushed.error)
            return False
        if self._failures:
            logger.info("Honcho is answering again")
        self._failures = 0
        self._handed = 0
        self._save()
        return True

    def _save(self) -> None:
        repos = dict(self._loaded)
        repos.update(
            {name: state.saved for name, state in self.repos.items() if state.saved}
        )
        position: dict[str, Any] = {"repos": dict(sorted(repos.items()))}
        problems = [f"{name}: {why}" for name, why in sorted(self._problems.items())]
        error = "; ".join(problems)[:2000] or None
        if self._saved == (position, error):
            return
        newest = (
            datetime.datetime.fromisoformat(self._newest.replace("Z", "+00:00"))
            if self._newest
            else None
        )
        if self.sink.save_position(
            self.instance, position, last_record_at=newest, error=error
        ):
            self._saved = (position, error)


__all__ = ["FOLLOW_METHODS", "FollowSink", "FollowStats", "GitHubFollower"]
