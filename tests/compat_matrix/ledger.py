"""What every write promised, checked through every API image.

Each expectation is recorded when something is written (by any image) and is
checked again after every later phase, through both the old and the new API.
That is the property rollback depends on: anything either image wrote stays
readable, the same way, by both. A candidate that stops writing a location
the old image still reads fails here as soon as the old API reads a row the
new code wrote. plastic-labs/honcho#952, which moved provenance out of
``source_ids``, is the case this was built to catch.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from tests.compat_matrix.api import HonchoAPI


class Expectation(Protocol):
    origin: str

    def check(self, api: HonchoAPI, cache: "ReadCache") -> list[str]: ...


class ReadCache:
    """Per-API listing cache, so one verification pass lists each workspace once."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str], Any] = {}

    def get(self, api: HonchoAPI, kind: str, key: str, load: Any) -> Any:
        slot = (api.label, kind, key)
        if slot not in self._data:
            self._data[slot] = load()
        return self._data[slot]


def _conclusions(api: HonchoAPI, cache: ReadCache, workspace: str) -> list[Any]:
    return cache.get(
        api, "conclusions", workspace, lambda: api.list_conclusions(workspace)
    )


def _diff(what: str, want: dict[str, Any], got: dict[str, Any]) -> list[str]:
    return [
        f"{what}: {key} = {got.get(key)!r}, expected {value!r}"
        for key, value in want.items()
        if got.get(key) != value
    ]


@dataclass
class ConclusionExpect:
    """A conclusion with known id: present with these fields, or gone."""

    origin: str
    workspace: str
    id: str
    fields: dict[str, Any] = field(default_factory=dict)
    present: bool = True

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        what = f"conclusion {self.id} ({self.origin})"
        got = api.get_conclusion(self.workspace, self.id)
        listed = any(
            c["id"] == self.id for c in _conclusions(api, cache, self.workspace)
        )
        if not self.present:
            problems: list[str] = []
            if got is not None:
                problems.append(f"{what}: deleted, but GET still returns it")
            if listed:
                problems.append(f"{what}: deleted, but list still returns it")
            return problems
        if got is None:
            return [f"{what}: missing from GET"]
        problems = _diff(what, self.fields, got)
        if not listed:
            problems.append(f"{what}: GET finds it but list does not")
        return problems


@dataclass
class DerivedExpect:
    """Queued work produced a conclusion mentioning ``marker``."""

    origin: str
    workspace: str
    observer: str
    observed: str
    marker: str
    present: bool = True

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        found = any(
            self.marker in c["content"]
            and c["observer_id"] == self.observer
            and c["observed_id"] == self.observed
            for c in _conclusions(api, cache, self.workspace)
        )
        if found == self.present:
            return []
        state = "missing" if self.present else "still present after deletion"
        return [
            f"derived conclusion for {self.marker} ({self.origin}): {state}"
            + f" in {self.observer}->{self.observed}"
        ]


@dataclass
class DeductionExpect:
    """A dream wrote a deductive conclusion citing two premises."""

    origin: str
    workspace: str
    observer: str
    observed: str
    # Filled by the first read that shows two premises; every later read
    # through any image must agree with it. A read missing its premises
    # never pins, so it is reported as the broken one.
    pinned: dict[str, Any] | None = None

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        what = f"dream deduction ({self.origin})"
        rows = [
            c
            for c in _conclusions(api, cache, self.workspace)
            if c["level"] == "deductive"
            and c["observer_id"] == self.observer
            and c["observed_id"] == self.observed
            and c["content"].startswith("cmx deduction")
        ]
        if not rows:
            return [f"{what}: no deductive conclusion"]
        row = rows[0]
        if len(row["source_ids"]) != 2:
            return [
                f"{what}: source_ids = {row['source_ids']!r}, expected two premises"
            ]
        if self.pinned is None:
            self.pinned = {"id": row["id"], "source_ids": row["source_ids"]}
            return []
        return _diff(what, self.pinned, row)


@dataclass
class MessageExpect:
    origin: str
    workspace: str
    session: str
    id: str
    fields: dict[str, Any]
    present: bool = True

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        del cache  # one GET per message; nothing to share
        what = f"message {self.id} ({self.origin})"
        got = api.get_message(self.workspace, self.session, self.id)
        if not self.present:
            return [f"{what}: session deleted, message still readable"] if got else []
        if got is None:
            return [f"{what}: missing"]
        return _diff(what, self.fields, got)


@dataclass
class SessionExpect:
    origin: str
    workspace: str
    name: str
    metadata: dict[str, Any] | None = None
    present: bool = True

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        what = f"session {self.name} ({self.origin})"
        sessions = cache.get(
            api, "sessions", self.workspace, lambda: api.list_sessions(self.workspace)
        )
        got = next((s for s in sessions if s["id"] == self.name), None)
        if not self.present:
            return [f"{what}: deleted, still listed"] if got else []
        if got is None:
            return [f"{what}: not listed"]
        if self.metadata is not None:
            return _diff(what, {"metadata": self.metadata}, got)
        return []


@dataclass
class PeerExpect:
    origin: str
    workspace: str
    name: str
    metadata: dict[str, Any]

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        what = f"peer {self.name} ({self.origin})"
        peers = cache.get(
            api, "peers", self.workspace, lambda: api.list_peers(self.workspace)
        )
        got = next((p for p in peers if p["id"] == self.name), None)
        if got is None:
            return [f"{what}: not listed"]
        return _diff(what, {"metadata": self.metadata}, got)


@dataclass
class WorkspaceExpect:
    origin: str
    name: str
    metadata: dict[str, Any]

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        what = f"workspace {self.name} ({self.origin})"
        workspaces = cache.get(api, "workspaces", "", api.list_workspaces)
        got = next((w for w in workspaces if w["id"] == self.name), None)
        if got is None:
            return [f"{what}: not listed"]
        return _diff(what, {"metadata": self.metadata}, got)


@dataclass
class SummaryExpect:
    """A session summary the worker wrote is readable, and the same everywhere."""

    origin: str
    workspace: str
    session: str
    pinned: dict[str, Any] | None = None

    def check(self, api: HonchoAPI, cache: ReadCache) -> list[str]:
        del cache  # one GET per session; nothing to share
        what = f"summary of {self.session} ({self.origin})"
        got = api.summaries(self.workspace, self.session).get("short_summary")
        if not got:
            return [f"{what}: no short summary"]
        view = {"content": got.get("content"), "message_id": got.get("message_id")}
        if self.pinned is None:
            self.pinned = view
            return []
        return _diff(what, self.pinned, view)


class Ledger:
    """Every expectation recorded so far, in write order."""

    def __init__(self) -> None:
        self.items: list[Expectation] = []

    def add(self, *expectations: Expectation) -> None:
        self.items.extend(expectations)

    def retract(self, predicate: Any) -> None:
        """Drop expectations a later, deliberate write superseded."""
        self.items = [e for e in self.items if not predicate(e)]

    def verify(self, apis: list[HonchoAPI]) -> list[str]:
        problems: list[str] = []
        for api in apis:
            cache = ReadCache()
            for expectation in self.items:
                try:
                    found = expectation.check(api, cache)
                except Exception as exc:  # an unreadable row is a finding too
                    found = [f"{expectation.origin}: {type(exc).__name__}: {exc}"]
                problems += [f"[read via {api.label}] {p}" for p in found]
        return problems
