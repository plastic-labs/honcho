"""Send a benchmark dataset through the ingest API, and check what ingest built.

    python -m tests.bench.auto_workspaces.replay --base-url http://127.0.0.1:8000 \\
        --workspace bench-1 [--data DIR] [--split all]

Records go in with `derive: "later"`, so nothing is derived; the report gives
the token count of what was deferred, to approve before deriving. Ingest still
embeds new messages right away when the server has `EMBED_MESSAGES` on, which
calls the embedding provider: for a replay that costs nothing, run the server
with `EMBED_MESSAGES=false` or against the mock provider (`src/mock_provider`).

History is sent oldest first, in batches of 100, as a connector should. Edits
are sent after and must be rejected `edit_not_supported` until edits ship;
deletes have no API yet and are only counted.

Then it checks ingest's mapping against the dataset's accounts: every account
of a stored record (authors and participants) must have exactly one peer, an
explicit `peer_id` must be that peer, and no two accounts may share a peer.
Without an organizer every account is its own peer, so scoring the mapping is
scoring "no organizer". Any other outcome makes the run fail.

Only a local server is accepted unless `--allow-remote` is given: a replay
creates sessions and peers in the target workspace. Datasets whose mapping
depends on what ingest stored first (a source record created twice, an
explicit `peer_id` naming a generated peer) are refused before anything is
sent; see `check_supported`.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args
from urllib.parse import urlparse

import httpx

from src import schemas
from src.crud.ingest import BindingKey, author_key_for, peer_key_for, record_identity
from tests.bench.auto_workspaces import fixtures
from tests.bench.auto_workspaces.model import (
    AccountKey,
    BenchRecord,
    Dataset,
    Prediction,
    account_key,
    event_time,
    record_accounts,
)
from tests.bench.auto_workspaces.scoring import count_tokens, score
from tests.bench.auto_workspaces.split import Split, select

BATCH = 100
_LOCAL = {"127.0.0.1", "localhost", "::1", "testserver"}


@dataclass
class ReplayReport:
    workspace: str
    sent: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    deletes_not_sent: int = 0
    sessions: int = 0
    peers: int = 0
    # Tokens of the stored records left deferred: what deriving would process.
    deferred_tokens: int = 0
    # Failures; any of these makes the run fail.
    unexpected_outcomes: list[str] = field(default_factory=list)
    accounts_without_a_peer: list[AccountKey] = field(default_factory=list)
    accounts_with_several_peers: list[AccountKey] = field(default_factory=list)
    explicit_peers_renamed: list[AccountKey] = field(default_factory=list)
    peers_shared_by_accounts: dict[str, list[AccountKey]] = field(default_factory=dict)
    # Identity recall of ingest's mapping as-is: the "no organizer" floor.
    unorganized_identity_recall: float = 0.0

    def failed(self) -> bool:
        return bool(
            self.unexpected_outcomes
            or self.accounts_without_a_peer
            or self.accounts_with_several_peers
            or self.explicit_peers_renamed
            or self.peers_shared_by_accounts
        )


def _ordered(records: list[BenchRecord]) -> list[BenchRecord]:
    """Creates oldest first, then edits by revision time; deletes are left out."""
    creates = [r for r in records if r.op == "create"]
    edits = [r for r in records if r.op == "edit"]
    return sorted(creates, key=event_time) + sorted(edits, key=event_time)


def _members(client: httpx.Client, workspace: str, session: str) -> set[str]:
    """Every peer of a session, page by page."""
    names: set[str] = set()
    page = 1
    while True:
        response = client.get(
            f"/v3/workspaces/{workspace}/sessions/{session}/peers",
            params={"page": page, "size": 100},
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        names.update(item["id"] for item in body["items"])
        if page >= int(body.get("pages") or 1):
            return names
        page += 1


def _candidates(
    record: dict[str, Any],
) -> tuple[list[tuple[AccountKey, list[str]]], list[AccountKey]]:
    """Every person the record names, with the names ingest could give them,
    and which of them are participants (whose peer replay has to find)."""
    validated = schemas.IngestRecordCreate.model_validate(record)
    container = validated.container.id if validated.container else None

    def key(person: schemas.IngestParticipant) -> AccountKey:
        return account_key(
            validated.source,
            validated.source_instance,
            external_id=person.id,
            name=person.name,
            container=container,
        )

    people = [
        (key(person), peer_key_for(validated, person).candidates())
        # An explicit peer_id replaces the author: ingest makes no peer for it.
        for person in [
            *([validated.author] if validated.author and not validated.peer_id else []),
            *validated.participants,
        ]
    ]
    return people, [key(person) for person in validated.participants]


class UnsupportedDataset(ValueError):
    """A dataset whose mapping replay can't check soundly."""


def check_supported(dataset: Dataset) -> None:
    """Refuse datasets whose expected mapping depends on ingest's history.

    Replay predicts each account's peer from the records alone. That holds
    when every source record is created once (a resend with other authors or
    participants is a duplicate, and ingest keeps the first version's), when
    no explicit `peer_id` names a peer ingest could generate for an inferred
    account (an alias, legitimate, but only knowable from the order things
    were stored), and when no two accounts could be given the same generated
    name (ingest gives the second one the long name, again by order).

    Raises:
        UnsupportedDataset: Which records break which rule.
    """
    problems: list[str] = []
    seen: dict[tuple[str, str, str, str], str] = {}
    generated: set[str] = set()
    claimants: dict[str, set[AccountKey]] = {}
    explicit: dict[str, str] = {}
    for r in dataset.records:
        if r.op != "create":
            continue
        record = r.record
        validated = schemas.IngestRecordCreate.model_validate(record)
        # The identity ingest dedups on: the source_id, or a content hash.
        kind, value = record_identity(validated)
        identity = (validated.source, validated.source_instance, kind, value)
        if identity in seen:
            problems.append(f"{r.key} creates {seen[identity]}'s source record again")
        seen.setdefault(identity, r.key)
        people, _ = _candidates(record)
        for account, names in people:
            generated.update(names)
            for name in names:
                claimants.setdefault(name, set()).add(account)
        author = author_key_for(validated)
        if isinstance(author, BindingKey):  # the source peer of author-less records
            generated.update(author.candidates())
        if validated.peer_id:
            explicit.setdefault(validated.peer_id, r.key)
    problems += [
        f"{key}: explicit peer_id {name!r} is a name ingest generates"
        for name, key in explicit.items()
        if name in generated
    ]
    problems += [
        f"accounts {sorted(accounts)} could both be named {name!r}"
        for name, accounts in claimants.items()
        if len(accounts) > 1
    ]
    if problems:
        raise UnsupportedDataset("; ".join(problems[:20]))


def replay(
    client: httpx.Client, dataset: Dataset, workspace: str, *, derive: str = "later"
) -> tuple[ReplayReport, Prediction]:
    """Ingest the dataset into `workspace`; return the report and ingest's mapping.

    Raises:
        UnsupportedDataset: The dataset is one replay can't check (see
            `check_supported`); nothing was sent.
    """
    check_supported(dataset)
    # Ingest writes only to auto workspaces, and only a new workspace can be
    # made auto, so `workspace` must be new or already auto (else a 409).
    response = client.post("/v3/workspaces", json={"id": workspace, "auto": True})
    response.raise_for_status()

    report = ReplayReport(workspace=workspace)
    report.deletes_not_sent = sum(1 for r in dataset.records if r.op == "delete")
    outcomes: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    peers_of: dict[AccountKey, set[str]] = defaultdict(set)
    expected: set[AccountKey] = set()
    session_of: dict[str, str] = {}
    # Every name ingest could have given each account, from all its
    # occurrences as author or participant: ingest names a peer after the first
    # display name it saw and keeps that name when the person renames.
    names_of: dict[AccountKey, set[str]] = defaultdict(set)
    # Per session, the participant accounts whose peer must be a member.
    participants_in: dict[str, set[AccountKey]] = defaultdict(set)
    deferred: dict[str, int] = {}

    ordered = _ordered(dataset.records)
    for start in range(0, len(ordered), BATCH):
        batch = ordered[start : start + BATCH]
        response = client.post(
            f"/v3/workspaces/{workspace}/ingest",
            json={"derive": derive, "records": [r.record for r in batch]},
            timeout=120,
        )
        response.raise_for_status()
        results: list[dict[str, Any]] = response.json()["records"]
        report.sent += len(batch)
        for r, result in zip(batch, results, strict=True):
            outcome: str = result["outcome"]
            reason: str | None = result.get("reason")
            outcomes[outcome] += 1
            if reason:
                rejections[reason] += 1
            if r.op == "edit":
                if (outcome, reason) != ("rejected", "edit_not_supported"):
                    report.unexpected_outcomes.append(f"{r.key}: edit {outcome}")
                continue
            if outcome == "rejected":
                report.unexpected_outcomes.append(f"{r.key}: rejected {reason}")
                continue
            missing = [
                f for f in ("record_id", "session_id", "peer_id") if not result.get(f)
            ]
            if missing:
                report.unexpected_outcomes.append(
                    f"{r.key}: {outcome} without {', '.join(missing)}"
                )
                continue
            session: str = result["session_id"]
            session_of[r.key] = session
            accounts = record_accounts(r.record)
            expected.update(accounts)
            peers_of[accounts[0]].add(result["peer_id"])
            people, participants = _candidates(r.record)
            for account, names in people:
                names_of[account].update(names)
            participants_in[session].update(participants)
            if result.get("derivation") == "deferred":
                deferred[result["record_id"]] = count_tokens(r.record["content"])

    # Participants: in every session they took part in, a member must be one of
    # the names ingest could have given them.
    # A name an author's result gave another account isn't this participant's,
    # even when it is one of their candidates (two people whose generated
    # names collide get the short and the long one).
    owner: dict[str, AccountKey] = {
        peer: account for account, ps in peers_of.items() for peer in ps
    }
    unresolved: list[str] = []
    for session, accounts_here in participants_in.items():
        members = _members(client, workspace, session)
        for account in accounts_here:
            found = {
                n
                for n in names_of[account] & members
                if owner.get(n, account) == account
            }
            if not found:
                unresolved.append(f"{account} in session {session}")
            peers_of[account].update(found)

    report.outcomes = dict(outcomes)
    report.rejections = dict(rejections)
    report.sessions = len(set(session_of.values()))
    report.peers = len({p for ps in peers_of.values() for p in ps})
    report.deferred_tokens = sum(deferred.values())
    report.accounts_without_a_peer = sorted(
        {a for a in expected if not peers_of[a]} | set(unresolved)
    )
    report.accounts_with_several_peers = sorted(
        a for a, ps in peers_of.items() if len(ps) > 1
    )
    report.explicit_peers_renamed = sorted(
        a
        for a, ps in peers_of.items()
        if a.startswith("explicit:") and ps != {a.removeprefix("explicit:")}
    )
    accounts_of: dict[str, list[AccountKey]] = defaultdict(list)
    for account, ps in peers_of.items():
        for peer in ps:
            accounts_of[peer].append(account)
    report.peers_shared_by_accounts = {
        peer: sorted(accounts)
        for peer, accounts in accounts_of.items()
        if len(accounts) > 1
    }

    # Ingest's mapping as a prediction: each account is the peer it got, and
    # each record is grouped by its session.
    prediction = Prediction(
        person_of={a: min(ps) for a, ps in peers_of.items() if ps},
        groups_of={k: frozenset({s}) for k, s in session_of.items()},
    )
    report.unorganized_identity_recall = score(dataset, prediction).identity.recall
    return report, prediction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send a benchmark dataset through the ingest API."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--data", type=Path, help="a dataset written to DIR")
    parser.add_argument("--split", choices=get_args(Split), default="all")
    parser.add_argument("--derive", choices=("later", "now"), default="later")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument("--report", type=Path, help="write the report as JSON")
    args = parser.parse_args(argv)

    base_url: str = args.base_url
    host = urlparse(base_url).hostname or ""
    if host not in _LOCAL and not args.allow_remote:
        parser.error(f"{host} isn't local; pass --allow-remote to replay into it")
    if args.derive == "now":
        print("derive=now derives everything: it calls the configured models.")

    dataset = Dataset.read(args.data) if args.data else fixtures.generate()
    dataset = select(dataset, args.split)
    headers: dict[str, str] = {}
    if key := os.environ.get("HONCHO_BENCH_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    try:
        check_supported(dataset)
    except UnsupportedDataset as error:
        parser.error(f"can't check this dataset's mapping: {error}")
    with httpx.Client(base_url=base_url, headers=headers) as client:
        report, _ = replay(client, dataset, args.workspace, derive=args.derive)
    text = json.dumps(asdict(report), indent=1)
    if args.report:
        args.report.write_text(text)
    print(text)
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
