"""Replaying the fixtures through the real ingest route, and what a replay refuses."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src import schemas
from src.config import settings
from src.crud.ingest import BindingKey, author_key_for, peer_key_for
from tests.bench.auto_workspaces import fixtures
from tests.bench.auto_workspaces.model import BenchRecord, Dataset, record_accounts
from tests.bench.auto_workspaces.replay import (
    UnsupportedDataset,
    check_supported,
    replay,
)


@pytest.fixture(autouse=True)
def _no_embedding(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    # Ingest embeds new messages right away; nothing here needs vectors.
    monkeypatch.setattr(settings, "EMBED_MESSAGES", False)


def _workspace() -> str:
    return f"bench-{generate_nanoid(size=8).replace('_', 'a')}"


def test_every_fixture_ingests_and_maps_one_peer_per_account(
    client: TestClient,
) -> None:
    dataset = fixtures.generate(instances=1)

    report, prediction = replay(client, dataset, _workspace())

    creates = sum(1 for r in dataset.records if r.op == "create")
    edits = sum(1 for r in dataset.records if r.op == "edit")
    assert report.outcomes == {"created": creates, "rejected": edits}
    assert report.rejections == {"edit_not_supported": edits}
    assert report.deletes_not_sent > 0
    assert report.accounts_with_several_peers == []
    assert report.peers_shared_by_accounts == {}
    # Authors and participants alike.
    accounts = {
        a
        for r in dataset.records
        if r.op == "create"
        for a in record_accounts(r.record)
    }
    assert set(prediction.person_of) == accounts
    assert report.peers == len(accounts)
    assert report.accounts_without_a_peer == []
    assert not report.failed()
    assert report.deferred_tokens > 0
    # Ingest links nobody, so it splits every multi-account person.
    assert report.unorganized_identity_recall < 1


def test_replaying_again_creates_nothing(client: TestClient) -> None:
    dataset = fixtures.generate(instances=1).subset(["cross_source#0"])
    workspace = _workspace()
    replay(client, dataset, workspace)

    again, _ = replay(client, dataset, workspace)
    assert set(again.outcomes) == {"duplicate"}


# ---------------------------------------------------------------------------
# What the replay refuses, against a fake server
# ---------------------------------------------------------------------------


def _session_for(record: dict[str, Any]) -> str:
    container: dict[str, Any] = record.get("container") or {}
    return str(container.get("id") or record["source_id"])


def _fake_server(
    *,
    reject: str | None = None,
    rename_explicit: bool = False,
    null_peer: bool = False,
    members: Callable[[str], list[str]] = lambda _session: [],
) -> httpx.Client:
    """An ingest API that maps each account to its own peer, unless told not to.

    Each container (or standalone record) is its own session; `members` says
    who a session's members are.
    """

    def handle(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v3/workspaces":
            # Ingest writes only to auto workspaces.
            auto = json.loads(request.content).get("auto") is True
            return httpx.Response(200 if auto else 409, json={})
        if path.endswith("/peers"):
            session = path.split("/sessions/")[1].removesuffix("/peers")
            names = members(session)
            return httpx.Response(
                200, json={"items": [{"id": m} for m in names], "pages": 1}
            )
        body: dict[str, Any] = json.loads(request.content)
        results: list[dict[str, Any]] = []
        for index, record in enumerate(body["records"]):
            if record.get("revised_at"):
                results.append(
                    {
                        "index": index,
                        "outcome": "rejected",
                        "reason": "edit_not_supported",
                    }
                )
                continue
            if reject:
                results.append(
                    {"index": index, "outcome": "rejected", "reason": reject}
                )
                continue
            if record.get("peer_id"):
                peer = "wrong-name" if rename_explicit else record["peer_id"]
            else:
                author: dict[str, Any] = record.get("author") or {}
                peer = f"p-{author.get('id') or author.get('name') or 'source'}"
            results.append(
                {
                    "index": index,
                    "outcome": "created",
                    "record_id": record["source_id"],
                    "message_id": record["source_id"],
                    "session_id": _session_for(record),
                    "peer_id": None if null_peer else peer,
                    "derivation": "deferred",
                }
            )
        return httpx.Response(200, json={"records": results})

    return httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handle)
    )


def _all_candidates(dataset: Dataset) -> dict[str, list[str]]:
    """Per session, the names ingest gives its participants (the short name,
    as when nothing else holds it)."""
    names: dict[str, list[str]] = {}
    for r in dataset.records:
        record = schemas.IngestRecordCreate.model_validate(r.record)
        for person in record.participants:
            names.setdefault(_session_for(r.record), []).extend(
                peer_key_for(record, person).candidates()[:1]
            )
    return names


def test_a_replay_where_every_create_is_rejected_fails() -> None:
    dataset = fixtures.generate(instances=1).subset(["cross_source#0"])
    report, _ = replay(_fake_server(reject="session_inactive"), dataset, "w")
    assert report.failed()
    creates = sum(1 for r in dataset.records if r.op == "create")
    assert len(report.unexpected_outcomes) == creates
    assert report.deferred_tokens == 0


def test_an_explicit_peer_mapped_to_another_name_fails() -> None:
    dataset = fixtures.generate(instances=1).subset(["shared_handle#0"])
    report, _ = replay(_fake_server(rename_explicit=True), dataset, "w")
    assert report.failed()
    assert len(report.explicit_peers_renamed) == 2
    assert "wrong-name" in report.peers_shared_by_accounts


def test_participants_without_a_peer_fail() -> None:
    dataset = fixtures.generate(instances=1).subset(["no_container#0"])
    report, _ = replay(_fake_server(), dataset, "w")
    assert report.failed()
    participants = {a for r in dataset.records for a in record_accounts(r.record)[1:]}
    assert participants and participants <= set(report.accounts_without_a_peer)


def test_participants_found_in_every_session_pass() -> None:
    dataset = fixtures.generate(instances=1).subset(["no_container#0"])
    names = _all_candidates(dataset)
    report, _ = replay(_fake_server(members=lambda s: names.get(s, [])), dataset, "w")
    assert not report.failed(), report


def test_a_participant_missing_from_one_session_fails() -> None:
    """Being a member of one meeting's session doesn't cover another's."""
    dataset = fixtures.generate(instances=1).subset(["no_container#0"])
    names = _all_candidates(dataset)
    first = sorted(names)[0]
    report, _ = replay(
        _fake_server(members=lambda s: names[s] if s == first else []), dataset, "w"
    )
    assert report.failed()
    assert any(" in session " in a for a in report.accounts_without_a_peer)


def test_a_null_peer_id_fails() -> None:
    dataset = fixtures.generate(instances=1).subset(["edits_deletes#0"])
    report, _ = replay(_fake_server(null_peer=True), dataset, "w")
    assert report.failed()
    assert all("without peer_id" in o for o in report.unexpected_outcomes)


def test_a_participant_who_renames_keeps_their_first_peer() -> None:
    """Ingest names a peer after the first display name it saw, and keeps it."""

    def record(n: int, name: str) -> BenchRecord:
        return BenchRecord(
            key=str(n),
            case="c",
            unit="u",
            record={
                "source": "s",
                "source_instance": "i",
                "source_id": str(n),
                "created_at": f"2026-11-0{n}T10:00:00Z",
                "content": "notes",
                "container": {"id": "room"},
                "participants": [{"id": "p1", "name": name}],
            },
        )

    first, renamed = record(1, "Alice"), record(2, "Alicia")
    dataset = Dataset(records=[first, renamed])
    for r in dataset.records:
        for account in record_accounts(r.record):
            dataset.truth.person_of[account] = "x"
    validated = schemas.IngestRecordCreate.model_validate(first.record)
    kept = peer_key_for(validated, validated.participants[0]).candidates()[0]
    report, _ = replay(_fake_server(members=lambda _s: [kept]), dataset, "w")
    assert report.accounts_without_a_peer == []


def test_a_participant_who_renames_across_sessions_keeps_their_first_peer() -> None:
    """Alice in one room, then Alicia in another: one person, one kept name."""

    def record(n: int, name: str, room: str) -> BenchRecord:
        return BenchRecord(
            key=str(n),
            case="c",
            unit="u",
            record={
                "source": "s",
                "source_instance": "i",
                "source_id": str(n),
                "created_at": f"2026-11-0{n}T10:00:00Z",
                "content": "notes",
                "container": {"id": room},
                "participants": [{"id": "p1", "name": name}],
            },
        )

    first = record(1, "Alice", "room-1")
    dataset = Dataset(records=[first, record(2, "Alicia", "room-2")])
    for r in dataset.records:
        for account in record_accounts(r.record):
            dataset.truth.person_of[account] = "x"
    validated = schemas.IngestRecordCreate.model_validate(first.record)
    kept = peer_key_for(validated, validated.participants[0]).candidates()[0]
    report, _ = replay(_fake_server(members=lambda _s: [kept]), dataset, "w")
    assert report.accounts_without_a_peer == []
    assert not report.failed()


def test_a_name_ingest_sanitizes_is_keyed_as_ingest_keys_it() -> None:
    raw = {
        "source": "s",
        "source_instance": "i",
        "source_id": "1",
        "created_at": "2026-11-01T10:00:00Z",
        "content": "notes",
        "container": {"id": "room"},
        "participants": [{"name": "Ali\u0000ce"}],
    }
    clean = {**raw, "participants": [{"name": "Alice"}]}
    assert record_accounts(raw) == record_accounts(clean)
    dataset = Dataset(records=[BenchRecord(key="1", case="c", unit="u", record=raw)])
    for account in record_accounts(raw):
        dataset.truth.person_of[account] = "x"
    validated = schemas.IngestRecordCreate.model_validate(raw)
    kept = peer_key_for(validated, validated.participants[0]).candidates()[0]
    report, _ = replay(_fake_server(members=lambda _s: [kept]), dataset, "w")
    assert not report.failed(), report


def _bench(n: int, **record: Any) -> BenchRecord:
    base: dict[str, Any] = {
        "source": "s",
        "source_instance": "i",
        "source_id": "1",
        "created_at": f"2026-11-0{n}T10:00:00Z",
        "content": "notes",
        "container": {"id": "room"},
    }
    return BenchRecord(key=str(n), case="c", unit="u", record={**base, **record})


def test_a_source_record_created_twice_is_refused_before_sending() -> None:
    dataset = Dataset(
        records=[
            _bench(1, author={"id": "a", "name": "A"}),
            _bench(2, author={"id": "b", "name": "B"}),
        ]
    )
    sent: list[str] = []
    client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(
            lambda request: sent.append(request.url.path) or httpx.Response(500)
        ),
    )
    with pytest.raises(UnsupportedDataset, match="source record again"):
        replay(client, dataset, "w")
    assert sent == []


def test_an_explicit_peer_naming_a_generated_peer_is_refused() -> None:
    first = _bench(1, participants=[{"id": "p1", "name": "Alice"}])
    validated = schemas.IngestRecordCreate.model_validate(first.record)
    short = peer_key_for(validated, validated.participants[0]).candidates()[0]
    dataset = Dataset(records=[first, _bench(2, source_id="2", peer_id=short)])
    with pytest.raises(UnsupportedDataset, match="is a name ingest generates"):
        check_supported(dataset)


def test_the_fixtures_are_supported() -> None:
    check_supported(fixtures.generate())


def test_records_without_a_source_id_are_told_apart_by_content() -> None:
    one = _bench(1, author={"id": "a", "name": "A"})
    two = _bench(2, author={"id": "a", "name": "A"}, content="other notes")
    for r in (one, two):
        del r.record["source_id"]
    check_supported(Dataset(records=[one, two]))
    same = _bench(3, author={"id": "a", "name": "A"})
    del same.record["source_id"]
    same.record["created_at"] = one.record["created_at"]
    with pytest.raises(UnsupportedDataset, match="source record again"):
        check_supported(Dataset(records=[one, same]))


def test_an_explicit_peer_naming_the_source_peer_is_refused() -> None:
    authorless = _bench(1)
    validated = schemas.IngestRecordCreate.model_validate(authorless.record)
    source_peer = author_key_for(validated)
    assert isinstance(source_peer, BindingKey)
    explicit = _bench(2, source_id="2", peer_id=source_peer.candidates()[0])
    with pytest.raises(UnsupportedDataset, match="is a name ingest generates"):
        check_supported(Dataset(records=[authorless, explicit]))


def test_an_author_replaced_by_an_explicit_peer_generates_no_name() -> None:
    record = _bench(1, author={"id": "a", "name": "Alice"})
    validated = schemas.IngestRecordCreate.model_validate(record.record)
    assert validated.author is not None
    would_be = peer_key_for(validated, validated.author).candidates()[0]
    check_supported(
        Dataset(
            records=[_bench(1, author={"id": "a", "name": "Alice"}, peer_id=would_be)]
        )
    )


def test_two_accounts_that_could_get_one_name_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which of two colliding accounts gets the short name depends on order."""
    from src.crud import ingest

    monkeypatch.setattr(ingest, "_SHORT_SUFFIX", 0)  # every Alice collides
    record = _bench(
        1,
        participants=[{"id": "p1", "name": "Alice"}, {"id": "p2", "name": "Alice"}],
    )
    with pytest.raises(UnsupportedDataset, match="could both be named"):
        check_supported(Dataset(records=[record]))
