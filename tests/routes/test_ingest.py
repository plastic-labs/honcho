"""POST /v3/workspaces/{workspace_id}/ingest: mapping, dedup, outcomes, atomicity."""

import asyncio
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from src import models, schemas
from src.crud import ingest as ingest_crud
from src.crud.ingest_fence import take_session_fences, take_workspace_fence
from src.crud.session import upsert_session_peers
from src.exceptions import ObserverException


def _workspace(client: TestClient) -> str:
    name = f"ws-{generate_nanoid(size=10).replace('_', 'a')}"
    response = client.post("/v3/workspaces", json={"id": name})
    assert response.status_code in (200, 201), response.text
    return name


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source": "discord",
        "source_instance": "guild1",
        "source_id": f"msg-{generate_nanoid(size=8)}",
        "created_at": "2026-11-04T20:47:00Z",
        "container": {"id": "chan-1", "name": "dev"},
        "author": {"id": "user-1", "name": "Jessica"},
        "content": "Northstar needs SSO before rollout.",
    }
    record.update(overrides)
    return record


def _ingest(
    client: TestClient, workspace: str, *records: dict[str, Any], **body: Any
) -> list[dict[str, Any]]:
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest", json={"records": list(records), **body}
    )
    assert response.status_code == 200, response.text
    return response.json()["records"]


async def _count(db: AsyncSession, model: Any, workspace: str) -> int:
    return int(
        (
            await db.execute(
                select(func.count())
                .select_from(model)
                .where(model.workspace_name == workspace)
            )
        ).scalar_one()
    )


def test_a_new_record_becomes_a_message_in_a_mapped_session(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    [result] = _ingest(client, workspace, _record())

    assert result["outcome"] == "created"
    assert result["derivation"] == "pending"
    assert result["session_id"].startswith("dev-")
    assert result["peer_id"].startswith("jessica-")

    messages = client.post(
        f"/v3/workspaces/{workspace}/sessions/{result['session_id']}/messages/list",
        json={},
    ).json()["items"]
    assert [(m["id"], m["peer_id"]) for m in messages] == [
        (result["message_id"], result["peer_id"])
    ]
    # The message keeps the source's timestamp.
    assert messages[0]["created_at"].startswith("2026-11-04T20:47:00")


def test_resending_a_batch_is_a_duplicate_with_the_same_message(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    record = _record()
    [first] = _ingest(client, workspace, record)
    [again] = _ingest(client, workspace, record)

    assert again["outcome"] == "duplicate"
    for key in ("record_id", "message_id", "session_id", "peer_id"):
        assert again[key] == first[key]
    messages = client.post(
        f"/v3/workspaces/{workspace}/sessions/{first['session_id']}/messages/list",
        json={},
    ).json()["items"]
    assert len(messages) == 1


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"revised_at": "2026-11-05T00:00:00Z", "content": "edited"},
            "edit_not_supported",
        ),
        ({"content": "same revision, different words"}, "content_conflict"),
    ],
)
def test_a_changed_record_is_rejected_with_a_reason(
    client: TestClient, changes: dict[str, Any], reason: str
) -> None:
    workspace = _workspace(client)
    record = _record()
    _ingest(client, workspace, record)
    [result] = _ingest(client, workspace, {**record, **changes})
    assert result["outcome"] == "rejected"
    assert result["reason"] == reason


def test_an_older_revision_is_stale(client: TestClient) -> None:
    workspace = _workspace(client)
    record = _record(revised_at="2026-11-06T00:00:00Z")
    _ingest(client, workspace, record)
    [result] = _ingest(
        client, workspace, {**record, "revised_at": "2026-11-05T00:00:00Z"}
    )
    assert result == {**result, "outcome": "rejected", "reason": "stale_revision"}


def test_a_repeated_record_inside_one_batch_is_a_duplicate_of_the_first(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    record = _record()
    first, repeat = _ingest(client, workspace, record, record)
    assert first["outcome"] == "created"
    assert repeat["outcome"] == "duplicate"
    assert repeat["message_id"] == first["message_id"]
    assert repeat["index"] == 1


def test_records_without_a_source_id_dedup_on_their_content(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    record = _record(source_id=None)
    [first] = _ingest(client, workspace, record)
    [again] = _ingest(client, workspace, record)
    [other] = _ingest(client, workspace, {**record, "content": "different words"})
    assert (first["outcome"], again["outcome"], other["outcome"]) == (
        "created",
        "duplicate",
        "created",
    )
    assert again["message_id"] == first["message_id"]


def test_renamed_containers_and_people_keep_their_session_and_peer(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    [first] = _ingest(client, workspace, _record())
    [renamed] = _ingest(
        client,
        workspace,
        _record(
            container={"id": "chan-1", "name": "dev-renamed"},
            author={"id": "user-1", "name": "Jess D"},
        ),
    )
    assert renamed["session_id"] == first["session_id"]
    assert renamed["peer_id"] == first["peer_id"]


def test_people_without_ids_are_never_merged_across_containers(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    a, b, a_again = _ingest(
        client,
        workspace,
        _record(container={"id": "c-a"}, author={"name": "Alex"}),
        _record(container={"id": "c-b"}, author={"name": "Alex"}),
        _record(container={"id": "c-a"}, author={"name": "alex"}),
    )
    assert a["peer_id"] != b["peer_id"]
    assert a_again["peer_id"] == a["peer_id"]


def test_standalone_and_authorless_records(client: TestClient) -> None:
    workspace = _workspace(client)
    note, other_note = _ingest(
        client,
        workspace,
        _record(
            container=None, author=None, participants=[{"id": "u2", "name": "Maya"}]
        ),
        _record(container=None, author=None),
    )
    # Each standalone record gets its own session; author-less records share the
    # instance's source peer.
    assert note["session_id"] != other_note["session_id"]
    assert note["peer_id"] == other_note["peer_id"]
    assert note["peer_id"].startswith("discord-source-")


def test_explicit_session_and_peer_bypass_mapping(client: TestClient) -> None:
    workspace = _workspace(client)
    [result] = _ingest(
        client, workspace, _record(session_id="my-session", peer_id="my-peer")
    )
    assert (result["session_id"], result["peer_id"]) == ("my-session", "my-peer")


def test_a_generated_name_taken_by_an_explicit_session_is_never_adopted(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    # Learn the generated name, then occupy it in a fresh workspace.
    [probe] = _ingest(client, workspace, _record())
    taken = probe["session_id"]
    other = _workspace(client)
    response = client.post(f"/v3/workspaces/{other}/sessions", json={"id": taken})
    assert response.status_code in (200, 201), response.text

    [result] = _ingest(client, other, _record())
    assert result["outcome"] == "created"
    assert result["session_id"] != taken
    assert result["session_id"].startswith(taken)  # same slug and digest, longer


def test_derive_later_defers_the_records(client: TestClient) -> None:
    workspace = _workspace(client)
    [result] = _ingest(client, workspace, _record(), derive="later")
    assert (result["outcome"], result["derivation"]) == ("created", "deferred")


def test_an_unknown_workspace_is_404(client: TestClient) -> None:
    response = client.post(
        "/v3/workspaces/no-such-workspace/ingest", json={"records": [_record()]}
    )
    assert response.status_code == 404


def test_more_than_100_records_is_422(client: TestClient) -> None:
    workspace = _workspace(client)
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"records": [_record() for _ in range(101)]},
    )
    assert response.status_code == 422


def test_a_refused_session_rejects_only_its_own_records(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late refusal rolls back that session's group and keeps the others."""
    workspace = _workspace(client)
    real_upsert = upsert_session_peers

    async def upsert(
        db: Any, ws: str, session_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        if session_name.startswith("crowded-"):
            raise ObserverException(session_name, 99)
        return await real_upsert(db, ws, session_name, *args, **kwargs)

    monkeypatch.setattr("src.crud.ingest.upsert_session_peers", upsert)
    refused_record = _record(container={"id": "x", "name": "crowded"})
    crowded, quiet = _ingest(
        client,
        workspace,
        refused_record,
        _record(container={"id": "y", "name": "quiet"}),
    )
    assert (crowded["outcome"], crowded["reason"]) == ("rejected", "observers_limit")
    assert quiet["outcome"] == "created"
    # The session created for the refused group was undone with it.
    sessions = client.post(f"/v3/workspaces/{workspace}/sessions/list", json={})
    names = [item["id"] for item in sessions.json()["items"]]
    assert not [n for n in names if n.startswith("crowded-")]
    # The refused group's claim and session binding rolled back with it, so the
    # record can be ingested once the session accepts it.
    monkeypatch.setattr("src.crud.ingest.upsert_session_peers", real_upsert)
    [retried] = _ingest(client, workspace, refused_record)
    assert retried["outcome"] == "created"


@pytest.mark.asyncio
async def test_a_failure_mid_transaction_writes_nothing(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = sample_data[0].name

    async def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("fail after the claims and bindings")

    monkeypatch.setattr("src.crud.ingest.insert_messages_locked", boom)
    batch = schemas.IngestBatch.model_validate({"records": [_record()]})
    with pytest.raises(RuntimeError):
        await ingest_crud.ingest_batch(db_session, workspace, batch)
    await db_session.rollback()

    for model in (
        models.IngestRecord,
        models.IngestSessionBinding,
        models.IngestPeerBinding,
        models.Message,
        models.Session,
    ):
        assert await _count(db_session, model, workspace) == 0


@pytest.mark.asyncio
async def test_concurrent_overlapping_batches_create_each_record_once(
    db_engine: AsyncEngine, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    workspace = sample_data[0].name
    shared = [_record(container={"id": f"c{i % 3}"}) for i in range(6)]
    batches = [
        schemas.IngestBatch.model_validate({"records": shared}),
        schemas.IngestBatch.model_validate({"records": list(reversed(shared))}),
    ]
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)

    async def run(batch: schemas.IngestBatch) -> list[schemas.IngestRecordResult]:
        async with factory() as db:
            return (await ingest_crud.ingest_batch(db, workspace, batch)).results

    first, second = await asyncio.gather(*(run(b) for b in batches))
    outcomes = [r.outcome for r in first + second]
    assert outcomes.count("created") == len(shared)
    assert outcomes.count("duplicate") == len(shared)
    async with factory() as db:
        assert await _count(db, models.Message, workspace) == len(shared)
        assert await _count(db, models.IngestRecord, workspace) == len(shared)


def test_a_rename_inside_one_batch_keeps_one_session(client: TestClient) -> None:
    """The first record to name a container sets its session's name."""
    workspace = _workspace(client)
    zulu, alpha = _ingest(
        client,
        workspace,
        _record(container={"id": "chan-9", "name": "zulu"}),
        _record(container={"id": "chan-9", "name": "alpha"}),
    )
    assert (zulu["outcome"], alpha["outcome"]) == ("created", "created")
    assert zulu["session_id"] == alpha["session_id"]
    assert zulu["session_id"].startswith("zulu-")


def test_the_same_instant_in_another_offset_is_a_duplicate(
    client: TestClient,
) -> None:
    workspace = _workspace(client)
    [first] = _ingest(
        client, workspace, _record(source_id=None, created_at="2026-11-04T20:47:00Z")
    )
    [again] = _ingest(
        client,
        workspace,
        _record(source_id=None, created_at="2026-11-04T15:47:00-05:00"),
    )
    assert again["outcome"] == "duplicate"
    assert again["message_id"] == first["message_id"]


def test_provisional_people_cannot_be_spelled_across_containers() -> None:
    """Container "a:b" with person "c" is not container "a" with person "b:c"."""

    def key(container: str, name: str) -> ingest_crud.BindingKey:
        record = schemas.IngestRecordCreate.model_validate(
            _record(container={"id": container}, author={"name": name})
        )
        assert record.author is not None
        return ingest_crud.peer_key_for(record, record.author)

    assert key("a:b", "c") != key("a", "b:c")


def test_a_repeat_matching_the_stored_record_is_a_duplicate(
    client: TestClient,
) -> None:
    """A repeat is classified against the stored row its first copy was."""
    workspace = _workspace(client)
    record = _record(revised_at="2026-11-06T00:00:00Z")
    [stored] = _ingest(client, workspace, record)
    older, same = _ingest(
        client, workspace, {**record, "revised_at": "2026-11-05T00:00:00Z"}, record
    )
    assert (older["outcome"], older["reason"]) == ("rejected", "stale_revision")
    assert same["outcome"] == "duplicate"
    assert same["message_id"] == stored["message_id"]


@pytest.mark.parametrize(
    "peer_id",
    ["no spaces allowed", "trailing-newline\n", "scope.not-yet-a-scope"],
    ids=["invalid", "newline", "scope"],
)
def test_an_explicit_peer_ingest_cannot_use_refuses_the_request(
    client: TestClient, peer_id: str
) -> None:
    workspace = _workspace(client)
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest",
        json={"records": [_record(), _record(peer_id=peer_id)]},
    )
    assert response.status_code == 422, response.text
    assert peer_id.strip() in response.text


@pytest.mark.asyncio
async def test_stale_bindings_are_replaced(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """A binding whose session or peer is gone maps afresh, as a first sighting."""
    workspace = sample_data[0].name

    async def ingest(record: dict[str, Any]) -> schemas.IngestRecordResult:
        batch = schemas.IngestBatch.model_validate({"records": [record]})
        [result] = (
            await ingest_crud.ingest_batch(db_session, workspace, batch)
        ).results
        return result

    first = await ingest(_record())
    assert first.outcome == "created"
    # Point both bindings at resources that don't exist.
    for model, column in (
        (models.IngestSessionBinding, "session_name"),
        (models.IngestPeerBinding, "peer_name"),
    ):
        await db_session.execute(
            update(model)
            .where(model.workspace_name == workspace)
            .values({column: f"gone-{column}"})
        )
    await db_session.commit()

    second = await ingest(_record(content="after the bindings went stale"))
    assert second.outcome == "created"
    assert second.session_id == first.session_id
    assert second.peer_id == first.peer_id
    sessions = await db_session.execute(
        select(models.IngestSessionBinding.session_name).where(
            models.IngestSessionBinding.workspace_name == workspace
        )
    )
    assert sessions.scalars().all() == [first.session_id]


@pytest.mark.asyncio
async def test_the_fence_covers_the_bound_session_after_a_rename(
    db_engine: AsyncEngine, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """A renamed container's records wait on the session they really write."""
    workspace = sample_data[0].name
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as db:
        batch = schemas.IngestBatch.model_validate({"records": [_record()]})
        [first] = (await ingest_crud.ingest_batch(db, workspace, batch)).results
    assert first.session_id is not None

    async with factory() as holder, factory() as db:
        # Session deletion, mid-flight on the bound session.
        await take_workspace_fence(holder, workspace, mode="shared")
        await take_session_fences(holder, workspace, [first.session_id])

        await db.execute(text("SET LOCAL lock_timeout = '200ms'"))
        renamed = schemas.IngestBatch.model_validate(
            {"records": [_record(container={"id": "chan-1", "name": "renamed"})]}
        )
        with pytest.raises(DBAPIError, match="lock timeout"):
            await ingest_crud.ingest_batch(db, workspace, renamed)
        await db.rollback()
        await holder.rollback()


def test_pending_dreams_of_the_authors_are_cancelled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancelled: list[tuple[str, str]] = []

    class Scheduler:
        async def cancel_dreams_for_observed(
            self, workspace_name: str, observed: str
        ) -> set[str]:
            cancelled.append((workspace_name, observed))
            return set()

    monkeypatch.setattr("src.routers.ingest.get_dream_scheduler", Scheduler)
    workspace = _workspace(client)
    [result] = _ingest(client, workspace, _record())
    assert cancelled == [(workspace, result["peer_id"])]


def test_a_raced_batch_is_retried_and_then_refused_with_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(client)
    real = ingest_crud.ingest_batch
    calls = 0

    async def race_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ingest_crud.IngestRaced
        return await real(*args, **kwargs)

    monkeypatch.setattr("src.routers.ingest.ingest_batch", race_once)
    [result] = _ingest(client, workspace, _record())
    assert (result["outcome"], calls) == ("created", 2)

    async def always(*_args: Any, **_kwargs: Any) -> Any:
        raise ingest_crud.IngestRaced

    monkeypatch.setattr("src.routers.ingest.ingest_batch", always)
    response = client.post(
        f"/v3/workspaces/{workspace}/ingest", json={"records": [_record()]}
    )
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"


@pytest.mark.asyncio
async def test_batches_recovering_one_stale_peer_binding_agree_on_one_peer(
    db_engine: AsyncEngine, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """A batch waits for one already recovering the binding, then reuses its peer."""
    workspace = sample_data[0].name
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    author = {"id": "user-7", "name": "Alice"}
    async with factory() as db:
        batch = schemas.IngestBatch.model_validate(
            {"records": [_record(author=author)]}
        )
        [first] = (await ingest_crud.ingest_batch(db, workspace, batch)).results
        binding = (
            await db.execute(
                select(models.IngestPeerBinding).where(
                    models.IngestPeerBinding.workspace_name == workspace,
                    models.IngestPeerBinding.key_value == "user-7",
                )
            )
        ).scalar_one()
        binding_id, key_kind = binding.id, binding.key_kind
        binding.peer_name = "gone-peer"
        await db.commit()
    assert first.peer_id is not None

    async with factory() as recovering, factory() as db:
        # Another batch is recovering the stale binding: it holds the binding row.
        await recovering.execute(
            select(models.IngestPeerBinding)
            .where(models.IngestPeerBinding.id == binding_id)
            .with_for_update()
        )
        recovered = "alice-recovered"
        await recovering.execute(
            update(models.IngestPeerBinding)
            .where(models.IngestPeerBinding.id == binding_id)
            .values(peer_name=recovered)
        )
        recovering.add(
            models.Peer(
                workspace_name=workspace,
                name=recovered,
                internal_metadata={
                    "ingest": {"binding_id": binding_id, "key_kind": key_kind}
                },
            )
        )
        await recovering.flush()

        renamed = schemas.IngestBatch.model_validate(
            {
                "records": [
                    _record(
                        container={"id": "chan-2", "name": "other"},
                        author={"id": "user-7", "name": "Alicia"},
                    )
                ]
            }
        )
        waiting = asyncio.create_task(ingest_crud.ingest_batch(db, workspace, renamed))
        await asyncio.sleep(0.3)
        assert not waiting.done()  # blocked on the binding row
        await recovering.commit()
        [result] = (await waiting).results

    assert result.outcome == "created"
    assert result.peer_id == recovered


@pytest.mark.asyncio
async def test_bindings_keep_the_newest_display_name(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """A rename updates the display name; an older record arriving late doesn't."""
    workspace = sample_data[0].name

    async def ingest(**over: Any) -> None:
        batch = schemas.IngestBatch.model_validate({"records": [_record(**over)]})
        await ingest_crud.ingest_batch(db_session, workspace, batch)

    await ingest(source_id="a", created_at="2026-11-04T10:00:00Z")
    await ingest(
        source_id="b",
        created_at="2026-11-05T10:00:00Z",
        container={"id": "chan-1", "name": "dev-renamed"},
        author={"id": "user-1", "name": "Jess D"},
    )
    await ingest(
        source_id="c",
        created_at="2026-11-03T10:00:00Z",
        container={"id": "chan-1", "name": "old-name"},
        author={"id": "user-1", "name": "J"},
    )

    session_binding = (
        await db_session.execute(
            select(
                models.IngestSessionBinding.display_name,
                models.IngestSessionBinding.display_name_at,
            ).where(models.IngestSessionBinding.workspace_name == workspace)
        )
    ).one()
    peer_binding = (
        await db_session.execute(
            select(models.IngestPeerBinding.display_name).where(
                models.IngestPeerBinding.workspace_name == workspace,
                models.IngestPeerBinding.key_value == "user-1",
            )
        )
    ).scalar_one()
    assert session_binding[0] == "dev-renamed"
    assert session_binding[1] is not None
    assert session_binding[1].isoformat().startswith("2026-11-05T10:00:00")
    assert peer_binding == "Jess D"


@pytest.mark.asyncio
async def test_records_keep_their_author_and_participant_bindings(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """Which account wrote a record survives linking people, so a link can split."""
    workspace = sample_data[0].name
    batch = schemas.IngestBatch.model_validate(
        {
            "records": [
                _record(
                    source_id="with-people",
                    participants=[{"id": "u2", "name": "Maya"}, {"name": "Alex"}],
                ),
                _record(source_id="explicit", peer_id="explicit-author"),
            ]
        }
    )
    await ingest_crud.ingest_batch(db_session, workspace, batch)

    bindings = dict(
        (
            await db_session.execute(
                select(
                    models.IngestPeerBinding.key_value, models.IngestPeerBinding.id
                ).where(models.IngestPeerBinding.workspace_name == workspace)
            )
        )
        .tuples()
        .all()
    )
    records = {
        row.identity: row
        for row in (
            await db_session.execute(
                select(models.IngestRecord).where(
                    models.IngestRecord.workspace_name == workspace
                )
            )
        ).scalars()
    }
    with_people = records["with-people"]
    assert with_people.author_binding_id == bindings["user-1"]
    assert with_people.participant_binding_ids == [
        bindings["u2"],
        bindings["container:6:chan-1:alex"],
    ]
    # An explicit peer has no binding.
    assert records["explicit"].author_binding_id is None
    assert records["explicit"].participant_binding_ids is None


@pytest.mark.asyncio
async def test_ingest_follows_bindings_a_person_confirmed(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """A confirmed link or move is used as is; an unconfirmed one is stale."""
    workspace = sample_data[0].name

    async def ingest(**over: Any) -> schemas.IngestRecordResult:
        batch = schemas.IngestBatch.model_validate({"records": [_record(**over)]})
        [result] = (
            await ingest_crud.ingest_batch(db_session, workspace, batch)
        ).results
        return result

    jess = await ingest(source_id="1")
    other_account = await ingest(
        source_id="2",
        container={"id": "chan-2", "name": "other"},
        author={"id": "slack-jess", "name": "Jess D"},
    )
    assert jess.peer_id != other_account.peer_id
    await db_session.execute(
        update(models.IngestSessionBinding)
        .where(
            models.IngestSessionBinding.workspace_name == workspace,
            models.IngestSessionBinding.key_value == "chan-2",
        )
        .values(session_name="explicit-target", confirmed_at=func.now())
    )
    db_session.add(models.Session(workspace_name=workspace, name="explicit-target"))
    await db_session.commit()

    # Unconfirmed: a binding pointing at someone else's peer is stale and maps back.
    await db_session.execute(
        update(models.IngestPeerBinding)
        .where(
            models.IngestPeerBinding.workspace_name == workspace,
            models.IngestPeerBinding.key_value == "slack-jess",
        )
        .values(peer_name=jess.peer_id)
    )
    await db_session.commit()
    unconfirmed = await ingest(
        source_id="3",
        container={"id": "chan-2"},
        author={"id": "slack-jess", "name": "Jess D"},
    )
    assert unconfirmed.peer_id == other_account.peer_id

    # Confirmed: the link is followed.
    await db_session.execute(
        update(models.IngestPeerBinding)
        .where(
            models.IngestPeerBinding.workspace_name == workspace,
            models.IngestPeerBinding.key_value == "slack-jess",
        )
        .values(peer_name=jess.peer_id, confirmed_at=func.now())
    )
    await db_session.commit()
    linked = await ingest(
        source_id="4",
        container={"id": "chan-2"},
        author={"id": "slack-jess", "name": "Jess D"},
    )
    assert linked.outcome == "created"
    assert linked.peer_id == jess.peer_id
    # A channel a person moved into an explicit session lands there.
    assert linked.session_id == "explicit-target"


@pytest.mark.asyncio
async def test_one_confirmed_binding_does_not_vouch_for_another(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """A channel never lands in an explicit session just because another
    channel a person moved there shares the batch."""
    workspace = sample_data[0].name

    async def ingest(*records: dict[str, Any]) -> list[schemas.IngestRecordResult]:
        batch = schemas.IngestBatch.model_validate({"records": list(records)})
        return (await ingest_crud.ingest_batch(db_session, workspace, batch)).results

    await ingest(_record(source_id="a0", container={"id": "chan-a", "name": "a"}))
    # Explicit sessions hold both of channel B's generated names.
    b_short, b_long = ingest_crud.BindingKey(
        "discord", "guild1", "container", "chan-b", "b"
    ).candidates()
    for name in (b_short, b_long):
        db_session.add(models.Session(workspace_name=workspace, name=name))
    # A person moved channel A into the session holding B's long name.
    await db_session.execute(
        update(models.IngestSessionBinding)
        .where(
            models.IngestSessionBinding.workspace_name == workspace,
            models.IngestSessionBinding.key_value == "chan-a",
        )
        .values(session_name=b_long, confirmed_at=func.now())
    )
    await db_session.commit()

    a, b = await ingest(
        _record(source_id="a1", container={"id": "chan-a", "name": "a"}),
        _record(source_id="b1", container={"id": "chan-b", "name": "b"}),
    )
    assert (a.outcome, a.session_id) == ("created", b_long)
    assert (b.outcome, b.reason) == ("rejected", "name_conflict")
    bindings = (
        await db_session.execute(
            select(models.IngestSessionBinding.key_value).where(
                models.IngestSessionBinding.workspace_name == workspace
            )
        )
    ).scalars()
    assert sorted(bindings) == ["chan-a"]


def test_containers_whose_short_names_collide_keep_separate_sessions(
    client: TestClient,
) -> None:
    """Two channels whose 10-hex digests collide, first seen in one batch."""
    workspace = _workspace(client)
    one, two = _ingest(
        client,
        workspace,
        _record(source_id="1", container={"id": "chan-592032", "name": "dev"}),
        _record(source_id="2", container={"id": "chan-1343149", "name": "dev"}),
    )
    assert (one["outcome"], two["outcome"]) == ("created", "created")
    assert one["session_id"] != two["session_id"]
    # Later records keep going to the session each channel was given.
    later_one, later_two = _ingest(
        client,
        workspace,
        _record(source_id="3", container={"id": "chan-592032", "name": "dev"}),
        _record(source_id="4", container={"id": "chan-1343149", "name": "dev"}),
    )
    assert later_one["session_id"] == one["session_id"]
    assert later_two["session_id"] == two["session_id"]


@pytest.mark.asyncio
async def test_a_confirmed_shared_session_is_not_separated(
    db_session: AsyncSession, sample_data: tuple[models.Workspace, models.Peer]
) -> None:
    """Colliding channels a person put in one session both keep going there."""
    workspace = sample_data[0].name

    async def ingest(*records: dict[str, Any]) -> list[schemas.IngestRecordResult]:
        batch = schemas.IngestBatch.model_validate({"records": list(records)})
        return (await ingest_crud.ingest_batch(db_session, workspace, batch)).results

    one, two = await ingest(
        _record(source_id="1", container={"id": "chan-592032", "name": "dev"}),
        _record(source_id="2", container={"id": "chan-1343149", "name": "dev"}),
    )
    assert two.session_id == "dev-4613682f95"
    await db_session.execute(
        update(models.IngestSessionBinding)
        .where(
            models.IngestSessionBinding.workspace_name == workspace,
            models.IngestSessionBinding.key_value == "chan-592032",
        )
        .values(session_name=two.session_id, confirmed_at=func.now())
    )
    await db_session.commit()

    later_one, later_two = await ingest(
        _record(source_id="3", container={"id": "chan-592032", "name": "dev"}),
        _record(source_id="4", container={"id": "chan-1343149", "name": "dev"}),
    )
    assert (later_one.outcome, later_two.outcome) == ("created", "created")
    assert later_one.session_id == later_two.session_id == two.session_id
    assert one.session_id != two.session_id
