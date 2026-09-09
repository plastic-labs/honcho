"""Tests for the read-only count of collections whose next dream is due."""

import datetime
from unittest.mock import patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.dreamer.dream_due import count_due_dreams
from src.schemas import DreamType
from src.utils.work_unit import construct_work_unit_key


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def test_observed_key_includes_tenant_only_under_multi_tenant(
    monkeypatch: pytest.MonkeyPatch,
):
    # The API scheduler runs cross-tenant, so the "is representation work still
    # pending?" check must be keyed per tenant — otherwise one tenant's pending work
    # suppresses another tenant's due dream (both share names like default/observed).
    from src.dreamer.dream_due import _observed_key  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(settings, "MULTI_TENANT", True)
    assert _observed_key("tenant-a", "ws", "obs") == ("tenant-a", "ws", "obs")
    # Same (workspace, observed), different tenants -> distinct keys, no cross-suppress.
    assert _observed_key("tenant-a", "ws", "obs") != _observed_key("tenant-b", "ws", "obs")

    # Flag off: the tenant is dropped, so a parsed (unnamespaced -> tenant_id=None) key
    # and the collection's real tenant still line up.
    monkeypatch.setattr(settings, "MULTI_TENANT", False)
    assert _observed_key("tenant-a", "ws", "obs") == ("ws", "obs")
    assert _observed_key(None, "ws", "obs") == _observed_key("tenant-a", "ws", "obs")


async def _make_collection(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    internal_metadata: dict[str, object] | None = None,
) -> models.Collection:
    workspace, peer = sample_data
    collection = models.Collection(
        observer=peer.name,
        observed=peer.name,
        workspace_name=workspace.name,
        internal_metadata=internal_metadata or {},
    )
    db_session.add(collection)
    await db_session.commit()
    return collection


async def _make_session(
    db_session: AsyncSession,
    workspace_name: str,
    configuration: dict[str, object] | None = None,
) -> str:
    session = models.Session(
        name=f"s-{generate_nanoid()}",
        workspace_name=workspace_name,
        configuration=configuration or {},
    )
    db_session.add(session)
    await db_session.commit()
    return session.name


async def _insert_docs(
    db_session: AsyncSession,
    collection: models.Collection,
    level: str,
    count: int,
    *,
    age_minutes: int = 0,
    session_name: str | None = None,
    sessionless: bool = False,
) -> None:
    if session_name is None and not sessionless:
        session_name = await _make_session(db_session, collection.workspace_name)
    created_at = _now() - datetime.timedelta(minutes=age_minutes)
    for _ in range(count):
        db_session.add(
            models.Document(
                content="test",
                level=level,
                workspace_name=collection.workspace_name,
                observer=collection.observer,
                observed=collection.observed,
                session_name=session_name,
                created_at=created_at,
            )
        )
    await db_session.commit()


async def _insert_dream_item(
    db_session: AsyncSession,
    collection: models.Collection,
    *,
    age_minutes: int,
    processed: bool,
    error: str | None = None,
) -> None:
    work_unit_key = construct_work_unit_key(
        collection.workspace_name,
        {
            "task_type": "dream",
            "observer": collection.observer,
            "observed": collection.observed,
            "dream_type": DreamType.OMNI.value,
        },
    )
    db_session.add(
        models.QueueItem(
            work_unit_key=work_unit_key,
            payload={"task_type": "dream"},
            task_type="dream",
            workspace_name=collection.workspace_name,
            processed=processed,
            error=error,
            created_at=_now() - datetime.timedelta(minutes=age_minutes),
        )
    )
    await db_session.commit()


@pytest.fixture(autouse=True)
def _pin_dream_config():  # pyright: ignore[reportUnusedFunction]
    with (
        patch("src.dreamer.dream_due.settings.DREAM.ENABLED", True),
        patch("src.dreamer.dream_due.settings.DREAM.DOCUMENT_THRESHOLD", 50),
        patch("src.dreamer.dream_due.settings.DREAM.ENABLED_TYPES", ["omni"]),
        patch("src.dreamer.dream_due.settings.DREAM.IDLE_TIMEOUT_MINUTES", 60),
        patch("src.dreamer.dream_due.settings.DREAM.MIN_HOURS_BETWEEN_DREAMS", 8),
    ):
        yield


@pytest.mark.asyncio
class TestCountDueDreams:
    async def test_below_threshold_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 30, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_derived_levels_do_not_count(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 30, age_minutes=90)
        await _insert_docs(db_session, collection, "deductive", 40, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_threshold_met_but_not_idle_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A collection still receiving documents is not idle yet."""
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=1)

        assert await count_due_dreams(db_session) == 0

    async def test_threshold_met_and_idle_is_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 1

    async def test_pending_representation_work_blocks_a_due_dream(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        session_name = await _make_session(db_session, collection.workspace_name)
        representation_key = construct_work_unit_key(
            collection.workspace_name,
            {
                "task_type": "representation",
                "session_name": session_name,
                "observed": collection.observed,
            },
        )
        item = models.QueueItem(
            work_unit_key=representation_key,
            payload={"task_type": "representation"},
            task_type="representation",
            workspace_name=collection.workspace_name,
            processed=False,
        )
        db_session.add(item)
        await db_session.commit()

        assert await count_due_dreams(db_session) == 0

        item.processed = True
        await db_session.commit()

        assert await count_due_dreams(db_session) == 1

    async def test_documents_since_last_dream_uses_stored_count(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(
            db_session, sample_data, {"dream": {"last_dream_document_count": 40}}
        )
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_min_hours_gate_blocks_a_recent_dream(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        last_dream_at = (_now() - datetime.timedelta(hours=2)).isoformat()
        collection = await _make_collection(
            db_session, sample_data, {"dream": {"last_dream_at": last_dream_at}}
        )
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_naive_last_dream_at_is_read_as_utc(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A stored timestamp with no offset must gate, not raise."""
        naive = (_now() - datetime.timedelta(hours=2)).replace(tzinfo=None).isoformat()
        collection = await _make_collection(
            db_session, sample_data, {"dream": {"last_dream_at": naive}}
        )
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        assert await count_due_dreams(db_session) == 0

    async def test_pending_dream_item_blocks(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)
        await _insert_dream_item(
            db_session, collection, age_minutes=10, processed=False
        )

        assert await count_due_dreams(db_session) == 0

    async def test_failed_dream_waits_for_new_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """Without this the count never returns to zero."""
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)
        await _insert_dream_item(
            db_session, collection, age_minutes=80, processed=True, error="boom"
        )

        assert await count_due_dreams(db_session) == 0

    async def test_failed_dream_retries_after_new_documents(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)
        await _insert_dream_item(
            db_session, collection, age_minutes=80, processed=True, error="boom"
        )
        await _insert_docs(db_session, collection, "explicit", 1, age_minutes=70)

        assert await count_due_dreams(db_session) == 1

    async def test_sessionless_documents_are_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """The deriver's own enqueue path refuses these, so they must not count."""
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(
            db_session, collection, "explicit", 60, age_minutes=90, sessionless=True
        )

        assert await count_due_dreams(db_session) == 0

    async def test_newest_document_decides_the_session(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=120)

        assert await count_due_dreams(db_session) == 1

        await _insert_docs(
            db_session, collection, "explicit", 1, age_minutes=90, sessionless=True
        )

        assert await count_due_dreams(db_session) == 0

    async def test_session_with_dreams_disabled_is_not_due(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        """A dream the enqueue path would refuse must not be counted."""
        collection = await _make_collection(db_session, sample_data)
        session_name = await _make_session(
            db_session,
            collection.workspace_name,
            {"dream": {"enabled": False}},
        )
        await _insert_docs(
            db_session,
            collection,
            "explicit",
            60,
            age_minutes=90,
            session_name=session_name,
        )

        assert await count_due_dreams(db_session) == 0

    async def test_dreams_disabled_globally_returns_zero(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        with patch("src.dreamer.dream_due.settings.DREAM.ENABLED", False):
            assert await count_due_dreams(db_session) == 0

    async def test_card_refresh_is_never_counted(
        self,
        db_session: AsyncSession,
        sample_data: tuple[models.Workspace, models.Peer],
    ):
        collection = await _make_collection(db_session, sample_data)
        await _insert_docs(db_session, collection, "explicit", 60, age_minutes=90)

        with patch(
            "src.dreamer.dream_due.settings.DREAM.ENABLED_TYPES", ["card_refresh"]
        ):
            assert await count_due_dreams(db_session) == 0


async def _seed_tenant_collection(
    db_session: AsyncSession,
    *,
    tenant_id: str,
    workspace_name: str,
    peer_name: str,
) -> None:
    """Seed the composite-FK parent chain for a self-observation collection.

    Mirrors _make_collection but pins tenant_id explicitly across
    tenants -> workspaces -> peers -> collections, so two tenants can share the
    same (workspace_name, observer, observed) triple. tenants is preserved between
    tests (see conftest _clear_all_tables), so the tenant row is inserted
    idempotently.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    await db_session.execute(
        pg_insert(models.Tenant).values(tenant_id=tenant_id).on_conflict_do_nothing()
    )
    db_session.add(models.Workspace(name=workspace_name, tenant_id=tenant_id))
    db_session.add(
        models.Peer(name=peer_name, workspace_name=workspace_name, tenant_id=tenant_id)
    )
    await db_session.flush()
    db_session.add(
        models.Collection(
            observer=peer_name,
            observed=peer_name,
            workspace_name=workspace_name,
            tenant_id=tenant_id,
            internal_metadata={},
        )
    )
    await db_session.commit()


async def _seed_tenant_docs(
    db_session: AsyncSession,
    *,
    tenant_id: str,
    workspace_name: str,
    peer_name: str,
    count: int,
    age_minutes: int,
) -> str:
    """Seed a Session and `count` idle explicit Documents for a tenant; return the session name."""
    session_name = f"s-{generate_nanoid()}"
    db_session.add(
        models.Session(
            name=session_name,
            workspace_name=workspace_name,
            tenant_id=tenant_id,
            configuration={},
        )
    )
    await db_session.flush()
    created_at = _now() - datetime.timedelta(minutes=age_minutes)
    for _ in range(count):
        db_session.add(
            models.Document(
                content="test",
                level="explicit",
                workspace_name=workspace_name,
                observer=peer_name,
                observed=peer_name,
                session_name=session_name,
                tenant_id=tenant_id,
                created_at=created_at,
            )
        )
    await db_session.commit()
    return session_name


@pytest.mark.asyncio
class TestMultiTenantDueDreams:
    """Under MULTI_TENANT the due-dream scan runs cross-tenant on the service
    session, so its explicit-count grouping and every downstream lookup must be
    keyed by tenant. Otherwise two tenants sharing (workspace, observer, observed)
    merge into one row: tenant A trips the threshold on tenant B's documents and
    inherits B's newest session.
    """

    async def test_due_dreams_are_isolated_by_tenant(
        self,
        db_session: AsyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from src.dreamer.dream_due import DueDream, list_due_dreams

        workspace_name = "ws-shared"
        peer_name = "peer-shared"

        # Two tenants share the SAME (workspace_name, observer, observed) triple.
        await _seed_tenant_collection(
            db_session,
            tenant_id="tenant-a",
            workspace_name=workspace_name,
            peer_name=peer_name,
        )
        await _seed_tenant_collection(
            db_session,
            tenant_id="tenant-b",
            workspace_name=workspace_name,
            peer_name=peer_name,
        )
        # Only tenant-b has documents — past the threshold (50) and idle (> 60m).
        # tenant-a has ZERO documents.
        b_session = await _seed_tenant_docs(
            db_session,
            tenant_id="tenant-b",
            workspace_name=workspace_name,
            peer_name=peer_name,
            count=60,
            age_minutes=90,
        )

        monkeypatch.setattr(settings, "MULTI_TENANT", True)
        due = await list_due_dreams(db_session)

        # Exactly tenant-b is due. Pre-fix the tenant-blind join merged the two
        # triples, so tenant-a's empty collection inherited tenant-b's count AND
        # session_name and both came back due.
        assert len(due) == 1
        (due_dream,) = due
        assert isinstance(due_dream, DueDream)
        assert due_dream.tenant_id == "tenant-b"
        assert due_dream.session_name == b_session
        assert due_dream.workspace_name == workspace_name
        assert due_dream.observer == peer_name
        assert due_dream.observed == peer_name
        assert all(d.tenant_id != "tenant-a" for d in due)
