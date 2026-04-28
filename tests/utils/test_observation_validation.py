from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from src import schemas
from src.crud import representation as representation_crud
from src.crud.representation import RepresentationManager
from src.utils import agent_tools
from src.utils.observation_validation import (
    ObservationValidationResult,
    validate_observation_candidate,
)
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)


def _disabled_dream_configuration() -> schemas.ResolvedConfiguration:
    return schemas.ResolvedConfiguration(
        reasoning=schemas.ResolvedReasoningConfiguration(enabled=False),
        peer_card=schemas.ResolvedPeerCardConfiguration(use=False, create=False),
        summary=schemas.ResolvedSummaryConfiguration(
            enabled=False,
            messages_per_short_summary=20,
            messages_per_long_summary=100,
        ),
        dream=schemas.ResolvedDreamConfiguration(enabled=False),
    )


def test_accepts_explicit_observation_with_provenance():
    result = validate_observation_candidate(
        content="Chris prefers durable infrastructure fixes.",
        level="explicit",
        origin="minimal_deriver",
        source_message_ids=[101, 102],
    )

    assert result.accepted is True
    assert result.provenance.origin == "minimal_deriver"
    assert result.provenance.validation_status == "accepted"
    assert result.provenance.source_message_ids == [101, 102]
    assert result.provenance.validation_errors == []


@pytest.mark.parametrize("content", ["", "   \n\t  ", "contains\x00nul"])
def test_rejects_empty_whitespace_and_nul_content(content: str):
    result = validate_observation_candidate(
        content=content,
        level="explicit",
        origin="minimal_deriver",
        source_message_ids=[1],
    )

    assert result.accepted is False
    assert result.provenance.validation_status == "rejected"
    assert result.errors


def test_observation_input_rejects_nul_content():
    with pytest.raises(ValidationError, match="NUL"):
        schemas.ObservationInput(content="foo\x00bar", level="explicit")


def test_rejects_reasoned_observation_with_missing_source_ids():
    result = validate_observation_candidate(
        content="Chris prefers ZFS because it supports snapshots.",
        level="deductive",
        origin="agent_tool",
        source_ids=[],
        available_source_ids={"doc-1"},
    )

    assert result.accepted is False
    assert "source_ids" in " ".join(result.errors)


def test_rejects_source_ids_outside_available_context():
    result = validate_observation_candidate(
        content="Chris tends to prefer conservative infrastructure changes.",
        level="inductive",
        origin="agent_tool",
        source_ids=["doc-1", "missing-doc"],
        available_source_ids={"doc-1"},
    )

    assert result.accepted is False
    assert "missing-doc" in " ".join(result.errors)


def test_accepts_reasoned_observation_when_sources_exist():
    result = validate_observation_candidate(
        content="Chris tends to prefer conservative infrastructure changes.",
        level="inductive",
        origin="agent_tool",
        source_ids=["doc-1", "doc-2"],
        available_source_ids={"doc-1", "doc-2"},
    )

    assert result.accepted is True
    assert result.provenance.validation_status == "accepted"
    assert result.provenance.source_document_ids == ["doc-1", "doc-2"]


def test_accepts_contradiction_observation_with_two_sources():
    result = validate_observation_candidate(
        content="Chris both prefers and avoids automatic deploys depending on risk.",
        level="contradiction",
        origin="agent_tool",
        source_ids=["doc-1", "doc-2"],
        available_source_ids={"doc-1", "doc-2"},
    )

    assert result.accepted is True
    assert result.provenance.validation_status == "accepted"
    assert result.provenance.source_document_ids == ["doc-1", "doc-2"]


def test_rejects_contradiction_observation_without_two_distinct_sources():
    result = validate_observation_candidate(
        content="Chris both prefers and avoids automatic deploys depending on risk.",
        level="contradiction",
        origin="agent_tool",
        source_ids=["doc-1", "doc-1"],
        available_source_ids={"doc-1", "doc-2"},
    )

    assert result.accepted is False
    assert "at least 2 distinct source_ids" in " ".join(result.errors)


@pytest.mark.parametrize("source_ids", [None, [], ["doc-1"]])
def test_rejects_contradiction_observation_without_two_sources(
    source_ids: list[str] | None,
):
    result = validate_observation_candidate(
        content="Chris both prefers and avoids automatic deploys depending on risk.",
        level="contradiction",
        origin="agent_tool",
        source_ids=source_ids,
        available_source_ids={"doc-1", "doc-2"},
    )

    assert result.accepted is False
    assert "at least 2 distinct source_ids" in " ".join(result.errors)


@pytest.mark.asyncio
async def test_save_representation_validates_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
):
    embedded_texts: list[str] = []
    captured_observations: list[
        tuple[ExplicitObservation | DeductiveObservation, ObservationValidationResult]
    ] = []

    async def fake_batch_embed(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    @asynccontextmanager
    async def fake_tracked_db(_: str):
        yield object()

    async def fake_save_internal(
        _self: RepresentationManager,
        _db: object,
        all_observations: list[
            tuple[ExplicitObservation | DeductiveObservation, ObservationValidationResult]
        ],
        *_args: object,
    ) -> int:
        captured_observations.extend(all_observations)
        return len(all_observations)

    monkeypatch.setattr(
        representation_crud.embedding_client,  # pyright: ignore[reportPrivateLocalImportUsage]
        "simple_batch_embed",
        fake_batch_embed,
    )
    monkeypatch.setattr(representation_crud, "tracked_db", fake_tracked_db)
    monkeypatch.setattr(
        RepresentationManager,
        "_save_representation_internal",
        fake_save_internal,
    )

    manager = RepresentationManager("workspace", observer="observer", observed="observed")
    representation = Representation(
        explicit=[
            ExplicitObservation(
                content="Valid observation",
                created_at=datetime(2026, 4, 26, tzinfo=UTC),
                message_ids=[1],
                session_name="session",
            ),
            ExplicitObservation(
                content="   \n",
                created_at=datetime(2026, 4, 26, tzinfo=UTC),
                message_ids=[1],
                session_name="session",
            ),
        ]
    )

    saved = await manager.save_representation(
        representation,
        message_ids=[1],
        session_name="session",
        message_created_at=datetime(2026, 4, 26, tzinfo=UTC),
        message_level_configuration=_disabled_dream_configuration(),
    )

    assert saved == 1
    assert embedded_texts == ["Valid observation"]
    assert [
        obs.conclusion if isinstance(obs, DeductiveObservation) else obs.content
        for obs, _validation in captured_observations
    ] == ["Valid observation"]


@pytest.mark.asyncio
async def test_save_representation_rejects_missing_deductive_source_ids(
    monkeypatch: pytest.MonkeyPatch,
):
    embedded_texts: list[str] = []

    async def fake_fetch_documents_by_ids(
        *_args: object,
        **_kwargs: object,
    ) -> list[object]:
        return []

    async def fake_batch_embed(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    @asynccontextmanager
    async def fake_tracked_db(_: str):
        yield object()

    monkeypatch.setattr(representation_crud, "tracked_db", fake_tracked_db)
    monkeypatch.setattr(
        representation_crud.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "fetch_documents_by_ids",
        fake_fetch_documents_by_ids,
    )
    monkeypatch.setattr(
        representation_crud.embedding_client,  # pyright: ignore[reportPrivateLocalImportUsage]
        "simple_batch_embed",
        fake_batch_embed,
    )

    manager = RepresentationManager("workspace", observer="observer", observed="observed")
    representation = Representation(
        deductive=[
            DeductiveObservation(
                conclusion="Derived conclusion",
                source_ids=["missing-doc"],
                premises=["Missing premise"],
                created_at=datetime(2026, 4, 26, tzinfo=UTC),
                message_ids=[1],
                session_name="session",
            )
        ]
    )

    saved = await manager.save_representation(
        representation,
        message_ids=[1],
        session_name="session",
        message_created_at=datetime(2026, 4, 26, tzinfo=UTC),
        message_level_configuration=_disabled_dream_configuration(),
    )

    assert saved == 0
    assert embedded_texts == []


@pytest.mark.asyncio
async def test_save_representation_attaches_minimal_deriver_provenance(
    monkeypatch: pytest.MonkeyPatch,
):
    created_documents: list[schemas.DocumentCreate] = []

    async def fake_get_or_create_collection(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        return SimpleNamespace(id="collection-id")

    async def fake_create_documents(
        _db: object,
        documents: list[schemas.DocumentCreate],
        *_args: object,
        **_kwargs: object,
    ) -> list[schemas.DocumentCreate]:
        created_documents.extend(documents)
        return documents

    monkeypatch.setattr(
        representation_crud.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "get_or_create_collection",
        fake_get_or_create_collection,
    )
    monkeypatch.setattr(
        representation_crud.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "create_documents",
        fake_create_documents,
    )

    manager = RepresentationManager("workspace", observer="observer", observed="observed")
    observation = ExplicitObservation(
        content="Valid observation",
        created_at=datetime(2026, 4, 26, tzinfo=UTC),
        message_ids=[1],
        session_name="session",
    )

    validation = validate_observation_candidate(
        content=observation.content,
        level="explicit",
        origin="minimal_deriver",
        source_message_ids=[1],
    )

    saved = await manager._save_representation_internal(  # pyright: ignore[reportPrivateUsage]
        db=cast(AsyncSession, object()),
        all_observations=[(observation, validation)],
        embeddings=[[0.1, 0.2, 0.3]],
        message_ids=[1],
        session_name="session",
        message_created_at=datetime(2026, 4, 26, tzinfo=UTC),
        message_level_configuration=_disabled_dream_configuration(),
    )

    assert saved == 1
    assert created_documents[0].metadata.provenance is not None
    assert created_documents[0].metadata.provenance.origin == "minimal_deriver"
    assert created_documents[0].metadata.provenance.validation_status == "accepted"
    assert created_documents[0].metadata.provenance.source_message_ids == [1]


@pytest.mark.asyncio
async def test_create_observations_rejects_missing_source_ids_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
):
    embedded_texts: list[str] = []
    created_documents: list[schemas.DocumentCreate] = []

    @asynccontextmanager
    async def fake_tracked_db(_: str):
        yield object()

    async def fake_get_or_create_collection(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        return SimpleNamespace(id="collection-id")

    async def fake_fetch_documents_by_ids(
        *_args: object,
        **_kwargs: object,
    ) -> list[object]:
        return []

    async def fake_batch_embed(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def fake_create_documents(
        _db: object,
        documents: list[schemas.DocumentCreate],
        *_args: object,
        **_kwargs: object,
    ) -> list[schemas.DocumentCreate]:
        created_documents.extend(documents)
        return documents

    monkeypatch.setattr(agent_tools, "tracked_db", fake_tracked_db)
    monkeypatch.setattr(
        agent_tools.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "get_or_create_collection",
        fake_get_or_create_collection,
    )
    monkeypatch.setattr(
        agent_tools.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "fetch_documents_by_ids",
        fake_fetch_documents_by_ids,
    )
    monkeypatch.setattr(
        agent_tools.embedding_client,  # pyright: ignore[reportPrivateLocalImportUsage]
        "simple_batch_embed",
        fake_batch_embed,
    )
    monkeypatch.setattr(
        agent_tools.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "create_documents",
        fake_create_documents,
    )

    result = await agent_tools.create_observations(
        observations=[
            schemas.ObservationInput(content="Explicit observation", level="explicit"),
            schemas.ObservationInput(
                content="Deductive observation",
                level="deductive",
                source_ids=["missing-doc"],
                premises=["missing premise"],
            ),
        ],
        observer="observer",
        observed="observed",
        session_name="session",
        workspace_name="workspace",
        message_ids=[1],
        message_created_at="2026-04-26T00:00:00Z",
    )

    assert result.created_count == 1
    assert embedded_texts == ["Explicit observation"]
    assert [doc.content for doc in created_documents] == ["Explicit observation"]
    assert len(result.failed) == 1
    assert "missing-doc" in result.failed[0].error


@pytest.mark.asyncio
async def test_create_observations_does_not_create_collection_when_all_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    get_or_create_calls = 0
    create_documents_calls = 0
    embed_calls = 0

    @asynccontextmanager
    async def fake_tracked_db(_: str):
        yield object()

    async def fake_get_or_create_collection(
        *_args: object,
        **_kwargs: object,
    ) -> object:
        nonlocal get_or_create_calls
        get_or_create_calls += 1
        return SimpleNamespace(id="collection-id")

    async def fake_fetch_documents_by_ids(
        *_args: object,
        **_kwargs: object,
    ) -> list[object]:
        return []

    async def fake_batch_embed(texts: list[str]) -> list[list[float]]:
        nonlocal embed_calls
        embed_calls += 1
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def fake_create_documents(
        _db: object,
        documents: list[schemas.DocumentCreate],
        *_args: object,
        **_kwargs: object,
    ) -> list[schemas.DocumentCreate]:
        nonlocal create_documents_calls
        create_documents_calls += 1
        return documents

    monkeypatch.setattr(agent_tools, "tracked_db", fake_tracked_db)
    monkeypatch.setattr(
        agent_tools.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "get_or_create_collection",
        fake_get_or_create_collection,
    )
    monkeypatch.setattr(
        agent_tools.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "fetch_documents_by_ids",
        fake_fetch_documents_by_ids,
    )
    monkeypatch.setattr(
        agent_tools.embedding_client,  # pyright: ignore[reportPrivateLocalImportUsage]
        "simple_batch_embed",
        fake_batch_embed,
    )
    monkeypatch.setattr(
        agent_tools.crud,  # pyright: ignore[reportPrivateLocalImportUsage]
        "create_documents",
        fake_create_documents,
    )

    result = await agent_tools.create_observations(
        observations=[
            schemas.ObservationInput(
                content="Deductive observation",
                level="deductive",
                source_ids=["missing-doc"],
                premises=["missing premise"],
            )
        ],
        observer="observer",
        observed="observed",
        session_name="session",
        workspace_name="workspace",
        message_ids=[1],
        message_created_at="2026-04-26T00:00:00Z",
    )

    assert result.created_count == 0
    assert len(result.failed) == 1
    assert "missing-doc" in result.failed[0].error
    assert get_or_create_calls == 0
    assert embed_calls == 0
    assert create_documents_calls == 0
