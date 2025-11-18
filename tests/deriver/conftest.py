import asyncio
from collections.abc import Awaitable, Callable, Generator, Sequence
from datetime import datetime, timezone
from typing import Any, Literal, TypeAlias, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.utils.queue_payload import create_payload
from src.utils.work_unit import construct_work_unit_key

QueuePayload: TypeAlias = dict[str, Any]
QueuePayloadEntry: TypeAlias = QueuePayload | tuple[QueuePayload, int | None]


@pytest.fixture
def mock_critical_analysis_call() -> Generator[Callable[..., Any], None, None]:
    """Mock the critical analysis call to avoid actual LLM calls"""

    async def mock_critical_analysis_call(*_args: Any, **_kwargs: Any) -> MagicMock:
        # Create a mock response that matches the expected structure
        mock_response = MagicMock()
        mock_response.explicit = ["Test explicit observation"]
        mock_response.deductive = []
        mock_response.thinking = "Test thinking content"
        mock_response._response = MagicMock()
        mock_response._response.thinking = "Test thinking content"
        return mock_response

    # Patch the actual function in the deriver module
    with patch(
        "src.deriver.deriver.critical_analysis_call", mock_critical_analysis_call
    ):
        yield mock_critical_analysis_call


@pytest.fixture
async def sample_session_with_peers(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
) -> tuple[models.Session, list[models.Peer]]:
    """Create a sample session with multiple peers for testing deriver functionality"""
    workspace, peer1 = sample_data

    # Create additional peers
    peer2 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    peer3 = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add_all([peer2, peer3])
    await db_session.flush()

    # Create session with peer configurations
    session = await crud.get_or_create_session(
        db_session,
        schemas.SessionCreate(
            name=str(generate_nanoid()),
            peers={
                peer1.name: schemas.SessionPeerConfig(observe_me=True),
                peer2.name: schemas.SessionPeerConfig(observe_others=True),
                peer3.name: schemas.SessionPeerConfig(),  # No special observation settings
            },
        ),
        workspace.name,
    )
    await db_session.commit()

    return session, [peer1, peer2, peer3]


@pytest.fixture
async def sample_messages(
    db_session: AsyncSession,
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
) -> list[models.Message]:
    """Create sample messages for testing deriver functionality"""
    session, peers = sample_session_with_peers
    peer1, peer2, peer3 = peers

    # Create multiple messages from different peers
    messages_data = [
        {
            "session_name": session.name,
            "content": "Hello, this is the first message from peer1",
            "peer_name": peer1.name,
            "workspace_name": session.workspace_name,
            "seq_in_session": 1,
        },
        {
            "session_name": session.name,
            "content": "Hi there! This is a response from peer2",
            "peer_name": peer2.name,
            "workspace_name": session.workspace_name,
            "seq_in_session": 2,
        },
        {
            "session_name": session.name,
            "content": "I'm just observing this conversation as peer3",
            "peer_name": peer3.name,
            "workspace_name": session.workspace_name,
            "seq_in_session": 3,
        },
    ]

    messages: list[models.Message] = []
    for msg_data in messages_data:
        message = models.Message(**msg_data)
        db_session.add(message)
        messages.append(message)

    await db_session.commit()

    # Query the messages again to get the committed versions
    result = await db_session.execute(
        select(models.Message)
        .where(models.Message.session_name == session.name)
        .order_by(models.Message.id)
    )
    messages = list(result.scalars().all())

    return messages


@pytest.fixture
def create_queue_payload() -> Callable[..., Any]:
    """Helper function to create queue payloads for testing"""

    def _create_payload(
        message: models.Message,
        task_type: Literal["representation", "summary"],
        observer: str | None = None,
        observed: str | None = None,
        message_seq_in_session: int | None = None,
    ) -> dict[str, Any]:
        """Create a queue payload for testing"""
        message_dict = {
            "workspace_name": message.workspace_name,
            "session_name": message.session_name,
            "message_id": message.id,
            "content": message.content,
            "created_at": message.created_at or datetime.now(timezone.utc),
            "message_public_id": message.public_id,
        }

        configuration = schemas.ResolvedConfiguration(
            deriver=schemas.ResolvedDeriverConfiguration(enabled=True),
            peer_card=schemas.ResolvedPeerCardConfiguration(use=True, create=True),
            summary=schemas.ResolvedSummaryConfiguration(
                enabled=True,
                messages_per_short_summary=10,
                messages_per_long_summary=20,
            ),
            dream=schemas.ResolvedDreamConfiguration(enabled=True),
        )

        return create_payload(
            message=message_dict,
            configuration=configuration,
            task_type=task_type,
            message_seq_in_session=message_seq_in_session,
            observer=observer,
            observed=observed,
        )

    return _create_payload


@pytest.fixture
async def add_queue_items(
    db_session: AsyncSession,
) -> Callable[
    [Sequence[QueuePayloadEntry], str, str], Awaitable[list[models.QueueItem]]
]:
    """Helper function to add queue items to the database"""

    async def _add_items(
        payloads: Sequence[QueuePayloadEntry],
        session_id: str,
        workspace_name: str,
    ) -> list[models.QueueItem]:
        """Add queue items to the database and return them"""
        queue_items: list[models.QueueItem] = []
        for payload_entry in payloads:
            payload: QueuePayload
            message_id: int | None
            if isinstance(payload_entry, tuple):
                payload, message_id = payload_entry
            else:
                payload = payload_entry
                message_id = cast(int | None, payload.get("message_id"))
            # Generate work_unit_key from the payload
            task_type = cast(str, payload.get("task_type", "unknown"))
            work_unit_key = construct_work_unit_key(workspace_name, payload)

            queue_item = models.QueueItem(
                session_id=session_id,
                task_type=task_type,
                work_unit_key=work_unit_key,
                payload=payload,
                processed=False,
                workspace_name=workspace_name,
                message_id=message_id,
            )
            db_session.add(queue_item)
            queue_items.append(queue_item)

        await db_session.commit()

        # Refresh to get the actual IDs
        for item in queue_items:
            await db_session.refresh(item)

        return queue_items

    return _add_items


@pytest.fixture
async def sample_queue_items(
    db_session: AsyncSession,  # noqa: ARG001  # pyright: ignore[reportUnusedParameter]
    sample_session_with_peers: tuple[models.Session, list[models.Peer]],
    sample_messages: list[models.Message],
    create_queue_payload: Callable[..., Any],
    add_queue_items: Callable[..., Any],
) -> list[models.QueueItem]:
    """Create sample queue items for testing"""
    session, peers = sample_session_with_peers
    _peer1, peer2, _peer3 = peers
    messages = sample_messages

    # Create various types of queue payloads
    payloads: list[tuple[dict[str, Any], int]] = []

    # Create representation payloads for each message
    for message in messages:
        # Self-representation (peer observing themselves)
        payload1 = create_queue_payload(
            message=message,
            task_type="representation",
            observer=message.peer_name,
            observed=message.peer_name,
        )
        payloads.append((payload1, message.id))

        # Representation for observer peer
        payload2 = create_queue_payload(
            message=message,
            task_type="representation",
            observer=peer2.name,  # peer2 observes others
            observed=message.peer_name,
        )
        payloads.append((payload2, message.id))

    # Create summary payloads for session
    for i, message in enumerate(messages):
        payload = create_queue_payload(
            message=message,
            task_type="summary",
            message_seq_in_session=i + 1,
        )
        payloads.append((payload, message.id))

    # Add all payloads as queue items
    queue_items = await add_queue_items(payloads, session.id, session.workspace_name)

    return queue_items


@pytest.fixture
async def create_active_queue_session(db_session: AsyncSession) -> Callable[..., Any]:
    """Helper function to create active queue sessions for testing work unit tracking"""

    async def _create_active_session(
        work_unit_key: str,
    ) -> models.ActiveQueueSession:
        """Create an active queue session"""
        active_session = models.ActiveQueueSession(
            work_unit_key=work_unit_key,
        )
        db_session.add(active_session)
        await db_session.commit()
        await db_session.refresh(active_session)
        return active_session

    return _create_active_session


@pytest.fixture
def mock_queue_manager(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:  # pyright: ignore[reportUnusedParameter]
    """Mock the queue manager to avoid actual queue processing"""
    from src.deriver.queue_manager import QueueManager

    # Create a mock queue manager
    mock_manager = AsyncMock(spec=QueueManager)

    # Mock the methods we might need to test
    mock_manager.initialize = AsyncMock()
    mock_manager.shutdown = AsyncMock()
    mock_manager.process_work_unit = AsyncMock()
    mock_manager.get_available_work_units = AsyncMock(return_value=[])
    mock_manager.add_task = MagicMock()
    mock_manager.track_work_unit = MagicMock()
    mock_manager.untrack_work_unit = MagicMock()

    # Mock the attributes
    mock_manager.shutdown_event = asyncio.Event()
    mock_manager.active_tasks = set()
    mock_manager.owned_work_units = set()
    mock_manager.queue_empty_flag = asyncio.Event()
    mock_manager.workers = 1
    mock_manager.semaphore = asyncio.Semaphore(1)

    return mock_manager


@pytest.fixture
def mock_representation_manager(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:  # pyright: ignore[reportUnusedParameter]
    """Mock the representation manager to avoid actual embedding operations"""
    from src.crud.representation import RepresentationManager

    mock_manager = AsyncMock(spec=RepresentationManager)
    mock_manager.save_representation.return_value = 0

    return mock_manager
