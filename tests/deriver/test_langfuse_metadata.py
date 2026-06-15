from datetime import datetime, timezone
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config import settings
from src.deriver.deriver import (
    _build_deriver_langfuse_metadata,  # pyright: ignore[reportPrivateUsage]
    process_representation_tasks_batch,
)
from src.llm import HonchoLLMCallResponse
from src.models import Message
from src.utils.representation import PromptRepresentation


def _message(
    *,
    message_id: int,
    public_id: str,
    workspace_name: str = "myah-user-123",
    session_name: str = "chat-abc",
    peer_name: str = "user-123",
    content: str = "private content should not appear",
) -> Mock:
    return Mock(
        id=message_id,
        public_id=public_id,
        session_name=session_name,
        workspace_name=workspace_name,
        peer_name=peer_name,
        content=content,
        token_count=5,
        created_at=datetime.now(timezone.utc),
    )


def _messages_for_helper(messages: list[Mock]) -> list[Message]:
    return cast(list[Message], messages)


def test_build_deriver_langfuse_metadata_uses_public_ids_and_omits_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_WORKSPACE_PREFIX", "myah-")
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_PLATFORM", "myah")
    messages = [
        _message(message_id=1, public_id="msg-public-1"),
        _message(message_id=2, public_id="msg-public-2"),
    ]

    metadata, trace_attrs = _build_deriver_langfuse_metadata(
        messages=_messages_for_helper(messages),
        observers=["myah", "support-agent"],
        observed="user-123",
        queue_item_message_ids=[1, 2],
    )

    assert metadata["honcho_operation"] == "minimal_deriver"
    assert metadata["honcho_workspace_id"] == "myah-user-123"
    assert metadata["honcho_session_id"] == "chat-abc"
    assert metadata["honcho_observed_peer"] == "user-123"
    assert metadata["honcho_observer_peers"] == ["myah", "support-agent"]
    assert metadata["honcho_message_count"] == 2
    assert metadata["honcho_queue_item_count"] == 2
    assert metadata["honcho_message_public_ids"] == ["msg-public-1", "msg-public-2"]
    assert metadata["honcho_latest_message_public_id"] == "msg-public-2"
    assert metadata["tenant_user_id"] == "user-123"
    assert metadata["tenant_platform"] == "myah"
    assert trace_attrs == {
        "user_id": "user-123",
        "session_id": "chat-abc",
        "tags": ["honcho", "memory", "minimal_deriver", "myah"],
    }
    assert "private content" not in str(metadata)
    assert "content" not in str(metadata).lower()
    assert "honcho_message_db_ids" not in metadata


def test_build_deriver_langfuse_metadata_bounds_and_sanitizes_lists() -> None:
    messages = [
        _message(message_id=i, public_id=f"msg-public-{i}")
        for i in range(30)
    ]
    messages.append(_message(message_id=31, public_id="Bearer secret-token"))

    metadata, _ = _build_deriver_langfuse_metadata(
        messages=_messages_for_helper(messages),
        observers=["myah", "lf_sk_secret"],
        observed="user-123",
        queue_item_message_ids=list(range(31)),
    )

    assert metadata["honcho_observer_peers"] == ["myah"]
    assert metadata["honcho_message_public_ids"] == [
        f"msg-public-{i}" for i in range(25)
    ]
    assert "Bearer secret-token" not in str(metadata)
    assert "lf_sk_secret" not in str(metadata)


@pytest.mark.asyncio
async def test_process_representation_tasks_batch_passes_langfuse_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_WORKSPACE_PREFIX", "myah-")
    monkeypatch.setattr(settings, "LANGFUSE_TENANT_PLATFORM", "myah")
    messages = [
        _message(message_id=1, public_id="msg-public-1"),
        _message(message_id=2, public_id="msg-public-2"),
    ]
    configuration = Mock()
    configuration.reasoning.enabled = True
    configuration.reasoning.custom_instructions = None
    mock_response = HonchoLLMCallResponse(
        content=PromptRepresentation(explicit=[]),
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["STOP"],
    )

    with patch(
        "src.deriver.deriver.honcho_llm_call",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_llm_call:
        await process_representation_tasks_batch(
            messages=_messages_for_helper(messages),
            message_level_configuration=configuration,
            observers=["myah"],
            observed="user-123",
            queue_item_message_ids=[1, 2],
        )

    await_args = mock_llm_call.await_args
    if await_args is None:
        raise AssertionError("Expected deriver LLM call")
    kwargs = await_args.kwargs

    assert kwargs["track_name"] == "Minimal Deriver"
    assert kwargs["langfuse_metadata"]["honcho_operation"] == "minimal_deriver"
    assert kwargs["langfuse_metadata"]["honcho_workspace_id"] == "myah-user-123"
    assert kwargs["langfuse_metadata"]["honcho_session_id"] == "chat-abc"
    assert kwargs["langfuse_metadata"]["honcho_queue_item_count"] == 2
    assert kwargs["langfuse_trace_user_id"] == "user-123"
    assert kwargs["langfuse_trace_session_id"] == "chat-abc"
    assert kwargs["langfuse_trace_tags"] == [
        "honcho",
        "memory",
        "minimal_deriver",
        "myah",
    ]
