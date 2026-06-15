from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.llm import HonchoLLMCallResponse
from src.utils.summarizer import (
    SummaryType,
    _build_summary_langfuse_metadata,  # pyright: ignore[reportPrivateUsage]
    _create_summary,  # pyright: ignore[reportPrivateUsage]
    create_long_summary,
    create_short_summary,
)

_FORMATTED_MESSAGES = "user: hello\nassistant: hi there"
_INPUT_TOKENS = 100
_MESSAGE_PUBLIC_ID = "msg_abc123"
_LAST_MESSAGE_ID = 42
_LAST_MESSAGE_CONTENT_PREVIEW = "hello there how are you"
_MESSAGE_COUNT = 5


class TestSummaryLangfuseMetadata:
    def test_build_summary_langfuse_metadata_is_safe_and_joinable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(settings, "LANGFUSE_TENANT_WORKSPACE_PREFIX", "acme-")
        monkeypatch.setattr(settings, "LANGFUSE_TENANT_PLATFORM", "acme")

        metadata, trace_attrs = _build_summary_langfuse_metadata(
            workspace_name="acme-user-123",
            session_name="chat-abc",
            summary_type=SummaryType.SHORT,
            message_count=5,
            latest_message_public_id="msg-public-5",
        )

        assert metadata["honcho_operation"] == "short_summary"
        assert metadata["honcho_workspace_id"] == "acme-user-123"
        assert metadata["honcho_session_id"] == "chat-abc"
        assert metadata["honcho_message_count"] == 5
        assert metadata["honcho_latest_message_public_id"] == "msg-public-5"
        assert metadata["tenant_user_id"] == "user-123"
        assert metadata["tenant_platform"] == "acme"
        assert trace_attrs == {
            "user_id": "user-123",
            "session_id": "chat-abc",
            "tags": ["honcho", "memory", "short_summary", "acme"],
        }
        assert "content" not in str(metadata).lower()

    def test_build_summary_langfuse_metadata_uses_long_operation(self):
        metadata, trace_attrs = _build_summary_langfuse_metadata(
            workspace_name="workspace-1",
            session_name="session-1",
            summary_type=SummaryType.LONG,
            message_count=10,
            latest_message_public_id="msg-public-10",
        )

        assert metadata["honcho_operation"] == "long_summary"
        assert trace_attrs == {
            "session_id": "session-1",
            "tags": ["honcho", "memory", "long_summary"],
        }


@pytest.mark.asyncio
class TestCreateSummaryLangfuseMetadata:
    async def test_create_summary_passes_langfuse_metadata_to_summary_call(self):
        mock_response = HonchoLLMCallResponse(
            content="summary",
            input_tokens=100,
            output_tokens=15,
            finish_reasons=["STOP"],
        )
        trace_attrs = {
            "user_id": "user-123",
            "session_id": "chat-abc",
            "tags": ["honcho", "memory", "short_summary"],
        }

        with patch(
            "src.utils.summarizer.create_short_summary",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_short:
            await _create_summary(
                formatted_messages=_FORMATTED_MESSAGES,
                previous_summary_text=None,
                summary_type=SummaryType.SHORT,
                input_tokens=_INPUT_TOKENS,
                message_public_id=_MESSAGE_PUBLIC_ID,
                last_message_id=_LAST_MESSAGE_ID,
                last_message_content_preview=_LAST_MESSAGE_CONTENT_PREVIEW,
                message_count=_MESSAGE_COUNT,
                langfuse_metadata={"honcho_operation": "short_summary"},
                langfuse_trace_attrs=trace_attrs,
            )

        await_args = mock_short.await_args
        if await_args is None:
            raise AssertionError("Expected short summary call")
        kwargs = await_args.kwargs
        assert kwargs["langfuse_metadata"] == {"honcho_operation": "short_summary"}
        assert kwargs["langfuse_trace_user_id"] == "user-123"
        assert kwargs["langfuse_trace_session_id"] == "chat-abc"
        assert kwargs["langfuse_trace_tags"] == [
            "honcho",
            "memory",
            "short_summary",
        ]

    async def test_create_short_summary_passes_langfuse_metadata(self):
        mock_response = HonchoLLMCallResponse(
            content="short summary",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.utils.summarizer.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm_call:
            await create_short_summary(
                formatted_messages=_FORMATTED_MESSAGES,
                input_tokens=_INPUT_TOKENS,
                previous_summary=None,
                langfuse_metadata={"honcho_operation": "short_summary"},
                langfuse_trace_user_id="user-123",
                langfuse_trace_session_id="chat-abc",
                langfuse_trace_tags=["honcho", "memory", "short_summary"],
            )

        await_args = mock_llm_call.await_args
        if await_args is None:
            raise AssertionError("Expected summary LLM call")
        kwargs = await_args.kwargs
        assert kwargs["track_name"] == "Create Short Summary"
        assert kwargs["trace_name"] == "short_summary"
        assert kwargs["langfuse_metadata"] == {"honcho_operation": "short_summary"}
        assert kwargs["langfuse_trace_user_id"] == "user-123"
        assert kwargs["langfuse_trace_session_id"] == "chat-abc"
        assert kwargs["langfuse_trace_tags"] == [
            "honcho",
            "memory",
            "short_summary",
        ]

    async def test_create_long_summary_passes_langfuse_metadata(self):
        mock_response = HonchoLLMCallResponse(
            content="long summary",
            input_tokens=10,
            output_tokens=5,
            finish_reasons=["STOP"],
        )

        with patch(
            "src.utils.summarizer.honcho_llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_llm_call:
            await create_long_summary(
                formatted_messages=_FORMATTED_MESSAGES,
                previous_summary=None,
                langfuse_metadata={"honcho_operation": "long_summary"},
                langfuse_trace_user_id="user-123",
                langfuse_trace_session_id="chat-abc",
                langfuse_trace_tags=["honcho", "memory", "long_summary"],
            )

        await_args = mock_llm_call.await_args
        if await_args is None:
            raise AssertionError("Expected summary LLM call")
        kwargs = await_args.kwargs
        assert kwargs["track_name"] == "Create Long Summary"
        assert kwargs["trace_name"] == "long_summary"
        assert kwargs["langfuse_metadata"] == {"honcho_operation": "long_summary"}
        assert kwargs["langfuse_trace_user_id"] == "user-123"
        assert kwargs["langfuse_trace_session_id"] == "chat-abc"
        assert kwargs["langfuse_trace_tags"] == [
            "honcho",
            "memory",
            "long_summary",
        ]
