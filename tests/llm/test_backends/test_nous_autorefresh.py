"""Integration tests for Nous auto-refresh in OpenAIBackend."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import AuthenticationError

from src.llm.backends.openai import OpenAIBackend


@pytest.mark.asyncio
async def test_nous_backend_auto_refresh_on_401() -> None:
    """OpenAIBackend with is_nous=True should refresh credentials on 401 and retry."""
    # Arrange
    mock_client = Mock()
    mock_client.api_key = "old_key"

    mock_success_response = Mock()
    mock_success_response.choices = [
        Mock(
            message=Mock(
                content="Hello, world!",
                tool_calls=[],  # required by _normalize_response
                refusal=None,
                parsed=None,
            )
        )
    ]
    mock_success_response.usage = Mock(completion_tokens=5)

    # First call raises AuthError, second returns success
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
        AuthenticationError(
            message="401 Unauthorized",
            response=Mock(status_code=401, request=Mock()),
            body={"error": "invalid_api_key"},
        ),
        mock_success_response,
    ]
    )

    backend = OpenAIBackend(mock_client, is_nous=True)

    # Patch refresh_nous_credentials to return a fake new key
    with patch(
        "src.llm.nous_refresh.refresh_nous_credentials",
        new=AsyncMock(return_value="new_refreshed_key"),
    ):
        result = await backend.complete(
            model="nous-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        )

    # Assert: client api_key updated
    assert mock_client.api_key == "new_refreshed_key"
    # Assert: create called twice (original + retry)
    assert mock_client.chat.completions.create.call_count == 2
    # Assert: result content from the second response
    assert result.content == "Hello, world!"


@pytest.mark.asyncio
async def test_openai_backend_no_refresh_on_401() -> None:
    """Non-Nous backends should not intercept 401; error bubbles up."""
    mock_client = Mock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=AuthenticationError(
        message="401",
        response=Mock(status_code=401, request=Mock()),
        body={"error": "invalid_api_key"},
    )
    )

    backend = OpenAIBackend(mock_client, is_nous=False)

    with pytest.raises(AuthenticationError):
        await backend.complete(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )

    # Should call create exactly once (no retry)
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_nous_backend_refresh_fails_propagates_error() -> None:
    """If refresh_nous_credentials returns None, original 401 is raised."""
    mock_client = Mock()
    mock_client.api_key = "old_key"
    mock_client.chat.completions.create = AsyncMock(
        side_effect=AuthenticationError(
        message="401",
        response=Mock(status_code=401, request=Mock()),
        body={"error": "invalid_api_key"},
    )
    )

    backend = OpenAIBackend(mock_client, is_nous=True)

    with patch(
        "src.llm.nous_refresh.refresh_nous_credentials",
        new=AsyncMock(return_value=None),  # refresh fails
    ):
        with pytest.raises(AuthenticationError):
            await backend.complete(
                model="nous-model",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )

    # Still only one call because refresh failed before retry
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_nous_backend_stream_auto_refresh() -> None:
    """Stream path also triggers auto-refresh on 401."""

    class FakeAsyncIterator:
        """Simple async iterator yielding predetermined chunks."""
        def __init__(self, chunks: list):
            self.chunks = chunks
            self.idx = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.idx >= len(self.chunks):
                raise StopAsyncIteration
            chunk = self.chunks[self.idx]
            self.idx += 1
            return chunk

    mock_client = Mock()
    mock_client.api_key = "old_key"

    call_count = 0

    async def create_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise AuthenticationError(
                message="401",
                response=Mock(status_code=401, request=Mock()),
                body={"error": "invalid_api_key"},
            )
        # Return an async iterator simulating a stream
        return FakeAsyncIterator(
            [Mock(choices=[Mock(delta=Mock(content="Hello"))], usage=Mock(completion_tokens=1))]
        )

    mock_client.chat.completions.create = AsyncMock(side_effect=create_side_effect)

    backend = OpenAIBackend(mock_client, is_nous=True)

    with patch(
        "src.llm.nous_refresh.refresh_nous_credentials",
        new=AsyncMock(return_value="new_key"),
    ):
        chunks = []
        async for chunk in backend.stream(
            model="nous-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
        ):
            chunks.append(chunk)

    assert mock_client.api_key == "new_key"
    assert call_count == 2
    # Stream yields one content chunk and one final done chunk
    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].is_done is True
    assert chunks[1].output_tokens == 1
