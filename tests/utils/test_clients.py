"""
Comprehensive tests for src/utils/clients.py

Tests cover:
- All supported LLM providers (Anthropic, OpenAI, Google/Gemini, Groq)
- Streaming and non-streaming responses
- Response models (structured output)
- Error handling and retries
- Provider-specific features
- Client initialization
- Langfuse integration
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, Usage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field

from src.utils.clients import (
    CLIENTS,
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    handle_streaming_response,
    honcho_llm_call,
    honcho_llm_call_inner,
    retry,
    stop_after_attempt,
    wait_exponential,
    with_langfuse,
)


class SampleTestModel(BaseModel):
    """Test Pydantic model for structured output"""

    name: str
    age: int
    active: bool = Field(default=True)


class TestLLMCallResponse:
    """Tests for HonchoLLMCallResponse and HonchoLLMCallStreamChunk models"""

    def test_llm_call_response_creation(self):
        """Test creating HonchoLLMCallResponse with string content"""
        response = HonchoLLMCallResponse(
            content="Hello world", output_tokens=10, finish_reasons=["stop"]
        )
        assert response.content == "Hello world"
        assert response.output_tokens == 10
        assert response.finish_reasons == ["stop"]

    def test_llm_call_response_with_model(self):
        """Test creating HonchoLLMCallResponse with Pydantic model content"""
        model = SampleTestModel(name="John", age=30)
        response = HonchoLLMCallResponse[SampleTestModel](
            content=model, output_tokens=15, finish_reasons=["stop"]
        )
        assert response.content.name == "John"
        assert response.content.age == 30
        assert response.content.active is True

    def test_stream_chunk_creation(self):
        """Test creating HonchoLLMCallStreamChunk"""
        chunk = HonchoLLMCallStreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.is_done is False
        assert chunk.finish_reasons == []

    def test_stream_chunk_done(self):
        """Test creating final HonchoLLMCallStreamChunk"""
        chunk = HonchoLLMCallStreamChunk(
            content="", is_done=True, finish_reasons=["stop"]
        )
        assert chunk.content == ""
        assert chunk.is_done is True
        assert chunk.finish_reasons == ["stop"]

    def test_stream_chunk_default_finish_reasons(self):
        """Test that finish_reasons defaults to empty list"""
        chunk = HonchoLLMCallStreamChunk(content="test")
        assert isinstance(chunk.finish_reasons, list)
        assert chunk.finish_reasons == []


class TestRetryMechanisms:
    """Tests for retry logic and backoff strategies"""

    def test_stop_after_attempt(self):
        """Test stop_after_attempt predicate"""
        stop_func = stop_after_attempt(3)

        # Should retry on attempts 1 and 2
        assert stop_func(1) is True
        assert stop_func(2) is True
        # Should not retry on attempt 3 and beyond
        assert stop_func(3) is False
        assert stop_func(4) is False

    def test_wait_exponential_basic(self):
        """Test basic exponential backoff"""
        wait_func = wait_exponential(multiplier=1.0)

        # Should return 1 * 2^(attempt-1)
        assert wait_func(1) == 1.0  # 1 * 2^0 = 1
        assert wait_func(2) == 2.0  # 1 * 2^1 = 2
        assert wait_func(3) == 4.0  # 1 * 2^2 = 4

    def test_wait_exponential_with_bounds(self):
        """Test exponential backoff with min/max bounds"""
        wait_func = wait_exponential(multiplier=1.0, min=1.5, max=3.0)

        # Should be clamped to min=1.5 for attempt 1
        assert wait_func(1) == 1.5  # max(1.0, 1.5) = 1.5
        # Should be normal for attempt 2
        assert wait_func(2) == 2.0
        # Should be clamped to max=3.0 for attempt 3
        assert wait_func(3) == 3.0  # min(4.0, 3.0) = 3.0

    def test_wait_exponential_multiplier(self):
        """Test exponential backoff with different multiplier"""
        wait_func = wait_exponential(multiplier=2.0)

        assert wait_func(1) == 2.0  # 2 * 2^0 = 2
        assert wait_func(2) == 4.0  # 2 * 2^1 = 4
        assert wait_func(3) == 8.0  # 2 * 2^2 = 8

    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """Test retry decorator with successful function"""
        call_count = 0

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.1, max=0.1),  # Fast for testing
        )
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_decorator_eventual_success(self):
        """Test retry decorator with eventual success"""
        call_count = 0

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.01, max=0.01),  # Fast for testing
        )
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_decorator_max_attempts_exceeded(self):
        """Test retry decorator when max attempts are exceeded"""
        call_count = 0

        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=0.01, max=0.01),  # Fast for testing
        )
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")

        with pytest.raises(Exception, match="Persistent failure"):
            await test_func()
        assert call_count == 2

    def test_retry_decorator_sync_function(self):
        """Test retry decorator with synchronous function"""
        call_count = 0

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.001, max=0.001),  # Fast for testing
        )
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"

        result = test_func()
        assert result == "success"
        assert call_count == 2


class TestLangfuseIntegration:
    """Tests for Langfuse integration"""

    @pytest.mark.asyncio
    async def test_with_langfuse_decorator(self):
        """Test Langfuse decorator functionality"""

        @with_langfuse
        async def test_func():
            return "decorated"

        # Mock the langfuse client
        with patch("src.utils.clients.lf") as mock_lf:
            result = await test_func()
            assert result == "decorated"
            mock_lf.start_as_current_generation.assert_called_once_with(name="LLM Call")


@pytest.mark.asyncio
class TestAnthropicClient:
    """Tests for Anthropic client functionality"""

    async def test_anthropic_basic_call(self):
        """Test basic Anthropic API call"""
        from anthropic import AsyncAnthropic

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text="Hello from Anthropic", type="text")]
        mock_response.usage = Usage(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            response = await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Hello",
                max_tokens=100,
            )

            assert isinstance(response, HonchoLLMCallResponse)
            assert response.content == "Hello from Anthropic"
            assert response.output_tokens == 5
            assert response.finish_reasons == ["stop"]

    async def test_anthropic_multiple_text_blocks(self):
        """Test Anthropic response with multiple text blocks"""
        from anthropic import AsyncAnthropic

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [
            TextBlock(text="First block", type="text"),
            TextBlock(text="Second block", type="text"),
        ]
        mock_response.usage = Usage(input_tokens=10, output_tokens=8)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            response = await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Hello",
                max_tokens=100,
            )

            assert response.content == "First block\nSecond block"
            assert response.output_tokens == 8

    async def test_anthropic_json_mode(self):
        """Test Anthropic with JSON mode"""
        from anthropic import AsyncAnthropic

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text='{"result": "success"}', type="text")]
        mock_response.usage = Usage(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Generate JSON",
                max_tokens=100,
                json_mode=True,
            )

            # Verify assistant message was added for JSON mode
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            assert any(
                msg["role"] == "assistant" and msg["content"] == "{" for msg in messages
            )

    async def test_anthropic_thinking_budget(self):
        """Test Anthropic with thinking budget tokens"""
        from anthropic import AsyncAnthropic

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text="Thoughtful response", type="text")]
        mock_response.usage = Usage(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Think about this",
                max_tokens=100,
                thinking_budget_tokens=1000,
            )

            # Verify thinking parameter was passed
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            thinking_config = call_args.kwargs["thinking"]
            assert thinking_config == {"type": "enabled", "budget_tokens": 1000}

    async def test_anthropic_response_model_not_supported(self):
        """Test that Anthropic raises error for response models"""
        mock_client = AsyncMock(spec=AsyncAnthropic)

        with (
            patch.dict(CLIENTS, {"anthropic": mock_client}),
            pytest.raises(
                NotImplementedError,
                match="Response model is not supported for Anthropic",
            ),
        ):
            await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Hello",
                max_tokens=100,
                response_model=SampleTestModel,
            )

    async def test_anthropic_streaming(self):
        """Test Anthropic streaming response"""
        from anthropic import AsyncAnthropic

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_stream = AsyncMock()

        # Mock streaming chunks
        mock_chunks = [
            Mock(type="content_block_delta", delta=Mock(text="Hello")),
            Mock(type="content_block_delta", delta=Mock(text=" world")),
        ]

        # Set up the async context manager
        mock_stream.__aenter__.return_value = mock_stream
        mock_stream.__aexit__.return_value = None

        # Set up the async iterator (same as working test_streaming_call)
        mock_stream.__aiter__.return_value = iter(mock_chunks)

        # Mock final message
        mock_final_message = Mock(stop_reason="stop")
        mock_stream.get_final_message.return_value = mock_final_message

        mock_client.messages.stream.return_value = mock_stream

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            async for chunk in handle_streaming_response(
                client=mock_client,
                params={
                    "model": "claude-3-sonnet",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                json_mode=False,
                thinking_budget_tokens=None,
            ):
                chunks.append(chunk)

            assert len(chunks) == 3  # 2 content chunks + 1 final chunk
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"
            assert chunks[2].content == ""
            assert chunks[2].is_done is True
            assert chunks[2].finish_reasons == ["stop"]


@pytest.mark.asyncio
class TestOpenAIClient:
    """Tests for OpenAI client functionality"""

    async def test_openai_basic_call(self):
        """Test basic OpenAI API call"""
        from openai import AsyncOpenAI

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="Hello from OpenAI"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai", model="gpt-4", prompt="Hello", max_tokens=100
            )

            assert isinstance(response, HonchoLLMCallResponse)
            assert response.content == "Hello from OpenAI"
            assert response.output_tokens == 5
            assert response.finish_reasons == ["stop"]

    async def test_openai_gpt5_parameters(self):
        """Test OpenAI GPT-5 specific parameters"""
        from openai import AsyncOpenAI

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-5-turbo",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="GPT-5 response"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="openai",
                model="gpt-5-turbo",
                prompt="Hello",
                max_tokens=100,
                reasoning_effort="high",
                verbosity="medium",
            )

            # Verify GPT-5 specific parameters were used
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            kwargs = call_args.kwargs
            assert "max_completion_tokens" in kwargs
            assert kwargs["max_completion_tokens"] == 100
            assert kwargs["reasoning_effort"] == "high"
            assert kwargs["verbosity"] == "medium"

    async def test_openai_json_mode(self):
        """Test OpenAI with JSON mode"""
        from openai import AsyncOpenAI

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content='{"result": "success"}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="openai",
                model="gpt-4",
                prompt="Generate JSON",
                max_tokens=100,
                json_mode=True,
            )

            # Verify JSON mode was enabled
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["response_format"] == {"type": "json_object"}

    async def test_openai_response_model(self):
        """Test OpenAI with structured output (response model)"""
        from openai import AsyncOpenAI

        mock_client = AsyncMock(spec=AsyncOpenAI)

        # Create a mock parsed object
        mock_parsed = SampleTestModel(name="John", age=30)

        # Create a proper ChatCompletionMessage and add parsed attribute
        message = ChatCompletionMessage(role="assistant", content="")
        setattr(message, "parsed", mock_parsed)  # noqa: B010

        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=message,
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=15, total_tokens=25
            ),
        )
        mock_client.chat.completions.parse = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            response = await honcho_llm_call_inner(
                provider="openai",
                model="gpt-4",
                prompt="Generate a person",
                max_tokens=100,
                response_model=SampleTestModel,
            )

            assert isinstance(response, HonchoLLMCallResponse)
            assert isinstance(response.content, SampleTestModel)
            assert response.content.name == "John"
            assert response.content.age == 30
            assert response.output_tokens == 15

            # Verify parse was called instead of create
            mock_client.chat.completions.parse.assert_called_once()
            mock_client.chat.completions.create.assert_not_called()

    async def test_openai_streaming(self):
        """Test OpenAI streaming response"""
        from openai import AsyncOpenAI

        mock_client = AsyncMock(spec=AsyncOpenAI)

        # Create mock streaming chunks
        mock_chunks = [
            ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567890,
                model="gpt-4",
                choices=[
                    ChunkChoice(
                        index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None
                    )
                ],
            ),
            ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567890,
                model="gpt-4",
                choices=[
                    ChunkChoice(
                        index=0, delta=ChoiceDelta(content=" world"), finish_reason=None
                    )
                ],
            ),
            ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567890,
                model="gpt-4",
                choices=[
                    ChunkChoice(
                        index=0, delta=ChoiceDelta(content=None), finish_reason="stop"
                    )
                ],
            ),
        ]

        # Create async iterator
        async def async_chunk_iterator():
            for chunk in mock_chunks:
                yield chunk

        # OpenAI's create method returns an awaitable that resolves to an async iterator
        async def mock_create(**_kwargs: Any):
            return async_chunk_iterator()

        mock_client.chat.completions.create = mock_create

        with patch.dict(CLIENTS, {"openai": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            async for chunk in handle_streaming_response(
                client=mock_client,
                params={
                    "model": "gpt-4",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                json_mode=False,
                thinking_budget_tokens=None,
            ):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"
            assert chunks[2].content == ""
            assert chunks[2].is_done is True
            assert chunks[2].finish_reasons == ["stop"]


@pytest.mark.asyncio
class TestGoogleClient:
    """Tests for Google/Gemini client functionality"""

    async def test_google_basic_call(self):
        """Test basic Google/Gemini API call"""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_response.text = "Hello from Gemini"
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_response.candidates = [
            Mock(token_count=5, finish_reason=mock_finish_reason)
        ]
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(CLIENTS, {"google": mock_client}):
            response = await honcho_llm_call_inner(
                provider="google",
                model="gemini-1.5-pro",
                prompt="Hello",
                max_tokens=100,
            )

            assert isinstance(response, HonchoLLMCallResponse)
            assert response.content == "Hello from Gemini"
            assert response.output_tokens == 5
            assert response.finish_reasons == ["STOP"]

    async def test_google_json_mode(self):
        """Test Google/Gemini with JSON mode"""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_response.text = '{"result": "success"}'
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_response.candidates = [
            Mock(token_count=10, finish_reason=mock_finish_reason)
        ]
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(CLIENTS, {"google": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="google",
                model="gemini-1.5-pro",
                prompt="Generate JSON",
                max_tokens=100,
                json_mode=True,
            )

            # Verify JSON mode was set in config
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            assert (
                call_args.kwargs["config"]["response_mime_type"] == "application/json"
            )

    async def test_google_response_model(self):
        """Test Google/Gemini with structured output"""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_parsed = SampleTestModel(name="Alice", age=25)
        mock_response.parsed = mock_parsed
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_response.candidates = [
            Mock(token_count=15, finish_reason=mock_finish_reason)
        ]
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(CLIENTS, {"google": mock_client}):
            response = await honcho_llm_call_inner(
                provider="google",
                model="gemini-1.5-pro",
                prompt="Generate a person",
                max_tokens=100,
                response_model=SampleTestModel,
            )

            assert isinstance(response, HonchoLLMCallResponse)
            assert isinstance(response.content, SampleTestModel)
            assert response.content.name == "Alice"
            assert response.content.age == 25

            # Verify structured output config
            mock_client.models.generate_content.assert_called_once()
            call_args = mock_client.models.generate_content.call_args
            config = call_args.kwargs["config"]
            assert config["response_mime_type"] == "application/json"
            assert config["response_schema"] == SampleTestModel

    async def test_google_streaming(self):
        """Test Google/Gemini streaming response"""
        from google import genai

        mock_client = Mock(spec=genai.Client)

        # Mock streaming chunks
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_chunks = [
            Mock(text="Hello"),
            Mock(text=" world"),
            Mock(text="", candidates=[Mock(finish_reason=mock_finish_reason)]),
        ]

        mock_client.models.generate_content_stream.return_value = iter(mock_chunks)

        with patch.dict(CLIENTS, {"google": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            async for chunk in handle_streaming_response(
                client=mock_client,
                params={
                    "model": "gemini-1.5-pro",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                json_mode=False,
                thinking_budget_tokens=None,
            ):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"
            assert chunks[2].content == ""
            assert chunks[2].is_done is True
            assert chunks[2].finish_reasons == ["STOP"]

    async def test_google_no_candidates_fallback(self):
        """Test Google/Gemini fallback when no candidates"""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_response.text = "Response text"
        mock_response.candidates = []  # Empty candidates
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(CLIENTS, {"google": mock_client}):
            response = await honcho_llm_call_inner(
                provider="google",
                model="gemini-1.5-pro",
                prompt="Hello",
                max_tokens=100,
            )

            assert response.content == "Response text"
            assert response.output_tokens == 0  # Fallback value
            assert response.finish_reasons == ["stop"]  # Default fallback


@pytest.mark.asyncio
class TestGroqClient:
    """Tests for Groq client functionality"""

    async def test_groq_basic_call(self):
        """Test basic Groq API call"""
        from groq import AsyncGroq

        mock_client = AsyncMock(spec=AsyncGroq)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="llama-3.1-70b",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="Hello from Groq"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=8, total_tokens=18
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"groq": mock_client}):
            response = await honcho_llm_call_inner(
                provider="groq", model="llama-3.1-70b", prompt="Hello", max_tokens=100
            )

            assert isinstance(response, HonchoLLMCallResponse)
            assert response.content == "Hello from Groq"
            assert response.output_tokens == 8
            assert response.finish_reasons == ["stop"]

    async def test_groq_json_mode(self):
        """Test Groq with JSON mode"""
        from groq import AsyncGroq

        mock_client = AsyncMock(spec=AsyncGroq)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="llama-3.1-70b",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content='{"success": true}'
                    ),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"groq": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="groq",
                model="llama-3.1-70b",
                prompt="Generate JSON",
                max_tokens=100,
                json_mode=True,
            )

            # Verify JSON mode was set
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["response_format"] == {"type": "json_object"}

    async def test_groq_response_model(self):
        """Test Groq with response model (structured output)"""
        from groq import AsyncGroq

        mock_client = AsyncMock(spec=AsyncGroq)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="llama-3.1-70b",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="Bob"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=12, total_tokens=22
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"groq": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="groq",
                model="llama-3.1-70b",
                prompt="Generate a person",
                max_tokens=100,
                response_model=SampleTestModel,
            )

            # Verify response_format was set to the model
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["response_format"] == SampleTestModel

    async def test_groq_no_content_error(self):
        """Test Groq error handling when no content in response"""
        from groq import AsyncGroq

        mock_client = AsyncMock(spec=AsyncGroq)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="llama-3.1-70b",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content=None),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=0, total_tokens=10
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with (
            patch.dict(CLIENTS, {"groq": mock_client}),
            pytest.raises(ValueError, match="No content in response"),
        ):
            await honcho_llm_call_inner(
                provider="groq",
                model="llama-3.1-70b",
                prompt="Hello",
                max_tokens=100,
            )

    async def test_groq_streaming(self):
        """Test Groq streaming response"""
        from groq import AsyncGroq

        mock_client = AsyncMock(spec=AsyncGroq)

        # Create mock streaming chunks
        mock_chunks = [
            ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567890,
                model="llama-3.1-70b",
                choices=[
                    ChunkChoice(
                        index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None
                    )
                ],
            ),
            ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567890,
                model="llama-3.1-70b",
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChoiceDelta(content=" from Groq"),
                        finish_reason=None,
                    )
                ],
            ),
            ChatCompletionChunk(
                id="test-id",
                object="chat.completion.chunk",
                created=1234567890,
                model="llama-3.1-70b",
                choices=[
                    ChunkChoice(
                        index=0, delta=ChoiceDelta(content=None), finish_reason="stop"
                    )
                ],
            ),
        ]

        # Create async iterator
        async def async_chunk_iterator():
            for chunk in mock_chunks:
                yield chunk

        # Mock the create method to return the async generator when awaited
        mock_client.chat.completions.create = AsyncMock(
            return_value=async_chunk_iterator()
        )

        with patch.dict(CLIENTS, {"groq": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            async for chunk in handle_streaming_response(
                client=mock_client,
                params={
                    "model": "llama-3.1-70b",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                json_mode=False,
                thinking_budget_tokens=None,
            ):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " from Groq"
            assert chunks[2].content == ""
            assert chunks[2].is_done is True
            assert chunks[2].finish_reasons == ["stop"]


@pytest.mark.asyncio
class TestMainLLMCallFunction:
    """Tests for the main honcho_llm_call function"""

    async def test_streaming_call(self):
        """Test streaming LLM call"""
        from anthropic import AsyncAnthropic

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_stream = AsyncMock()

        # Mock streaming chunks
        mock_chunks = [
            Mock(type="content_block_delta", delta=Mock(text="Stream")),
            Mock(type="content_block_delta", delta=Mock(text=" test")),
        ]
        mock_stream.__aenter__.return_value = mock_stream
        mock_stream.__aiter__.return_value = iter(mock_chunks)

        # Mock final message
        mock_final_message = Mock(stop_reason="stop")
        mock_stream.get_final_message.return_value = mock_final_message

        mock_client.messages.stream.return_value = mock_stream

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            async for chunk in await honcho_llm_call(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Hello",
                max_tokens=100,
                stream=True,
                enable_retry=False,  # Disable retry for simpler testing
            ):
                chunks.append(chunk)

            assert len(chunks) == 3  # 2 content + 1 final
            assert chunks[0].content == "Stream"
            assert chunks[1].content == " test"
            assert chunks[2].is_done is True

    async def test_retry_disabled(self):
        """Test that retry can be disabled"""
        from anthropic import AsyncAnthropic

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text="No retry response", type="text")]
        mock_response.usage = Usage(input_tokens=5, output_tokens=5)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with (
            patch.dict(CLIENTS, {"anthropic": mock_client}),
            patch("src.utils.clients.retry") as mock_retry,
        ):
            response = await honcho_llm_call(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

            # Retry decorator should not have been applied
            mock_retry.assert_not_called()
            assert response.content == "No retry response"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_wait_exponential_zero_multiplier(self):
        """Test exponential backoff with zero multiplier"""
        wait_func = wait_exponential(multiplier=0.0)
        assert wait_func(1) == 0.0
        assert wait_func(2) == 0.0
        assert wait_func(5) == 0.0

    def test_wait_exponential_very_large_attempt(self):
        """Test exponential backoff with very large attempt number"""
        wait_func = wait_exponential(multiplier=1.0, max=100.0)
        # Should be clamped to max
        assert wait_func(20) == 100.0  # 2^19 would be huge but clamped to 100

    def test_stop_after_attempt_edge_cases(self):
        """Test stop predicate edge cases"""
        # Zero attempts should never retry
        stop_func = stop_after_attempt(0)
        assert stop_func(1) is False

        # One attempt should never retry (first call fails, no retry)
        stop_func = stop_after_attempt(1)
        assert stop_func(1) is False

    def test_stream_chunk_with_no_finish_reasons(self):
        """Test stream chunk creation without finish reasons"""
        chunk = HonchoLLMCallStreamChunk(content="test")
        # Should use default_factory for empty list
        assert chunk.finish_reasons == []
        # Modifying the list shouldn't affect other instances
        chunk.finish_reasons.append("stop")

        new_chunk = HonchoLLMCallStreamChunk(content="test2")
        assert new_chunk.finish_reasons == []  # Should still be empty


# Test fixtures and utilities
@pytest.fixture
def sample_test_model():
    """Fixture providing a sample SampleTestModel instance"""
    return SampleTestModel(name="Test User", age=25, active=True)


@pytest.fixture
def mock_anthropic_client():
    """Fixture providing a mocked Anthropic client"""
    mock_client = AsyncMock()
    mock_response = Mock()
    mock_response.content = [TextBlock(text="Mocked Anthropic response", type="text")]
    mock_response.usage = Usage(input_tokens=10, output_tokens=5)
    mock_response.stop_reason = "stop"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Fixture providing a mocked OpenAI client"""
    mock_client = AsyncMock()
    mock_response = ChatCompletion(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant", content="Mocked OpenAI response"
                ),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    return mock_client
