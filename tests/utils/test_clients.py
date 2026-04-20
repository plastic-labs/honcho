"""
Comprehensive tests for the public src.llm orchestration surface.

Tests cover:
- All supported LLM providers (Anthropic, OpenAI, Google/Gemini)
- Streaming and non-streaming responses
- Response models (structured output)
- Error handling and retries
- Provider-specific features
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

from src.config import ConfiguredModelSettings, ModelConfig, ResolvedFallbackConfig
from src.exceptions import LLMError, ValidationException
from src.llm import (
    CLIENTS,
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    honcho_llm_call,
    honcho_llm_call_inner,
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
                thinking_budget_tokens=1024,
            )

            # Verify thinking parameter was passed
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args
            thinking_config = call_args.kwargs["thinking"]
            assert thinking_config == {"type": "enabled", "budget_tokens": 1024}

    async def test_anthropic_streaming(self):
        """Test Anthropic streaming response"""

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

        # Mock final message with usage tokens
        mock_usage = Mock(output_tokens=42)
        mock_final_message = Mock(stop_reason="stop", usage=mock_usage)
        mock_stream.get_final_message.return_value = mock_final_message

        mock_client.messages.stream.return_value = mock_stream

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            stream = await honcho_llm_call_inner(
                provider="anthropic",
                model="claude-3-sonnet",
                prompt="Hello",
                max_tokens=100,
                stream=True,
                client_override=mock_client,
                messages=[{"role": "user", "content": "Hello"}],
            )
            async for chunk in stream:
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
            stream = await honcho_llm_call_inner(
                provider="openai",
                model="gpt-4",
                prompt="Hello",
                max_tokens=100,
                stream=True,
                client_override=mock_client,
                messages=[{"role": "user", "content": "Hello"}],
            )
            async for chunk in stream:
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
        # Mock the parts structure that the code expects
        mock_part = Mock()
        mock_part.text = "Hello from Gemini"
        mock_part.function_call = None
        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = mock_finish_reason
        mock_response.candidates = [mock_candidate]
        # Mock the usage_metadata with both prompt_token_count and candidates_token_count
        mock_usage_metadata = Mock()
        mock_usage_metadata.prompt_token_count = 3
        mock_usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata = mock_usage_metadata
        # Mock the async aio interface
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            response = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-1.5-pro",
                prompt="Hello",
                max_tokens=100,
            )

            assert isinstance(response, HonchoLLMCallResponse)
            assert response.content == "Hello from Gemini"
            assert response.input_tokens == 3
            assert response.output_tokens == 5
            assert response.finish_reasons == ["STOP"]

            # Verify max output token cap is passed through
            mock_aio.models.generate_content.assert_called_once()
            call_args = mock_aio.models.generate_content.call_args
            assert call_args.kwargs["config"]["max_output_tokens"] == 100

    async def test_google_json_mode(self):
        """Test Google/Gemini with JSON mode"""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        # Mock the parts structure that the code expects
        mock_part = Mock()
        mock_part.text = '{"result": "success"}'
        mock_part.function_call = None
        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = mock_finish_reason
        mock_response.candidates = [mock_candidate]
        # Mock the usage_metadata with both prompt_token_count and candidates_token_count
        mock_usage_metadata = Mock()
        mock_usage_metadata.prompt_token_count = 5
        mock_usage_metadata.candidates_token_count = 10
        mock_response.usage_metadata = mock_usage_metadata
        # Mock the async aio interface
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            _response = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-1.5-pro",
                prompt="Generate JSON",
                max_tokens=100,
                json_mode=True,
            )

            # Verify JSON mode was set in config
            mock_aio.models.generate_content.assert_called_once()
            call_args = mock_aio.models.generate_content.call_args
            config = call_args.kwargs["config"]
            assert config["response_mime_type"] == "application/json"
            assert config["max_output_tokens"] == 100

    async def test_google_response_model(self):
        """Test Google/Gemini with structured output"""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_parsed = SampleTestModel(name="Alice", age=25)
        mock_response.parsed = mock_parsed
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_response.candidates = [Mock(finish_reason=mock_finish_reason)]
        # Mock the usage_metadata with both prompt_token_count and candidates_token_count
        mock_usage_metadata = Mock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 15
        mock_response.usage_metadata = mock_usage_metadata
        # Mock the async aio interface
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            response = await honcho_llm_call_inner(
                provider="gemini",
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
            mock_aio.models.generate_content.assert_called_once()
            call_args = mock_aio.models.generate_content.call_args
            config = call_args.kwargs["config"]
            assert config["response_mime_type"] == "application/json"
            assert config["response_schema"] == SampleTestModel
            assert config["max_output_tokens"] == 100

    async def test_google_streaming(self):
        """Test Google/Gemini streaming response"""
        from google import genai

        mock_client = Mock(spec=genai.Client)

        # Mock streaming chunks
        mock_finish_reason = Mock()
        mock_finish_reason.name = "STOP"
        mock_usage_metadata = Mock(candidates_token_count=35)
        mock_chunks = [
            Mock(text="Hello"),
            Mock(text=" world"),
            Mock(
                text="",
                candidates=[Mock(finish_reason=mock_finish_reason)],
                usage_metadata=mock_usage_metadata,
            ),
        ]

        # Create async iterator for the chunks
        async def async_chunk_iterator():
            for chunk in mock_chunks:
                yield chunk

        # Mock the aio.models.generate_content_stream method to return an awaitable async iterator
        mock_aio = Mock()
        mock_aio.models.generate_content_stream = AsyncMock(
            return_value=async_chunk_iterator()
        )
        mock_client.aio = mock_aio

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            stream = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-1.5-pro",
                prompt="Hello",
                max_tokens=100,
                stream=True,
                client_override=mock_client,
                messages=[{"role": "user", "content": "Hello"}],
            )
            async for chunk in stream:
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0].content == "Hello"
            assert chunks[1].content == " world"
            assert chunks[2].content == ""
            assert chunks[2].is_done is True
            assert chunks[2].finish_reasons == ["STOP"]

            # Verify streaming call includes max output token cap
            mock_aio.models.generate_content_stream.assert_called_once()
            call_args = mock_aio.models.generate_content_stream.call_args
            assert call_args.kwargs["config"]["max_output_tokens"] == 100

    async def test_google_no_candidates_fallback(self):
        """Test Google/Gemini fallback when no candidates"""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_response.candidates = []  # Empty candidates
        # Mock usage_metadata as None to test fallback
        mock_response.usage_metadata = None
        # Mock the async aio interface
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            response = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-1.5-pro",
                prompt="Hello",
                max_tokens=100,
            )

            # With empty candidates, content should be empty and defaults should be used
            assert response.content == ""
            assert response.output_tokens == 0  # Fallback value
            assert response.finish_reasons == ["stop"]  # Default fallback

    @pytest.mark.parametrize(
        "finish_reason", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"]
    )
    async def test_google_blocked_response_raises_error(self, finish_reason: str):
        """Test that blocked Gemini responses raise LLMError for retry/failover."""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        # Blocked responses typically have candidates with no content
        mock_finish_reason = Mock()
        mock_finish_reason.name = finish_reason
        mock_candidate = Mock()
        mock_candidate.content = None
        mock_candidate.finish_reason = mock_finish_reason
        mock_response.candidates = [mock_candidate]
        mock_usage_metadata = Mock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 0
        mock_response.usage_metadata = mock_usage_metadata
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with (
            patch.dict(CLIENTS, {"gemini": mock_client}),
            pytest.raises(LLMError, match=f"finish_reason={finish_reason}"),
        ):
            await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="Summarize this",
                max_tokens=1000,
            )

    async def test_google_max_tokens_empty_does_not_raise(self):
        """Test that MAX_TOKENS with empty content returns normally (not a blocked response)."""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_finish_reason = Mock()
        mock_finish_reason.name = "MAX_TOKENS"
        mock_candidate = Mock()
        mock_candidate.content = None
        mock_candidate.finish_reason = mock_finish_reason
        mock_response.candidates = [mock_candidate]
        mock_usage_metadata = Mock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 0
        mock_response.usage_metadata = mock_usage_metadata
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            response = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="Hello",
                max_tokens=100,
            )
            # MAX_TOKENS is not a blocked reason — returns empty content without raising
            assert response.content == ""
            assert response.finish_reasons == ["MAX_TOKENS"]

    async def test_google_blocked_response_model_raises_error(self):
        """Test that blocked responses in the response_model path raise LLMError."""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_response.parsed = None
        mock_finish_reason = Mock()
        mock_finish_reason.name = "SAFETY"
        mock_response.candidates = [Mock(finish_reason=mock_finish_reason)]
        mock_usage_metadata = Mock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 0
        mock_response.usage_metadata = mock_usage_metadata
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with (
            patch.dict(CLIENTS, {"gemini": mock_client}),
            pytest.raises(LLMError, match="finish_reason=SAFETY"),
        ):
            await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="Generate a person",
                max_tokens=100,
                response_model=SampleTestModel,
            )

    async def test_google_blocked_finish_reason_with_valid_parsed_does_not_raise(self):
        """Blocked finish_reason should not raise if parsed content is valid."""
        from google import genai

        mock_client = Mock(spec=genai.Client)
        mock_response = Mock()
        mock_response.parsed = SampleTestModel(name="Alice", age=30, active=True)
        mock_finish_reason = Mock()
        mock_finish_reason.name = "SAFETY"
        mock_response.candidates = [Mock(finish_reason=mock_finish_reason)]
        mock_usage_metadata = Mock()
        mock_usage_metadata.prompt_token_count = 10
        mock_usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata = mock_usage_metadata
        mock_aio = Mock()
        mock_aio.models.generate_content = AsyncMock(return_value=mock_response)
        mock_client.aio = mock_aio

        with patch.dict(CLIENTS, {"gemini": mock_client}):
            response = await honcho_llm_call_inner(
                provider="gemini",
                model="gemini-2.5-flash",
                prompt="Generate a person",
                max_tokens=100,
                response_model=SampleTestModel,
            )

        assert isinstance(response.content, SampleTestModel)
        assert response.content.name == "Alice"
        assert response.finish_reasons == ["SAFETY"]


@pytest.mark.asyncio
class TestMainLLMCallFunction:
    """Tests for the main honcho_llm_call function"""

    async def test_streaming_call(self):
        """Test streaming LLM call"""

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_stream = AsyncMock()

        # Mock streaming chunks
        mock_chunks = [
            Mock(type="content_block_delta", delta=Mock(text="Stream")),
            Mock(type="content_block_delta", delta=Mock(text=" test")),
        ]
        mock_stream.__aenter__.return_value = mock_stream
        mock_stream.__aiter__.return_value = iter(mock_chunks)

        # Mock final message with usage tokens
        mock_usage = Mock(output_tokens=28)
        mock_final_message = Mock(stop_reason="stop", usage=mock_usage)
        mock_stream.get_final_message.return_value = mock_final_message

        mock_client.messages.stream.return_value = mock_stream

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            chunks: list[HonchoLLMCallStreamChunk] = []
            async for chunk in await honcho_llm_call(
                model_config=ConfiguredModelSettings(
                    model="claude-4-sonnet",
                    transport="anthropic",
                ),
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

        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text="No retry response", type="text")]
        mock_response.usage = Usage(input_tokens=5, output_tokens=5)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            response = await honcho_llm_call(
                model_config=ConfiguredModelSettings(
                    model="claude-4-sonnet",
                    transport="anthropic",
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

            assert response.content == "No retry response"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_stream_chunk_with_no_finish_reasons(self):
        """Test stream chunk creation without finish reasons"""
        chunk = HonchoLLMCallStreamChunk(content="test")
        # Should use default_factory for empty list
        assert chunk.finish_reasons == []
        # Modifying the list shouldn't affect other instances
        chunk.finish_reasons.append("stop")

        new_chunk = HonchoLLMCallStreamChunk(content="test2")
        assert new_chunk.finish_reasons == []  # Should still be empty


@pytest.mark.asyncio
class TestModelConfigCalls:
    async def test_honcho_llm_call_accepts_model_config(self):
        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text="ModelConfig response", type="text")]
        mock_response.usage = Usage(input_tokens=8, output_tokens=4)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            response = await honcho_llm_call(
                model_config=ModelConfig(
                    model="claude-haiku-4-5",
                    transport="anthropic",
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

            assert response.content == "ModelConfig response"
            await_args = mock_client.messages.create.await_args
            if await_args is None:
                raise AssertionError("Expected Anthropic create call")
            call_args = await_args.kwargs
            assert call_args["model"] == "claude-haiku-4-5"

    async def test_honcho_llm_call_accepts_configured_model_settings(self):
        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [
            TextBlock(text="ConfiguredModelSettings response", type="text")
        ]
        mock_response.usage = Usage(input_tokens=8, output_tokens=4)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            response = await honcho_llm_call(
                model_config=ConfiguredModelSettings(
                    model="claude-haiku-4-5",
                    transport="anthropic",
                    thinking_budget_tokens=1024,
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

            assert response.content == "ConfiguredModelSettings response"
            await_args = mock_client.messages.create.await_args
            if await_args is None:
                raise AssertionError("Expected Anthropic create call")
            call_args = await_args.kwargs
            assert call_args["model"] == "claude-haiku-4-5"
            assert call_args["thinking"] == {
                "type": "enabled",
                "budget_tokens": 1024,
            }


@pytest.mark.asyncio
class TestModelConfigExtraParamsPropagation:
    """Regression tests — config knobs must reach the backend.

    Prior to the fix, honcho_llm_call_inner built extra_params from only
    {json_mode, verbosity}, silently dropping top_p/top_k/frequency_penalty/
    presence_penalty/seed/provider_params off the ModelConfig. These tests
    lock in that each backend now receives them.
    """

    async def test_openai_propagates_top_p_frequency_seed(self):
        from openai import AsyncOpenAI

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4.1",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"openai": mock_client}):
            await honcho_llm_call(
                model_config=ModelConfig(
                    model="gpt-4.1",
                    transport="openai",
                    top_p=0.92,
                    frequency_penalty=0.5,
                    presence_penalty=0.1,
                    seed=42,
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

            mock_client.chat.completions.create.assert_called_once()
            kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert kwargs["top_p"] == 0.92
            assert kwargs["frequency_penalty"] == 0.5
            assert kwargs["presence_penalty"] == 0.1
            assert kwargs["seed"] == 42

    async def test_anthropic_propagates_top_p_top_k(self):
        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text="ok", type="text")]
        mock_response.usage = Usage(input_tokens=8, output_tokens=4)
        mock_response.stop_reason = "stop"
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            await honcho_llm_call(
                model_config=ModelConfig(
                    model="claude-haiku-4-5",
                    transport="anthropic",
                    top_p=0.85,
                    top_k=40,
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

            await_args = mock_client.messages.create.await_args
            if await_args is None:
                raise AssertionError("Expected Anthropic create call")
            kwargs = await_args.kwargs
            assert kwargs["top_p"] == 0.85
            assert kwargs["top_k"] == 40

    async def test_provider_params_passthrough(self):
        """Operator-supplied provider_params must reach the backend's extra_params.

        Scope: verifies the ModelConfig.provider_params → backend.extra_params
        boundary inside honcho_llm_call_inner. This is NOT a guarantee that
        arbitrary keys reach the provider SDK — each backend's _build_params
        forwards only an allowlist (top_p, top_k, frequency_penalty, seed,
        etc.). We assert only that the sentinel key arrives in extra_params
        at the backend boundary, which is the internal contract this test
        exists to protect.
        """
        from openai import AsyncOpenAI

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4.1",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        captured_extra: dict[str, Any] = {}

        from src.llm.backends.openai import OpenAIBackend

        original_complete = OpenAIBackend.complete

        async def capture_extra(self: Any, **kwargs: Any) -> Any:
            captured_extra.update(kwargs.get("extra_params") or {})
            return await original_complete(self, **kwargs)

        with (
            patch.dict(CLIENTS, {"openai": mock_client}),
            patch.object(OpenAIBackend, "complete", capture_extra),
        ):
            await honcho_llm_call(
                model_config=ModelConfig(
                    model="gpt-4.1",
                    transport="openai",
                    provider_params={"honcho_sentinel": "zap"},
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

            assert captured_extra.get("honcho_sentinel") == "zap"

    async def test_cache_policy_reaches_gemini_backend(self):
        """PromptCachePolicy set on ModelConfig must reach the Gemini backend's
        extra_params as a typed object (so gemini_cached_content reuse fires)."""
        from google import genai

        from src.config import PromptCachePolicy
        from src.llm.backends.gemini import GeminiBackend

        mock_client = Mock(spec=genai.Client)
        mock_client.__class__ = genai.Client  # pyright: ignore[reportAttributeAccessIssue]

        import contextlib

        captured_extra: dict[str, Any] = {}

        async def capture_extra(_self: Any, **kwargs: Any) -> Any:
            captured_extra.update(kwargs.get("extra_params") or {})
            return None

        policy = PromptCachePolicy(mode="gemini_cached_content", ttl_seconds=300)

        with (
            patch.dict(CLIENTS, {"gemini": mock_client}),
            patch.object(GeminiBackend, "complete", capture_extra),
            # capture_extra returns None, so downstream normalization will raise;
            # we only care that extra_params was observed pre-raise.
            contextlib.suppress(Exception),
        ):
            await honcho_llm_call(
                model_config=ModelConfig(
                    model="gemini-2.5-flash",
                    transport="gemini",
                    cache_policy=policy,
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=False,
            )

        assert captured_extra.get("cache_policy") is policy

    async def test_per_call_kwargs_override_provider_params(self):
        """json_mode/verbosity from honcho_llm_call must win over provider_params defaults."""
        from openai import AsyncOpenAI

        from src.llm.backends.openai import OpenAIBackend

        mock_client = AsyncMock(spec=AsyncOpenAI)
        mock_response = ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="gpt-4.1",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="{}"),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        captured_extra: dict[str, Any] = {}
        original_complete = OpenAIBackend.complete

        async def capture_extra(self: Any, **kwargs: Any) -> Any:
            captured_extra.update(kwargs.get("extra_params") or {})
            return await original_complete(self, **kwargs)

        with (
            patch.dict(CLIENTS, {"openai": mock_client}),
            patch.object(OpenAIBackend, "complete", capture_extra),
        ):
            await honcho_llm_call(
                model_config=ModelConfig(
                    model="gpt-4.1",
                    transport="openai",
                    provider_params={"json_mode": False, "verbosity": "low"},
                ),
                prompt="Hello",
                max_tokens=100,
                json_mode=True,
                verbosity="high",
                enable_retry=False,
            )

            assert captured_extra["json_mode"] is True
            assert captured_extra["verbosity"] == "high"

    async def test_fallback_config_thinking_params_applied_on_final_retry(
        self,
    ) -> None:
        """When primary fails, the FALLBACK ModelConfig's own temperature and
        thinking_budget_tokens must reach the backend on the final retry —
        not the primary's values, and not whatever the caller never set.

        Regression for the 'default caller kwargs from runtime_model_config too
        early' bug: if honcho_llm_call pre-populated temperature from
        runtime_model_config (the primary) before attempt selection, those
        primary values would clobber the fallback's own thinking params via
        effective_config_for_call(update={...}).
        """
        mock_client = AsyncMock(spec=AsyncAnthropic)
        mock_response = Mock()
        mock_response.content = [TextBlock(text="from fallback", type="text")]
        mock_response.usage = Usage(input_tokens=5, output_tokens=3)
        mock_response.stop_reason = "stop"

        # Primary fails twice, then fallback succeeds on attempt 3.
        mock_client.messages.create = AsyncMock(
            side_effect=[
                RuntimeError("primary attempt 1"),
                RuntimeError("primary attempt 2"),
                mock_response,
            ]
        )

        fallback = ResolvedFallbackConfig(
            model="claude-haiku-4-5",
            transport="anthropic",
            temperature=0.9,
            thinking_budget_tokens=2048,
        )

        with patch.dict(CLIENTS, {"anthropic": mock_client}):
            await honcho_llm_call(
                model_config=ModelConfig(
                    model="claude-sonnet-4-5",
                    transport="anthropic",
                    temperature=0.1,
                    thinking_budget_tokens=1024,
                    fallback=fallback,
                ),
                prompt="Hello",
                max_tokens=100,
                enable_retry=True,
                retry_attempts=3,
            )

            # Final call should carry the FALLBACK's values, not primary's.
            final_call = mock_client.messages.create.await_args_list[-1]
            kwargs = final_call.kwargs
            assert kwargs["model"] == "claude-haiku-4-5"
            assert kwargs["temperature"] == 0.9
            assert kwargs["thinking"] == {
                "type": "enabled",
                "budget_tokens": 2048,
            }


@pytest.mark.asyncio
class TestToolLoopValidation:
    """Lock in the fail-fast behavior on max_tool_iterations out of range."""

    @pytest.mark.parametrize("bad_value", [0, -1, 101, 1_000])
    async def test_invalid_max_tool_iterations_raises(self, bad_value: int) -> None:
        from src.llm.tool_loop import execute_tool_loop

        def _noop_plan() -> Any:  # pragma: no cover - never called
            raise AssertionError("plan should not be invoked for invalid input")

        def _noop_executor(
            _name: str, _input: dict[str, Any]
        ) -> str:  # pragma: no cover
            return "ok"

        def _noop_retry_callback(_state: Any) -> None:  # pragma: no cover
            return None

        with pytest.raises(ValidationException, match="max_tool_iterations"):
            await execute_tool_loop(
                prompt="x",
                max_tokens=10,
                messages=None,
                tools=[{"name": "t", "description": "d", "input_schema": {}}],
                tool_choice=None,
                tool_executor=_noop_executor,
                max_tool_iterations=bad_value,
                response_model=None,
                json_mode=False,
                temperature=None,
                stop_seqs=None,
                verbosity=None,
                enable_retry=False,
                retry_attempts=3,
                max_input_tokens=None,
                get_attempt_plan=_noop_plan,
                before_retry_callback=_noop_retry_callback,
            )
