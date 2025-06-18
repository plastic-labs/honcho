import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.model_client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODELS,
    DEFAULT_TEMPERATURE,
    ModelClient,
    ModelProvider,
)


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "mock-anthropic-api-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "mock-openai-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-openai-api-key")
    monkeypatch.setenv("GROQ_API_KEY", "mock-groq-api-key")
    monkeypatch.setenv("CEREBRAS_API_KEY", "mock-cerebras-api-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-api-key")


# Test fixtures
@pytest.fixture
def mock_anthropic_response():
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.type = "text"
    mock_content.text = "Test response"
    mock_response.content = [mock_content]
    return mock_response


@pytest.fixture
def mock_openai_response():
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Test response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


@pytest.fixture
def mock_anthropic_client():
    with patch("src.utils.model_client.AsyncAnthropic") as mock:
        client = MagicMock()
        client.messages.create = AsyncMock()
        client.messages.stream = AsyncMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_openai_client():
    with patch("src.utils.model_client.AsyncOpenAI") as mock:
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        mock.return_value = client
        yield client


# Initialization Tests
def test_default_initialization(mock_env):
    """Test default initialization with Anthropic provider."""
    client = ModelClient()
    assert client.provider == ModelProvider.ANTHROPIC
    assert client.model == DEFAULT_MODELS[ModelProvider.ANTHROPIC]
    assert client.base_url is None


def test_custom_model_initialization():
    """Test initialization with custom model name."""
    custom_model = "custom-model"
    client = ModelClient(model=custom_model)
    assert client.model == custom_model


def test_custom_api_key_initialization():
    """Test initialization with custom API key."""
    custom_key = "test-api-key"
    client = ModelClient(api_key=custom_key)
    assert client.api_key == custom_key


def test_custom_base_url_initialization():
    """Test initialization with custom base URL."""
    custom_url = "https://custom-api.example.com"
    client = ModelClient(base_url=custom_url)
    assert client.base_url == custom_url


def test_unsupported_provider_initialization():
    """Test initialization with unsupported provider."""
    with pytest.raises(ValueError, match="is not a valid ModelProvider"):
        ModelClient(provider=ModelProvider("unsupported"))


def test_missing_api_key_initialization():
    """Test initialization without required API key."""
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="API key is required"),
    ):
        ModelClient()


# Message Creation Tests
def test_create_message():
    """Test message creation with different roles."""
    client = ModelClient()
    message = client.create_message("user", "Hello")
    assert message == {"role": "user", "content": "Hello"}


# Generation Tests
@pytest.mark.asyncio
async def test_generate_anthropic(mock_anthropic_client, mock_anthropic_response):
    """Test generation with Anthropic provider."""
    mock_anthropic_client.messages.create.return_value = mock_anthropic_response

    client = ModelClient(provider=ModelProvider.ANTHROPIC)
    messages = [{"role": "user", "content": "Hello"}]
    response = await client.generate(messages)
    assert response == "Test response"

    # Verify the API call
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args[1]
    assert call_args["model"] == DEFAULT_MODELS[ModelProvider.ANTHROPIC]
    assert call_args["messages"] == messages
    assert call_args["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call_args["temperature"] == DEFAULT_TEMPERATURE


@pytest.mark.asyncio
async def test_generate_openai(mock_openai_client, mock_openai_response, mock_env):
    """Test generation with OpenAI provider."""
    mock_openai_client.chat.completions.create.return_value = mock_openai_response

    client = ModelClient(provider=ModelProvider.OPENAI)
    messages = [{"role": "user", "content": "Hello"}]
    response = await client.generate(messages)
    assert response == "Test response"

    # Verify the API call
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args[1]
    assert call_args["model"] == DEFAULT_MODELS[ModelProvider.OPENAI]
    assert call_args["messages"] == messages
    assert call_args["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call_args["temperature"] == DEFAULT_TEMPERATURE


@pytest.mark.asyncio
async def test_generate_with_system_prompt(
    mock_anthropic_client, mock_anthropic_response
):
    """Test generation with system prompt."""
    mock_anthropic_client.messages.create.return_value = mock_anthropic_response

    client = ModelClient(provider=ModelProvider.ANTHROPIC)
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are a helpful assistant"
    response = await client.generate(messages, system=system)
    assert response == "Test response"

    # Verify the API call
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args[1]
    assert call_args["system"] == system


@pytest.mark.asyncio
async def test_generate_with_caching(mock_anthropic_client, mock_anthropic_response):
    """Test generation with caching enabled."""
    mock_anthropic_client.messages.create.return_value = mock_anthropic_response

    client = ModelClient(provider=ModelProvider.ANTHROPIC)
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are a helpful assistant"
    response = await client.generate(messages, system=system, use_caching=True)
    assert response == "Test response"

    # Verify the API call
    mock_anthropic_client.messages.create.assert_called_once()
    call_args = mock_anthropic_client.messages.create.call_args[1]
    assert call_args["system"] == [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
