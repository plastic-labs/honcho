import os

import pytest
from dotenv import load_dotenv

from src.utils.model_client import DEFAULT_MODELS, ModelClient, ModelProvider

load_dotenv()

@pytest.mark.asyncio
async def test_anthropic_api_behavior():
    """Test that our ModelClient correctly maps to Anthropic's API behavior."""
    model = DEFAULT_MODELS[ModelProvider.ANTHROPIC]
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = ModelClient(provider=ModelProvider.ANTHROPIC, model=model, api_key=api_key)
    
    # Test message format
    messages = [{"role": "user", "content": "Hello"}]
    response = await client.generate(messages)
    assert response is not None
    
    # Test system prompt format
    system = "You are a helpful assistant"
    response = await client.generate(messages, system=system)
    assert response is not None
    
    # Test streaming behavior
    stream = await client.stream(messages)  # Get the stream first
    async with stream as stream_manager:  # Use it as a context manager
        async for chunk in stream_manager:  # Then iterate over it
            assert chunk is not None
            # Verify the chunk has the expected structure
            if hasattr(chunk, 'delta'):
                assert chunk.delta is not None
                if hasattr(chunk.delta, 'text'):
                    assert chunk.delta.text is not None
            elif hasattr(chunk, 'content'):
                assert chunk.content is not None

@pytest.mark.asyncio
async def test_openai_api_behavior():
    """Test that our ModelClient correctly maps to OpenAI's API behavior."""
    model = DEFAULT_MODELS[ModelProvider.OPENAI]
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = ModelClient(provider=ModelProvider.OPENAI, model=model, api_key=api_key, base_url=base_url)
    
    # Test message format
    messages = [{"role": "user", "content": "Hello"}]
    response = await client.generate(messages)
    assert response is not None
    
    # Test system prompt format
    system = "You are a helpful assistant"
    response = await client.generate(messages, system=system)
    assert response is not None
    
    # Test streaming behavior
    stream = await client.stream(messages)
    async for chunk in stream:
        assert chunk is not None

@pytest.mark.asyncio
async def test_multiple_providers():
    providers = [ModelProvider.ANTHROPIC, ModelProvider.OPENAI, ModelProvider.GROQ, ModelProvider.CEREBRAS, ModelProvider.OPENROUTER]

    api_key_env_vars = {
        ModelProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        ModelProvider.OPENAI: "OPENAI_API_KEY",
        ModelProvider.GROQ: "GROQ_API_KEY",
        ModelProvider.CEREBRAS: "CEREBRAS_API_KEY",
        ModelProvider.OPENROUTER: "OPENROUTER_API_KEY",
    }

    base_url_env_vars = {
        ModelProvider.OPENAI: "OPENAI_BASE_URL",
        ModelProvider.OPENROUTER: "OPENROUTER_BASE_URL",
        ModelProvider.CEREBRAS: "CEREBRAS_BASE_URL",
        ModelProvider.GROQ: "GROQ_BASE_URL",
    }
    
    for provider in providers:
        model = DEFAULT_MODELS[provider]
        api_key = os.getenv(api_key_env_vars[provider])
        base_url_env_var = base_url_env_vars.get(provider)
        base_url = os.getenv(base_url_env_var) if base_url_env_var else None
        client = ModelClient(provider=provider, model=model, api_key=api_key, base_url=base_url)
        messages = [{"role": "user", "content": "Hello"}]
        response = await client.generate(messages)
        assert response is not None