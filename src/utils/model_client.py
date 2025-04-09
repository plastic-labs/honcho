"""
Utility functions for interacting with various language model APIs.
"""

import os
from enum import Enum
from typing import Any, Optional, Protocol

import sentry_sdk
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe

# from openai import AsyncOpenAI
from langfuse.openai import AsyncOpenAI

# Load environment variables
load_dotenv()


# Supported model providers
class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"
    GROQ = "groq"
    # Add other providers as needed


# Default models for each provider
DEFAULT_MODELS = {
    ModelProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    ModelProvider.OPENAI: "gpt-4o",
    ModelProvider.OPENROUTER: "meta-llama/Llama-3.3-70B-Instruct",
    ModelProvider.CEREBRAS: "llama-3.3-70b",
    ModelProvider.GROQ: "llama-3.3-70b-versatile",
}

OPENAI_COMPATIBLE_PROVIDERS = [
    ModelProvider.OPENAI,
    ModelProvider.OPENROUTER,
    ModelProvider.CEREBRAS,
    ModelProvider.GROQ,
]

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1000


class Message(Protocol):
    """Protocol for a message that works with any provider."""

    role: str
    content: str


class ModelClient:
    """A client for interacting with various language model APIs."""

    def __init__(
        self,
        provider: ModelProvider = ModelProvider.ANTHROPIC,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the model client.

        Args:
            provider: The model provider to use
            model: The specific model to use, or None to use the default model for the provider
            api_key: The API key to use, or None to read from environment variables
            base_url: Custom base URL for the API endpoints (used for OpenRouter)
        """
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.base_url = base_url
        self.openai_client = None

        # Setup provider-specific clients
        if provider == ModelProvider.ANTHROPIC:
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key is required")
            self.client = AsyncAnthropic(api_key=self.api_key)
        elif provider in OPENAI_COMPATIBLE_PROVIDERS:
            self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
            self.base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            if not self.api_key:
                raise ValueError("OpenAI-compatible API key is required")
            self.openai_client = AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def create_message(self, role: str, content: str) -> dict[str, str]:
        """
        Create a message that works with the current provider.

        Args:
            role: The role of the message (e.g., "user", "assistant")
            content: The message content

        Returns:
            A message compatible with the current provider
        """
        # For now, just return a dictionary that works with both Anthropic and OpenAI
        return {"role": role, "content": content}

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        extra_headers: Optional[dict[str, str]] = None,
        use_caching: bool = False,
    ) -> str:
        """
        Generate a response using the configured model.

        Args:
            messages: The conversation history
            system: Optional system prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            extra_headers: Optional headers to add to the request
            use_caching: Whether to use provider-side caching for the response

        Returns:
            The generated text
        """
        with sentry_sdk.start_transaction(
            op="llm-api", name=f"{self.provider} API Call"
        ):
            # Log to langfuse
            langfuse_context.update_current_observation(
                input=messages, model=self.model
            )

            if self.provider == ModelProvider.ANTHROPIC:
                return await self._generate_anthropic(
                    messages,
                    system,
                    max_tokens,
                    temperature,
                    extra_headers,
                    use_caching,
                )
            elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                return await self._generate_openai(
                    messages, system, max_tokens, temperature
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

    @observe(as_type="generation")
    async def _generate_anthropic(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        extra_headers: Optional[dict[str, str]] = None,
        use_caching: bool = False,
    ) -> str:
        """Generate a response using the Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized.")

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Handle system prompt with caching if enabled
        if system:
            if use_caching:
                params["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                params["system"] = system

        response = await self.client.messages.create(**params)

        # Extract the text from the response
        if response.content and len(response.content) > 0:
            content_block = response.content[0]
            # Check content_block by checking for attribute 'type' instead of using isinstance
            if hasattr(content_block, "type") and content_block.type == "text":
                return content_block.text
            return str(content_block)
        return ""

    @observe(as_type="generation")
    async def _generate_openai(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Generate text using OpenAI or OpenRouter API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        # Prepare messages
        formatted_messages = []

        # Add system message if provided
        if system:
            formatted_messages.append({"role": "system", "content": system})

        # Add the rest of the messages
        formatted_messages.extend(messages)

        # Make the API call with the OpenAI client
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract the generated text
        choice = response.choices[0]
        if choice and choice.message and choice.message.content:
            return choice.message.content
        return ""

    @observe(as_type="generation")
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        extra_headers: Optional[dict[str, str]] = None,
        use_caching: bool = False,
    ) -> Any:
        """
        Stream a response using the configured model.

        Args:
            messages: The conversation history
            system: Optional system prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            extra_headers: Optional headers to add to the request
            use_caching: Whether to use provider-side caching for the response

        Returns:
            A streaming response from the provider
        """
        with sentry_sdk.start_transaction(
            op="llm-api-stream", name=f"{self.provider} API Stream"
        ):
            # Log to langfuse
            langfuse_context.update_current_observation(
                input=messages, model=self.model
            )

            if self.provider == ModelProvider.ANTHROPIC:
                return await self._stream_anthropic(
                    messages,
                    system,
                    max_tokens,
                    temperature,
                    extra_headers,
                    use_caching,
                )
            elif self.provider in OPENAI_COMPATIBLE_PROVIDERS:
                return await self._stream_openai(
                    messages, system, max_tokens, temperature
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

    async def _stream_anthropic(
        self,
        messages: list[dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        extra_headers: Optional[dict[str, str]] = None,
        use_caching: bool = False,
    ) -> Any:
        """Stream text using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized.")

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Handle system prompt with caching if enabled
        if system:
            if use_caching:
                params["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                params["system"] = system

        # Return the stream directly without awaiting it
        return self.client.messages.stream(**params)

    async def _stream_openai(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Any:
        """Stream text using OpenAI or OpenRouter API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        # Prepare messages
        formatted_messages = []

        # Add system message if provided
        if system:
            formatted_messages.append({"role": "system", "content": system})

        # Add the rest of the messages
        formatted_messages.extend(messages)

        # Make the API call with the OpenAI client
        stream = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        return stream
