"""
Utility functions for interacting with various language model APIs.
"""

import os
from enum import Enum
from typing import Any, Optional, Protocol

import sentry_sdk
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
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
    GEMINI = "gemini"
    # Add other providers as needed


# Default models for each provider
DEFAULT_MODELS = {
    ModelProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    ModelProvider.OPENAI: "gpt-4o",
    ModelProvider.OPENROUTER: "meta-llama/Llama-3.3-70B-Instruct",
    ModelProvider.CEREBRAS: "llama-3.3-70b",
    ModelProvider.GROQ: "llama-3.3-70b-versatile",
    ModelProvider.GEMINI: "gemini-2.0-flash-lite",
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
    
    _instances: dict[tuple[ModelProvider, str, str | None, str | None], 'ModelClient'] = {}

    def __new__(
        cls,
        provider: ModelProvider = ModelProvider.ANTHROPIC,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Create or return a cached ModelClient instance.
        """
        # Create a cache key from our ModelClient parameters
        cache_key = (
            provider,
            model or DEFAULT_MODELS[provider],
            api_key or cls._get_default_api_key(provider),
            base_url or cls._get_default_base_url(provider),
        )
        
        # Simple cache lookup - no locking needed in asyncio single-threaded context
        if cache_key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cache_key] = instance
        
        return cls._instances[cache_key]

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
        # Only initialize once per instance (guard against re-initialization of cached instances)
        if hasattr(self, '_initialized'):
            return
        
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.base_url = base_url
        self.openai_client = None
        self.gemini_client = None

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
        elif provider == ModelProvider.GEMINI:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("Gemini API key is required")
            self.gemini_client = genai.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Mark as initialized to prevent re-initialization
        self._initialized = True

    @staticmethod
    def _get_provider_defaults(provider: ModelProvider) -> tuple[str | None, str | None]:
        """Get default API key and base URL for provider."""
        if provider == ModelProvider.ANTHROPIC:
            return os.getenv("ANTHROPIC_API_KEY"), None
        elif provider in OPENAI_COMPATIBLE_PROVIDERS:
            return os.getenv("OPENAI_COMPATIBLE_API_KEY"), os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        elif provider == ModelProvider.GEMINI:
            return os.getenv("GEMINI_API_KEY"), None
        return None, None

    @staticmethod
    def _get_default_api_key(provider: ModelProvider) -> Optional[str]:
        """Get default API key for provider."""
        return ModelClient._get_provider_defaults(provider)[0]
    
    @staticmethod
    def _get_default_base_url(provider: ModelProvider) -> Optional[str]:
        """Get default base URL for provider."""
        return ModelClient._get_provider_defaults(provider)[1]

    @classmethod
    def test_only_clear_cache(cls) -> None:
        """
        Clear all cached clients - intended for testing purposes only to clear state between tests.
        """
        cls._instances.clear()

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

    @observe()
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
            elif self.provider == ModelProvider.GEMINI:
                return await self._generate_gemini(
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

        langfuse_context.update_current_observation(input=messages, model=self.model)

        response = await self.client.messages.create(**params)

        # Extract the text from the response
        if response.content and len(response.content) > 0:
            content_block = response.content[0]
            # Check content_block by checking for attribute 'type' instead of using isinstance
            if (
                content_block
                and hasattr(content_block, "type")
                and content_block.type == "text"
            ):
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

    @observe()
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
            elif self.provider == ModelProvider.GEMINI:
                return await self._stream_gemini(
                    messages, system, max_tokens, temperature
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

    @observe(as_type="generation")
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

        langfuse_context.update_current_observation(input=messages, model=self.model)

        # Return the stream directly without awaiting it
        return self.client.messages.stream(**params)

    @observe(as_type="generation")
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

    @observe(as_type="generation")
    async def _generate_gemini(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Generate text using Gemini API."""
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")

        # Format messages for Gemini
        gemini_messages = []

        # Convert messages to Gemini format
        for message in messages:
            role = message["role"]
            # Map roles to what Gemini expects
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                # Skip system messages as they're handled through config
                continue

            gemini_messages.append(
                genai_types.Content(
                    role=gemini_role,
                    parts=[genai_types.Part.from_text(text=message["content"])],
                )
            )

        # Set generation config
        generate_content_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain",
        )

        # Add system instruction if provided
        if system:
            generate_content_config.system_instruction = system

        # Make the API call
        if not gemini_messages:
            # If we have no messages but have a system prompt, create a default user message
            if system:
                default_content = genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_text(
                            text="Please respond based on the system instructions."
                        )
                    ],
                )
                # model = self.gemini_client.get_model(self.model)
                response = await self.gemini_client.aio.models.generate_content(
                    model=self.model,
                    contents=default_content,
                    config=generate_content_config,
                )
            else:
                raise ValueError("No messages provided for Gemini generation")
        else:
            # Normal case with messages
            # model = get_model(self.model)
            response = await self.gemini_client.aio.models.generate_content(
                model=self.model,
                contents=gemini_messages
                if len(gemini_messages) > 1
                else gemini_messages[0],
                config=generate_content_config,
            )

        # Extract text from response
        if response and response.text:
            return response.text
        return ""

    @observe(as_type="generation")
    async def _stream_gemini(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Any:
        """Stream text using Gemini API."""
        if not self.gemini_client:
            raise ValueError("Gemini client not initialized")

        # Format messages for Gemini
        gemini_messages = []

        # Convert messages to Gemini format
        for message in messages:
            role = message["role"]
            # Map roles to what Gemini expects
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            else:
                # Skip system messages as they're handled through config
                continue

            gemini_messages.append(
                genai_types.Content(
                    role=gemini_role,
                    parts=[genai_types.Part.from_text(text=message["content"])],
                )
            )

        # Set generation config
        generate_content_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="text/plain",
        )

        # Add system instruction if provided
        if system:
            generate_content_config.system_instruction = system

        # Make the streaming API call
        if not gemini_messages:
            # If we have no messages but have a system prompt, create a default user message
            if system:
                default_content = genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_text(
                            text="Please respond based on the system instructions."
                        )
                    ],
                )
                stream = await self.gemini_client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=default_content,
                    config=generate_content_config,
                )
            else:
                raise ValueError("No messages provided for Gemini streaming")
        else:
            # Normal case with messages
            stream = await self.gemini_client.aio.models.generate_content_stream(
                model=self.model,
                contents=gemini_messages
                if len(gemini_messages) > 1
                else gemini_messages[0],
                config=generate_content_config,
            )

        return stream
