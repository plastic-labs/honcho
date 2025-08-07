from collections.abc import AsyncGenerator
from typing import Any, TypeVar

from anthropic import AsyncAnthropic
from google import genai
from groq import AsyncGroq
from openai import AsyncOpenAI
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.utils.types import Providers

clients: dict[Providers, AsyncAnthropic | AsyncOpenAI | genai.Client | AsyncGroq] = {}

if settings.LLM.ANTHROPIC_API_KEY:
    anthropic = AsyncAnthropic(api_key=settings.LLM.ANTHROPIC_API_KEY)
    clients["anthropic"] = anthropic

if settings.LLM.OPENAI_API_KEY:
    openai_client = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
    )
    clients["openai"] = openai_client

if settings.LLM.OPENAI_COMPATIBLE_BASE_URL:
    clients["custom"] = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_COMPATIBLE_API_KEY,
        base_url=settings.LLM.OPENAI_COMPATIBLE_BASE_URL,
    )

if settings.LLM.GEMINI_API_KEY:
    google = genai.Client(api_key=settings.LLM.GEMINI_API_KEY)
    clients["google"] = google

if settings.LLM.GROQ_API_KEY:
    groq = AsyncGroq(api_key=settings.LLM.GROQ_API_KEY)
    clients["groq"] = groq

providers = [
    ("Dialectic", settings.DIALECTIC.PROVIDER),
    ("Summary", settings.SUMMARY.PROVIDER),
    ("Deriver", settings.DERIVER.PROVIDER),
    ("Query Generation Provider", settings.DIALECTIC.QUERY_GENERATION_PROVIDER),
]

for provider_name, provider_value in providers:
    if provider_value not in clients:
        raise ValueError(f"Missing client for {provider_name}: {provider_value}")

T = TypeVar("T", bound=BaseModel)


async def direct_llm_call(
    prompt: str,
    provider: Providers,
    model: str,
    response_model: type[T] | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    stream: bool = False,
    # track_name: str | None = None,
) -> T | str | AsyncGenerator[str, None]:
    """
    Direct LLM call using native client libraries.

    Args:
        prompt: The prompt text
        provider: LLM provider to use
        model: Model name
        response_model: Pydantic model for structured responses
        json_mode: Enable JSON mode
        max_tokens: Maximum tokens for response
        thinking_budget_tokens: Budget for thinking tokens (Anthropic only)
        stream: Enable streaming
        track_name: Name for AI tracking

    Returns:
        Response model instance, string, or streaming generator
    """
    client = clients[provider]

    # if track_name:
    # Wrap with AI tracking
    # from functools import wraps

    # def ai_track_decorator(func):
    #     @wraps(func)
    #     async def wrapper(*args, **kwargs):
    #         return await ai_track(track_name)(func)(*args, **kwargs)
    #
    #     return wrapper

    # Handle custom provider (OpenAI-compatible)
    resolved_provider = "openai" if provider == "custom" else provider

    if resolved_provider == "google":
        return await _call_google(
            client=client,  # pyright: ignore
            prompt=prompt,
            model=model,
            response_model=response_model,
            json_mode=json_mode,
            max_tokens=max_tokens,
            stream=stream,
        )
    elif resolved_provider == "anthropic":
        return await _call_anthropic(
            client=client,  # pyright: ignore
            prompt=prompt,
            model=model,
            response_model=response_model,
            json_mode=json_mode,
            max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
            stream=stream,
        )
    else:  # openai, groq
        return await _call_openai_compatible(
            client=client,  # pyright: ignore
            prompt=prompt,
            model=model,
            response_model=response_model,
            json_mode=json_mode,
            max_tokens=max_tokens,
            stream=stream,
        )


async def _call_google(
    client: genai.Client,
    prompt: str,
    model: str,
    response_model: type[T] | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    stream: bool = False,
) -> T | str | AsyncGenerator[str, None]:
    """Google Gemini API call."""
    config: dict[str, Any] = {}
    if max_tokens:
        config["max_output_tokens"] = max_tokens

    if response_model:
        config["response_schema"] = response_model

    if json_mode:
        config["response_mime_type"] = "application/json"

    if stream:
        response = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        )

        async def stream_generator() -> AsyncGenerator[str, None]:
            async for chunk in response:
                if chunk.text:
                    yield chunk.text

        return stream_generator()
    else:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        if response_model:
            return response_model.model_validate_json(response.text)
        elif response.text is None:
            return ""
        else:
            return response.text


async def _call_anthropic(
    client: AsyncAnthropic,
    prompt: str,
    model: str,
    response_model: type[T] | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    stream: bool = False,
) -> T | str | AsyncGenerator[str, None]:
    """Anthropic Claude API call."""
    messages = [{"role": "user", "content": prompt}]

    call_params = {}
    if max_tokens:
        call_params["max_tokens"] = max_tokens

    if thinking_budget_tokens:
        call_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget_tokens,
        }

    if response_model or json_mode:
        call_params["response_format"] = {"type": "json_object"}

    if stream:
        response = await client.messages.create(
            model=model,
            messages=messages,
            stream=True,
            **call_params,
        )

        async def stream_generator():
            async for chunk in response:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    yield chunk.delta.text

        return stream_generator()
    else:
        response = await client.messages.create(
            model=model,
            messages=messages,
            **call_params,
        )

        content = response.content[0].text

        if response_model:
            return response_model.model_validate_json(content)
        else:
            return content


async def _call_openai_compatible(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    response_model: type[T] | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    stream: bool = False,
) -> T | str | AsyncGenerator[str, None]:
    """OpenAI-compatible API call (OpenAI, Groq, custom providers)."""
    messages = [{"role": "user", "content": prompt}]

    call_params: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if max_tokens:
        call_params["max_tokens"] = max_tokens

    if response_model:
        call_params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": response_model.model_json_schema(),
                "strict": True,
            },
        }
    elif json_mode:
        call_params["response_format"] = {"type": "json_object"}

    if stream:
        call_params["stream"] = True
        response = await client.chat.completions.create(**call_params)

        async def stream_generator():
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return stream_generator()
    else:
        response = await client.chat.completions.create(**call_params)
        content = response.choices[0].message.content

        if response_model:
            return response_model.model_validate_json(content or "")
        else:
            return content or ""


def create_retry_wrapper(max_attempts: int = 3):
    """Create retry decorator with exponential backoff."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )


# Keep the old honcho_llm_call for now, but mark as deprecated
# We'll remove it after all usages are migrated
