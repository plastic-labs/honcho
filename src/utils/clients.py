from collections.abc import Awaitable, Callable
from typing import (
    Any,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

from anthropic import AsyncAnthropic
from google import genai
from groq import AsyncGroq
from mirascope import llm
from mirascope.core import ResponseModelConfigDict
from mirascope.integrations.langfuse import with_langfuse
from mirascope.llm import Stream
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

P = ParamSpec("P")
T = TypeVar("T", bound=BaseModel)
T_co = TypeVar("T_co", bound=BaseModel, covariant=True)
F = TypeVar("F", bound=Callable[..., Any])


# Define protocols for different return types
@runtime_checkable
class AsyncResponseModelCallable(Protocol[P, T_co]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co: ...


@runtime_checkable
class SyncResponseModelCallable(Protocol[P, T_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co: ...


@runtime_checkable
class AsyncStreamCallable(Protocol[P]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Stream: ...


@runtime_checkable
class SyncStreamCallable(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Stream: ...


@runtime_checkable
class AsyncStringCallable(Protocol[P]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> str: ...


@runtime_checkable
class SyncStringCallable(Protocol[P]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> str: ...


@runtime_checkable
class AsyncCallResponseCallable(Protocol[P]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> llm.CallResponse: ...


# Overload for stream=True with async function
@overload
def honcho_llm_call(
    *,
    provider: Providers | None = None,
    model: str | None = None,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[True],
    **extra_call_params: Any,
) -> Callable[[Callable[P, Awaitable[Any]]], AsyncStreamCallable[P]]: ...


# Overload for response_model with async function
@overload
def honcho_llm_call(
    *,
    provider: Providers | None = None,
    model: str | None = None,
    track_name: str | None = None,
    response_model: type[T],
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    **extra_call_params: Any,
) -> Callable[[Callable[P, Awaitable[Any]]], AsyncResponseModelCallable[P, T]]: ...


# Overload for return_call_response=True with async function
@overload
def honcho_llm_call(
    *,
    provider: Providers | None = None,
    model: str | None = None,
    track_name: str | None = None,
    response_model: None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    return_call_response: Literal[True],
    **extra_call_params: Any,
) -> Callable[[Callable[P, Awaitable[Any]]], AsyncCallResponseCallable[P]]: ...


# Overload for no response_model with async function (string return)
@overload
def honcho_llm_call(
    *,
    provider: Providers | None = None,
    model: str | None = None,
    track_name: str | None = None,
    response_model: None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    return_call_response: Literal[False],
    **extra_call_params: Any,
) -> Callable[[Callable[P, Awaitable[Any]]], AsyncStringCallable[P]]: ...


# Generic overload for sync functions (fallback)
@overload
def honcho_llm_call(
    *,
    provider: Providers | None = None,
    model: str | None = None,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: bool = False,
    **extra_call_params: Any,
) -> Callable[[Callable[P, Any]], Callable[P, Any]]: ...


def honcho_llm_call(
    provider: Providers | None = None,
    model: str | None = None,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: bool = False,
    return_call_response: bool = False,  # pyright: ignore
    **extra_call_params: Any,
) -> Any:
    """
    Consolidated decorator for LLM calls that handles provider-specific configurations.

    This decorator automatically:
    - Handles both sync and async functions seamlessly
    - Applies retry logic with exponential backoff
    - Adds AI tracking for Sentry
    - Integrates with Langfuse for observability
    - Builds provider-specific call parameters
    - Handles client selection from the global clients dict

    Args:
        provider: The LLM provider to use (e.g., "anthropic", "google", "openai")
        model: The model to use
        track_name: Name for AI tracking (e.g., "Critical Analysis Call")
        response_model: Optional Pydantic model for structured responses
        json_mode: Whether to enable JSON mode (for providers that support it)
        max_tokens: Maximum tokens for the response
        thinking_budget_tokens: Budget for thinking tokens (Anthropic only)
        enable_retry: Whether to enable retry logic (default: True)
        retry_attempts: Number of retry attempts (default: 3)
        stream: Whether to enable streaming responses (default: False)
        _return_call_response: Whether to return the full CallResponse object (default: False)
        **extra_call_params: Additional provider-specific parameters

    Returns:
        A decorator that returns:
        - For async functions: Callable[P, Awaitable[T]] where T is Stream, response_model, CallResponse, or str
        - For sync functions: Callable[P, T] where T is Stream, response_model, CallResponse, or str

    Note: Type annotations may be needed at the call site for proper type checking.

    Example (async function):
        @honcho_llm_call(
            provider=settings.DERIVER.PROVIDER,
            model=settings.DERIVER.MODEL,
            track_name="Critical Analysis Call",
            response_model=ReasoningResponse,
            json_mode=True,
            max_tokens=settings.DERIVER.MAX_OUTPUT_TOKENS,
        )
        async def analyze(context: str, query: str):
            return prompt_template(context, query)

    Example (sync function):
        @honcho_llm_call(
            provider="openai",
            model="gpt-4",
            max_tokens=1000,
        )
        def generate_summary(text: str) -> str:
            return f"Summarize: {text}"

        # Call synchronously
        result = generate_summary("Long text here...")
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Handle special case for custom provider
        # Custom providers use OpenAI-compatible endpoints, so we resolve to "openai" for the provider name
        # but keep the original "custom" for client lookup
        resolved_provider = "openai" if provider == "custom" else provider

        # Build provider-specific call params
        call_params: dict[str, Any] = {}

        if resolved_provider == "google":
            # Google uses 'config' parameter
            config: dict[str, Any] = {}
            if max_tokens:
                config["max_output_tokens"] = max_tokens

            if response_model:
                config["response_schema"] = response_model

            if json_mode:
                config["response_mime_type"] = "application/json"

            if config:
                call_params["config"] = config
        elif resolved_provider == "anthropic":
            # Anthropic uses thinking params and max_tokens
            if thinking_budget_tokens:
                call_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
            if max_tokens:
                call_params["max_tokens"] = max_tokens
        else:
            # Other providers just use max_tokens
            if max_tokens:
                call_params["max_tokens"] = max_tokens

        # Merge with any extra call params
        # Remove return_call_response from extra_call_params --
        # that one is just for our type system.
        extra_call_params.pop("return_call_response", None)
        call_params.update(extra_call_params)

        # Build kwargs for llm.call
        llm_kwargs: dict[str, Any] = {}
        if resolved_provider and provider:
            llm_kwargs["provider"] = resolved_provider
            llm_kwargs["client"] = clients[
                provider
            ]  # Use original provider for client lookup
        if model:
            llm_kwargs["model"] = model
        if response_model:
            # https://mirascope.com/docs/mirascope/learn/provider-specific/openai#response-models
            if resolved_provider == "openai":
                response_model.model_config = ResponseModelConfigDict(strict=True)
            llm_kwargs["response_model"] = response_model
        if json_mode:
            llm_kwargs["json_mode"] = json_mode
        if stream:
            llm_kwargs["stream"] = stream
        if call_params:
            llm_kwargs["call_params"] = call_params

        # Apply decorators in order
        decorated: Any = func

        # Apply llm.call
        decorated = llm.call(**llm_kwargs)(decorated)  # pyright: ignore

        # Apply langfuse if enabled
        if settings.LANGFUSE_PUBLIC_KEY:
            decorated = with_langfuse()(decorated)  # pyright: ignore

        # Apply AI tracking if name provided
        if track_name:
            decorated = ai_track(track_name)(decorated)

        # Apply retry logic if enabled
        if enable_retry:
            decorated = retry(  # pyright: ignore
                stop=stop_after_attempt(retry_attempts),
                wait=wait_exponential(multiplier=1, min=4, max=10),
            )(decorated)  # pyright: ignore

        return decorated  # pyright: ignore

    return decorator
