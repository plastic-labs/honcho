import asyncio
import builtins as _builtins
import inspect
import time
from collections.abc import AsyncIterator, Callable
from functools import wraps
from typing import Any, Generic, Literal, TypeVar, cast, overload

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock
from anthropic.types.message import Message as AnthropicMessage
from google import genai
from google.genai.types import GenerateContentResponse
from groq import AsyncGroq
from langfuse import get_client
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel
from sentry_sdk.ai.monitoring import ai_track

from src.config import settings
from src.utils.types import SupportedProviders

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)

lf = get_client()

CLIENTS: dict[
    SupportedProviders,
    AsyncAnthropic | AsyncOpenAI | genai.Client | AsyncGroq,
] = {}

if settings.LLM.ANTHROPIC_API_KEY:
    anthropic = AsyncAnthropic(api_key=settings.LLM.ANTHROPIC_API_KEY)
    CLIENTS["anthropic"] = anthropic

if settings.LLM.OPENAI_API_KEY:
    openai_client = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
    )
    CLIENTS["openai"] = openai_client

if settings.LLM.OPENAI_COMPATIBLE_BASE_URL:
    CLIENTS["custom"] = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_COMPATIBLE_API_KEY,
        base_url=settings.LLM.OPENAI_COMPATIBLE_BASE_URL,
    )

if settings.LLM.GEMINI_API_KEY:
    google = genai.client.Client(api_key=settings.LLM.GEMINI_API_KEY)
    CLIENTS["google"] = google

if settings.LLM.GROQ_API_KEY:
    groq = AsyncGroq(api_key=settings.LLM.GROQ_API_KEY)
    CLIENTS["groq"] = groq

SELECTED_PROVIDERS = [
    ("Dialectic", settings.DIALECTIC.PROVIDER),
    ("Summary", settings.SUMMARY.PROVIDER),
    ("Deriver", settings.DERIVER.PROVIDER),
    ("Query Generation Provider", settings.DIALECTIC.QUERY_GENERATION_PROVIDER),
]

for provider_name, provider_value in SELECTED_PROVIDERS:
    if provider_value not in CLIENTS:
        raise ValueError(f"Missing client for {provider_name}: {provider_value}")


class HonchoLLMCallResponse(BaseModel, Generic[T]):
    """
    Response object for LLM calls.

    Args:
        content: The response content. When a response_model is provided, this will be
                the parsed object of that type. Otherwise, it will be a string.
        output_tokens: Number of tokens generated in the response.
        finish_reasons: List of finish reasons for the response.
    """

    content: T
    output_tokens: int
    finish_reasons: list[str]


class HonchoLLMCallStreamChunk(BaseModel):
    """
    A single chunk in a streaming LLM response.

    Args:
        content: The text content for this chunk. Empty for chunks that only contain metadata.
        is_done: Whether this is the final chunk in the stream.
        finish_reasons: List of finish reasons if the stream is complete.
    """

    content: str
    is_done: bool = False
    finish_reasons: list[str] = []


@overload
async def honcho_llm_call(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    *,
    response_model: type[M],
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: None = None,
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[True] = ...,
) -> AsyncIterator[HonchoLLMCallStreamChunk]: ...


async def honcho_llm_call(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: bool = False,
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    client = CLIENTS.get(provider)
    if not client:
        raise ValueError(f"Missing client for {provider}")

    decorated = honcho_llm_call_inner

    # apply langfuse if enabled
    if settings.LANGFUSE_PUBLIC_KEY:
        decorated = with_langfuse(decorated)

    # apply tracking
    if track_name:
        decorated = ai_track(track_name)(decorated)

    # apply retry logic
    if enable_retry:
        decorated = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )(decorated)

    if stream:
        return await decorated(
            provider,
            model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
            True,
        )
    else:
        return await decorated(
            provider,
            model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
            False,
        )


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[M],
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: Literal[False] = False,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: None = None,
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: Literal[False] = False,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: Literal[True] = ...,
) -> AsyncIterator[HonchoLLMCallStreamChunk]: ...


async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: bool = False,
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    # has already been validated by honcho_llm_call
    client = CLIENTS[provider]

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }

    if stream:
        # Return async generator for streaming responses
        return _handle_streaming_response(
            client, params, json_mode, thinking_budget_tokens, response_model
        )

    # Remove stream parameter for non-streaming calls as some providers don't accept it
    params.pop("stream", None)

    match client:
        case AsyncAnthropic():
            anthropic_params: dict[str, Any] = {
                "model": params["model"],
                "max_tokens": params["max_tokens"],
                "messages": list(params["messages"]),
            }
            if json_mode:
                anthropic_params["messages"].append(
                    {"role": "assistant", "content": "{"}
                )
            if thinking_budget_tokens:
                anthropic_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
            anthropic_response: AnthropicMessage = await client.messages.create(  # pyright: ignore
                **anthropic_params
            )
            # Extract text content from content blocks
            text_blocks: list[str] = []
            for block in anthropic_response.content:  # pyright: ignore
                if isinstance(block, TextBlock):
                    text_blocks.append(block.text)

            # Safely extract usage and stop_reason
            usage = anthropic_response.usage  # pyright: ignore
            stop_reason = anthropic_response.stop_reason  # pyright: ignore

            return HonchoLLMCallResponse(
                content="\n".join(text_blocks),
                output_tokens=usage.output_tokens if usage else 0,  # pyright: ignore
                finish_reasons=[stop_reason] if stop_reason else [],
            )
        case AsyncOpenAI():
            openai_params: dict[str, Any] = {
                "model": params["model"],
                "messages": params["messages"],
            }
            if "gpt-5" in model:
                openai_params["max_completion_tokens"] = params["max_tokens"]
                if reasoning_effort:
                    openai_params["reasoning_effort"] = reasoning_effort
                if verbosity:
                    openai_params["verbosity"] = verbosity
            else:
                openai_params["max_tokens"] = params["max_tokens"]
            if json_mode:
                openai_params["response_format"] = {"type": "json_object"}
            if response_model:
                openai_params["response_format"] = response_model
                response: ChatCompletion = await client.chat.completions.parse(  # pyright: ignore
                    **openai_params
                )
                # Extract the parsed object for structured output
                parsed_content = response.choices[0].message.parsed
                if parsed_content is None:
                    raise ValueError("No parsed content in structured response")

                # Safely extract usage and finish_reason
                usage = response.usage
                finish_reason = response.choices[0].finish_reason

                return HonchoLLMCallResponse(
                    content=parsed_content,
                    output_tokens=usage.completion_tokens if usage else 0,
                    finish_reasons=[finish_reason] if finish_reason else [],
                )
            else:
                response: ChatCompletion = await client.chat.completions.create(  # pyright: ignore
                    **openai_params
                )
                if response.choices[0].message.content is None:  # pyright: ignore
                    raise ValueError("No content in response")

                # Safely extract usage and finish_reason
                usage = response.usage  # pyright: ignore
                finish_reason = response.choices[0].finish_reason  # pyright: ignore

                return HonchoLLMCallResponse(
                    content=response.choices[0].message.content,  # pyright: ignore
                    output_tokens=usage.completion_tokens if usage else 0,  # pyright: ignore
                    finish_reasons=[finish_reason] if finish_reason else [],
                )
        case genai.Client():
            if response_model is None:
                gemini_response: GenerateContentResponse = (
                    client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json"
                            if json_mode
                            else None,
                        },
                    )
                )

                # Safely extract response data
                text_content = gemini_response.text if gemini_response.text else ""
                token_count = (
                    gemini_response.candidates[0].token_count or 0
                    if gemini_response.candidates
                    else 0
                )
                finish_reason = (
                    gemini_response.candidates[0].finish_reason.name
                    if gemini_response.candidates
                    and gemini_response.candidates[0].finish_reason
                    else "stop"
                )

                return HonchoLLMCallResponse(
                    content=text_content,
                    output_tokens=token_count,
                    finish_reasons=[finish_reason],
                )

            else:
                gemini_response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": response_model,
                    },
                )

                token_count = (
                    gemini_response.candidates[0].token_count or 0
                    if gemini_response.candidates
                    else 0
                )
                finish_reason = (
                    gemini_response.candidates[0].finish_reason.name
                    if gemini_response.candidates
                    and gemini_response.candidates[0].finish_reason
                    else "stop"
                )

                return HonchoLLMCallResponse(
                    content=gemini_response.parsed,
                    output_tokens=token_count,
                    finish_reasons=[finish_reason],
                )

        case AsyncGroq():
            groq_params: dict[str, Any] = {
                "model": params["model"],
                "max_tokens": params["max_tokens"],
                "messages": params["messages"],
            }

            if response_model:
                groq_params["response_format"] = response_model
            elif json_mode:
                groq_params["response_format"] = {"type": "json_object"}

            response: ChatCompletion = await client.chat.completions.create(  # pyright: ignore
                **groq_params
            )
            if response.choices[0].message.content is None:  # pyright: ignore
                raise ValueError("No content in response")

            # Safely extract usage and finish_reason
            usage = response.usage  # pyright: ignore
            finish_reason = response.choices[0].finish_reason  # pyright: ignore

            return HonchoLLMCallResponse(
                content=response.choices[0].message.content,  # pyright: ignore
                output_tokens=usage.completion_tokens if usage else 0,  # pyright: ignore
                finish_reasons=[finish_reason] if finish_reason else [],
            )


async def _handle_streaming_response(
    client: AsyncAnthropic | AsyncOpenAI | genai.Client | AsyncGroq,
    params: dict[str, Any],
    json_mode: bool,
    thinking_budget_tokens: int | None,
    response_model: type[BaseModel] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"] | None = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Handle streaming responses for all supported providers.

    Args:
        client: The LLM client instance
        params: Request parameters including stream=True
        json_mode: Whether to use JSON mode
        thinking_budget_tokens: Anthropic thinking budget tokens
        response_model: Pydantic model for structured output
        reasoning_effort: OpenAI reasoning effort level (GPT-5 only)
        verbosity: OpenAI verbosity level (GPT-5 only)

    Yields:
        HonchoLLMCallStreamChunk: Individual chunks of the streaming response
    """
    match client:
        case AsyncAnthropic():
            anthropic_params: dict[str, Any] = {
                "model": params["model"],
                "max_tokens": params["max_tokens"],
                "messages": list(params["messages"]),
            }
            if json_mode:
                anthropic_params["messages"].append(
                    {"role": "assistant", "content": "{"}
                )
            if thinking_budget_tokens:
                anthropic_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
            async with client.messages.stream(**anthropic_params) as anthropic_stream:
                async for chunk in anthropic_stream:
                    if (
                        chunk.type == "content_block_delta"
                        and hasattr(chunk, "delta")
                        and hasattr(chunk.delta, "text")
                    ):
                        text_content = getattr(chunk.delta, "text", "")
                        yield HonchoLLMCallStreamChunk(content=text_content)
                final_message = await anthropic_stream.get_final_message()
                yield HonchoLLMCallStreamChunk(
                    content="",
                    is_done=True,
                    finish_reasons=[final_message.stop_reason]
                    if final_message.stop_reason
                    else [],
                )

        case AsyncOpenAI():
            openai_params: dict[str, Any] = {
                "model": params["model"],
                "messages": params["messages"],
                "stream": True,
            }

            model_name = params["model"]
            if "gpt-5" in model_name:
                openai_params["max_completion_tokens"] = params["max_tokens"]
                if reasoning_effort:
                    openai_params["reasoning_effort"] = reasoning_effort
                if verbosity:
                    openai_params["verbosity"] = verbosity
            else:
                openai_params["max_tokens"] = params["max_tokens"]

            if response_model:
                openai_params["response_format"] = response_model
            elif json_mode:
                openai_params["response_format"] = {"type": "json_object"}

            openai_stream = await client.chat.completions.create(**openai_params)  # pyright: ignore
            async for chunk in openai_stream:  # pyright: ignore
                chunk = cast(ChatCompletionChunk, chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield HonchoLLMCallStreamChunk(
                        content=chunk.choices[0].delta.content
                    )
                if chunk.choices and chunk.choices[0].finish_reason:
                    yield HonchoLLMCallStreamChunk(
                        content="",
                        is_done=True,
                        finish_reasons=[chunk.choices[0].finish_reason],
                    )

        case genai.Client():
            prompt_text = params["messages"][0]["content"] if params["messages"] else ""

            if response_model is not None:
                response_stream = client.models.generate_content_stream(
                    model=params["model"],
                    contents=prompt_text,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": response_model,
                    },
                )
            else:
                response_stream = client.models.generate_content_stream(
                    model=params["model"],
                    contents=prompt_text,
                    config={
                        "response_mime_type": "application/json" if json_mode else None,
                    },
                )

            final_chunk = None
            for chunk in response_stream:
                if chunk.text:
                    yield HonchoLLMCallStreamChunk(content=chunk.text)
                final_chunk = chunk

            finish_reason = "stop"  # Default fallback
            if (
                final_chunk
                and hasattr(final_chunk, "candidates")
                and final_chunk.candidates
                and hasattr(final_chunk.candidates[0], "finish_reason")
                and final_chunk.candidates[0].finish_reason
            ):
                finish_reason = final_chunk.candidates[0].finish_reason.name

            yield HonchoLLMCallStreamChunk(
                content="", is_done=True, finish_reasons=[finish_reason]
            )

        case AsyncGroq():
            groq_params: dict[str, Any] = {
                "model": params["model"],
                "max_tokens": params["max_tokens"],
                "messages": params["messages"],
                "stream": True,
            }

            if response_model:
                groq_params["response_format"] = response_model
            elif json_mode:
                groq_params["response_format"] = {"type": "json_object"}

            groq_stream = await client.chat.completions.create(**groq_params)  # pyright: ignore
            async for chunk in groq_stream:  # pyright: ignore
                chunk = cast(ChatCompletionChunk, chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    yield HonchoLLMCallStreamChunk(
                        content=chunk.choices[0].delta.content
                    )
                if chunk.choices and chunk.choices[0].finish_reason:
                    yield HonchoLLMCallStreamChunk(
                        content="",
                        is_done=True,
                        finish_reasons=[chunk.choices[0].finish_reason],
                    )


def retry(
    stop: Callable[[int], bool],
    wait: Callable[[int], float],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that retries a function up to a maximum number of attempts.

    Args:
        stop: A function that returns True if the function should be retried.
        wait: A function that returns the wait time in seconds.

    Returns:
        A decorator that returns a function that retries the input function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt_number = 1
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except Exception:
                        if not stop(attempt_number):
                            raise
                        delay_seconds = max(0.0, float(wait(attempt_number)))
                        if delay_seconds:
                            await asyncio.sleep(delay_seconds)
                        attempt_number += 1

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt_number = 1
                while True:
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        if not stop(attempt_number):
                            raise
                        delay_seconds = max(0.0, float(wait(attempt_number)))
                        if delay_seconds:
                            time.sleep(delay_seconds)
                        attempt_number += 1

            return sync_wrapper

    return decorator


def stop_after_attempt(max_attempts: int) -> Callable[[int], bool]:
    """Create a stop predicate that allows retries up to a maximum number of attempts.

    The predicate receives the 1-based attempt number that just failed and returns
    True if another retry should be performed, otherwise False.

    Args:
        max_attempts: Maximum number of attempts to perform, including the first call.

    Returns:
        A callable that decides whether to continue retrying based on attempt count.
    """

    def _should_retry(attempt_number: int) -> bool:
        return attempt_number < max_attempts

    return _should_retry


def wait_exponential(
    *, multiplier: float = 1.0, min: float | None = None, max: float | None = None
) -> Callable[[int], float]:
    """Create an exponential backoff wait strategy.

    Computes wait time as multiplier * 2^(attempt-1), clamped to [min, max] if provided.

    Args:
        multiplier: Base multiplier applied to the exponential growth.
        min: Optional minimum wait bound in seconds.
        max: Optional maximum wait bound in seconds.

    Returns:
        A callable that maps 1-based attempt number to a wait duration in seconds.
    """

    clamped_min = float(min) if min is not None else None
    clamped_max = float(max) if max is not None else None

    def _wait_time(attempt_number: int) -> float:
        base = float(multiplier) * (2.0 ** _builtins.max(0, attempt_number - 1))
        if clamped_min is not None:
            base = _builtins.max(clamped_min, base)
        if clamped_max is not None:
            base = _builtins.min(clamped_max, base)
        return float(base)

    return _wait_time


def with_langfuse(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        lf.start_as_current_generation(name="LLM Call")
        return await func(*args, **kwargs)

    return wrapper
