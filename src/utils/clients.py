import json
import logging
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    assert_never,
    cast,
    overload,
)

from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from sentry_sdk.ai.monitoring import ai_track
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    ConfiguredModelSettings,
    ModelConfig,
    ModelTransport,
    resolve_model_config,
    settings,
)
from src.llm.backend import CompletionResult as BackendCompletionResult
from src.llm.backend import StreamChunk as BackendStreamChunk
from src.llm.backend import ToolCallResult
from src.llm.backends.anthropic import AnthropicBackend
from src.llm.backends.gemini import GeminiBackend
from src.llm.backends.openai import OpenAIBackend
from src.llm.history_adapters import (
    AnthropicHistoryAdapter,
    GeminiHistoryAdapter,
    OpenAIHistoryAdapter,
)
from src.telemetry.logging import conditional_observe
from src.telemetry.reasoning_traces import log_reasoning_trace
from src.utils.json_parser import validate_and_repair_json
from src.utils.representation import PromptRepresentation
from src.utils.tokens import estimate_tokens
from src.utils.types import SupportedProviders, set_current_iteration

logger = logging.getLogger(__name__)

# Gemini finish reasons that indicate the response was blocked by safety or policy
# filters. When these occur, the response typically has no usable text content and
# retrying with a backup provider is appropriate.
GEMINI_BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
}


@dataclass
class IterationData:
    """Data passed to iteration callbacks after each tool execution loop iteration."""

    iteration: int
    """1-indexed iteration number."""
    tool_calls: list[str]
    """List of tool names called in this iteration."""
    input_tokens: int
    """Input tokens used in this iteration's LLM call."""
    output_tokens: int
    """Output tokens generated in this iteration's LLM call."""
    cache_read_tokens: int = 0
    """Tokens read from cache in this iteration."""
    cache_creation_tokens: int = 0
    """Tokens written to cache in this iteration."""


# Type alias for iteration callback
IterationCallback = Callable[[IterationData], None]

T = TypeVar("T")

# Type aliases for OpenAI GPT-5 specific parameters
ReasoningEffortType = (
    Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
)
VerbosityType = Literal["low", "medium", "high"] | None

ProviderClient = AsyncAnthropic | AsyncOpenAI | genai.Client


class ProviderSelection(NamedTuple):
    """Result of selecting a provider for the current attempt."""

    provider: "SupportedProviders"
    model: str
    client: ProviderClient
    thinking_budget_tokens: int | None
    reasoning_effort: ReasoningEffortType


def _provider_for_model_config(
    transport: ModelTransport,
) -> SupportedProviders:
    provider_map: dict[ModelTransport, SupportedProviders] = {
        "anthropic": "anthropic",
        "openai": "openai",
        "gemini": "google",
    }
    return provider_map[transport]


def _resolve_runtime_model_config(
    model_config: ModelConfig | ConfiguredModelSettings,
) -> ModelConfig:
    if isinstance(model_config, ModelConfig):
        return model_config
    return resolve_model_config(model_config)


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a list of messages using tiktoken."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Handle Anthropic-style content blocks
            total += estimate_tokens(json.dumps(content))
        # Also count parts for Google format
        if "parts" in msg:
            try:
                total += estimate_tokens(json.dumps(msg["parts"]))
            except TypeError:
                # Handle non-JSON-serializable content (e.g., bytes) by estimating based on string representation
                total += estimate_tokens(str(msg["parts"]))
    return total


def _is_tool_use_message(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool calls (any format)."""
    # Anthropic format: content is a list with tool_use blocks
    content = msg.get("content")
    if isinstance(content, list):
        for block in cast(list[dict[str, Any]], content):
            if block.get("type") == "tool_use":
                return True

    # OpenAI format: tool_calls field on assistant message
    return bool(msg.get("tool_calls"))


def _is_tool_result_message(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool results (any format)."""
    # Anthropic format: content is a list with tool_result blocks
    content = msg.get("content")
    if isinstance(content, list):
        for block in cast(list[dict[str, Any]], content):
            if block.get("type") == "tool_result":
                return True

    # OpenAI format: role is "tool"
    return msg.get("role") == "tool"


def _group_into_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """
    Group messages into logical conversation units.

    A unit is either:
    - A tool_use message + ALL consecutive tool_result messages that follow
    - A single non-tool message

    This ensures tool_use and tool_results stay together.
    """
    units: list[list[dict[str, Any]]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if _is_tool_use_message(msg):
            # Collect this tool_use and ALL following tool_results
            j = i + 1
            while j < len(messages) and _is_tool_result_message(messages[j]):
                j += 1

            # Create unit with tool_use + all tool_results
            unit = messages[i:j]
            if len(unit) > 1:  # Has at least one tool_result
                units.append(unit)
                i = j
            else:
                # Orphaned tool_use (no results) - skip it
                logger.debug(f"Skipping orphaned tool_use at index {i}")
                i += 1
        elif _is_tool_result_message(msg):
            # Orphaned tool_result - skip it
            logger.debug(f"Skipping orphaned tool_result at index {i}")
            i += 1
        else:
            # Regular message - its own unit
            units.append([msg])
            i += 1

    return units


def truncate_messages_to_fit(
    messages: list[dict[str, Any]],
    max_tokens: int,
    preserve_system: bool = True,
) -> list[dict[str, Any]]:
    """
    Truncate messages to fit within a token limit while maintaining valid structure.

    Strategy:
    1. Group messages into units (tool_use + results together, or single messages)
    2. Remove oldest units first to preserve recent context
    3. Units stay intact so tool_use/tool_result pairs are never broken
    """
    current_tokens = count_message_tokens(messages)
    if current_tokens <= max_tokens:
        return messages

    logger.info(f"Truncating: {current_tokens} tokens exceeds {max_tokens} limit")

    # Separate system messages from conversation
    system_messages: list[dict[str, Any]] = []
    conversation: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") == "system" and preserve_system:
            system_messages.append(msg)
        else:
            conversation.append(msg)

    system_tokens = count_message_tokens(system_messages)
    available_tokens = max_tokens - system_tokens

    if available_tokens <= 0:
        logger.warning("System message exceeds max_input_tokens")
        return messages

    # Group messages into units
    units = _group_into_units(conversation)

    if not units:
        logger.warning("No valid conversation units")
        return system_messages

    # Remove oldest units until we fit
    while len(units) > 1:  # Keep at least one unit
        # Calculate current token count
        flat_messages = [msg for unit in units for msg in unit]
        if count_message_tokens(flat_messages) <= available_tokens:
            break

        # Remove the oldest unit
        removed_unit = units.pop(0)
        logger.debug(
            f"Removed unit with {len(removed_unit)} messages "
            + f"(~{count_message_tokens(removed_unit)} tokens)"
        )

    # Flatten remaining units
    result_conversation = [msg for unit in units for msg in unit]

    result = system_messages + result_conversation
    result_tokens = count_message_tokens(result)
    logger.info(
        f"Truncation complete: {len(messages)} -> {len(result)} messages, "
        + f"{current_tokens} -> {result_tokens} tokens, "
        + f"{len(units)} units kept"
    )
    return result


M = TypeVar("M", bound=BaseModel)

# Context variable to track retry attempts for provider switching
_current_attempt: ContextVar[int] = ContextVar("current_attempt", default=0)


def _get_effective_temperature(temperature: float | None) -> float | None:
    """Adjust temperature on retries - bump 0.0 to 0.2 to get different results."""
    if temperature == 0.0 and _current_attempt.get() > 1:
        logger.debug("Bumping temperature from 0.0 to 0.2 on retry")
        return 0.2
    return temperature


CLIENTS: dict[SupportedProviders, ProviderClient] = {}

if settings.LLM.ANTHROPIC_API_KEY:
    anthropic = AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        timeout=600.0,  # 10 minutes timeout for long-running operations
    )
    CLIENTS["anthropic"] = anthropic

if settings.LLM.OPENAI_API_KEY:
    openai_client = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
    )
    CLIENTS["openai"] = openai_client

if settings.LLM.GEMINI_API_KEY:
    google = genai.client.Client(
        api_key=settings.LLM.GEMINI_API_KEY,
    )
    CLIENTS["google"] = google


def _default_credentials_for_provider(
    provider: SupportedProviders,
) -> tuple[str | None, str | None]:
    if provider == "anthropic":
        return settings.LLM.ANTHROPIC_API_KEY, None
    if provider == "openai":
        return settings.LLM.OPENAI_API_KEY, None
    if provider == "google":
        return settings.LLM.GEMINI_API_KEY, None
    assert_never(provider)


def _build_client(
    provider: SupportedProviders,
    *,
    api_key: str | None,
    base_url: str | None,
) -> ProviderClient:
    if provider == "anthropic":
        if not api_key:
            raise ValueError("Missing API key for Anthropic model config")
        return AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=600.0,
        )
    if provider == "openai":
        if not api_key:
            raise ValueError("Missing API key for OpenAI model config")
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    if provider == "google":
        if not api_key:
            raise ValueError("Missing API key for Gemini model config")
        http_options = genai_types.HttpOptions(base_url=base_url) if base_url else None
        return genai.client.Client(api_key=api_key, http_options=http_options)
    assert_never(provider)


def _client_for_model_config(
    provider: SupportedProviders,
    model_config: ModelConfig,
) -> ProviderClient:
    if model_config.api_key is None and model_config.base_url is None:
        existing_client = CLIENTS.get(provider)
        if existing_client is not None:
            return existing_client

    default_api_key, default_base_url = _default_credentials_for_provider(provider)
    api_key = model_config.api_key or default_api_key
    base_url = model_config.base_url or default_base_url
    return _build_client(
        provider,
        api_key=api_key,
        base_url=base_url,
    )


def _select_model_config_for_attempt(
    model_config: ModelConfig,
    *,
    attempt: int,
    retry_attempts: int,
) -> ModelConfig:
    if attempt != retry_attempts or model_config.fallback is None:
        return model_config

    fb = model_config.fallback
    return ModelConfig(
        model=fb.model,
        transport=fb.transport,
        fallback=None,
        api_key=fb.api_key,
        base_url=fb.base_url,
        temperature=fb.temperature,
        top_p=fb.top_p,
        top_k=fb.top_k,
        frequency_penalty=fb.frequency_penalty,
        presence_penalty=fb.presence_penalty,
        seed=fb.seed,
        thinking_effort=fb.thinking_effort,
        thinking_budget_tokens=fb.thinking_budget_tokens,
        provider_params=fb.provider_params,
        max_output_tokens=fb.max_output_tokens,
        stop_sequences=fb.stop_sequences,
    )


def extract_openai_reasoning_content(response: Any) -> str | None:
    """
    Extract reasoning/thinking content from an OpenAI ChatCompletion response.

    GPT-5 and o1 models include reasoning_details in the response message.
    OpenAI-compatible endpoints may also include this field.

    Args:
        response: OpenAI ChatCompletion response object

    Returns:
        Concatenated reasoning content string, or None if not present
    """
    try:
        message = response.choices[0].message
        # Check for reasoning_details (GPT-5/o1 models)
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            # reasoning_details is a list of reasoning steps
            reasoning_parts: list[Any] = []
            for detail in message.reasoning_details:
                if hasattr(detail, "content") and detail.content:
                    reasoning_parts.append(detail.content)
                elif isinstance(detail, dict) and detail.get("content"):  # pyright: ignore[reportUnknownMemberType]
                    reasoning_parts.append(detail["content"])
            if reasoning_parts:
                return "\n".join(reasoning_parts)
        # Check for reasoning_content (some compatible endpoints)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content
    except (AttributeError, IndexError, TypeError):
        pass
    return None


def extract_openai_reasoning_details(response: Any) -> list[dict[str, Any]]:
    """
    Extract reasoning_details array from an OpenAI/OpenRouter ChatCompletion response.

    OpenRouter returns reasoning blocks in reasoning_details that must be preserved
    and passed back in subsequent requests for Gemini models with tool use.

    Args:
        response: OpenAI ChatCompletion response object

    Returns:
        List of reasoning detail objects, or empty list if not present
    """
    try:
        message = response.choices[0].message
        # Check for reasoning_details (OpenRouter/Gemini)
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            # Return the full array for preservation
            return [
                detail.model_dump() if hasattr(detail, "model_dump") else dict(detail)
                for detail in message.reasoning_details
            ]
    except (AttributeError, IndexError, TypeError):
        pass
    return []


def extract_openai_cache_tokens(usage: Any) -> tuple[int, int]:
    """
    Extract cache token counts from OpenAI-style usage objects.

    OpenAI reports cached tokens in usage.prompt_tokens_details.cached_tokens.
    OpenRouter and some proxies may report in different locations.

    Args:
        usage: OpenAI CompletionUsage object or similar

    Returns:
        Tuple of (cache_creation_tokens, cache_read_tokens).
        For OpenAI-style APIs, cache_creation is always 0 (automatic caching),
        and cache_read is the cached_tokens count.
    """
    if not usage:
        return 0, 0

    cache_read = 0

    # OpenAI native: usage.prompt_tokens_details.cached_tokens
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        details = usage.prompt_tokens_details
        if hasattr(details, "cached_tokens") and details.cached_tokens:
            cache_read = details.cached_tokens

    # OpenRouter style: usage.cache_read_input_tokens or usage.cached_tokens
    if cache_read == 0:
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            cache_read = usage.cache_read_input_tokens
        elif hasattr(usage, "cached_tokens") and usage.cached_tokens:
            cache_read = usage.cached_tokens

    # OpenRouter/Anthropic-proxy style: cache_creation_input_tokens
    cache_creation = 0
    if (
        hasattr(usage, "cache_creation_input_tokens")
        and usage.cache_creation_input_tokens
    ):
        cache_creation = usage.cache_creation_input_tokens

    return cache_creation, cache_read


class HonchoLLMCallResponse(BaseModel, Generic[T]):
    """
    Response object for LLM calls.

    Args:
        content: The response content. When a response_model is provided, this will be
                the parsed object of that type. Otherwise, it will be a string.
        input_tokens: Total number of input tokens (including cached).
        output_tokens: Number of tokens generated in the response.
        cache_creation_input_tokens: Number of tokens written to cache.
        cache_read_input_tokens: Number of tokens read from cache.
        finish_reasons: List of finish reasons for the response.
        tool_calls_made: Optional list of all tool calls executed during the request.

    Note:
        Uncached input tokens = input_tokens - cache_read_input_tokens + cache_creation_input_tokens
        (cache_creation costs 25% more, cache_read costs 90% less)
    """

    content: T
    input_tokens: int = 0
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    finish_reasons: list[str]
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    """Number of LLM calls made in the tool execution loop (1 = single response, 2+ = tool use iterations plus final synthesis)."""
    thinking_content: str | None = None
    # Full thinking blocks with signatures for multi-turn conversation replay (Anthropic only)
    thinking_blocks: list[dict[str, Any]] = Field(default_factory=list)
    # OpenRouter reasoning_details for Gemini models - must be preserved across turns
    reasoning_details: list[dict[str, Any]] = Field(default_factory=list)


class HonchoLLMCallStreamChunk(BaseModel):
    """
    A single chunk in a streaming LLM response.

    Args:
        content: The text content for this chunk. Empty for chunks that only contain metadata.
        is_done: Whether this is the final chunk in the stream.
        finish_reasons: List of finish reasons if the stream is complete.
        output_tokens: Number of tokens generated in the response. Only set on the final chunk.
    """

    content: str
    is_done: bool = False
    finish_reasons: list[str] = Field(default_factory=list)
    output_tokens: int | None = None


class StreamingResponseWithMetadata:
    """
    Wrapper for streaming responses that includes metadata from the tool execution phase.

    This allows callers to access tool call counts, token usage, and thinking content
    from the tool loop while still streaming the final response.
    """

    _stream: AsyncIterator[HonchoLLMCallStreamChunk]
    tool_calls_made: list[dict[str, Any]]
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    thinking_content: str | None
    iterations: int

    def __init__(
        self,
        stream: AsyncIterator[HonchoLLMCallStreamChunk],
        tool_calls_made: list[dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        thinking_content: str | None = None,
        iterations: int = 0,
    ):
        self._stream = stream
        self.tool_calls_made = tool_calls_made
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.thinking_content = thinking_content
        self.iterations = iterations

    def __aiter__(self) -> AsyncIterator[HonchoLLMCallStreamChunk]:
        return self._stream.__aiter__()

    async def __anext__(self) -> HonchoLLMCallStreamChunk:
        return await self._stream.__anext__()


def _get_backend_for_provider(
    provider: SupportedProviders,
    client: AsyncAnthropic | AsyncOpenAI | genai.Client,
) -> AnthropicBackend | OpenAIBackend | GeminiBackend:
    if provider == "anthropic":
        return AnthropicBackend(client)
    if provider == "openai":
        return OpenAIBackend(client)
    if provider == "google":
        return GeminiBackend(client)
    assert_never(provider)


def _history_adapter_for_provider(
    provider: SupportedProviders,
) -> AnthropicHistoryAdapter | GeminiHistoryAdapter | OpenAIHistoryAdapter:
    if provider == "anthropic":
        return AnthropicHistoryAdapter()
    if provider == "google":
        return GeminiHistoryAdapter()
    return OpenAIHistoryAdapter()


def _tool_call_result_to_dict(tool_call: ToolCallResult) -> dict[str, Any]:
    result = {
        "id": tool_call.id,
        "name": tool_call.name,
        "input": tool_call.input,
    }
    if tool_call.thought_signature is not None:
        result["thought_signature"] = tool_call.thought_signature
    return result


def _completion_result_to_response(
    result: BackendCompletionResult,
) -> HonchoLLMCallResponse[Any]:
    return HonchoLLMCallResponse(
        content=result.content,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
        finish_reasons=[result.finish_reason] if result.finish_reason else [],
        tool_calls_made=[_tool_call_result_to_dict(tc) for tc in result.tool_calls],
        thinking_content=result.thinking_content,
        thinking_blocks=result.thinking_blocks,
        reasoning_details=result.reasoning_details,
    )


def _stream_chunk_to_response_chunk(
    chunk: BackendStreamChunk,
) -> HonchoLLMCallStreamChunk:
    return HonchoLLMCallStreamChunk(
        content=chunk.content,
        is_done=chunk.is_done,
        finish_reasons=[chunk.finish_reason] if chunk.finish_reason else [],
        output_tokens=chunk.output_tokens,
    )


# Bounds for max_tool_iterations to prevent runaway loops
MIN_TOOL_ITERATIONS = 1
MAX_TOOL_ITERATIONS = 100


async def _stream_final_response(
    model_config: ModelConfig,
    prompt: str,
    max_tokens: int,
    conversation_messages: list[dict[str, Any]],
    response_model: type[BaseModel] | None,
    json_mode: bool,
    temperature: float | None,
    stop_seqs: list[str] | None,
    reasoning_effort: ReasoningEffortType,
    verbosity: VerbosityType,
    thinking_budget_tokens: int | None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Stream the final response after tool execution is complete.

    Makes a streaming LLM call with the accumulated conversation messages
    (which include all tool call results) to generate the final answer.

    Args:
        model_config: Model configuration for the LLM provider
        prompt: Original prompt (used as fallback)
        max_tokens: Maximum tokens to generate
        conversation_messages: Full conversation history including tool results
        response_model: Optional Pydantic model for structured output
        json_mode: Whether to use JSON mode
        temperature: Temperature for the LLM
        stop_seqs: Stop sequences
        reasoning_effort: OpenAI reasoning effort (GPT-5 only)
        verbosity: OpenAI verbosity (GPT-5 only)
        thinking_budget_tokens: Anthropic / Gemini thinking budget

    Yields:
        HonchoLLMCallStreamChunk objects containing the streaming response
    """
    provider = _provider_for_model_config(model_config.transport)
    model = model_config.model
    client = _client_for_model_config(provider, model_config)

    # Make a streaming call without tools
    stream_response = await honcho_llm_call_inner(
        provider,
        model,
        prompt,
        max_tokens,
        response_model,
        json_mode,
        _get_effective_temperature(temperature),
        stop_seqs,
        reasoning_effort,
        verbosity,
        thinking_budget_tokens,
        stream=True,
        client_override=client,
        tools=None,
        tool_choice=None,
        messages=conversation_messages,
    )

    # Yield chunks from the streaming response
    async for chunk in stream_response:
        yield chunk


async def _execute_tool_loop(
    model_config: ModelConfig,
    prompt: str,
    max_tokens: int,
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None,
    tool_executor: Callable[[str, dict[str, Any]], Any],
    max_tool_iterations: int,
    response_model: type[BaseModel] | None,
    json_mode: bool,
    temperature: float | None,
    stop_seqs: list[str] | None,
    reasoning_effort: ReasoningEffortType,
    verbosity: VerbosityType,
    thinking_budget_tokens: int | None,
    enable_retry: bool,
    retry_attempts: int,
    max_input_tokens: int | None,
    get_provider_and_model: Callable[[], ProviderSelection],
    before_retry_callback: Callable[[Any], None],
    stream_final: bool = False,
    iteration_callback: IterationCallback | None = None,
) -> HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata:
    """
    Execute the tool calling loop for agentic LLM interactions.

    This function handles the iterative process of:
    1. Making an LLM call with tools available
    2. Executing any tool calls the LLM requests
    3. Feeding tool results back to the LLM
    4. Repeating until the LLM stops calling tools or max iterations reached

    Args:
        model_config: Runtime model configuration
        prompt: Initial prompt (used if messages is None)
        max_tokens: Maximum tokens to generate per call
        messages: Conversation history
        tools: Tool definitions in Anthropic format
        tool_choice: Tool selection strategy
        tool_executor: Async function to execute tools
        max_tool_iterations: Maximum iterations before forcing completion
        response_model: Optional Pydantic model for structured output
        json_mode: Whether to use JSON mode
        temperature: Temperature for the LLM (default **none**, only some models support this)
        stop_seqs: Stop sequences
        reasoning_effort: OpenAI reasoning effort (GPT-5 only)
        verbosity: OpenAI verbosity (GPT-5 only)
        thinking_budget_tokens: Anthropic / Gemini thinking budget
        enable_retry: Whether to enable retry with exponential backoff
        retry_attempts: Number of retry attempts
        max_input_tokens: Maximum input tokens (for truncation)
        get_provider_and_model: Function to get current provider/model based on attempt
        before_retry_callback: Callback for retry events
        stream_final: If True, stream the final response instead of returning it synchronously
        iteration_callback: Optional callback invoked after each iteration with IterationData

    Returns:
        Final HonchoLLMCallResponse with accumulated token counts and tool call history,
        or an AsyncIterator of HonchoLLMCallStreamChunk if stream_final=True
    """
    # Initialize conversation messages
    conversation_messages: list[dict[str, Any]] = (
        messages.copy() if messages else [{"role": "user", "content": prompt}]
    )

    iteration = 0
    all_tool_calls: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    empty_response_retries = 0
    # Track effective tool_choice - switches from "required" to "auto" after first iteration
    effective_tool_choice = tool_choice

    while iteration < max_tool_iterations:
        # Reset attempt counter so each iteration starts with the primary provider
        _current_attempt.set(1)
        logger.debug(f"Tool execution iteration {iteration + 1}/{max_tool_iterations}")

        # Truncate BEFORE making the API call to avoid context length errors
        if max_input_tokens is not None:
            conversation_messages = truncate_messages_to_fit(
                conversation_messages, max_input_tokens
            )

        # Create a wrapper that injects our messages
        async def _call_with_messages(
            effective_tool_choice: str | dict[str, Any] | None = effective_tool_choice,
            conversation_messages: list[dict[str, Any]] = conversation_messages,
        ) -> HonchoLLMCallResponse[Any]:
            # Use shared provider selection helper — per-attempt reasoning params
            sel = get_provider_and_model()

            return await honcho_llm_call_inner(
                sel.provider,
                sel.model,
                prompt,  # Will be ignored since we pass messages
                max_tokens,
                response_model,
                json_mode,
                _get_effective_temperature(temperature),
                stop_seqs,
                sel.reasoning_effort,
                verbosity,
                sel.thinking_budget_tokens,
                stream=False,
                client_override=sel.client,
                tools=tools,
                tool_choice=effective_tool_choice,
                messages=conversation_messages,
            )

        # Apply retry if enabled
        if enable_retry:
            call_func = retry(
                stop=stop_after_attempt(retry_attempts),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                before_sleep=before_retry_callback,
            )(_call_with_messages)
        else:
            call_func = _call_with_messages

        # Make the call
        response = await call_func()

        # Accumulate tokens from this iteration
        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        total_cache_creation_tokens += response.cache_creation_input_tokens
        total_cache_read_tokens += response.cache_read_input_tokens

        # Check if there are tool calls
        if not response.tool_calls_made:
            # No tool calls, return final response
            logger.debug("No tool calls in response, finishing")

            if (
                isinstance(response.content, str)
                and not response.content.strip()
                and empty_response_retries < 1
                and iteration < max_tool_iterations - 1
            ):
                empty_response_retries += 1
                conversation_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your last response was empty. Provide a concise answer "
                            "to the original query using the available context."
                        ),
                    }
                )
                iteration += 1
                continue

            if stream_final:
                # Stream the final response with metadata from tool execution
                stream = _stream_final_response(
                    model_config=model_config,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    conversation_messages=conversation_messages,
                    response_model=response_model,
                    json_mode=json_mode,
                    temperature=temperature,
                    stop_seqs=stop_seqs,
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    thinking_budget_tokens=thinking_budget_tokens,
                )
                return StreamingResponseWithMetadata(
                    stream=stream,
                    tool_calls_made=all_tool_calls,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_creation_input_tokens=total_cache_creation_tokens,
                    cache_read_input_tokens=total_cache_read_tokens,
                    thinking_content=response.thinking_content,
                    iterations=iteration + 1,
                )

            response.tool_calls_made = all_tool_calls
            response.input_tokens = total_input_tokens
            response.output_tokens = total_output_tokens
            response.cache_creation_input_tokens = total_cache_creation_tokens
            response.cache_read_input_tokens = total_cache_read_tokens
            response.iterations = iteration + 1
            return response

        # Determine which provider we're using (reuse the helper)
        current_provider = get_provider_and_model().provider

        # Add assistant message with tool calls to conversation
        assistant_message = _format_assistant_tool_message(
            current_provider,
            response.content,
            response.tool_calls_made,
            response.thinking_blocks,
            response.reasoning_details,
        )
        conversation_messages.append(assistant_message)

        # Set current iteration for telemetry context (1-indexed)
        set_current_iteration(iteration + 1)

        # Execute tools and add results
        tool_results: list[dict[str, Any]] = []
        for tool_call in response.tool_calls_made:
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id = tool_call.get("id", "")

            logger.debug(f"Executing tool: {tool_name}")

            try:
                # Execute the tool
                tool_result = await tool_executor(tool_name, tool_input)

                # Store for Anthropic format
                tool_results.append(
                    {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "result": tool_result,
                    }
                )

                all_tool_calls.append(
                    {
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_result": tool_result,
                    }
                )

            except Exception as e:
                logger.error(f"Tool execution failed for {tool_name}: {e}")
                tool_results.append(
                    {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "result": f"Error: {str(e)}",
                        "is_error": True,
                    }
                )

        # Add tool result message in provider-specific format
        _append_tool_results(current_provider, tool_results, conversation_messages)

        # Call iteration callback if provided
        if iteration_callback is not None:
            try:
                iteration_data = IterationData(
                    iteration=iteration + 1,  # 1-indexed
                    tool_calls=[tc["name"] for tc in response.tool_calls_made],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cache_read_tokens=response.cache_read_input_tokens or 0,
                    cache_creation_tokens=response.cache_creation_input_tokens or 0,
                )
                iteration_callback(iteration_data)
            except Exception:
                logger.warning("iteration_callback failed", exc_info=True)

        # After first iteration, switch from "required" to "auto" to allow model to stop
        if iteration == 0 and effective_tool_choice in ("required", "any"):
            effective_tool_choice = "auto"
            logger.debug(
                "Switched tool_choice from 'required'/'any' to 'auto' after first iteration"
            )

        iteration += 1

    # Max iterations reached
    logger.warning(
        f"Tool execution loop reached max iterations ({max_tool_iterations})"
    )

    # Add a synthesis prompt to help the model generate a response
    # without tool calls - the conversation currently ends with tool results
    # and the model may not know to produce text output
    synthesis_prompt = (
        "You have reached the maximum number of tool calls. "
        "Based on all the information you have gathered, provide your final response now. "
        "Do not attempt to call any more tools."
    )
    conversation_messages.append({"role": "user", "content": synthesis_prompt})

    # If streaming the final response, use the streaming helper with metadata
    if stream_final:
        stream = _stream_final_response(
            model_config=model_config,
            prompt=prompt,
            max_tokens=max_tokens,
            conversation_messages=conversation_messages,
            response_model=response_model,
            json_mode=json_mode,
            temperature=temperature,
            stop_seqs=stop_seqs,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            thinking_budget_tokens=thinking_budget_tokens,
        )
        return StreamingResponseWithMetadata(
            stream=stream,
            tool_calls_made=all_tool_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_creation_input_tokens=total_cache_creation_tokens,
            cache_read_input_tokens=total_cache_read_tokens,
            thinking_content=None,  # No thinking content at max iterations
            iterations=iteration + 1,  # +1 for the synthesis call
        )

    # Make one final call to get a text response
    _current_attempt.set(1)  # Reset attempt counter

    async def _final_call() -> HonchoLLMCallResponse[Any]:
        # Use shared provider selection helper for backup failover support
        sel = get_provider_and_model()

        # No tools for final call
        return await honcho_llm_call_inner(
            sel.provider,
            sel.model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            _get_effective_temperature(temperature),
            stop_seqs,
            sel.reasoning_effort,
            verbosity,
            sel.thinking_budget_tokens,
            stream=False,
            client_override=sel.client,
            tools=None,
            tool_choice=None,
            messages=conversation_messages,
        )

    if enable_retry:
        final_call_func = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(_final_call)
    else:
        final_call_func = _final_call

    final_response = await final_call_func()
    final_response.tool_calls_made = all_tool_calls
    final_response.iterations = iteration + 1  # +1 for the synthesis call
    # Include accumulated tokens from all iterations plus the final call
    final_response.input_tokens = total_input_tokens + final_response.input_tokens
    final_response.output_tokens = total_output_tokens + final_response.output_tokens
    final_response.cache_creation_input_tokens = (
        total_cache_creation_tokens + final_response.cache_creation_input_tokens
    )
    final_response.cache_read_input_tokens = (
        total_cache_read_tokens + final_response.cache_read_input_tokens
    )
    return final_response


def _format_assistant_tool_message(
    provider: SupportedProviders,
    content: Any,
    tool_calls: list[dict[str, Any]],
    thinking_blocks: list[dict[str, Any]] | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Format an assistant message with tool calls for a specific provider.

    Args:
        provider: The LLM provider
        content: The text content from the response
        tool_calls: List of tool call dicts with id, name, input keys
        thinking_blocks: Full thinking blocks with signatures for multi-turn replay (Anthropic only)
        reasoning_details: OpenRouter reasoning_details for Gemini models (must be preserved)

    Returns:
        Provider-formatted assistant message dict
    """
    adapter = _history_adapter_for_provider(provider)
    result = BackendCompletionResult(
        content=content,
        tool_calls=[
            ToolCallResult(
                id=tool_call["id"],
                name=tool_call["name"],
                input=tool_call["input"],
                thought_signature=tool_call.get("thought_signature"),
            )
            for tool_call in tool_calls
        ],
        thinking_blocks=thinking_blocks or [],
        reasoning_details=reasoning_details or [],
    )
    return adapter.format_assistant_tool_message(result)


def _append_tool_results(
    provider: SupportedProviders,
    tool_results: list[dict[str, Any]],
    conversation_messages: list[dict[str, Any]],
) -> None:
    """
    Append tool results to conversation messages in provider-specific format.

    Args:
        provider: The LLM provider
        tool_results: List of tool result dicts with tool_id, tool_name, result, is_error keys
        conversation_messages: The conversation to append to (modified in place)
    """
    adapter = _history_adapter_for_provider(provider)
    conversation_messages.extend(adapter.format_tool_results(tool_results))


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[M],
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[True] = ...,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk] | StreamingResponseWithMetadata: ...


@conditional_observe(name="LLM Call")
async def honcho_llm_call(
    *,
    model_config: ModelConfig | ConfiguredModelSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: bool = False,
    stream_final_only: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
    trace_name: str | None = None,
    iteration_callback: IterationCallback | None = None,
) -> (
    HonchoLLMCallResponse[Any]
    | AsyncIterator[HonchoLLMCallStreamChunk]
    | StreamingResponseWithMetadata
):
    """
    Make an LLM call with automatic backup provider failover. Backup provider/model
    is used on the final retry attempt, which is 3 by default.

    Args:
        model_config: Runtime or configured model settings
        prompt: The prompt to send to the LLM (used if messages is None)
        max_tokens: Maximum tokens to generate
        track_name: Optional name for AI tracking
        response_model: Optional Pydantic model for structured output
        json_mode: Whether to use JSON mode
        temperature: Temperature for the LLM (default **none**, only some models support this)
        stop_seqs: Stop sequences
        reasoning_effort: OpenAI reasoning effort (GPT-5 only)
        verbosity: OpenAI verbosity (GPT-5 only)
        thinking_budget_tokens: Anthropic / Gemini thinking budget
        enable_retry: Whether to enable retry with exponential backoff
        retry_attempts: Number of retry attempts
        stream: Whether to stream the response
        stream_final_only: If True with tools, run tool loop non-streaming then stream final answer
        tools: Tool definitions for tool calling (Anthropic/OpenAI format)
        tool_choice: Tool selection strategy (auto/required/specific tool)
        tool_executor: Async callable to execute tools, receives (tool_name, tool_input)
        max_tool_iterations: Maximum number of tool execution loops
        messages: Optional message list for multi-turn conversations (overrides prompt)
        iteration_callback: Optional callback invoked after each tool iteration with IterationData

    Returns:
        HonchoLLMCallResponse or AsyncIterator depending on stream parameter

    Raises:
        ValueError: If provider is not configured
    """
    runtime_model_config = _resolve_runtime_model_config(model_config)
    if temperature is None:
        temperature = runtime_model_config.temperature
    if stop_seqs is None:
        stop_seqs = runtime_model_config.stop_sequences
    if thinking_budget_tokens is None:
        thinking_budget_tokens = runtime_model_config.thinking_budget_tokens
    if reasoning_effort is None and runtime_model_config.thinking_effort is not None:
        reasoning_effort = cast(
            ReasoningEffortType,
            runtime_model_config.thinking_effort,
        )

    # Validate that streaming and tools are not used together
    # (unless stream_final_only is set, which streams only the final response after tool calls)
    if stream and tools and not stream_final_only:
        raise ValueError(
            "Streaming is not supported with tool calling. Set stream=False when using tools, "
            + "or use stream_final_only=True to stream only the final response after tool calls."
        )

    # Set attempt counter to 1 for first call (tenacity uses 1-indexed attempts)
    _current_attempt.set(1)

    def _get_provider_and_model() -> ProviderSelection:
        """
        Get the provider, model, client, and per-attempt reasoning params.

        Reasoning params are resolved from the selected ModelConfig (primary
        or fallback) so that cross-transport fallbacks get appropriate params.
        """
        attempt = _current_attempt.get()
        selected_model_config = _select_model_config_for_attempt(
            runtime_model_config,
            attempt=attempt,
            retry_attempts=retry_attempts,
        )
        provider = _provider_for_model_config(selected_model_config.transport)
        model = selected_model_config.model
        client = _client_for_model_config(provider, selected_model_config)

        # Resolve reasoning params from the selected config (not the primary)
        attempt_thinking_budget = (
            thinking_budget_tokens
            if selected_model_config is runtime_model_config
            else selected_model_config.thinking_budget_tokens
        )
        attempt_reasoning_effort: ReasoningEffortType = (
            reasoning_effort
            if selected_model_config is runtime_model_config
            else selected_model_config.thinking_effort
        )

        if attempt == retry_attempts and runtime_model_config.fallback is not None:
            primary_provider = _provider_for_model_config(
                runtime_model_config.transport
            )
            primary_model = runtime_model_config.model
            logger.warning(
                f"Final retry attempt {attempt}/{retry_attempts}: switching from "
                + f"{primary_provider}/{primary_model} to "
                + f"backup {provider}/{model}"
            )
        return ProviderSelection(
            provider=provider,
            model=model,
            client=client,
            thinking_budget_tokens=attempt_thinking_budget,
            reasoning_effort=attempt_reasoning_effort,
        )

    async def _call_with_provider_selection() -> (
        HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
    ):
        """
        Inner function that selects provider/model based on current attempt.
        This function is retried, so provider selection happens on each attempt.
        """
        sel = _get_provider_and_model()

        if stream:
            return await honcho_llm_call_inner(
                sel.provider,
                sel.model,
                prompt,
                max_tokens,
                response_model,
                json_mode,
                _get_effective_temperature(temperature),
                stop_seqs,
                sel.reasoning_effort,
                verbosity,
                sel.thinking_budget_tokens,
                stream=True,
                client_override=sel.client,
                tools=tools,
                tool_choice=tool_choice,
            )
        else:
            return await honcho_llm_call_inner(
                sel.provider,
                sel.model,
                prompt,
                max_tokens,
                response_model,
                json_mode,
                _get_effective_temperature(temperature),
                stop_seqs,
                sel.reasoning_effort,
                verbosity,
                sel.thinking_budget_tokens,
                stream=False,
                client_override=sel.client,
                tools=tools,
                tool_choice=tool_choice,
            )

    decorated = _call_with_provider_selection

    # apply tracking
    if track_name:
        decorated = ai_track(track_name)(decorated)

    # Define retry callback for updating attempt counter and logging
    def before_retry_callback(retry_state: Any) -> None:
        """Update attempt counter before each retry.

        Note: before_sleep is called AFTER an attempt fails and BEFORE sleeping,
        so we need to increment to the next attempt number.
        """
        next_attempt = retry_state.attempt_number + 1
        _current_attempt.set(next_attempt)
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc:
            primary_provider = _provider_for_model_config(
                runtime_model_config.transport
            )
            primary_model = runtime_model_config.model
            logger.warning(
                f"Error on attempt {retry_state.attempt_number}/{retry_attempts} with "
                + f"{primary_provider}/{primary_model}: {exc}"
            )
            logger.info(f"Will retry with attempt {next_attempt}/{retry_attempts}")

    # apply retry logic - retries on ANY exception
    if enable_retry:
        decorated = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(decorated)

    # If no tools or no tool_executor, just call once and return
    if not tools or not tool_executor:
        result: (
            HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
        ) = await decorated()
        if trace_name and isinstance(result, HonchoLLMCallResponse):
            log_reasoning_trace(
                task_type=trace_name,
                model_config=runtime_model_config,
                prompt=prompt,
                response=result,
                max_tokens=max_tokens,
                thinking_budget_tokens=thinking_budget_tokens,
                reasoning_effort=reasoning_effort,
                json_mode=json_mode,
                stop_seqs=stop_seqs,
                messages=messages,
            )
        return result

    # Validate and clamp max_tool_iterations
    clamped_iterations = max(
        MIN_TOOL_ITERATIONS, min(max_tool_iterations, MAX_TOOL_ITERATIONS)
    )
    if clamped_iterations != max_tool_iterations:
        logger.warning(
            f"max_tool_iterations {max_tool_iterations} clamped to {clamped_iterations} "
            + f"(valid range: {MIN_TOOL_ITERATIONS}-{MAX_TOOL_ITERATIONS})"
        )

    # Delegate to the tool execution loop
    result = await _execute_tool_loop(
        model_config=runtime_model_config,
        prompt=prompt,
        max_tokens=max_tokens,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        tool_executor=tool_executor,
        max_tool_iterations=clamped_iterations,
        response_model=response_model,
        json_mode=json_mode,
        temperature=temperature,
        stop_seqs=stop_seqs,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        thinking_budget_tokens=thinking_budget_tokens,
        enable_retry=enable_retry,
        retry_attempts=retry_attempts,
        max_input_tokens=max_input_tokens,
        get_provider_and_model=_get_provider_and_model,
        before_retry_callback=before_retry_callback,
        stream_final=stream_final_only,
        iteration_callback=iteration_callback,
    )
    if trace_name and isinstance(result, HonchoLLMCallResponse):
        log_reasoning_trace(
            task_type=trace_name,
            model_config=runtime_model_config,
            prompt=prompt,
            response=result,
            max_tokens=max_tokens,
            thinking_budget_tokens=thinking_budget_tokens,
            reasoning_effort=reasoning_effort,
            json_mode=json_mode,
            stop_seqs=stop_seqs,
            messages=messages,
        )
    return result


def _repair_response_model_json(  # pyright: ignore[reportUnusedFunction]
    raw_content: str,
    response_model: type[BaseModel],
    model: str,
) -> BaseModel:
    """Attempt to repair truncated/malformed JSON and validate against response_model.

    Used by all provider paths when structured output parsing fails.
    For PromptRepresentation, falls back to an empty instance.
    For other models, re-raises ValidationError.
    """
    try:
        final = validate_and_repair_json(raw_content)
        repaired_data = json.loads(final)

        # Schema-aware repair for PromptRepresentation
        if (
            response_model is PromptRepresentation
            and "deductive" in repaired_data
            and isinstance(repaired_data["deductive"], list)
        ):
            for i, item in enumerate(repaired_data["deductive"]):
                if isinstance(item, dict):
                    if "conclusion" not in item and "premises" in item:
                        logger.warning(
                            f"Deductive observation {i} missing conclusion, adding placeholder"
                        )
                        if item["premises"]:
                            item["conclusion"] = (
                                f"[Incomplete reasoning from premises: {item['premises'][0][:100]}...]"
                            )
                        else:
                            item["conclusion"] = (
                                "[Incomplete reasoning - conclusion missing]"
                            )
                    if "premises" not in item:
                        item["premises"] = []

        final = json.dumps(repaired_data)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as repair_err:
        final = ""
        logger.warning(
            f"Could not perform JSON repair on truncated output from {model}: {repair_err}"
        )

    try:
        return response_model.model_validate_json(final)
    except ValidationError as ve:
        logger.error(
            f"Validation error after repair of truncated output from {model}: {ve}"
        )
        logger.debug(f"Problematic JSON: {final}")

        if response_model is PromptRepresentation:
            logger.warning(
                "Using fallback empty Representation due to truncated output"
            )
            return PromptRepresentation(explicit=[])
        else:
            raise


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[M],
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic / Gemini
    stream: Literal[False] = False,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic / Gemini
    stream: Literal[False] = False,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic / Gemini
    stream: Literal[True] = ...,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]: ...


async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    temperature: float | None = None,
    stop_seqs: list[str] | None = None,
    reasoning_effort: ReasoningEffortType = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic / Gemini
    stream: bool = False,
    client_override: ProviderClient | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    client = client_override or CLIENTS.get(provider)
    if client is None:
        raise ValueError(f"Missing client for {provider}")

    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    backend = _get_backend_for_provider(
        provider,
        client,
    )
    if stream:

        async def _stream() -> AsyncIterator[HonchoLLMCallStreamChunk]:
            async for chunk in backend.stream(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop_seqs,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_model,
                thinking_budget_tokens=thinking_budget_tokens,
                thinking_effort=reasoning_effort,
                max_output_tokens=max_tokens,
                extra_params={
                    "json_mode": json_mode,
                    "verbosity": verbosity,
                },
            ):
                yield _stream_chunk_to_response_chunk(chunk)

        return _stream()

    result = await backend.complete(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_seqs,
        tools=tools,
        tool_choice=tool_choice,
        response_format=response_model,
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_effort=reasoning_effort,
        max_output_tokens=max_tokens,
        extra_params={
            "json_mode": json_mode,
            "verbosity": verbosity,
        },
    )
    return _completion_result_to_response(result)


def _provider_for_streaming_client(
    client: AsyncAnthropic | AsyncOpenAI | genai.Client,
) -> SupportedProviders:
    if isinstance(client, AsyncAnthropic):
        return "anthropic"
    if isinstance(client, genai.Client):
        return "google"
    return "openai"


async def handle_streaming_response(
    client: AsyncAnthropic | AsyncOpenAI | genai.Client,
    params: dict[str, Any],
    json_mode: bool,
    thinking_budget_tokens: int | None,
    response_model: type[BaseModel] | None = None,
    reasoning_effort: ReasoningEffortType = None,
    verbosity: Literal["low", "medium", "high"] | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Handle streaming responses for all supported providers.

    Args:
        client: The LLM client instance
        params: Request parameters including stream=True
        json_mode: Whether to use JSON mode
        thinking_budget_tokens: Anthropic / Gemini thinking budget tokens
        response_model: Pydantic model for structured output
        reasoning_effort: OpenAI reasoning effort level (GPT-5 only)
        verbosity: OpenAI verbosity level (GPT-5 only)

    Yields:
        HonchoLLMCallStreamChunk: Individual chunks of the streaming response
    """
    provider = _provider_for_streaming_client(client)
    backend = _get_backend_for_provider(provider, client)
    extra_params: dict[str, Any] = {"json_mode": json_mode, "verbosity": verbosity}
    async for chunk in backend.stream(
        model=cast(str, params["model"]),
        messages=cast(list[dict[str, Any]], params["messages"]),
        max_tokens=cast(int, params["max_tokens"]),
        temperature=cast(float | None, params.get("temperature")),
        stop=cast(list[str] | None, params.get("stop")),
        tools=cast(list[dict[str, Any]] | None, params.get("tools")),
        tool_choice=cast(str | dict[str, Any] | None, params.get("tool_choice")),
        response_format=response_model,
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_effort=reasoning_effort,
        max_output_tokens=cast(int, params["max_tokens"]),
        extra_params=extra_params,
    ):
        yield _stream_chunk_to_response_chunk(chunk)
