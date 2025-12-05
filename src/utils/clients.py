import json
import logging
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Any, Generic, Literal, TypeVar, cast, overload

from anthropic import AsyncAnthropic
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from anthropic.types.message import Message as AnthropicMessage
from anthropic.types.usage import Usage
from google import genai
from google.genai.types import (
    ContentListUnionDict,
    GenerateContentConfigDict,
    GenerateContentResponse,
)
from groq import AsyncGroq
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel, Field, ValidationError
from sentry_sdk.ai.monitoring import ai_track
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import LLMComponentSettings, settings
from src.utils.json_parser import validate_and_repair_json
from src.utils.logging import conditional_observe
from src.utils.representation import PromptRepresentation
from src.utils.tokens import estimate_tokens
from src.utils.types import SupportedProviders

logger = logging.getLogger(__name__)

T = TypeVar("T")


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

CLIENTS: dict[
    SupportedProviders,
    AsyncAnthropic | AsyncOpenAI | genai.Client | AsyncGroq,
] = {}

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

if settings.LLM.OPENAI_COMPATIBLE_API_KEY and settings.LLM.OPENAI_COMPATIBLE_BASE_URL:
    CLIENTS["custom"] = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_COMPATIBLE_API_KEY,
        base_url=settings.LLM.OPENAI_COMPATIBLE_BASE_URL,
    )

# NOTE: user must know whether they want to use 'custom' or 'vllm'
if settings.LLM.OPENAI_COMPATIBLE_API_KEY and settings.LLM.OPENAI_COMPATIBLE_BASE_URL:
    CLIENTS["vllm"] = AsyncOpenAI(
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
]

for provider_name, provider_value in SELECTED_PROVIDERS:
    if provider_value not in CLIENTS:
        raise ValueError(f"Missing client for {provider_name}: {provider_value}")

# Validate backup providers are initialized if configured
BACKUP_PROVIDERS = [
    ("Deriver", settings.DERIVER),
    ("Dialectic", settings.DIALECTIC),
    ("Summary", settings.SUMMARY),
    ("Dream", settings.DREAM),
]

for component_name, component_settings in BACKUP_PROVIDERS:
    if (
        hasattr(component_settings, "BACKUP_PROVIDER")
        and component_settings.BACKUP_PROVIDER is not None
        and component_settings.BACKUP_PROVIDER not in CLIENTS
    ):
        raise ValueError(
            f"Backup provider for {component_name} is set to {component_settings.BACKUP_PROVIDER}, "
            + "but this provider is not initialized. Please set the required API key/URL environment "
            + "variables or remove the backup configuration."
        )


def convert_tools_for_provider(
    tools: list[dict[str, Any]],
    provider: SupportedProviders,
) -> list[dict[str, Any]]:
    """
    Convert tool definitions to provider-specific format.

    Args:
        tools: List of tool definitions in Anthropic format (with input_schema)
        provider: The target provider to convert tools for

    Returns:
        List of tool definitions in the provider's native format
    """
    if provider == "anthropic":
        # Anthropic format: input_schema
        return tools
    elif provider in ("openai", "custom", "vllm"):
        # OpenAI format: parameters instead of input_schema
        # custom and vllm use AsyncOpenAI client so need OpenAI format
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]
    elif provider == "google":
        # Google format: function_declarations wrapped in a tool object
        return [
            {
                "function_declarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    }
                    for tool in tools
                ]
            }
        ]
    else:
        # For unsupported providers, return as-is (will likely error if tools are used)
        logger.warning(
            f"Tool calling not implemented for provider {provider}, returning tools as-is"
        )
        return tools


def extract_openai_reasoning_content(response: Any) -> str | None:
    """
    Extract reasoning/thinking content from an OpenAI ChatCompletion response.

    GPT-5 and o1 models include reasoning_details in the response message.
    Custom OpenAI-compatible providers may also include this field.

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
        # Check for reasoning_content (some custom providers)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content
    except (AttributeError, IndexError, TypeError):
        pass
    return None


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
    thinking_content: str | None = None


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


@overload
async def honcho_llm_call(
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    *,
    response_model: type[M],
    json_mode: bool = False,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
) -> HonchoLLMCallResponse[M]: ...


@overload
async def honcho_llm_call(
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: None = None,
    json_mode: bool = False,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[False] = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
) -> HonchoLLMCallResponse[str]: ...


@overload
async def honcho_llm_call(
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: Literal[True] = ...,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]: ...


@conditional_observe(name="LLM Call")
async def honcho_llm_call(
    llm_settings: LLMComponentSettings,
    prompt: str,
    max_tokens: int,
    track_name: str | None = None,
    response_model: type[BaseModel] | None = None,
    json_mode: bool = False,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,
    enable_retry: bool = True,
    retry_attempts: int = 3,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    max_tool_iterations: int = 10,
    messages: list[dict[str, Any]] | None = None,
    max_input_tokens: int | None = None,
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    """
    Make an LLM call with automatic backup provider failover. Backup provider/model
    is used on the final retry attempt, which is 3 by default.

    Args:
        llm_settings: Settings object containing PROVIDER, MODEL,
                     BACKUP_PROVIDER, and BACKUP_MODEL
        prompt: The prompt to send to the LLM (used if messages is None)
        max_tokens: Maximum tokens to generate
        track_name: Optional name for AI tracking
        response_model: Optional Pydantic model for structured output
        json_mode: Whether to use JSON mode
        stop_seqs: Stop sequences
        reasoning_effort: OpenAI reasoning effort (GPT-5 only)
        verbosity: OpenAI verbosity (GPT-5 only)
        thinking_budget_tokens: Anthropic thinking budget
        enable_retry: Whether to enable retry with exponential backoff
        retry_attempts: Number of retry attempts
        stream: Whether to stream the response
        tools: Tool definitions for tool calling (Anthropic/OpenAI format)
        tool_choice: Tool selection strategy (auto/required/specific tool)
        tool_executor: Async callable to execute tools, receives (tool_name, tool_input)
        max_tool_iterations: Maximum number of tool execution loops
        messages: Optional message list for multi-turn conversations (overrides prompt)

    Returns:
        HonchoLLMCallResponse or AsyncIterator depending on stream parameter

    Raises:
        ValueError: If provider is not configured
    """
    # Validate that streaming and tools are not used together
    if stream and tools:
        raise ValueError(
            "Streaming is not supported with tool calling. Set stream=False when using tools."
        )

    # Set attempt counter to 1 for first call (tenacity uses 1-indexed attempts)
    _current_attempt.set(1)

    async def _call_with_provider_selection() -> (
        HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]
    ):
        """
        Inner function that selects provider/model based on current attempt.
        This function is retried, so provider selection happens on each attempt.
        """
        attempt = _current_attempt.get()

        # Use backup on final retry attempt (when attempt == retry_attempts)
        if (
            attempt == retry_attempts
            and llm_settings.BACKUP_PROVIDER is not None
            and llm_settings.BACKUP_MODEL is not None
            and llm_settings.BACKUP_PROVIDER in CLIENTS
        ):
            provider: SupportedProviders = llm_settings.BACKUP_PROVIDER
            model: str = llm_settings.BACKUP_MODEL
            logger.warning(
                f"Final retry attempt {attempt}/{retry_attempts}: switching from "
                + f"{llm_settings.PROVIDER}/{llm_settings.MODEL} to "
                + f"backup {provider}/{model}"
            )

            # Filter out incompatible parameters when using backup
            thinking_budget = thinking_budget_tokens
            gpt5_reasoning_effort = reasoning_effort
            gpt5_verbosity = verbosity

            if provider != "anthropic" and thinking_budget:
                logger.warning(
                    f"thinking_budget_tokens not supported by {provider}, ignoring"
                )
                thinking_budget = None

            if "gpt-5" not in model and (gpt5_reasoning_effort or gpt5_verbosity):
                logger.warning(
                    "reasoning_effort/verbosity only supported by GPT-5 models, ignoring"
                )
                gpt5_reasoning_effort = None
                gpt5_verbosity = None
        else:
            provider = llm_settings.PROVIDER
            model = llm_settings.MODEL
            thinking_budget = thinking_budget_tokens
            gpt5_reasoning_effort = reasoning_effort
            gpt5_verbosity = verbosity

        # Validate client exists
        client = CLIENTS.get(provider)
        if not client:
            raise ValueError(f"Missing client for {provider}")

        # Convert tools to provider-specific format if provided
        converted_tools = convert_tools_for_provider(tools, provider) if tools else None

        if stream:
            return await honcho_llm_call_inner(
                provider,
                model,
                prompt,
                max_tokens,
                response_model,
                json_mode,
                stop_seqs,
                gpt5_reasoning_effort,
                gpt5_verbosity,
                thinking_budget,
                True,  # type: ignore[arg-type]
                converted_tools,
                tool_choice,
            )
        else:
            return await honcho_llm_call_inner(
                provider,
                model,
                prompt,
                max_tokens,
                response_model,
                json_mode,
                stop_seqs,
                gpt5_reasoning_effort,
                gpt5_verbosity,
                thinking_budget,
                False,  # type: ignore[arg-type]
                converted_tools,
                tool_choice,
            )

    decorated = _call_with_provider_selection

    # apply tracking
    if track_name:
        decorated = ai_track(track_name)(decorated)

    # Define retry callback
    def before_retry_callback(retry_state: Any) -> None:
        """Update attempt counter before each retry.

        Note: before_sleep is called AFTER an attempt fails and BEFORE sleeping,
        so we need to increment to the next attempt number.
        """
        next_attempt = retry_state.attempt_number + 1
        _current_attempt.set(next_attempt)
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if exc:
            logger.warning(
                f"Error on attempt {retry_state.attempt_number}/{retry_attempts} with "
                + f"{llm_settings.PROVIDER}/{llm_settings.MODEL}: {exc}"
            )
            logger.info(f"Will retry with attempt {next_attempt}/{retry_attempts}")

    # apply retry logic - retries on ANY exception
    if enable_retry:

        def before_retry_callback(retry_state: Any) -> None:
            """Update attempt counter before each retry.

            Note: before_sleep is called AFTER an attempt fails and BEFORE sleeping,
            so we need to increment to the next attempt number.
            """
            next_attempt = retry_state.attempt_number + 1
            _current_attempt.set(next_attempt)
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc:
                logger.warning(
                    f"Error on attempt {retry_state.attempt_number}/{retry_attempts} with "
                    + f"{llm_settings.PROVIDER}/{llm_settings.MODEL}: {exc}"
                )
                logger.info(f"Will retry with attempt {next_attempt}/{retry_attempts}")

        decorated = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(decorated)

    # If no tools or no tool_executor, just call once and return
    if not tools or not tool_executor:
        return await decorated()

    # Tool execution loop
    # Initialize conversation messages
    conversation_messages: list[dict[str, Any]] = (
        messages.copy() if messages else [{"role": "user", "content": prompt}]
    )

    iteration = 0
    all_tool_calls: list[dict[str, Any]] = []
    total_input_tokens = 0  # Accumulate input tokens across all tool iterations
    total_output_tokens = 0  # Accumulate output tokens across all tool iterations
    total_cache_creation_tokens = 0  # Accumulate cache creation tokens
    total_cache_read_tokens = 0  # Accumulate cache read tokens
    # Track effective tool_choice - switches from "required" to "auto" after first iteration
    effective_tool_choice = tool_choice

    while iteration < max_tool_iterations:
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
            # Reimplement provider selection with messages
            attempt = _current_attempt.get()

            if (
                attempt == retry_attempts
                and llm_settings.BACKUP_PROVIDER is not None
                and llm_settings.BACKUP_MODEL is not None
                and llm_settings.BACKUP_PROVIDER in CLIENTS
            ):
                provider: SupportedProviders = llm_settings.BACKUP_PROVIDER
                model: str = llm_settings.BACKUP_MODEL
                thinking_budget = thinking_budget_tokens
                gpt5_reasoning_effort = reasoning_effort
                gpt5_verbosity = verbosity

                if provider != "anthropic" and thinking_budget:
                    thinking_budget = None
                if "gpt-5" not in model and (gpt5_reasoning_effort or gpt5_verbosity):
                    gpt5_reasoning_effort = None
                    gpt5_verbosity = None
            else:
                provider = llm_settings.PROVIDER
                model = llm_settings.MODEL
                thinking_budget = thinking_budget_tokens
                gpt5_reasoning_effort = reasoning_effort
                gpt5_verbosity = verbosity

            client = CLIENTS.get(provider)
            if not client:
                raise ValueError(f"Missing client for {provider}")

            converted_tools = (
                convert_tools_for_provider(tools, provider) if tools else None
            )

            return await honcho_llm_call_inner(
                provider,
                model,
                prompt,  # Will be ignored since we pass messages
                max_tokens,
                response_model,
                json_mode,
                stop_seqs,
                gpt5_reasoning_effort,
                gpt5_verbosity,
                thinking_budget,
                False,
                converted_tools,
                effective_tool_choice,
                conversation_messages,
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
            response.tool_calls_made = all_tool_calls
            response.input_tokens = total_input_tokens
            response.output_tokens = total_output_tokens
            response.cache_creation_input_tokens = total_cache_creation_tokens
            response.cache_read_input_tokens = total_cache_read_tokens
            return response

        # Determine which provider we're using
        attempt = _current_attempt.get()
        current_provider = (
            llm_settings.BACKUP_PROVIDER
            if (
                attempt == retry_attempts
                and llm_settings.BACKUP_PROVIDER is not None
                and llm_settings.BACKUP_PROVIDER in CLIENTS
            )
            else llm_settings.PROVIDER
        )

        # Add assistant message with tool calls to conversation
        # Format depends on provider
        if current_provider == "anthropic":
            # Anthropic requires content to be a list of blocks including tool use blocks
            content_blocks: list[dict[str, Any]] = []

            # Add text content if present
            if isinstance(response.content, str) and response.content:
                content_blocks.append({"type": "text", "text": response.content})

            # Add tool use blocks
            for tool_call in response.tool_calls_made:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "input": tool_call["input"],
                    }
                )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content_blocks,
            }
        elif current_provider == "google":
            # Google format: model role with function_call parts
            parts: list[dict[str, Any]] = []

            # Add text content if present
            if isinstance(response.content, str) and response.content:
                parts.append({"text": response.content})

            # Add function call parts with thought_signature if present
            for tool_call in response.tool_calls_made:
                part_data: dict[str, Any] = {
                    "function_call": {
                        "name": tool_call["name"],
                        "args": tool_call["input"],
                    }
                }
                # Include thought_signature if present (required by Gemini)
                if "thought_signature" in tool_call:
                    part_data["thought_signature"] = tool_call["thought_signature"]
                parts.append(part_data)

            assistant_message = {
                "role": "model",
                "parts": parts,
            }
        else:
            # OpenAI format - must include tool_calls in the assistant message
            openai_tool_calls: list[Any] = []
            for tool_call in response.tool_calls_made:
                openai_tool_calls.append(
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["input"]),
                        },
                    }
                )
            assistant_message = {
                "role": "assistant",
                "content": response.content
                if isinstance(response.content, str)
                else None,
                "tool_calls": openai_tool_calls,
            }

        conversation_messages.append(assistant_message)

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
        if current_provider == "anthropic":
            # Anthropic requires tool results in specific content blocks
            result_blocks: list[dict[str, Any]] = []
            for tr in tool_results:
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tr["tool_id"],
                        "content": str(tr["result"]),
                        "is_error": tr.get("is_error", False),
                    }
                )

            tool_result_message = {
                "role": "user",
                "content": result_blocks,
            }
        elif current_provider == "google":
            # Google format: user role with function_response parts
            response_parts: list[dict[str, Any]] = []
            for tr in tool_results:
                response_parts.append(
                    {
                        "function_response": {
                            "name": tr["tool_name"],
                            "response": {"result": str(tr["result"])},
                        }
                    }
                )

            tool_result_message = {
                "role": "user",
                "parts": response_parts,
            }
        else:
            # OpenAI format - add each tool result as a separate message with role="tool"
            for tr in tool_results:
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tr["tool_id"],
                    "content": str(tr["result"]),
                }
                conversation_messages.append(tool_result_message)
            # Skip the append below since we already added messages
            tool_result_message = None  # type: ignore

        if tool_result_message is not None:
            conversation_messages.append(tool_result_message)

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

    # Make one final call to get a text response
    _current_attempt.set(1)  # Reset attempt counter

    async def _final_call() -> HonchoLLMCallResponse[Any]:
        provider = llm_settings.PROVIDER
        model = llm_settings.MODEL

        client = CLIENTS.get(provider)
        if not client:
            raise ValueError(f"Missing client for {provider}")

        # No tools for final call
        return await honcho_llm_call_inner(
            provider,
            model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            stop_seqs,
            reasoning_effort,
            verbosity,
            thinking_budget_tokens,
            False,
            None,  # No tools
            None,  # No tool_choice
            conversation_messages,
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


@overload
async def honcho_llm_call_inner(
    provider: SupportedProviders,
    model: str,
    prompt: str,
    max_tokens: int,
    response_model: type[M],
    json_mode: bool = False,
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: Literal[False] = False,
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
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: Literal[False] = False,
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
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: Literal[True] = ...,
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
    stop_seqs: list[str] | None = None,
    reasoning_effort: Literal["low", "medium", "high", "minimal"]
    | None = None,  # OpenAI only
    verbosity: Literal["low", "medium", "high"] | None = None,  # OpenAI only
    thinking_budget_tokens: int | None = None,  # Anthropic only
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[Any] | AsyncIterator[HonchoLLMCallStreamChunk]:
    # has already been validated by honcho_llm_call
    client = CLIENTS[provider]

    # Use messages if provided, otherwise convert prompt to message
    if messages is None:
        messages = [{"role": "user", "content": prompt}]

    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": stream,
    }

    if stream:
        # Return async generator for streaming responses
        return handle_streaming_response(
            client,
            params,
            json_mode,
            thinking_budget_tokens,
            response_model,
            reasoning_effort,
            verbosity,
        )

    # Remove stream parameter for non-streaming calls as some providers don't accept it
    params.pop("stream", None)

    system_messages: list[str] = []
    non_system_messages: list[dict[str, Any]] = []

    match client:
        case AsyncAnthropic():
            # Anthropic requires system messages to be passed as a top-level parameter
            # Extract system messages and non-system messages
            for msg in params["messages"]:
                if msg.get("role") == "system":
                    system_messages.append(msg["content"])
                else:
                    non_system_messages.append(msg)

            anthropic_params: dict[str, Any] = {
                "model": params["model"],
                "max_tokens": params["max_tokens"],
                "messages": non_system_messages,
            }

            # Add system parameter if there are system messages
            # Use cache_control for prompt caching
            if system_messages:
                anthropic_params["system"] = [
                    {
                        "type": "text",
                        "text": "\n\n".join(system_messages),
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            # Add tools if provided
            if tools:
                anthropic_params["tools"] = tools
                if tool_choice:
                    # Convert tool_choice to Anthropic format
                    if isinstance(tool_choice, str):
                        if tool_choice == "auto":
                            anthropic_params["tool_choice"] = {"type": "auto"}
                        elif tool_choice in ("any", "required"):
                            anthropic_params["tool_choice"] = {"type": "any"}
                        elif tool_choice == "none":
                            # Don't set tool_choice, let Anthropic default
                            pass
                        else:
                            # Assume it's a tool name
                            anthropic_params["tool_choice"] = {
                                "type": "tool",
                                "name": tool_choice,
                            }
                    else:
                        # Already in dict format, use as-is
                        anthropic_params["tool_choice"] = tool_choice

            # For response models, we need to request JSON and parse manually
            # Note: tools and response_model should not be used together
            if response_model or json_mode:
                # Add JSON schema instructions to the prompt if using response_model
                if response_model:
                    schema_json = json.dumps(
                        response_model.model_json_schema(), indent=2
                    )
                    anthropic_params["messages"][-1]["content"] += (
                        f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
                    )
                anthropic_params["messages"].append(
                    {"role": "assistant", "content": "{"}
                )

            if thinking_budget_tokens:
                anthropic_params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }

            anthropic_response: AnthropicMessage = cast(
                AnthropicMessage, await client.messages.create(**anthropic_params)
            )

            # Extract text content, thinking blocks, and tool use blocks from content blocks
            text_blocks: list[str] = []
            thinking_blocks: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in anthropic_response.content:
                if isinstance(block, TextBlock):
                    text_blocks.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    thinking_blocks.append(block.thinking)
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            # Safely extract usage and stop_reason
            usage: Any | Usage = anthropic_response.usage
            stop_reason = anthropic_response.stop_reason

            text_content = "\n".join(text_blocks)
            thinking_content = "\n".join(thinking_blocks) if thinking_blocks else None

            # Extract cache token counts from Anthropic usage
            # Anthropic's input_tokens = uncached tokens only
            # Total = input_tokens + cache_read + cache_creation
            cache_creation_tokens = (
                getattr(usage, "cache_creation_input_tokens", 0) or 0 if usage else 0
            )
            cache_read_tokens = (
                getattr(usage, "cache_read_input_tokens", 0) or 0 if usage else 0
            )
            uncached_tokens = usage.input_tokens if usage else 0
            # Calculate total input tokens for consistent reporting
            total_input_tokens = (
                uncached_tokens + cache_read_tokens + cache_creation_tokens
            )

            # If using response_model, parse the JSON response
            if response_model:
                try:
                    # Add back the opening brace that we prefilled
                    json_content = "{" + text_content
                    parsed_json = json.loads(json_content)
                    parsed_content = response_model.model_validate(parsed_json)

                    return HonchoLLMCallResponse(
                        content=parsed_content,
                        input_tokens=total_input_tokens,
                        output_tokens=usage.output_tokens if usage else 0,
                        cache_creation_input_tokens=cache_creation_tokens,
                        cache_read_input_tokens=cache_read_tokens,
                        finish_reasons=[stop_reason] if stop_reason else [],
                        tool_calls_made=tool_calls,
                        thinking_content=thinking_content,
                    )
                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    raise ValueError(
                        f"Failed to parse Anthropic response as {response_model}: {e}. Raw content: {text_content}"
                    ) from e

            return HonchoLLMCallResponse(
                content=text_content,
                input_tokens=total_input_tokens,
                output_tokens=usage.output_tokens if usage else 0,
                cache_creation_input_tokens=cache_creation_tokens,
                cache_read_input_tokens=cache_read_tokens,
                finish_reasons=[stop_reason] if stop_reason else [],
                tool_calls_made=tool_calls,
                thinking_content=thinking_content,
            )

        case AsyncOpenAI():
            # For custom providers (e.g., OpenRouter), add cache_control to system messages
            # This enables prompt caching for Anthropic models proxied via OpenAI-compatible APIs
            processed_messages: list[dict[str, Any]] = params["messages"]
            if provider == "custom":
                processed_messages = []
                for msg in params["messages"]:
                    if msg.get("role") == "system" and isinstance(
                        msg.get("content"), str
                    ):
                        # Convert system message to content block format with cache_control
                        processed_messages.append(
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": msg["content"],
                                        "cache_control": {"type": "ephemeral"},
                                    }
                                ],
                            }
                        )
                    else:
                        processed_messages.append(msg)

            openai_params: dict[str, Any] = {
                "model": params["model"],
                "messages": processed_messages,
            }
            if "gpt-5" in model:
                openai_params["max_completion_tokens"] = params["max_tokens"]
                if reasoning_effort:
                    openai_params["reasoning_effort"] = reasoning_effort
                if verbosity:
                    openai_params["verbosity"] = verbosity
            else:
                openai_params["max_tokens"] = params["max_tokens"]

            # Add tools if provided (not compatible with response_model for most cases)
            if tools and not response_model:
                openai_params["tools"] = tools
                if tool_choice:
                    openai_params["tool_choice"] = tool_choice

            if json_mode and provider != "vllm":
                openai_params["response_format"] = {"type": "json_object"}

            # custom shim for vLLM response model formatting
            # NOTE: this is all specific to the Representation model.
            # Do not call with any other response model.
            if provider == "vllm" and response_model:
                if response_model is not PromptRepresentation:
                    raise NotImplementedError(
                        "vLLM structured output currently supports only PromptRepresentation"
                    )
                openai_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": response_model.model_json_schema(),
                    },
                }
                if stop_seqs:
                    openai_params["stop"] = stop_seqs
                vllm_response: ChatCompletion = cast(
                    ChatCompletion,
                    await client.chat.completions.create(**openai_params),
                )

                usage = vllm_response.usage
                finish_reason = vllm_response.choices[0].finish_reason

                try:
                    test_rep = ""
                    if vllm_response.choices[0].message.content is not None:
                        test_rep = vllm_response.choices[0].message.content

                    final = validate_and_repair_json(test_rep)

                    # Schema-aware repair: ensure deductive observations have required fields

                    repaired_data = json.loads(final)

                    # Fix deductive observations that might be missing conclusion
                    if "deductive" in repaired_data and isinstance(
                        repaired_data["deductive"], list
                    ):
                        for i, item in enumerate(repaired_data["deductive"]):
                            if isinstance(item, dict):
                                # If conclusion is missing but premises exist, create a placeholder
                                if "conclusion" not in item and "premises" in item:
                                    logger.warning(
                                        f"Deductive observation {i} missing conclusion, adding placeholder"
                                    )
                                    # Try to generate a conclusion from premises if possible
                                    if item["premises"]:
                                        item["conclusion"] = (
                                            f"[Incomplete reasoning from premises: {item['premises'][0][:100]}...]"
                                        )
                                    else:
                                        item["conclusion"] = (
                                            "[Incomplete reasoning - conclusion missing]"
                                        )
                                # If premises is missing, add empty list (it's optional with default)
                                if "premises" not in item:
                                    item["premises"] = []

                    final = json.dumps(repaired_data)
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    final = ""
                    logger.warning(f"Could not perform schema-aware repair: {e}")
                    # Continue with original final value if repair fails

                try:
                    response_obj = PromptRepresentation.model_validate_json(final)
                except ValidationError as e:
                    logger.error(f"Validation error after repair: {e}")
                    logger.debug(f"Problematic JSON: {final}")

                    # Fallback: return empty response rather than failing
                    logger.warning(
                        "Using fallback empty Representation due to validation error"
                    )
                    response_obj = PromptRepresentation(explicit=[], deductive=[])

                cache_creation, cache_read = extract_openai_cache_tokens(usage)
                return HonchoLLMCallResponse(
                    content=response_obj,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                    finish_reasons=[finish_reason] if finish_reason else [],
                    tool_calls_made=[],
                    thinking_content=extract_openai_reasoning_content(vllm_response),
                )
            elif response_model:
                openai_params["response_format"] = response_model
                response: ChatCompletion = await client.chat.completions.parse(  # pyright: ignore
                    **openai_params
                )
                # Extract the parsed object for structured output
                parsed_content = response.choices[0].message.parsed
                if parsed_content is None:
                    raise ValueError("No parsed content in structured response")

                usage = response.usage
                finish_reason = response.choices[0].finish_reason

                # Validate that parsed content matches the response model
                if not isinstance(parsed_content, response_model):
                    raise ValueError(
                        f"Parsed content does not match the response model: {parsed_content} != {response_model}"
                    )

                # Extract tool calls if present (though unlikely with structured output)
                parsed_tool_calls: list[dict[str, Any]] = []
                if (
                    hasattr(response.choices[0].message, "tool_calls")
                    and response.choices[0].message.tool_calls
                ):
                    for tool_call in response.choices[0].message.tool_calls:
                        parsed_tool_calls.append(
                            {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": json.loads(tool_call.function.arguments)
                                if tool_call.function.arguments
                                else {},
                            }
                        )

                cache_creation, cache_read = extract_openai_cache_tokens(usage)
                return HonchoLLMCallResponse(
                    content=parsed_content,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                    finish_reasons=[finish_reason] if finish_reason else [],
                    tool_calls_made=parsed_tool_calls,
                    thinking_content=extract_openai_reasoning_content(response),
                )
            else:
                response: ChatCompletion = await client.chat.completions.create(  # pyright: ignore
                    **openai_params
                )

                usage = response.usage  # pyright: ignore
                finish_reason = response.choices[0].finish_reason  # pyright: ignore

                # Extract tool calls if present
                tool_calls_list: list[dict[str, Any]] = []
                if response.choices[0].message.tool_calls:  # pyright: ignore
                    for tool_call in response.choices[0].message.tool_calls:  # pyright: ignore
                        tool_calls_list.append(
                            {
                                "id": tool_call.id,  # pyright: ignore
                                "name": tool_call.function.name,  # pyright: ignore
                                "input": json.loads(tool_call.function.arguments)  # pyright: ignore
                                if tool_call.function.arguments  # pyright: ignore
                                else {},
                            }
                        )

                cache_creation, cache_read = extract_openai_cache_tokens(usage)
                return HonchoLLMCallResponse(
                    content=response.choices[0].message.content or "",  # pyright: ignore
                    input_tokens=usage.prompt_tokens if usage else 0,  # pyright: ignore
                    output_tokens=usage.completion_tokens if usage else 0,  # pyright: ignore
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                    finish_reasons=[finish_reason] if finish_reason else [],
                    tool_calls_made=tool_calls_list,
                    thinking_content=extract_openai_reasoning_content(response),
                )

        case genai.Client():
            # Build config for Gemini
            gemini_config: dict[str, Any] = {}

            # Add tools if provided
            if tools:
                gemini_config["tools"] = tools
                # Handle tool_choice
                if tool_choice:
                    if tool_choice == "auto":
                        gemini_config["tool_config"] = {
                            "function_calling_config": {"mode": "AUTO"}
                        }
                    elif tool_choice == "any" or tool_choice == "required":
                        gemini_config["tool_config"] = {
                            "function_calling_config": {"mode": "ANY"}
                        }
                    elif tool_choice == "none":
                        gemini_config["tool_config"] = {
                            "function_calling_config": {"mode": "NONE"}
                        }
                    elif isinstance(tool_choice, dict) and "name" in tool_choice:
                        # Specific tool selection
                        gemini_config["tool_config"] = {
                            "function_calling_config": {
                                "mode": "ANY",
                                "allowed_function_names": [tool_choice["name"]],
                            }
                        }

            if response_model is None:
                if json_mode and not tools:
                    gemini_config["response_mime_type"] = "application/json"

                # Use messages if provided, otherwise use prompt
                if messages:
                    # Extract system messages for system_instruction parameter
                    # Gemini doesn't support system role in contents - it causes
                    # consecutive user messages which results in empty responses
                    for msg in messages:
                        if msg.get("role") == "system":
                            if isinstance(msg.get("content"), str):
                                system_messages.append(msg["content"])
                        else:
                            non_system_messages.append(msg)

                    # Add system instruction if present
                    if system_messages:
                        gemini_config["system_instruction"] = "\n\n".join(
                            system_messages
                        )

                    # Convert non-system messages to Google format
                    gemini_contents: list[dict[str, Any]] = []
                    for msg in non_system_messages:
                        # Map roles to Google's expected values (user, model)
                        role = msg.get("role", "user")
                        if role == "assistant":
                            role = "model"

                        # Handle different content formats
                        if isinstance(msg.get("content"), str):
                            # Simple string content
                            gemini_contents.append(
                                {"role": role, "parts": [{"text": msg["content"]}]}
                            )
                        elif isinstance(msg.get("parts"), list):
                            # Already in Google format (from tool calling loop)
                            # But still need to ensure role is correct
                            msg_copy = msg.copy()
                            msg_copy["role"] = role
                            gemini_contents.append(msg_copy)
                        elif isinstance(msg.get("content"), list):
                            # Content is a list of parts (Anthropic format) - skip for now
                            # This shouldn't happen with Google provider in tool loop
                            continue
                        else:
                            # Empty or unknown format, skip
                            continue
                    contents: ContentListUnionDict = cast(
                        ContentListUnionDict, gemini_contents
                    )
                else:
                    contents = prompt

                gemini_response: GenerateContentResponse = (
                    await client.aio.models.generate_content(
                        model=model,
                        contents=contents,
                        config=cast(GenerateContentConfigDict, gemini_config)  # pyright: ignore[reportInvalidCast]
                        if gemini_config
                        else None,
                    )
                )

                # Extract text content and function calls from response
                text_parts: list[str] = []
                gemini_tool_calls: list[dict[str, Any]] = []

                if gemini_response.candidates and gemini_response.candidates[0].content:
                    for part in gemini_response.candidates[0].content.parts or []:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            tool_call_data: dict[str, Any] = {
                                "id": f"call_{fc.name}_{len(gemini_tool_calls)}",
                                "name": fc.name,
                                "input": dict(fc.args) if fc.args else {},
                            }
                            # Preserve thought_signature if present (required by Gemini)
                            if (
                                hasattr(part, "thought_signature")
                                and part.thought_signature
                            ):
                                tool_call_data["thought_signature"] = (
                                    part.thought_signature
                                )
                            gemini_tool_calls.append(tool_call_data)

                text_content = "\n".join(text_parts) if text_parts else ""
                input_token_count = (
                    gemini_response.usage_metadata.prompt_token_count or 0
                    if gemini_response.usage_metadata
                    else 0
                )
                output_token_count = (
                    gemini_response.usage_metadata.candidates_token_count or 0
                    if gemini_response.usage_metadata
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
                    input_tokens=input_token_count,
                    output_tokens=output_token_count,
                    finish_reasons=[finish_reason],
                    tool_calls_made=gemini_tool_calls,
                )

            else:
                gemini_config["response_mime_type"] = "application/json"
                gemini_config["response_schema"] = response_model

                gemini_response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=cast(GenerateContentConfigDict, gemini_config),  # pyright: ignore[reportInvalidCast]
                )

                input_token_count = (
                    gemini_response.usage_metadata.prompt_token_count or 0
                    if gemini_response.usage_metadata
                    else 0
                )
                output_token_count = (
                    gemini_response.usage_metadata.candidates_token_count or 0
                    if gemini_response.usage_metadata
                    else 0
                )
                finish_reason = (
                    gemini_response.candidates[0].finish_reason.name
                    if gemini_response.candidates
                    and gemini_response.candidates[0].finish_reason
                    else "stop"
                )

                # Validate that parsed content matches the response model
                if not isinstance(gemini_response.parsed, response_model):
                    raise ValueError(
                        f"Parsed content does not match the response model: {gemini_response.parsed} != {response_model}"
                    )

                return HonchoLLMCallResponse(
                    content=gemini_response.parsed,
                    input_tokens=input_token_count,
                    output_tokens=output_token_count,
                    finish_reasons=[finish_reason],
                    tool_calls_made=[],
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

            # TODO: figure out why groq returns unknown type and fix it
            response: ChatCompletion = await client.chat.completions.create(  # pyright: ignore
                **groq_params
            )
            if response.choices[0].message.content is None:  # pyright: ignore
                raise ValueError("No content in response")

            # Safely extract usage and finish_reason
            usage = response.usage  # pyright: ignore
            finish_reason = response.choices[0].finish_reason  # pyright: ignore

            # Handle response model parsing for Groq
            cache_creation, cache_read = extract_openai_cache_tokens(usage)
            if response_model:
                try:
                    json_content = json.loads(response.choices[0].message.content)  # pyright: ignore
                    parsed_content = response_model.model_validate(json_content)

                    return HonchoLLMCallResponse(
                        content=parsed_content,
                        input_tokens=usage.prompt_tokens if usage else 0,  # pyright: ignore
                        output_tokens=usage.completion_tokens if usage else 0,  # pyright: ignore
                        cache_creation_input_tokens=cache_creation,
                        cache_read_input_tokens=cache_read,
                        finish_reasons=[finish_reason] if finish_reason else [],
                        tool_calls_made=[],
                    )
                except (json.JSONDecodeError, ValidationError, ValueError) as e:
                    raise ValueError(
                        f"Failed to parse Groq response as {response_model}: {e}. Raw content: {response.choices[0].message.content}"  # pyright: ignore
                    ) from e
            else:
                return HonchoLLMCallResponse(
                    content=response.choices[0].message.content,  # pyright: ignore
                    input_tokens=usage.prompt_tokens if usage else 0,  # pyright: ignore
                    output_tokens=usage.completion_tokens if usage else 0,  # pyright: ignore
                    cache_creation_input_tokens=cache_creation,
                    cache_read_input_tokens=cache_read,
                    finish_reasons=[finish_reason] if finish_reason else [],
                    tool_calls_made=[],
                )


async def handle_streaming_response(
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

            # For response models, we need to request JSON and parse manually
            # Note: Streaming with response_model is not ideal but we'll accumulate and parse at the end
            if response_model or json_mode:
                # Add JSON schema instructions to the prompt if using response_model
                if response_model:
                    schema_json = json.dumps(
                        response_model.model_json_schema(), indent=2
                    )
                    anthropic_params["messages"][-1]["content"] += (
                        f"\n\nRespond with valid JSON matching this schema:\n{schema_json}"
                    )
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
                usage = final_message.usage
                output_tokens = usage.output_tokens if usage else None
                yield HonchoLLMCallStreamChunk(
                    content="",
                    is_done=True,
                    finish_reasons=[final_message.stop_reason]
                    if final_message.stop_reason
                    else [],
                    output_tokens=output_tokens,
                )

        case AsyncOpenAI():
            openai_params: dict[str, Any] = {
                "model": params["model"],
                "messages": params["messages"],
                "stream": True,
                "stream_options": {"include_usage": True},
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
            finish_reason: str | None = None
            usage_chunk_received = False
            async for chunk in openai_stream:  # pyright: ignore
                chunk = cast(ChatCompletionChunk, chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield HonchoLLMCallStreamChunk(content=content)
                # Track finish_reason when it appears (before usage chunk)
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                # Check for usage info in chunk (with include_usage, this is a separate chunk with empty choices)
                if hasattr(chunk, "usage") and chunk.usage:
                    yield HonchoLLMCallStreamChunk(
                        content="",
                        is_done=True,
                        finish_reasons=[finish_reason] if finish_reason else [],
                        output_tokens=chunk.usage.completion_tokens,
                    )
                    usage_chunk_received = True

            # If stream ended without usage chunk (interrupted), still yield final chunk
            if not usage_chunk_received and finish_reason:
                logger.warning("OpenAI stream ended without usage chunk (interrupted)")
                yield HonchoLLMCallStreamChunk(
                    content="",
                    is_done=True,
                    finish_reasons=[finish_reason],
                    output_tokens=None,
                )

        case genai.Client():
            prompt_text = params["messages"][0]["content"] if params["messages"] else ""

            if response_model is not None:
                response_stream = await client.aio.models.generate_content_stream(
                    model=params["model"],
                    contents=prompt_text,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": response_model,
                    },
                )
            else:
                response_stream = await client.aio.models.generate_content_stream(
                    model=params["model"],
                    contents=prompt_text,
                    config={
                        "response_mime_type": "application/json" if json_mode else None,
                    },
                )

            final_chunk = None
            async for chunk in response_stream:
                if chunk.text:
                    yield HonchoLLMCallStreamChunk(content=chunk.text)
                final_chunk = chunk

            finish_reason = "stop"  # Default fallback
            gemini_output_tokens: int | None = None
            if (
                final_chunk
                and hasattr(final_chunk, "candidates")
                and final_chunk.candidates
                and hasattr(final_chunk.candidates[0], "finish_reason")
                and final_chunk.candidates[0].finish_reason
            ):
                finish_reason = final_chunk.candidates[0].finish_reason.name

            # Extract output tokens from usage_metadata if available
            if (
                final_chunk
                and hasattr(final_chunk, "usage_metadata")
                and final_chunk.usage_metadata
                and hasattr(final_chunk.usage_metadata, "candidates_token_count")
            ):
                gemini_output_tokens = (
                    final_chunk.usage_metadata.candidates_token_count or None
                )

            yield HonchoLLMCallStreamChunk(
                content="",
                is_done=True,
                finish_reasons=[finish_reason],
                output_tokens=gemini_output_tokens,
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
