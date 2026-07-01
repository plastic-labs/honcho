from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from functools import cache
from typing import Any, cast

from openai import BadRequestError, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from src.exceptions import ValidationException
from src.llm.backend import CompletionResult, StreamChunk, ToolCallResult
from src.llm.request_builder import apply_sdk_passthroughs
from src.llm.structured_output import (
    StructuredOutputError,
    empty_structured_output,
    repair_response_model_json,
    validate_structured_output,
)

logger = logging.getLogger(__name__)


@cache
def _json_object_instruction(response_format: type[BaseModel]) -> str:
    """Schema-injection instruction for json_object mode.

    The JSON schema is static per response_format class, so cache the serialized
    instruction — the deriver issues one structured call per batch on the worker
    hot path and would otherwise re-walk the schema + re-serialize it every call.
    """
    # "JSON" must appear in the messages to satisfy the json_object contract.
    return (
        "You must respond with a single JSON object that conforms exactly to "
        "the following JSON schema. Do not include any text, markdown, or code "
        "fences outside the JSON object.\n\nJSON schema:\n"
        f"{json.dumps(response_format.model_json_schema())}"
    )


def _uses_max_completion_tokens(model: str) -> bool:
    """OpenAI reasoning models (gpt-5 family + o-series) require
    ``max_completion_tokens`` instead of the classic ``max_tokens`` parameter.

    Matches: gpt-5, gpt-5-anything, gpt-5.anything (incl. gpt-5.4, gpt-5.4-mini),
    o1*, o3*, o4*. Anything else (gpt-4.x, gpt-4o, chat models on proxies)
    stays on ``max_tokens``.
    """
    m = model.lower()
    if m == "gpt-5" or m.startswith("gpt-5-") or m.startswith("gpt-5."):
        return True
    for prefix in ("o1", "o3", "o4"):
        if m == prefix or m.startswith(prefix + "-"):
            return True
    return False


def extract_openai_reasoning_content(response: Any) -> str | None:
    try:
        message = response.choices[0].message
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            reasoning_parts: list[str] = []
            for detail in message.reasoning_details:
                detail_content = getattr(detail, "content", None)
                if isinstance(detail_content, str) and detail_content:
                    reasoning_parts.append(detail_content)
                elif isinstance(detail, dict):
                    detail_dict = cast(dict[str, Any], detail)
                    dict_content = detail_dict.get("content")
                    if isinstance(dict_content, str) and dict_content:
                        reasoning_parts.append(dict_content)
            if reasoning_parts:
                return "\n".join(reasoning_parts)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            return message.reasoning_content
    except (AttributeError, IndexError, TypeError):
        return None
    return None


def extract_openai_reasoning_details(response: Any) -> list[dict[str, Any]]:
    try:
        message = response.choices[0].message
        if hasattr(message, "reasoning_details") and message.reasoning_details:
            details: list[dict[str, Any]] = []
            for detail in message.reasoning_details:
                if hasattr(detail, "model_dump"):
                    dumped = detail.model_dump()
                    if isinstance(dumped, dict):
                        details.append(cast(dict[str, Any], dumped))
                elif isinstance(detail, dict):
                    details.append(cast(dict[str, Any], detail))
                else:
                    detail_content = getattr(detail, "content", None)
                    if isinstance(detail_content, str) and detail_content:
                        details.append({"content": detail_content})
            return details
    except (AttributeError, IndexError, TypeError):
        return []
    return []


def extract_openai_cache_tokens(usage: Any) -> tuple[int, int]:
    if not usage:
        return 0, 0

    cache_read = 0
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        details = usage.prompt_tokens_details
        if hasattr(details, "cached_tokens") and details.cached_tokens:
            cache_read = details.cached_tokens

    if cache_read == 0:
        if hasattr(usage, "cache_read_input_tokens") and usage.cache_read_input_tokens:
            cache_read = usage.cache_read_input_tokens
        elif hasattr(usage, "cached_tokens") and usage.cached_tokens:
            cache_read = usage.cached_tokens

    cache_creation = 0
    if (
        hasattr(usage, "cache_creation_input_tokens")
        and usage.cache_creation_input_tokens
    ):
        cache_creation = usage.cache_creation_input_tokens

    return cache_creation, cache_read


class OpenAIBackend:
    """Provider backend wrapping AsyncOpenAI."""

    def __init__(self, client: Any) -> None:
        self._client: Any = client

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> CompletionResult:
        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            thinking_effort=thinking_effort,
            thinking_budget_tokens=thinking_budget_tokens,
            extra_params=extra_params,
        )

        if isinstance(response_format, type):
            if self._structured_output_mode(extra_params) == "json_object":
                self._apply_json_object_mode(params, response_format)
                response = await self._client.chat.completions.create(**params)
                # A loose provider that returns nothing shouldn't crash the call.
                content = self._parse_or_repair_structured_content(
                    response, response_format, model, empty_on_missing=True
                )
                return self._normalize_response(response, content_override=content)
            params["response_format"] = response_format
            try:
                response = await self._client.chat.completions.parse(**params)
            except LengthFinishReasonError as exc:
                # Truncated output: repair the partial content directly. repair
                # handles empty/unrepairable JSON with its own model-aware fallback
                # (PromptRepresentation -> empty, others -> raise), which differs
                # from the parse-fallback terminal below, so it stays a direct call.
                truncated = exc.completion
                raw_content = truncated.choices[0].message.content or ""
                content = repair_response_model_json(
                    raw_content,
                    response_format,
                    model,
                )
                return self._normalize_response(
                    truncated,
                    content_override=content,
                )
            except BadRequestError:
                # A 400 means the provider rejected the request shape — most
                # often it doesn't support OpenAI Structured Outputs (json_schema).
                # Retrying or re-requesting won't help (it rejects the same shape
                # again, the latency trap of #797), so return empty rather than
                # erroring existing flows. The warning is the signal to set
                # structured_output_mode=json_object. There is no response body to
                # account for, so token usage is legitimately zero here.
                logger.warning(
                    "Structured output via json_schema rejected by model %s; "
                    + "set structured_output_mode=json_object if the provider does "
                    + "not support OpenAI Structured Outputs.",
                    model,
                )
                # empty_structured_output() validates {} against the model, which
                # itself raises if the model has required fields. Fall back to
                # empty string content rather than letting that escape the handler.
                try:
                    fallback_content: Any = empty_structured_output(response_format)
                except ValidationError:
                    fallback_content = ""
                return CompletionResult(content=fallback_content)
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return self._normalize_response(
                    response,
                    content_override=validate_structured_output(
                        parsed, response_format
                    ),
                )
            # parse() returned no model: repair raw content, surface a refusal,
            # or raise so the retry/fallback chain engages on a junk response.
            content = self._parse_or_repair_structured_content(
                response, response_format, model, empty_on_missing=False
            )
            return self._normalize_response(response, content_override=content)
        if response_format is not None:
            params["response_format"] = response_format

        if extra_params and extra_params.get("json_mode"):
            params["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(**params)
        return self._normalize_response(response)

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None = None,
        stop: list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_effort: str | None = None,
        max_output_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        params = self._build_params(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens or max_tokens,
            temperature=temperature,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            thinking_effort=thinking_effort,
            thinking_budget_tokens=thinking_budget_tokens,
            extra_params=extra_params,
        )
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        if isinstance(response_format, type):
            if self._structured_output_mode(extra_params) == "json_object":
                # Inject the schema into the prompt for providers without
                # json_schema support; repair happens downstream.
                self._apply_json_object_mode(params, response_format)
            else:
                # Streaming create() can't take a BaseModel like parse() does;
                # convert to a json_schema dict.
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": response_format.model_json_schema(),
                    },
                }
        elif response_format is not None:
            params["response_format"] = response_format
        elif extra_params and extra_params.get("json_mode"):
            params["response_format"] = {"type": "json_object"}

        response_stream = await self._client.chat.completions.create(**params)
        finish_reason: str | None = None
        usage_chunk_received = False
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield StreamChunk(content=chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if hasattr(chunk, "usage") and chunk.usage:
                yield StreamChunk(
                    is_done=True,
                    finish_reason=finish_reason,
                    output_tokens=chunk.usage.completion_tokens,
                )
                usage_chunk_received = True

        if not usage_chunk_received and finish_reason:
            yield StreamChunk(is_done=True, finish_reason=finish_reason)

    def _build_params(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float | None,
        stop: list[str] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        thinking_effort: str | None,
        thinking_budget_tokens: int | None,
        extra_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if _uses_max_completion_tokens(model):
            params["max_completion_tokens"] = max_tokens
            if extra_params and extra_params.get("verbosity"):
                params["verbosity"] = extra_params["verbosity"]
        else:
            params["max_tokens"] = max_tokens

        if temperature is not None:
            params["temperature"] = temperature

        if thinking_effort:
            params["reasoning_effort"] = thinking_effort

        # Token-budget style thinking is not part of the native OpenAI API, but
        # OpenAI-compatible proxies (OpenRouter, etc.) accept a `reasoning` object
        # on the request body. Pass through via extra_body so it reaches those
        # backends. Operators on providers that need a different shape (e.g.
        # Anthropic-via-Vertex behind litellm wants `thinking`, not `reasoning`)
        # supply that shape via ModelConfig.provider_params.extra_body and unset
        # thinking_budget_tokens themselves — Honcho does not try to translate.
        if thinking_budget_tokens is not None and thinking_budget_tokens > 0:
            params.setdefault("extra_body", {}).setdefault("reasoning", {})[
                "max_tokens"
            ] = thinking_budget_tokens

        if stop:
            params["stop"] = stop
        if tools:
            params["tools"] = self._convert_tools(tools)
            converted_tool_choice = self._convert_tool_choice(tool_choice)
            if converted_tool_choice is not None:
                params["tool_choice"] = converted_tool_choice
        if extra_params:
            for key in (
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "seed",
            ):
                if key in extra_params:
                    params[key] = extra_params[key]
            # Operator escape hatch: forward OpenAI SDK passthrough kwargs from
            # ModelConfig.provider_params. Shallow merge with operator-wins —
            # if the operator supplies `extra_body.reasoning`, it replaces any
            # value Honcho auto-injected above.
            apply_sdk_passthroughs(params, extra_params)
        return params

    def _normalize_response(
        self,
        response: Any,
        *,
        content_override: Any | None = None,
    ) -> CompletionResult:
        usage = response.usage
        finish_reason = response.choices[0].finish_reason
        tool_calls: list[ToolCallResult] = []
        message = response.choices[0].message
        if getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                tool_input: dict[str, Any] = {}
                if tool_call.function.arguments:
                    try:
                        tool_input = json.loads(tool_call.function.arguments)
                    except (json.JSONDecodeError, TypeError) as exc:
                        # Don't log the raw arguments payload — LLM-generated
                        # tool calls can mirror user PII from the prompt into
                        # their arguments, and this runs at WARN level.
                        logger.warning(
                            "Malformed tool arguments for %s (id=%s): %s",
                            tool_call.function.name,
                            tool_call.id,
                            exc.__class__.__name__,
                        )
                tool_calls.append(
                    ToolCallResult(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=tool_input,
                    )
                )

        cache_creation, cache_read = extract_openai_cache_tokens(usage)
        return CompletionResult(
            content=content_override
            if content_override is not None
            else (message.content or ""),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            finish_reason=finish_reason or "stop",
            tool_calls=tool_calls,
            thinking_content=extract_openai_reasoning_content(response),
            reasoning_details=extract_openai_reasoning_details(response),
            raw_response=response,
        )

    @staticmethod
    def _structured_output_mode(extra_params: dict[str, Any] | None) -> str | None:
        # Threaded in via extra_params (see build_config_extra_params).
        if not extra_params:
            return None
        return extra_params.get("structured_output_mode")

    def _apply_json_object_mode(
        self,
        params: dict[str, Any],
        response_format: type[BaseModel],
    ) -> None:
        """Configure params for json_object mode in place (shared by complete/stream).

        Injects the schema into the prompt and requests loose JSON, so providers
        without OpenAI Structured Outputs (json_schema) support still return JSON.
        """
        params["messages"] = self._with_json_schema_instructions(
            params["messages"], response_format
        )
        params["response_format"] = {"type": "json_object"}

    @staticmethod
    def _with_json_schema_instructions(
        messages: list[dict[str, Any]],
        response_format: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Add JSON-schema instructions to a copy of messages for json_object mode.

        The Anthropic backend has its own schema-into-prompt injection
        (``_append_text_to_last_message``); the two are intentionally kept
        separate since the providers want different placement and wording.
        """
        instruction = _json_object_instruction(response_format)
        new_messages = [dict(message) for message in messages]
        first = new_messages[0] if new_messages else None
        # Only merge into a leading system message when its content is a plain
        # string; non-string content (e.g. a list of content parts) would be
        # corrupted by f-string coercion, so prepend a fresh system message.
        if (
            first
            and first.get("role") == "system"
            and isinstance(first.get("content"), str)
        ):
            first["content"] = f"{first['content']}\n\n{instruction}".strip()
        else:
            new_messages.insert(0, {"role": "system", "content": instruction})
        return new_messages

    @staticmethod
    def _parse_or_repair_structured_content(
        response: Any,
        response_format: type[BaseModel],
        model: str,
        *,
        empty_on_missing: bool,
    ) -> BaseModel | str:
        """Validate (or repair) the raw structured content of a response.

        Shared by the json_object path and the json_schema parse() fallbacks
        (truncation, parsed=None). On a contentless response with no refusal,
        ``empty_on_missing`` selects the terminal behavior: json_object returns a
        graceful empty so a loose provider can't crash the call, while json_schema
        raises so the retry/fallback chain engages on a junk response.
        """
        message = response.choices[0].message
        raw_content = message.content or ""
        if raw_content:
            # Fast path: clean JSON validates directly. Only fall back to the
            # repair pipeline when validation fails — repair is comparatively
            # expensive and silently degrades malformed input to an empty model.
            try:
                return validate_structured_output(raw_content, response_format)
            except (StructuredOutputError, ValidationError):
                return repair_response_model_json(raw_content, response_format, model)
        refusal = getattr(message, "refusal", None)
        if refusal:
            return refusal
        if not empty_on_missing:
            raise ValidationException("No parsed content in structured response")
        # empty_structured_output() validates {} against the model, which itself
        # raises if the model has required fields. Fall back to empty string
        # content rather than letting that escape the handler.
        try:
            return empty_structured_output(response_format)
        except ValidationError:
            return ""

    @staticmethod
    def _convert_tool_choice(
        tool_choice: str | dict[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        # Translate Honcho's canonical tool_choice vocabulary to OpenAI's. This
        # mirrors the Anthropic/Gemini backends so a single TOOL_CHOICE value
        # works regardless of which provider a fallback chain lands on. Notably
        # OpenAI has no "any" — it spells the same intent "required".
        if tool_choice is None:
            return None
        if isinstance(tool_choice, dict):
            if "name" in tool_choice:
                return {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
            return tool_choice
        if tool_choice in {"any", "required"}:
            return "required"
        if tool_choice in {"auto", "none"}:
            return tool_choice
        return {"type": "function", "function": {"name": tool_choice}}

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not tools or tools[0].get("type") == "function":
            return tools
        # Tool schemas in src/utils/agent_tools.py use optional fields with
        # defaults and don't declare additionalProperties: false. OpenAI's
        # strict function-calling mode forbids both, so we intentionally
        # don't set strict: True. Standard function calling on GPT-4.x /
        # GPT-5 remains reliable, and this stays compatible with
        # OpenAI-compatible proxies (OpenRouter, Together, vLLM, Ollama)
        # whose strict-mode support is inconsistent.
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
