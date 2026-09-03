"""OpenAI-compatible ``/chat/completions``, answered without inference."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.mock_provider.coerce import as_dict, as_str
from src.mock_provider.schema_gen import generate
from src.mock_provider.schemas import ChatCompletionRequest, ChatMessage

router = APIRouter(tags=["mock-provider"])

# Honcho's json_object mode injects the schema into the prompt text rather than
# into response_format (see _apply_json_object_mode in the OpenAI backend), so
# the only machine-readable copy of the schema is inside a message.
_SCHEMA_HINT = re.compile(r"schema:\s*(\{)", re.IGNORECASE)


def _completion_id(body: ChatCompletionRequest) -> str:
    """Stable id, so a replayed request is byte-identical."""
    digest = hashlib.sha256(
        body.model_dump_json(exclude_none=True).encode()
    ).hexdigest()
    return f"chatcmpl-mock-{digest[:24]}"


def _extract_balanced_json(text: str, start: int) -> dict[str, Any] | None:
    """Read one balanced ``{...}`` beginning at ``start`` and parse it.

    A plain regex cannot do this — a JSON Schema contains nested objects, and
    braces inside string literals must not count toward the depth.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return as_dict(parsed)
    return None


def _schema_from_messages(messages: list[ChatMessage]) -> dict[str, Any] | None:
    """Recover an injected schema from the prompt, for json_object mode."""
    for message in reversed(messages):
        content = as_str(message.content)
        if content is None:
            continue
        for match in _SCHEMA_HINT.finditer(content):
            candidate = _extract_balanced_json(content, match.start(1))
            if candidate and ("properties" in candidate or "$defs" in candidate):
                return candidate
    return None


def _response_content(body: ChatCompletionRequest) -> str:
    """The assistant message body: schema-conforming JSON, or prose."""
    response_format = body.response_format

    if response_format is not None:
        kind = as_str(response_format.get("type"))
        if kind == "json_schema":
            wrapper = as_dict(response_format.get("json_schema"))
            if wrapper is not None:
                schema = as_dict(wrapper.get("schema"))
                if schema is not None:
                    return json.dumps(generate(schema))
            # A json_schema request whose schema we cannot read must not fall
            # through to prose — that is the silent-empty failure this mock
            # exists to avoid. An empty object at least parses.
            return "{}"
        if kind == "json_object":
            schema = _schema_from_messages(body.messages)
            return json.dumps(generate(schema)) if schema else "{}"

    return (
        "[mock] This is a synthetic response from Honcho's mock provider. "
        "No model was called."
    )


def _usage(body: ChatCompletionRequest, content: str) -> dict[str, int]:
    """Rough token accounting, so cost telemetry has plausible numbers."""
    prompt_chars = 0
    for message in body.messages:
        text = as_str(message.content)
        if text is not None:
            prompt_chars += len(text)
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = max(1, len(content) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _created() -> int:
    # Fixed rather than time-based: a mock that changes its output between
    # identical calls defeats the point.
    return 1577836800  # 2020-01-01T00:00:00Z


async def _stream(
    completion_id: str, model: str, content: str, usage: dict[str, int] | None
) -> AsyncIterator[bytes]:
    """Stream ``content``, ending on a usage chunk when ``usage`` is given."""

    def chunk(payload: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": _created(),
        "model": model,
    }
    yield chunk(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
    )
    yield chunk(
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}
            ],
        }
    )
    yield chunk(
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    # The usage chunk is conditional: the real API emits it only when
    # stream_options.include_usage is set, and ends the stream on it — so it
    # must come last and must carry choices: []. Honcho's own backend always
    # asks for it (_build_params in the OpenAI backend), but a caller that does
    # not must not receive a chunk it never requested.
    if usage is not None:
        yield chunk({**base, "choices": [], "usage": usage})
    yield b"data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(body: ChatCompletionRequest) -> Any:
    model = body.model or "mock-model"
    content = _response_content(body)
    usage = _usage(body, content)
    completion_id = _completion_id(body)

    if body.stream:
        include_usage = (
            body.stream_options is not None and body.stream_options.include_usage
        )
        return StreamingResponse(
            _stream(completion_id, model, content, usage if include_usage else None),
            media_type="text/event-stream",
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": _created(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": None,
                    "tool_calls": None,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }
