"""Single-capture point for replay-grade LLM call content + the exporter seam.

The executor boundary (`honcho_llm_call_inner`) is the one place that holds the
normalized `messages` array and the provider response together. Historically
three consumers each re-serialized that content their own way (Langfuse, the
JSONL reasoning trace, and — new in the CloudEvents tracing spec — the
`llm.call.traced` / `trace.content` events). Divergent canonicalization would
produce divergent `content_hash` values and silently break the O(N)
content-addressing dedup, so this module owns the ONE canonical form.

`build_captured_call` constructs a `CapturedLLMCall` once; `dispatch_captured_call`
fans it out to every registered `LLMCallExporter`. The CloudEvents trace exporter
(in `src/telemetry/`) registers itself at startup — telemetry depends on `src/llm/`,
never the reverse, so this module stays import-cycle free (it imports only stdlib,
`src.config`, and sibling `src/llm/` modules).

All capture is best-effort: `dispatch_captured_call` swallows exporter exceptions
so telemetry can never break the LLM call path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.config import settings

from .backend import CompletionResult as BackendCompletionResult
from .backend import ToolCallResult
from .types import LLMTelemetryContext

logger = logging.getLogger(__name__)

# Sentinel roles for non-message content stored in the shared content store so
# the same hash+dedup machinery covers them. They never collide with real
# conversation roles ("user"/"assistant"/"system"/"tool").
ROLE_OUTPUT = "assistant"
ROLE_TOOL_SCHEMA = "__tool_schema__"
ROLE_THINKING = "__thinking__"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for every content hash.

    `sort_keys` + tight separators + `default=str` make the output stable
    regardless of dict ordering or non-JSON-native values (e.g. a stray enum),
    so the same logical content always hashes to the same digest across
    processes and exporters.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def compute_content_hash(role: str, content: Any, tool_call_id: str | None) -> str:
    """Content hash covering the FULL message identity, not just the text.

    `role` and `tool_call_id` live inside the hash so identical text under
    different roles (or attached to different tool calls) can never collide.
    This is the single source of truth — `src/telemetry/events/trace.py` imports
    this exact function so producer and event agree byte-for-byte.
    """
    digest = hashlib.sha256(
        canonical_json(
            {"role": role, "content": content, "tool_call_id": tool_call_id}
        ).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def clip_for_trace(content: Any) -> tuple[Any, bool]:
    """Clip a content value to `TELEMETRY.TRACE_MAX_BYTES`, returning (content, truncated).

    Truncation happens BEFORE hashing so the hash matches the bytes actually
    shipped. Only oversized string content is clipped (with a marker); non-string
    structured content is left intact (its size is bounded in practice and
    clipping it would corrupt structure). Returns the input unchanged when it
    fits or when the cap is non-positive.
    """
    max_bytes = settings.TELEMETRY.TRACE_MAX_BYTES
    if max_bytes <= 0 or not isinstance(content, str):
        return content, False
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content, False
    marker = "…[truncated]"
    keep = max(0, max_bytes - len(marker.encode("utf-8")))
    clipped = encoded[:keep].decode("utf-8", errors="ignore") + marker
    return clipped, True


@dataclass(slots=True)
class CapturedMessage:
    """One input message, canonicalized + content-addressed.

    `content` is the (possibly clipped) structured message Honcho holds —
    multi-part blocks, tool results, cache hints — not flattened text.
    `content_hash` is computed over this exact `content` so the ref and the
    shipped `trace.content` always agree.
    """

    role: str
    content: Any
    tool_call_id: str | None
    content_hash: str
    truncated: bool = False


@dataclass(slots=True)
class CapturedLLMCall:
    """Everything one LLM call needs to be reconstructed, captured once.

    Built at the executor boundary from `LLMTelemetryContext` + the provider
    result. Exporters read this; they never re-extract from the raw response.
    """

    # Correlation (span tree)
    trace_id: str | None
    span_id: str | None
    parent_span_id: str | None
    iteration: int | None
    step_seq: int
    attempt: int
    was_fallback: bool
    run_id: str | None
    # Path identity
    workspace_name: str | None
    call_purpose: str | None
    parent_category: str | None
    agent_type: str | None
    # Honcho conversation grouping key (raw Session.id; NAMESPACE-prefixed only
    # at the Langfuse boundary). None for sessionless calls.
    session_id: str | None
    # Peer context + human-readable trace/run name. Carried so the Langfuse
    # projection has the same attribution the old inline `_base_metadata` did
    # (the CloudEvents TraceExporter ignores them).
    observer: str | None
    observed: str | None
    peer_name: str | None
    track_name: str | None
    transport: str
    provider_label: str | None
    model: str
    # Context window
    input_messages: list[CapturedMessage]
    tool_schemas: list[dict[str, Any]]
    tool_choice: Any
    # Output (replay-grade)
    output_content: Any
    output_tool_calls: list[dict[str, Any]]
    thinking_content: str | None
    thinking_blocks: list[dict[str, Any]]
    reasoning_details: list[dict[str, Any]]
    finish_reason: str | None
    # Accounting copy (so the trace stream stands alone)
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    was_stream: bool
    # True when any input message was clipped to TRACE_MAX_BYTES.
    input_truncated: bool = False


def build_captured_messages(
    messages: list[dict[str, Any]],
    memo: dict[int, CapturedMessage] | None,
) -> tuple[list[CapturedMessage], bool]:
    """Canonicalize + content-hash each input message, O(N) via the memo.

    The conversation is append-only across tool-loop iterations and the message
    dict objects are reused (`truncate_messages_to_fit` rebuilds the list but not
    the dicts), so memoizing the fully-built `CapturedMessage` by `id(message)`
    means each message is clipped and hashed exactly once across the whole span —
    a memo hit skips both `clip_for_trace` (a full re-encode of string content)
    and the sha256. `memo` is None for single-shot callers (one call — nothing to
    memoize). Returns (messages, any_truncated).
    """
    captured: list[CapturedMessage] = []
    any_truncated = False
    for message in messages:
        key = id(message)
        cached = memo.get(key) if memo is not None else None
        if cached is not None:
            captured.append(cached)
            any_truncated = any_truncated or cached.truncated
            continue
        role = str(message.get("role", ""))
        raw_content = message.get("content")
        tool_call_id = message.get("tool_call_id")
        content, truncated = clip_for_trace(raw_content)
        any_truncated = any_truncated or truncated
        captured_message = CapturedMessage(
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            content_hash=compute_content_hash(role, content, tool_call_id),
            truncated=truncated,
        )
        if memo is not None:
            memo[key] = captured_message
        captured.append(captured_message)
    return captured, any_truncated


def build_captured_call(
    *,
    telemetry: LLMTelemetryContext | None,
    transport: str,
    provider_label: str | None,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    result: BackendCompletionResult | None,
    attempt: int,
    was_fallback: bool,
    was_stream: bool,
    finish_reason: str | None,
) -> CapturedLLMCall:
    """Assemble a `CapturedLLMCall` from telemetry + the provider result.

    `result` is None on the error path — output fields collapse to empty and
    `finish_reason` carries the outcome ("error"/"cancelled"). The hash memo
    rides on `telemetry.hash_memo` (seeded once per span by the tool loop).
    """
    memo = telemetry.hash_memo if telemetry is not None else None
    captured_messages, input_truncated = build_captured_messages(messages, memo)

    output_tool_calls = [
        _tool_call_to_dict(tc) for tc in (result.tool_calls if result else [])
    ]

    return CapturedLLMCall(
        trace_id=telemetry.trace_id if telemetry else None,
        span_id=telemetry.span_id if telemetry else None,
        parent_span_id=telemetry.exported_parent_span_id() if telemetry else None,
        iteration=telemetry.iteration if telemetry else None,
        step_seq=telemetry.step_seq if telemetry else 0,
        attempt=attempt,
        was_fallback=was_fallback,
        run_id=telemetry.run_id if telemetry else None,
        workspace_name=telemetry.workspace_name if telemetry else None,
        call_purpose=telemetry.call_purpose if telemetry else None,
        parent_category=telemetry.parent_category if telemetry else None,
        agent_type=telemetry.agent_type if telemetry else None,
        session_id=telemetry.session_id if telemetry else None,
        observer=telemetry.observer if telemetry else None,
        observed=telemetry.observed if telemetry else None,
        peer_name=telemetry.peer_name if telemetry else None,
        track_name=telemetry.track_name if telemetry else None,
        transport=transport,
        provider_label=provider_label,
        model=model,
        input_messages=captured_messages,
        tool_schemas=list(tools) if tools else [],
        tool_choice=tool_choice,
        output_content=result.content if result else None,
        output_tool_calls=output_tool_calls,
        thinking_content=result.thinking_content if result else None,
        thinking_blocks=result.thinking_blocks if result else [],
        reasoning_details=result.reasoning_details if result else [],
        finish_reason=finish_reason,
        input_tokens=result.input_tokens if result else 0,
        output_tokens=result.output_tokens if result else 0,
        cache_read_tokens=result.cache_read_input_tokens if result else 0,
        cache_creation_tokens=result.cache_creation_input_tokens if result else 0,
        was_stream=was_stream,
        input_truncated=input_truncated,
    )


def _tool_call_to_dict(tool_call: ToolCallResult) -> dict[str, Any]:
    """Normalize a ToolCallResult to a replay-grade dict (incl thought_signature)."""
    out: dict[str, Any] = {
        "id": tool_call.id,
        "name": tool_call.name,
        "input": tool_call.input,
    }
    if tool_call.thought_signature is not None:
        out["thought_signature"] = tool_call.thought_signature
    return out


@runtime_checkable
class LLMCallExporter(Protocol):
    """A sink that consumes a `CapturedLLMCall`. Implementations must be
    best-effort internally; `dispatch_captured_call` also guards against raises."""

    def export(self, call: CapturedLLMCall) -> None: ...


_EXPORTERS: list[LLMCallExporter] = []


def register_exporter(exporter: LLMCallExporter) -> None:
    """Register an exporter (idempotent on identity). Called at startup."""
    if exporter not in _EXPORTERS:
        _EXPORTERS.append(exporter)


def clear_exporters() -> None:
    """Drop all exporters — used on shutdown and in tests."""
    _EXPORTERS.clear()


def has_exporters() -> bool:
    """True when at least one exporter is registered.

    The executor checks this BEFORE building a `CapturedLLMCall` so the O(N)
    hashing cost is never paid when payload capture is off.
    """
    return bool(_EXPORTERS)


def dispatch_captured_call(call: CapturedLLMCall) -> None:
    """Fan a captured call out to every exporter, swallowing any exporter error."""
    for exporter in _EXPORTERS:
        try:
            exporter.export(call)
        except Exception:  # pragma: no cover - best-effort telemetry
            logger.debug("LLM call exporter failed", exc_info=True)


__all__ = [
    "ROLE_OUTPUT",
    "ROLE_THINKING",
    "ROLE_TOOL_SCHEMA",
    "CapturedLLMCall",
    "CapturedMessage",
    "LLMCallExporter",
    "build_captured_call",
    "build_captured_messages",
    "canonical_json",
    "clear_exporters",
    "clip_for_trace",
    "compute_content_hash",
    "dispatch_captured_call",
    "has_exporters",
    "register_exporter",
]
