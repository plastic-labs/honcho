"""Replay-grade payload events for full-fidelity LLM tracing.

Two ground-truth (never-sampled) events that make the CloudEvents stream carry
the exact context a model saw, content-addressed to keep payload O(N):

- ``LLMCallTracedEvent`` (``llm.call.traced``) — one per LLM call. Carries
  span-tree correlation, path identity, content *references* (hashes, not bytes)
  for the context window and the replay-grade output, plus a self-contained
  accounting copy. It deliberately does NOT claim a join to
  ``llm.call.completed`` — cost is computable from the trace alone, so the
  billing and audit streams stay decoupled.
- ``TraceContentEvent`` (``trace.content``) — one per unique message, emitted
  once per run and referenced by hash. ``content_hash`` covers the full message
  identity ({role, content, tool_call_id}) so identical text under different
  roles can't collide. ``generate_id()`` is overridden to derive the CloudEvent
  id from the hash with NO timestamp, so accidental re-sends of identical
  content dedupe at the transport layer too.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field

from src.config import ModelTransport
from src.telemetry.events.base import BaseEvent

__all__ = ["EmbeddingCallTracedEvent", "LLMCallTracedEvent", "TraceContentEvent"]


class LLMCallTracedEvent(BaseEvent):
    """One replay-grade record per LLM call. Ground-truth, never sampled.

    Content fields are *references* (content hashes) into the ``trace.content``
    store, never inline bytes
    """

    _event_type: ClassVar[str] = "llm.call.traced"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "trace"
    _volume_class: ClassVar[str] = "ground_truth"

    # --- Correlation (span tree) ---
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    iteration: int | None = None
    step_seq: int = 0
    attempt: int = 1
    was_fallback: bool = False
    parent_event_id: str | None = None

    # --- Path identity ---
    call_purpose: str | None = None
    parent_category: str | None = None
    # Used for grouping traces
    session_id: str | None = None
    transport: ModelTransport
    provider_label: str | None = None
    model: str

    # --- Context window (content-addressed) ---
    input_message_refs: list[str] = Field(default_factory=list)
    system_prompt_ref: str | None = None
    tool_schema_refs: list[str] = Field(default_factory=list)
    tool_choice: Any = None

    # --- Output (replay-grade) ---
    output_content_ref: str | None = None
    output_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    output_thinking_ref: str | None = None
    output_signatures: list[str] = Field(default_factory=list)
    # Reserved: Honcho captures the normalized request/response, not wire bytes.
    raw_response_ref: str | None = None
    finish_reason: str | None = None

    # --- Accounting copy (stream stands alone; NOT joined to llm.call.completed) ---
    provider_input_tokens: int = 0
    provider_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    was_truncated: bool = False

    def get_resource_id(self) -> str:
        """Idempotency key. ``tool_call_seq`` is deliberately absent — it indexed
        tool *executions*, not LLM calls, and was never a valid join field."""
        return f"{self.span_id}:{self.iteration}:{self.attempt}:{self.step_seq}"


class EmbeddingCallTracedEvent(BaseEvent):
    """One trace-stream record per embedding-provider call."""

    _event_type: ClassVar[str] = "embedding.call.traced"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "trace"
    _volume_class: ClassVar[str] = "ground_truth"

    # --- Correlation (span tree) ---
    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None
    iteration: int | None = None
    step_seq: int = 0
    attempt: int = 1
    session_id: str | None = None

    # --- Path identity ---
    call_purpose: str | None = None
    parent_category: str | None = None
    provider: str
    model: str

    # --- Accounting copy ---
    # v1: input tokens are a tiktoken ESTIMATE (no authoritative provider count is
    # plumbed yet); output tokens are always 0 (embeddings produce none).
    provider_input_tokens: int = 0
    provider_output_tokens: int = 0
    input_count: int = 0
    was_truncated: bool = False

    def get_resource_id(self) -> str:
        return f"{self.span_id}:embedding:{self.call_purpose}:{self.input_count}"


class TraceContentEvent(BaseEvent):
    """One unique message in the content store. Ground-truth, never sampled.

    The hash covers the full message identity, and the event id derives from the
    hash with no timestamp.
    """

    _event_type: ClassVar[str] = "trace.content"
    _schema_version: ClassVar[int] = 1
    _category: ClassVar[str] = "trace"
    _volume_class: ClassVar[str] = "ground_truth"

    content_hash: str
    role: str
    # Message text, normalized across providers.
    content: Any = None
    tool_call_id: str | None = None
    # Tool calls in a unified {id, name, input} shape (provider-agnostic).
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    # Tags Honcho-authored content (system prompts, scaffold) so tenant-facing
    # views can withhold globally-shared content (the §6.3 access invariant —
    # dedup is global, the content store has no tenant column).
    honcho_authored: bool = False

    def get_resource_id(self) -> str:
        return self.content_hash

    def generate_id(self) -> str:
        """Content-addressed id with NO timestamp/version.

        Overrides the base (which folds timestamp + honcho_version) so any
        cross-process or cross-retry re-send of the same content collides on
        the same id and dedupes at the transport layer.
        """
        digest = self.content_hash.split(":", 1)[-1]
        return f"content_{digest[:22]}"
