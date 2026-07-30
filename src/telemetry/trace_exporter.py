"""CloudEvents exporter: turns a CapturedLLMCall into trace events.

Registered into the `src/llm/capture.py` exporter registry at startup when
`TELEMETRY.TRACE_PAYLOADS_ENABLED` is on. For each captured call it emits:
- one `trace.content` per unique message/output/thinking/tool-schema (deduped
  per run so each ships once), and
- one `llm.call.traced` carrying the span-tree correlation + content refs +
  a self-contained accounting copy.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.llm.capture import (
    ROLE_OUTPUT,
    ROLE_THINKING,
    ROLE_TOOL_SCHEMA,
    CapturedLLMCall,
    clip_for_trace,
    compute_content_hash,
)
from src.telemetry import trace_session
from src.telemetry.events import emit_trace
from src.telemetry.events.trace import LLMCallTracedEvent, TraceContentEvent

logger = logging.getLogger(__name__)


class TraceExporter:
    """`LLMCallExporter` that ships replay-grade content to the trace stream."""

    def export(self, call: CapturedLLMCall) -> None:
        # Double-gate (the exporter is only registered when on, but a config
        # flip or a stray registration shouldn't leak payloads).
        if not settings.TELEMETRY.TRACE_PAYLOADS_ENABLED:
            return
        purposes = settings.TELEMETRY.TRACE_PURPOSES
        if purposes and call.call_purpose not in purposes:
            return

        run_key = call.trace_id or call.span_id or call.run_id or ""
        was_truncated = call.input_truncated

        # --- Context window: reuse precomputed input-message hashes ---
        input_message_refs: list[str] = []
        for message in call.input_messages:
            input_message_refs.append(message.content_hash)
            self._emit_content(
                run_key,
                content_hash=message.content_hash,
                role=message.role,
                content=message.content,
                tool_call_id=message.tool_call_id,
                honcho_authored=message.role == "system",
                tool_calls=message.tool_calls,
            )

        # --- Tool schemas (Honcho-authored, content-addressed) ---
        tool_schema_refs: list[str] = []
        for schema in call.tool_schemas:
            ref, truncated = self._emit_hashed_content(
                run_key, ROLE_TOOL_SCHEMA, schema, honcho_authored=True
            )
            was_truncated = was_truncated or truncated
            tool_schema_refs.append(ref)

        # --- Output content / thinking ---
        output_content_ref: str | None = None
        if call.output_content not in (None, ""):
            output_content_ref, truncated = self._emit_hashed_content(
                run_key, ROLE_OUTPUT, call.output_content
            )
            was_truncated = was_truncated or truncated

        output_thinking_ref: str | None = None
        if call.thinking_content:
            output_thinking_ref, truncated = self._emit_hashed_content(
                run_key, ROLE_THINKING, call.thinking_content
            )
            was_truncated = was_truncated or truncated

        signatures = [
            block["signature"]
            for block in call.thinking_blocks
            if block.get("signature")
        ]

        emit_trace(
            LLMCallTracedEvent(
                trace_id=call.trace_id,
                span_id=call.span_id,
                parent_span_id=call.parent_span_id,
                iteration=call.iteration,
                step_seq=call.step_seq,
                attempt=call.attempt,
                was_fallback=call.was_fallback,
                call_purpose=call.call_purpose,
                parent_category=call.parent_category,
                session_id=call.session_id,
                transport=call.transport,  # pyright: ignore[reportArgumentType]
                provider_label=call.provider_label,
                model=call.model,
                input_message_refs=input_message_refs,
                tool_schema_refs=tool_schema_refs,
                tool_choice=call.tool_choice,
                output_content_ref=output_content_ref,
                output_tool_calls=call.output_tool_calls,
                output_thinking_ref=output_thinking_ref,
                output_signatures=signatures,
                finish_reason=call.finish_reason,
                provider_input_tokens=call.input_tokens,
                provider_output_tokens=call.output_tokens,
                cache_read_tokens=call.cache_read_tokens,
                cache_creation_tokens=call.cache_creation_tokens,
                was_truncated=was_truncated,
            )
        )

    def _emit_hashed_content(
        self,
        run_key: str,
        role: str,
        raw_content: Any,
        *,
        honcho_authored: bool = False,
    ) -> tuple[str, bool]:
        """Clip + hash a non-message content value and emit it. Returns (hash, truncated)."""
        content, truncated = clip_for_trace(raw_content)
        content_hash = compute_content_hash(role, content, None)
        self._emit_content(
            run_key,
            content_hash=content_hash,
            role=role,
            content=content,
            tool_call_id=None,
            honcho_authored=honcho_authored,
        )
        return content_hash, truncated

    def _emit_content(
        self,
        run_key: str,
        *,
        content_hash: str,
        role: str,
        content: Any,
        tool_call_id: str | None,
        honcho_authored: bool,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Emit one trace.content, deduped per run (skip if already shipped)."""
        if not trace_session.mark_emitted(run_key, content_hash):
            return
        emit_trace(
            TraceContentEvent(
                content_hash=content_hash,
                role=role,
                content=content,
                tool_call_id=tool_call_id,
                honcho_authored=honcho_authored,
                tool_calls=tool_calls or [],
            )
        )


__all__ = ["TraceExporter"]
