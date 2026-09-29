"""Langfuse projection over the captured LLM trace stream.

`LangfuseExporter` is an `LLMCallExporter` and a `SpanTreeExporter`, active when
Langfuse keys are configured. It receives captured calls, run/step
lifecycle records, and executed tool calls one at a time and rebuilds the
Langfuse trace tree from their ids, since there is no live span nesting to
inherit:

    Trace (id = create_trace_id(seed=honcho trace_id))
     └─ [dream root]  (multi-specialist agents only — one "Dream" span per trace)
         └─ run agent   (one per (run_id, agent_type); name = track_name)
            └─ step span (one per (agent_type, iteration); name = "<track> step")
               ├─ generation (one per CapturedLLMCall; name = "<track> generation")
               └─ tool span  (one per executed tool call; sibling of generation)

Root, run, and step spans open and close with their lifecycle records
(`CapturedSpan`), so they cover the whole run, including work before the first
LLM call. A call whose run was never reported gets its run/step spans created at
first use instead. Ids are tracked in `langfuse_session`. Single-shot callers
(deriver/summarizer, `run_id is None`) skip the run/step wrappers and put the
generation at the trace root.

The Dreamer runs two specialists (deduction + induction) under one run_id, so its
branches hang off a single synthetic "Dream" root to keep the trace
single-rooted; single-specialist agents (dialectic) let their run span be the
root. Every observation carries the trace-level user and name, and exactly one
per trace is exported as a root (see `_attach`).

Best-effort throughout: every export is wrapped so telemetry can never break the
LLM call path.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.config import settings
from src.llm.capture import CapturedLLMCall, CapturedSpan, CapturedToolCall
from src.telemetry import langfuse_session

logger = logging.getLogger(__name__)

# finish_reason values that mark the generation as failed.
_ERROR_FINISHES = frozenset({"error", "cancelled"})

# Records that carry the trace/run identity fields the mappers read.
_Traced = CapturedLLMCall | CapturedSpan


class LangfuseExporter:
    """Projects captured calls, spans, and tool calls onto Langfuse traces."""

    def export(self, call: CapturedLLMCall) -> None:
        if not settings.langfuse_exporter_enabled:
            return
        try:
            self._export(call)
        except Exception:  # pragma: no cover - best-effort telemetry
            logger.debug("Langfuse exporter failed", exc_info=True)

    def export_span(self, span: CapturedSpan) -> None:
        if not settings.langfuse_exporter_enabled:
            return
        try:
            self._export_span(span)
        except Exception:  # pragma: no cover - best-effort telemetry
            logger.debug("Langfuse span exporter failed", exc_info=True)

    def export_tool_call(self, tool_call: CapturedToolCall) -> None:
        if not settings.langfuse_exporter_enabled:
            return
        try:
            self._export_tool_call(tool_call)
        except Exception:  # pragma: no cover - best-effort telemetry
            logger.debug("Langfuse tool exporter failed", exc_info=True)

    def _export(self, call: CapturedLLMCall) -> None:
        from langfuse import get_client

        client = get_client()
        seed = call.trace_id or call.run_id or call.span_id
        if not seed:
            return
        lf_trace_id = client.create_trace_id(seed=seed)
        # Export runs right after the provider call returns, so "now" stands in
        # for the call's end and duration_ms backdates its start.
        end_ns = time.time_ns()
        start_ns = (
            end_ns - int(call.duration_ms * 1_000_000)
            if call.duration_ms is not None
            else None
        )

        # Agentic runs (run_id set: dialectic / dreamer) get a run span and
        # per-iteration step spans; single-shot calls put the generation at root.
        # Branch = agent_type so co-trace specialists (dreamer) don't collide.
        parent_span_id: str | None = None
        if call.run_id is not None:
            branch = call.agent_type or "_"
            langfuse_session.note_trace_name(lf_trace_id, self._trace_name(call))
            # Multi-specialist agents (the Dreamer runs deduction + induction in
            # ONE trace) hang every branch off a single synthetic trace root, so
            # the trace has one root instead of one per specialist. That root
            # also stamps the trace attrs. Single-specialist agents (dialectic)
            # get None here and let their run span be the root.
            root_span_id = self._ensure_trace_root(client, lf_trace_id, call, start_ns)
            run_span_id = langfuse_session.ensure_run_span(
                lf_trace_id,
                branch,
                lambda: self._create_span(
                    client,
                    lf_trace_id,
                    parent_span_id=root_span_id,
                    name=call.track_name or "LLM run",
                    metadata=self._metadata(call),
                    call=call,
                    start_ns=start_ns,
                    as_type="agent",
                ),
            )
            parent_span_id = run_span_id
            if call.iteration is not None and run_span_id is not None:
                if langfuse_session.is_open(lf_trace_id, run_span_id):
                    # The agent reports its own steps; a call outside one (the
                    # streamed tail) nests directly under the run.
                    parent_span_id = (
                        langfuse_session.step_span_id(
                            lf_trace_id, branch, call.iteration
                        )
                        or run_span_id
                    )
                else:
                    parent_span_id = langfuse_session.ensure_step_span(
                        lf_trace_id,
                        branch,
                        call.iteration,
                        lambda: self._create_span(
                            client,
                            lf_trace_id,
                            parent_span_id=run_span_id,
                            name=self._step_name(call),
                            metadata=self._step_metadata(call),
                            call=call,
                            start_ns=start_ns,
                        ),
                    )

        self._create_generation(
            client,
            lf_trace_id,
            parent_span_id=parent_span_id,
            call=call,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    def _export_span(self, span: CapturedSpan) -> None:
        from langfuse import get_client

        client = get_client()
        seed = span.trace_id or span.run_id or span.span_id
        if not seed:
            return
        lf_trace_id = client.create_trace_id(seed=seed)
        langfuse_session.note_trace_name(lf_trace_id, self._trace_name(span))
        branch = span.agent_type or "_"
        if span.phase == "end":
            self._close_span(lf_trace_id, branch, span)
            return

        if span.kind == "trace":
            obs = self._open_span(
                client,
                lf_trace_id,
                parent_span_id=None,
                name=self._trace_name(span) or "Dream",
                metadata=self._root_metadata(span),
                span=span,
            )
            if obs.id is not None:
                langfuse_session.register_root(lf_trace_id, obs.id, obs)
        elif span.kind == "run":
            root_span_id = self._ensure_trace_root(client, lf_trace_id, span, None)
            obs = self._open_span(
                client,
                lf_trace_id,
                parent_span_id=root_span_id,
                name=span.track_name or "LLM run",
                metadata=self._metadata(span),
                span=span,
                as_type="agent",
            )
            if obs.id is not None:
                langfuse_session.register_run(lf_trace_id, branch, obs.id, obs)
        elif span.iteration is not None:
            run_span_id = langfuse_session.run_span_id(lf_trace_id, branch)
            if run_span_id is None:
                return
            obs = self._open_span(
                client,
                lf_trace_id,
                parent_span_id=run_span_id,
                name=self._step_name(span),
                metadata={**self._metadata(span), "iteration": str(span.iteration)},
                span=span,
            )
            if obs.id is not None:
                langfuse_session.register_step(
                    lf_trace_id, branch, span.iteration, obs.id, obs
                )

    def _close_span(self, lf_trace_id: str, branch: str, span: CapturedSpan) -> None:
        """End the span a lifecycle end record refers to, plus any open children."""
        rest: list[Any] = []
        if span.kind == "trace":
            obs, rest = langfuse_session.close_root(lf_trace_id)
        elif span.kind == "run":
            obs, rest = langfuse_session.close_run(lf_trace_id, branch)
        elif span.iteration is not None:
            obs = langfuse_session.close_step(lf_trace_id, branch, span.iteration)
        else:
            obs = None
        for child in rest:
            child.end(end_time=span.time_ns)
        if obs is None:
            return
        if span.output is not None:
            obs.update(output=span.output)
        if span.is_error:
            obs.update(level="ERROR")
        obs.end(end_time=span.time_ns)

    def _export_tool_call(self, tool_call: CapturedToolCall) -> None:
        from langfuse import get_client

        client: Any = get_client()
        lf_trace_id = client.create_trace_id(seed=tool_call.run_id)
        step_span_id = (
            langfuse_session.step_span_id(
                lf_trace_id, tool_call.agent_type, tool_call.iteration
            )
            if tool_call.iteration is not None
            else None
        )
        parent_span_id = step_span_id or langfuse_session.run_span_id(
            lf_trace_id, tool_call.agent_type
        )
        if parent_span_id is None:
            return
        # Dispatched as the tool returns, so "now" is its end.
        end_ns = time.time_ns()
        obs = client.start_observation(
            trace_context=self._trace_context(lf_trace_id, parent_span_id),
            name=tool_call.name,
            as_type="tool",
            input=tool_call.input,
            output=tool_call.output,
            metadata=self._tool_metadata(tool_call),
            level="ERROR" if tool_call.is_error else None,
        )
        backdated = self._backdate_start(
            obs, end_ns - int(tool_call.duration_ms * 1_000_000)
        )
        self._attach(obs, parent_span_id, langfuse_session.trace_name(lf_trace_id))
        obs.end(end_time=end_ns if backdated else None)

    # -- observation builders ------------------------------------------------

    def _ensure_trace_root(
        self,
        client: Any,
        lf_trace_id: str,
        call: _Traced,
        start_ns: int | None,
    ) -> str | None:
        """Single branch-agnostic trace root for multi-specialist agents.

        The Dreamer's deduction + induction specialists share one trace (same
        run_id) but each builds its own run span with no parent — so Langfuse
        sees two roots, races the trace name between them, and renders the
        specialists as separate sub-traces. One synthetic "Dream" root gives the
        trace a single root with both specialists nested beneath.
        Single-specialist agents (dialectic) return None and let their run span
        be the root.
        """
        if call.parent_category != "dream":
            return None
        return langfuse_session.ensure_trace_root(
            lf_trace_id,
            lambda: self._create_span(
                client,
                lf_trace_id,
                parent_span_id=None,
                name=self._trace_name(call) or "Dream",
                metadata=self._root_metadata(call),
                call=call,
                start_ns=start_ns,
            ),
        )

    def _create_span(
        self,
        client: Any,
        lf_trace_id: str,
        *,
        parent_span_id: str | None,
        name: str,
        metadata: dict[str, str],
        call: _Traced,
        start_ns: int | None,
        as_type: str = "span",
    ) -> str | None:
        """Create a (run or step) span, returning its OTEL span id.

        Created-and-ended immediately: nesting is by id, so children link fine to
        an already-ended parent. The span starts with the first call that mints
        it and ends when that call does; later calls are not reflected, since
        the stream has no 'run finished' signal.
        """
        obs = client.start_observation(
            trace_context=self._trace_context(lf_trace_id, parent_span_id),
            name=name,
            as_type=as_type,
            metadata=metadata,
        )
        self._backdate_start(obs, start_ns)
        self._attach(obs, parent_span_id, self._trace_name(call))
        obs.end()
        return getattr(obs, "id", None)

    def _open_span(
        self,
        client: Any,
        lf_trace_id: str,
        *,
        parent_span_id: str | None,
        name: str,
        metadata: dict[str, str],
        span: CapturedSpan,
        as_type: str = "span",
    ) -> Any:
        """Start a span that stays open until its lifecycle end record."""
        obs = client.start_observation(
            trace_context=self._trace_context(lf_trace_id, parent_span_id),
            name=name,
            as_type=as_type,
            input=span.input,
            metadata=metadata,
        )
        self._attach(obs, parent_span_id, self._trace_name(span))
        return obs

    def _create_generation(
        self,
        client: Any,
        lf_trace_id: str,
        *,
        parent_span_id: str | None,
        call: CapturedLLMCall,
        start_ns: int | None,
        end_ns: int,
    ) -> None:
        level = "ERROR" if (call.finish_reason in _ERROR_FINISHES) else None
        obs = client.start_observation(
            trace_context=self._trace_context(lf_trace_id, parent_span_id),
            name=self._gen_name(call),
            as_type="generation",
            model=call.model,
            input=self._input(call),
            output=self._output(call),
            metadata=self._step_metadata(call),
            usage_details=self._usage(call),
            level=level,
        )
        backdated = self._backdate_start(obs, start_ns)
        self._attach(obs, parent_span_id, self._trace_name(call))
        # end_ns predates the SDK's own start stamp, so only pin it when backdated.
        obs.end(end_time=end_ns if backdated else None)

    @staticmethod
    def _backdate_start(obs: Any, start_ns: int | None) -> bool:
        """Move an observation's start back to when its captured call began.

        `start_observation` has no start-time parameter, so this rewrites the
        OTEL SDK span's start before `end()`; the span processor reads it only
        at export. No-op for non-recording spans (Langfuse disabled). Returns
        whether the start was moved.
        """
        span = getattr(obs, "_otel_span", None)
        if (
            span is None
            or start_ns is None
            or getattr(span, "_start_time", None) is None
        ):
            return False
        span._start_time = start_ns
        return True

    def _attach(
        self, obs: Any, parent_span_id: str | None, trace_name: str | None
    ) -> None:
        """Stamp trace attrs and export `obs` as the trace root or a plain child."""
        self._stamp_trace_attrs(obs, trace_name)
        if parent_span_id is None:
            self._detach_placeholder_parent(obs)
        else:
            self._demote_from_root(obs)

    @staticmethod
    def _detach_placeholder_parent(obs: Any) -> None:
        """Export a parentless observation as a true root.

        `start_observation(trace_context=...)` without a `parent_span_id` nests
        the span under a random, never-exported span id. This clears the OTEL SDK
        span's private `_parent` before `end()`.
        """
        span = getattr(obs, "_otel_span", None)
        if span is not None and getattr(span, "_parent", None) is not None:
            span._parent = None

    @staticmethod
    def _demote_from_root(obs: Any) -> None:
        """Clear the AS_ROOT flag the SDK stamps on every `trace_context` span.

        Several root-flagged spans in one trace make Langfuse pick the trace's
        root and name from whichever it ingests first.
        """
        span = getattr(obs, "_otel_span", None)
        if span is None:
            return
        from langfuse import LangfuseOtelSpanAttributes as Attr

        span.set_attribute(Attr.AS_ROOT, False)

    @staticmethod
    def _trace_context(lf_trace_id: str, parent_span_id: str | None) -> dict[str, str]:
        ctx: dict[str, str] = {"trace_id": lf_trace_id}
        if parent_span_id is not None:
            ctx["parent_span_id"] = parent_span_id
        return ctx

    @staticmethod
    def _stamp_trace_attrs(obs: Any, trace_name: str | None) -> None:
        """Stamp the trace-level user and name; applied to every observation.

        Deliberately does NOT set a Langfuse session: no Honcho construct is a
        conversation thread. A dialectic chat is a one-shot query scoped to a
        session, not a turn in a multi-turn dialectic exchange (no such primitive
        exists), so grouping independent queries under one Langfuse session would
        invent a conversation that isn't there. The Honcho session rides in
        metadata (`honcho_session`) instead — a correlation key, not a group."""
        span = getattr(obs, "_otel_span", None)
        if span is None:
            return
        from langfuse import LangfuseOtelSpanAttributes as Attr

        span.set_attribute(Attr.TRACE_USER_ID, str(settings.NAMESPACE))
        if trace_name:
            span.set_attribute(Attr.TRACE_NAME, trace_name)

    # -- field mappers (port of runtime._base_metadata/_step_metadata) -------

    @staticmethod
    def _metadata(call: _Traced) -> dict[str, str]:
        # `trace_id` is the run grouping key (also handy for cross-referencing the
        # CloudEvents stream). `span_id`/`parent_span_id` are intentionally omitted
        # until the source mints distinct per-call span ids: today every call in a
        # run shares span_id == trace_id == run_id, so surfacing them here only
        # duplicates trace_id and misleads. Re-add once the source differentiates.
        md: dict[str, str] = {"namespace": str(settings.NAMESPACE)}
        for key, value in (
            ("workspace_name", call.workspace_name),
            ("call_purpose", call.call_purpose),
            ("agent_type", call.agent_type),
            ("observer", call.observer),
            ("observed", call.observed),
            ("peer_name", call.peer_name),
            ("trace_id", call.trace_id),
            # Honcho session as a correlation key, NOT a Langfuse session — see
            # `_stamp_trace_attrs`. Lets you filter "queries scoped to session X"
            # without falsely grouping one-shot dialectic queries as a thread.
            ("honcho_session", call.session_id),
        ):
            if value is not None:
                md[key] = str(value)
        return md

    @staticmethod
    def _root_metadata(call: _Traced) -> dict[str, str]:
        # Branch-agnostic: the synthetic dream root spans both specialists, so it
        # carries only trace-level fields — not a single specialist's agent_type/
        # observer/observed/call_purpose.
        md: dict[str, str] = {"namespace": str(settings.NAMESPACE)}
        for key, value in (
            ("workspace_name", call.workspace_name),
            ("trace_id", call.trace_id),
        ):
            if value is not None:
                md[key] = str(value)
        return md

    def _step_metadata(self, call: CapturedLLMCall) -> dict[str, str]:
        md = self._metadata(call)
        if call.iteration is not None:
            md["iteration"] = str(call.iteration)
        md["step_seq"] = str(call.step_seq)
        md["attempt"] = str(call.attempt)
        md["provider"] = str(call.transport)
        md["model"] = str(call.model)
        return md

    @staticmethod
    def _tool_metadata(tool_call: CapturedToolCall) -> dict[str, str]:
        md: dict[str, str] = {
            "namespace": str(settings.NAMESPACE),
            "agent_type": tool_call.agent_type,
            "trace_id": tool_call.run_id,
            "tool_call_seq": str(tool_call.tool_call_seq),
        }
        for key, value in (
            ("workspace_name", tool_call.workspace_name),
            ("iteration", tool_call.iteration),
            ("tool_call_id", tool_call.tool_call_id),
        ):
            if value is not None:
                md[key] = str(value)
        return md

    @staticmethod
    def _trace_name(call: _Traced) -> str | None:
        # Branch-agnostic trace label: the Dreamer's two specialists share one
        # trace, so the trace name must not be pinned to whichever specialist's
        # run span stamped it first. Per-branch identity stays on the run spans.
        if call.parent_category == "dream":
            return "Dream"
        return call.track_name

    @staticmethod
    def _step_name(call: _Traced) -> str:
        # Canonical, index-free name: Langfuse aggregates step spans by name and
        # the iteration/step_seq/attempt ride on metadata (see _step_metadata).
        return f"{call.track_name} step" if call.track_name else "Agent step"

    @staticmethod
    def _gen_name(call: CapturedLLMCall) -> str:
        return f"{call.track_name} generation" if call.track_name else "generation"

    @staticmethod
    def _input(call: CapturedLLMCall) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for message in call.input_messages:
            entry: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_call_id is not None:
                entry["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                entry["tool_calls"] = message.tool_calls
            out.append(entry)
        return out

    @staticmethod
    def _output(call: CapturedLLMCall) -> Any:
        """Plain text when that's all there is, else one assistant message."""
        if not call.output_tool_calls and not call.thinking_content:
            return call.output_content
        message: dict[str, Any] = {"role": "assistant", "content": call.output_content}
        if call.thinking_content:
            message["thinking"] = call.thinking_content
        if call.output_tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name"),
                        "arguments": json.dumps(tc.get("input") or {}, default=str),
                    },
                }
                for tc in call.output_tool_calls
            ]
        return message

    @staticmethod
    def _usage(call: CapturedLLMCall) -> dict[str, int]:
        return {
            "input": call.input_tokens,
            "output": call.output_tokens,
            "cache_read_input_tokens": call.cache_read_tokens,
            "cache_creation_input_tokens": call.cache_creation_tokens,
        }


__all__ = ["LangfuseExporter"]
