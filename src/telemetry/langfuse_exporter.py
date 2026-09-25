"""Langfuse projection over the captured LLM trace stream.

`LangfuseExporter` is an `LLMCallExporter`, active when
`LANGFUSE_EXPORTER_MODE == "exporter"`. It receives one `CapturedLLMCall` at a
time and rebuilds the Langfuse trace tree from the ids on each call, since there
is no live span nesting to inherit:

    Trace (id = create_trace_id(seed=honcho trace_id))
     └─ [dream root]  (multi-specialist agents only — one "Dream" span per trace)
         └─ run span    (one per (run_id, agent_type); name = track_name)
            └─ step span (one per (agent_type, iteration); name = "<track> step")
               ├─ generation (one per CapturedLLMCall; name = "<track> generation")
               └─ tool span  (one per requested tool call; sibling of generation)

Run and step spans are created once per trace and reused as the `parent_span_id`
of later calls (tracked in `langfuse_session`). Single-shot callers
(deriver/summarizer, `run_id is None`) skip the run/step wrappers and put the
generation at the trace root.

The Dreamer runs two specialists (deduction + induction) under one run_id, so its
branches hang off a single synthetic "Dream" root to keep the trace
single-rooted; single-specialist agents (dialectic) let their run span be the
root. Keeping exactly one root is also why child spans are demoted from the SDK's
auto-root flag (see `_demote_from_root`).

Best-effort throughout: every export is wrapped so telemetry can never break the
LLM call path.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings
from src.llm.capture import CapturedLLMCall
from src.telemetry import langfuse_session

logger = logging.getLogger(__name__)

# finish_reason values that mark the generation as failed.
_ERROR_FINISHES = frozenset({"error", "cancelled"})


class LangfuseExporter:
    """`LLMCallExporter` that projects captured calls onto Langfuse traces."""

    def export(self, call: CapturedLLMCall) -> None:
        if not settings.langfuse_exporter_enabled:
            return
        try:
            self._export(call)
        except Exception:  # pragma: no cover - best-effort telemetry
            logger.debug("Langfuse exporter failed", exc_info=True)

    def _export(self, call: CapturedLLMCall) -> None:
        from langfuse import get_client

        client = get_client()
        seed = call.trace_id or call.run_id or call.span_id
        if not seed:
            return
        lf_trace_id = client.create_trace_id(seed=seed)

        # Agentic runs (run_id set: dialectic / dreamer) get a run span and
        # per-iteration step spans; single-shot calls put the generation at root.
        # Branch = agent_type so co-trace specialists (dreamer) don't collide.
        parent_span_id: str | None = None
        if call.run_id is not None:
            branch = call.agent_type or "_"
            # Multi-specialist agents (the Dreamer runs deduction + induction in
            # ONE trace) hang every branch off a single synthetic trace root, so
            # the trace has one root instead of one per specialist. That root
            # also stamps the trace attrs. Single-specialist agents (dialectic)
            # get None here and let their run span be the root.
            root_span_id = self._ensure_trace_root(client, lf_trace_id, call)
            run_span_id = langfuse_session.ensure_run_span(
                lf_trace_id,
                branch,
                lambda should_stamp: self._create_span(
                    client,
                    lf_trace_id,
                    parent_span_id=root_span_id,
                    name=call.track_name or "LLM run",
                    metadata=self._metadata(call),
                    # The synthetic root stamps the trace attrs when present;
                    # otherwise the first branch's run span does.
                    stamp_trace=should_stamp and root_span_id is None,
                    call=call,
                ),
            )
            parent_span_id = run_span_id
            if call.iteration is not None and run_span_id is not None:
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
                        stamp_trace=False,
                        call=call,
                    ),
                )

        self._create_generation(
            client,
            lf_trace_id,
            parent_span_id=parent_span_id,
            # Single-shot: the generation is the trace root, so it stamps the
            # trace attrs. Agentic: the first branch's run span already did.
            stamp_trace=call.run_id is None,
            call=call,
        )

        # Tool calls the model requested this iteration: siblings of the
        # generation under the step span. Skipped at the trace root (single-shot
        # callers don't use tools) since there's no step to anchor them.
        if parent_span_id is not None and call.output_tool_calls:
            for seq, tool_call in enumerate(call.output_tool_calls):
                self._create_tool_span(
                    client, lf_trace_id, parent_span_id, seq, tool_call, call
                )

    # -- observation builders ------------------------------------------------

    def _ensure_trace_root(
        self, client: Any, lf_trace_id: str, call: CapturedLLMCall
    ) -> str | None:
        """Single branch-agnostic trace root for multi-specialist agents.

        The Dreamer's deduction + induction specialists share one trace (same
        run_id) but each builds its own run span with no parent — so Langfuse
        sees two roots, races the trace name between them, and renders the
        specialists as separate sub-traces. One synthetic "Dream" root (which
        also stamps the trace attrs) gives the trace a single root with both
        specialists nested beneath. Single-specialist agents (dialectic) return
        None and let their run span be the root.
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
                stamp_trace=True,
                call=call,
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
        stamp_trace: bool,
        call: CapturedLLMCall,
    ) -> str | None:
        """Create a (run or step) span, returning its OTEL span id.

        Created-and-ended immediately: nesting is by id, so children link fine to
        an already-ended parent. Span durations are therefore approximate — an
        accepted v1 trade for not having a 'run finished' signal in the stream.
        """
        obs = client.start_observation(
            trace_context=self._trace_context(lf_trace_id, parent_span_id),
            name=name,
            as_type="span",
            metadata=metadata,
        )
        if stamp_trace:
            self._stamp_trace_attrs(obs, call)
        if parent_span_id is not None:
            self._demote_from_root(obs)
        obs.end()
        return getattr(obs, "id", None)

    def _create_generation(
        self,
        client: Any,
        lf_trace_id: str,
        *,
        parent_span_id: str | None,
        stamp_trace: bool,
        call: CapturedLLMCall,
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
        if stamp_trace:
            self._stamp_trace_attrs(obs, call)
        if parent_span_id is not None:
            self._demote_from_root(obs)
        obs.end()

    def _create_tool_span(
        self,
        client: Any,
        lf_trace_id: str,
        parent_span_id: str,
        seq: int,
        tool_call: dict[str, Any],
        call: CapturedLLMCall,
    ) -> None:
        """Create a tool span for one requested tool call, under the step span.

        Built from the model's request (`output_tool_calls`): tool name + input
        args. Result/duration/error aren't on the captured call (they live on
        AgentToolCallCompletedEvent) — a later enrichment, not v1.
        """
        obs = client.start_observation(
            trace_context=self._trace_context(lf_trace_id, parent_span_id),
            name=str(tool_call.get("name") or "tool"),
            as_type="tool",
            input=tool_call.get("input"),
            metadata=self._tool_metadata(call, seq),
        )
        self._demote_from_root(obs)  # always a child of the step span
        obs.end()

    @staticmethod
    def _demote_from_root(obs: Any) -> None:
        """Clear the AS_ROOT flag the SDK auto-stamps on a child observation.

        `start_observation(trace_context={"trace_id": ...})` marks EVERY span it
        mints with `AS_ROOT=True` (langfuse `_client/client.py`) — including the
        step/generation/tool spans we link under a run span by id. With several
        root-flagged spans in one trace, Langfuse resolves the trace's root (and
        therefore its name) from whichever it ingests first: a race that names a
        dialectic trace after a child ("... step"/"... generation") and renders
        children as if each were its own trace. Demoting every span that has a
        real parent leaves exactly one root, making name + nesting deterministic.
        Verified empirically against Langfuse cloud (the dangling remote-parent
        id on the surviving root is benign and unavoidable — it's present even
        with native context nesting).
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

    def _stamp_trace_attrs(self, obs: Any, call: CapturedLLMCall) -> None:
        """Stamp user/name on the trace via the root observation's span.

        Called once per trace, on the first branch's run span (decided by
        `langfuse_session.ensure_run_span`) or, for single-shot calls, on the
        generation (its own trace).

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
        trace_name = self._trace_name(call)
        if trace_name:
            span.set_attribute(Attr.TRACE_NAME, trace_name)

    # -- field mappers (port of runtime._base_metadata/_step_metadata) -------

    @staticmethod
    def _metadata(call: CapturedLLMCall) -> dict[str, str]:
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
    def _root_metadata(call: CapturedLLMCall) -> dict[str, str]:
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

    def _tool_metadata(self, call: CapturedLLMCall, seq: int) -> dict[str, str]:
        md = self._step_metadata(call)
        md["tool_call_seq"] = str(seq)
        return md

    @staticmethod
    def _trace_name(call: CapturedLLMCall) -> str | None:
        # Branch-agnostic trace label: the Dreamer's two specialists share one
        # trace, so the trace name must not be pinned to whichever specialist's
        # run span stamped it first. Per-branch identity stays on the run spans.
        if call.parent_category == "dream":
            return "Dream"
        return call.track_name

    @staticmethod
    def _step_name(call: CapturedLLMCall) -> str:
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
        if isinstance(call.output_content, str) and call.output_content.strip():
            return call.output_content
        if call.output_tool_calls:
            return {"tool_calls": [tc.get("name") for tc in call.output_tool_calls]}
        return call.output_content

    @staticmethod
    def _usage(call: CapturedLLMCall) -> dict[str, int]:
        return {
            "input": call.input_tokens,
            "output": call.output_tokens,
            "cache_read_input_tokens": call.cache_read_tokens,
            "cache_creation_input_tokens": call.cache_creation_tokens,
        }


__all__ = ["LangfuseExporter"]
