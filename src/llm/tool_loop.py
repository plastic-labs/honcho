"""Agentic/tool orchestration — the multi-iteration tool execution loop.

`execute_tool_loop` owns:
- initial tool-enabled call
- tool execution
- conversation augmentation with assistant messages + tool results
- max-iteration handling and synthesis call
- stream-final-only mode
- empty-response retry (one retry nudge when the model returns empty content)
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import ModelTransport
from src.exceptions import ValidationException
from src.utils.types import (
    get_last_tool_metadata,
    iteration_scope,
    set_current_iteration,
    set_current_tool_call_seq,
    set_last_tool_metadata,
)

from .executor import honcho_llm_call_inner
from .registry import history_adapter_for_provider
from .runtime import (
    AttemptPlan,
    current_attempt,
    effective_temperature,
)
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    IterationData,
    LLMTelemetryContext,
    StreamingResponseWithMetadata,
    VerbosityType,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _with_iteration_scope(
    fn: Callable[_P, Awaitable[_R]],
) -> Callable[_P, Awaitable[_R]]:
    """Wrap an async tool-loop entry point in `iteration_scope()` so the
    per-iteration ContextVars (iteration, tool_call_seq, provider id, last
    tool metadata) are reset to their pre-call values on exit. Defensive
    against subsequent loops in the same asyncio Task observing stale state.
    """

    @functools.wraps(fn)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with iteration_scope():
            return await fn(*args, **kwargs)

    return wrapper


def _telemetry_for_iteration(
    base: LLMTelemetryContext | None, iteration: int
) -> LLMTelemetryContext | None:
    """Return a copy of `base` with `iteration` set, or None if no base.

    We always copy rather than mutate the caller-supplied context so callers
    that pass the same context into multiple `honcho_llm_call` invocations
    don't see drift across concurrent runs.
    """
    if base is None:
        return None
    return LLMTelemetryContext(
        workspace_name=base.workspace_name,
        call_purpose=base.call_purpose,
        parent_category=base.parent_category,
        run_id=base.run_id,
        iteration=iteration,
        observer=base.observer,
        observed=base.observed,
        peer_name=base.peer_name,
        agent_type=base.agent_type,
        langfuse_session_id=base.langfuse_session_id,
    )


def _emit_agent_iteration(
    telemetry: LLMTelemetryContext | None,
    iteration: int,
    response: HonchoLLMCallResponse[Any],
) -> None:
    """emit AgentIterationEvent after each per-iteration LLM response.

    Fired immediately after `response = await call_func()` in the per-iteration
    loop AND after the max-iteration synthesis call. Emitted regardless of
    whether the model requested tool calls — the no-tool terminating iteration
    still counts as an iteration for cost calibration.

    Skipped when telemetry context is missing or lacks the required agent
    identifiers (no agent → no agent.iteration event).
    """
    if telemetry is None or not telemetry.run_id:
        return
    if not telemetry.parent_category or not telemetry.agent_type:
        # Without agent_type / parent_category we can't fill the event's
        # required fields. Skip rather than emit a half-populated event.
        return
    if not telemetry.workspace_name:
        return
    try:
        # Local import: keeps src/llm/ free of a hard dependency on telemetry
        # at import time so the LLM layer remains usable in unit tests that
        # don't initialize the telemetry stack.
        from src.telemetry.events import AgentIterationEvent, emit

        emit(
            AgentIterationEvent(
                run_id=telemetry.run_id,
                parent_category=telemetry.parent_category,
                agent_type=telemetry.agent_type,
                workspace_name=telemetry.workspace_name,
                observer=telemetry.observer,
                observed=telemetry.observed,
                peer_name=telemetry.peer_name,
                iteration=iteration,
                tool_calls=[tc["name"] for tc in response.tool_calls_made],
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cache_read_tokens=response.cache_read_input_tokens or 0,
                cache_creation_tokens=response.cache_creation_input_tokens or 0,
            )
        )
    except Exception:  # pragma: no cover - telemetry must not raise
        logger.debug("Failed to emit AgentIterationEvent", exc_info=True)


logger = logging.getLogger(__name__)

# Bounds for max_tool_iterations to prevent runaway loops.
MIN_TOOL_ITERATIONS = 1
MAX_TOOL_ITERATIONS = 100


def format_assistant_tool_message(
    provider: ModelTransport,
    content: Any,
    tool_calls: list[dict[str, Any]],
    thinking_blocks: list[dict[str, Any]] | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Format an assistant message with tool calls in provider-native shape."""
    from .backend import CompletionResult as BackendCompletionResult
    from .backend import ToolCallResult

    adapter = history_adapter_for_provider(provider)
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


def append_tool_results(
    provider: ModelTransport,
    tool_results: list[dict[str, Any]],
    conversation_messages: list[dict[str, Any]],
) -> None:
    """Append tool results to `conversation_messages` in provider-native shape."""
    adapter = history_adapter_for_provider(provider)
    conversation_messages.extend(adapter.format_tool_results(tool_results))


async def stream_final_response(
    *,
    winning_plan: AttemptPlan,
    prompt: str,
    max_tokens: int,
    conversation_messages: list[dict[str, Any]],
    response_model: type[BaseModel] | None,
    json_mode: bool,
    temperature: float | None,
    stop_seqs: list[str] | None,
    verbosity: VerbosityType,
    enable_retry: bool,
    retry_attempts: int,
    before_retry_callback: Callable[[Any], None],
    telemetry: LLMTelemetryContext | None = None,
) -> AsyncIterator[HonchoLLMCallStreamChunk]:
    """Stream the final response after tool execution is complete.

    Uses the AttemptPlan captured at the moment streaming began (typically
    the plan whose inner LLM call just succeeded) and pins it across any
    retries of the stream setup. Re-running provider selection here would
    bleed the outer current_attempt ContextVar into streaming retries,
    potentially rolling the selection back to primary after the tool loop
    had already settled on fallback. Tenacity retries re-issue the same
    streaming call against the same pinned model for transient errors.
    """

    # Bump the per-retry attempt index inside `_setup_stream`. The pinned
    # `winning_plan.attempt` is frozen from before retries started; without
    # this counter, every retried stream-setup emit reports the same attempt
    # value — telemetry can't tell the retry sequence apart.
    stream_attempt = 0

    async def _setup_stream() -> AsyncIterator[HonchoLLMCallStreamChunk]:
        nonlocal stream_attempt
        stream_attempt += 1
        # `dataclasses.replace` produces a per-attempt plan with the bumped
        # `attempt` and the real `retry_attempts` budget so the executor's
        # LLMCallCompletedEvent reports attempt=1/2/3 and is_final_attempt
        # correctly across the retry sequence.
        plan_for_attempt = dataclasses.replace(
            winning_plan,
            attempt=stream_attempt,
            retry_attempts=retry_attempts,
        )
        return await honcho_llm_call_inner(
            winning_plan.provider,
            winning_plan.model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            effective_temperature(temperature),
            stop_seqs,
            winning_plan.reasoning_effort,
            verbosity,
            winning_plan.thinking_budget_tokens,
            stream=True,
            client_override=winning_plan.client,
            tools=None,
            tool_choice=None,
            messages=conversation_messages,
            selected_config=winning_plan.selected_config,
            plan=plan_for_attempt,
            telemetry=telemetry,
        )

    if enable_retry:
        wrapped = retry(
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
        )(_setup_stream)
        stream = await wrapped()
    else:
        stream = await _setup_stream()

    async for chunk in stream:
        yield chunk


@_with_iteration_scope
async def execute_tool_loop(
    *,
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
    verbosity: VerbosityType,
    enable_retry: bool,
    retry_attempts: int,
    max_input_tokens: int | None,
    get_attempt_plan: Callable[[], AttemptPlan],
    before_retry_callback: Callable[[Any], None],
    stream_final: bool = False,
    iteration_callback: IterationCallback | None = None,
    telemetry: LLMTelemetryContext | None = None,
) -> HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata:
    """Run the iterative tool calling loop for agentic LLM interactions.

    Loop per iteration:
      1. Make an LLM call with tools available
      2. Execute any tool calls the LLM requests
      3. Append tool results to the conversation
      4. Repeat until the LLM stops calling tools or max iterations reached

    Returns:
        Final HonchoLLMCallResponse with accumulated token counts and tool call
        history, or a StreamingResponseWithMetadata if stream_final=True.
    """
    from .conversation import count_message_tokens, truncate_messages_to_fit

    if not MIN_TOOL_ITERATIONS <= max_tool_iterations <= MAX_TOOL_ITERATIONS:
        raise ValidationException(
            "max_tool_iterations must be in "
            + f"[{MIN_TOOL_ITERATIONS}, {MAX_TOOL_ITERATIONS}]; "
            + f"got {max_tool_iterations}"
        )

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
    # Latch — set when any iteration's input exceeded `max_input_tokens`.
    # Token-based rather than message-count-based: catches both "messages
    # got dropped" and "couldn't drop the last unit but still over cap."
    # Stamped onto the final response so
    # RepresentationCompletedEvent.hit_input_token_cap and
    # DialecticCompletedEvent.hit_input_token_cap reflect the cap hit
    # (the toolless path tracks this in src/llm/api.py:325-340).
    hit_input_token_cap = False
    # Track effective tool_choice — switches from "required"/"any" to "auto" after iter 1.
    effective_tool_choice = tool_choice

    while iteration < max_tool_iterations:
        # Reset attempt counter so each iteration starts with the primary provider.
        current_attempt.set(1)
        logger.debug(f"Tool execution iteration {iteration + 1}/{max_tool_iterations}")

        if max_input_tokens is not None:
            if count_message_tokens(conversation_messages) > max_input_tokens:
                hit_input_token_cap = True
            conversation_messages = truncate_messages_to_fit(
                conversation_messages, max_input_tokens
            )

        async def _call_with_messages(
            effective_tool_choice: str | dict[str, Any] | None = effective_tool_choice,
            conversation_messages: list[dict[str, Any]] = conversation_messages,
            iteration_for_call: int = iteration + 1,
        ) -> HonchoLLMCallResponse[Any]:
            plan = get_attempt_plan()
            return await honcho_llm_call_inner(
                plan.provider,
                plan.model,
                prompt,  # ignored when messages is passed
                max_tokens,
                response_model,
                json_mode,
                effective_temperature(temperature),
                stop_seqs,
                plan.reasoning_effort,
                verbosity,
                plan.thinking_budget_tokens,
                stream=False,
                client_override=plan.client,
                tools=tools,
                tool_choice=effective_tool_choice,
                messages=conversation_messages,
                selected_config=plan.selected_config,
                plan=plan,
                telemetry=_telemetry_for_iteration(telemetry, iteration_for_call),
            )

        if enable_retry:
            call_func = retry(
                stop=stop_after_attempt(retry_attempts),
                wait=wait_exponential(multiplier=1, min=4, max=10),
                before_sleep=before_retry_callback,
            )(_call_with_messages)
        else:
            call_func = _call_with_messages

        response = await call_func()

        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        total_cache_creation_tokens += response.cache_creation_input_tokens
        total_cache_read_tokens += response.cache_read_input_tokens

        # emit one AgentIterationEvent per LLM response BEFORE the
        # no-tool early return. The terminating iteration counts too — it has
        # an empty tool_calls list and is essential for cost calibration.
        _emit_agent_iteration(telemetry, iteration + 1, response)

        if not response.tool_calls_made:
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
                # Snapshot the plan that just succeeded — streaming retries
                # pin to this exact client/model so we don't bounce back to
                # primary after the tool loop settled on fallback.
                winning_plan = get_attempt_plan()
                stream = stream_final_response(
                    winning_plan=winning_plan,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    conversation_messages=conversation_messages,
                    response_model=response_model,
                    json_mode=json_mode,
                    temperature=temperature,
                    stop_seqs=stop_seqs,
                    verbosity=verbosity,
                    enable_retry=enable_retry,
                    retry_attempts=retry_attempts,
                    before_retry_callback=before_retry_callback,
                    telemetry=_telemetry_for_iteration(telemetry, iteration + 1),
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
                    hit_input_token_cap=hit_input_token_cap,
                )

            response.tool_calls_made = all_tool_calls
            response.input_tokens = total_input_tokens
            response.output_tokens = total_output_tokens
            response.cache_creation_input_tokens = total_cache_creation_tokens
            response.cache_read_input_tokens = total_cache_read_tokens
            response.iterations = iteration + 1
            response.hit_input_token_cap = (
                response.hit_input_token_cap or hit_input_token_cap
            )
            return response

        current_provider = get_attempt_plan().provider

        assistant_message = format_assistant_tool_message(
            current_provider,
            response.content,
            response.tool_calls_made,
            response.thinking_blocks,
            response.reasoning_details,
        )
        conversation_messages.append(assistant_message)

        # Telemetry context — 1-indexed iteration.
        set_current_iteration(iteration + 1)

        tool_results: list[dict[str, Any]] = []
        for seq, tool_call in enumerate(response.tool_calls_made):
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            tool_id = tool_call.get("id", "")

            logger.debug(f"Executing tool: {tool_name}")

            # the executor closure reads these from
            # ContextVars to populate AgentToolCallCompletedEvent. Set BEFORE
            # the executor call so two calls to the same tool in one iteration
            # get distinct seq values. Reset last-tool metadata so we never
            # observe stale state from a prior call.
            set_current_tool_call_seq(seq, tool_id or None)
            set_last_tool_metadata({})

            try:
                tool_result = await tool_executor(tool_name, tool_input)
                # Stash ToolResult.metadata on all_tool_calls so
                # specialist rollups can read created/deleted observation
                # counts without round-tripping through the event store.
                tool_result_metadata = get_last_tool_metadata()
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
                        "tool_result_metadata": tool_result_metadata,
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

        append_tool_results(current_provider, tool_results, conversation_messages)

        if iteration_callback is not None:
            try:
                iteration_data = IterationData(
                    iteration=iteration + 1,
                    tool_calls=[tc["name"] for tc in response.tool_calls_made],
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cache_read_tokens=response.cache_read_input_tokens or 0,
                    cache_creation_tokens=response.cache_creation_input_tokens or 0,
                )
                iteration_callback(iteration_data)
            except Exception:
                logger.warning("iteration_callback failed", exc_info=True)

        # After first iteration, switch "required"/"any" → "auto" so the model can stop.
        if iteration == 0 and effective_tool_choice in ("required", "any"):
            effective_tool_choice = "auto"
            logger.debug(
                "Switched tool_choice from 'required'/'any' to 'auto' after first iteration"
            )

        iteration += 1

    logger.warning(
        f"Tool execution loop reached max iterations ({max_tool_iterations})"
    )

    # The max-iteration synthesis call gets iteration N+1 in telemetry so 's
    # AgentIterationEvent and this LLMCallCompletedEvent line up sequentially.
    synthesis_iteration = iteration + 1

    synthesis_prompt = (
        "You have reached the maximum number of tool calls. "
        "Based on all the information you have gathered, provide your final response now. "
        "Do not attempt to call any more tools."
    )
    conversation_messages.append({"role": "user", "content": synthesis_prompt})

    # Truncate again — the per-iteration truncate ran before the last tool
    # call, so appending synthesis_prompt could nudge us back over the cap.
    if max_input_tokens is not None:
        if count_message_tokens(conversation_messages) > max_input_tokens:
            hit_input_token_cap = True
        conversation_messages = truncate_messages_to_fit(
            conversation_messages, max_input_tokens
        )

    if stream_final:
        # Snapshot the plan the loop settled on — streaming retries pin to
        # this exact client/model rather than re-running provider selection.
        winning_plan = get_attempt_plan()
        stream = stream_final_response(
            winning_plan=winning_plan,
            prompt=prompt,
            max_tokens=max_tokens,
            conversation_messages=conversation_messages,
            response_model=response_model,
            json_mode=json_mode,
            temperature=temperature,
            stop_seqs=stop_seqs,
            verbosity=verbosity,
            enable_retry=enable_retry,
            retry_attempts=retry_attempts,
            before_retry_callback=before_retry_callback,
            telemetry=_telemetry_for_iteration(telemetry, synthesis_iteration),
        )
        return StreamingResponseWithMetadata(
            stream=stream,
            tool_calls_made=all_tool_calls,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_creation_input_tokens=total_cache_creation_tokens,
            cache_read_input_tokens=total_cache_read_tokens,
            thinking_content=None,
            iterations=iteration + 1,
            hit_input_token_cap=hit_input_token_cap,
        )

    current_attempt.set(1)

    async def _final_call() -> HonchoLLMCallResponse[Any]:
        plan = get_attempt_plan()
        return await honcho_llm_call_inner(
            plan.provider,
            plan.model,
            prompt,
            max_tokens,
            response_model,
            json_mode,
            effective_temperature(temperature),
            stop_seqs,
            plan.reasoning_effort,
            verbosity,
            plan.thinking_budget_tokens,
            stream=False,
            client_override=plan.client,
            tools=None,
            tool_choice=None,
            messages=conversation_messages,
            selected_config=plan.selected_config,
            plan=plan,
            telemetry=_telemetry_for_iteration(telemetry, synthesis_iteration),
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

    # emit the synthesis-call iteration event BEFORE merging cumulative
    # totals onto final_response below — otherwise the event's per-iteration
    # token counts would double-count the running totals.
    _emit_agent_iteration(
        _telemetry_for_iteration(telemetry, synthesis_iteration),
        synthesis_iteration,
        final_response,
    )

    final_response.tool_calls_made = all_tool_calls
    final_response.iterations = iteration + 1
    final_response.input_tokens = total_input_tokens + final_response.input_tokens
    final_response.output_tokens = total_output_tokens + final_response.output_tokens
    final_response.cache_creation_input_tokens = (
        total_cache_creation_tokens + final_response.cache_creation_input_tokens
    )
    final_response.cache_read_input_tokens = (
        total_cache_read_tokens + final_response.cache_read_input_tokens
    )
    final_response.hit_input_token_cap = (
        final_response.hit_input_token_cap or hit_input_token_cap
    )
    return final_response


__all__ = [
    "MAX_TOOL_ITERATIONS",
    "MIN_TOOL_ITERATIONS",
    "append_tool_results",
    "execute_tool_loop",
    "format_assistant_tool_message",
    "stream_final_response",
]
