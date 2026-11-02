"""Agentic/tool orchestration — the multi-iteration tool execution loop.

`execute_tool_loop` owns:
- initial tool-enabled call
- tool execution
- conversation augmentation with assistant messages + tool results
- max-iteration handling and synthesis call
- stream-final-only mode
- empty-response retry (one retry nudge when the model returns empty content)
- structured-answer check (a final answer that misses `response_model` is
  retried like a provider error)
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Collection,
)
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel
from tenacity import RetryCallState, retry, stop_after_attempt, wait_exponential

from src.config import ModelTransport
from src.exceptions import ValidationException
from src.utils.types import (
    get_last_tool_error,
    get_last_tool_metadata,
    iteration_scope,
    set_current_iteration,
    set_current_tool_call_seq,
    set_last_tool_error,
    set_last_tool_metadata,
)

from .errors import llm_retry_condition
from .executor import honcho_llm_call_inner
from .registry import history_adapter_for_provider
from .runtime import (
    AttemptPlan,
    CapturedAgentSpan,
    current_attempt,
    effective_temperature,
    start_captured_span,
)
from .structured_output import StructuredOutputError
from .types import (
    BudgetLine,
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
    base: LLMTelemetryContext | None,
    iteration: int,
    *,
    step_seq: int,
) -> LLMTelemetryContext | None:
    """Return a copy of `base` with per-step correlation set, or None if no base.

    We always copy rather than mutate the caller-supplied context so callers
    that pass the same context into multiple `honcho_llm_call` invocations
    don't see drift across concurrent runs. `parent_span_id` is set to the
    span being looped over (`base.span_id`) so nested generations are correctly
    treated as children of the run span.
    """
    if base is None:
        return None
    return dataclasses.replace(
        base,
        iteration=iteration,
        step_seq=step_seq,
        parent_span_id=base.span_identity(),
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


def _require_structured_answer(
    response: HonchoLLMCallResponse[Any],
    response_model: type[BaseModel] | None,
    *,
    allow_empty: bool,
) -> None:
    """Raise when a final answer did not parse into `response_model`.

    A backend that degrades instead of raising (an OpenAI refusal, json_object
    mode's empty answer for a model with required fields) hands back a plain
    string. Runs inside the retried call, so the miss consumes an attempt and
    the last attempt reaches the fallback. `allow_empty` lets an empty answer
    through to the one-shot empty-response nudge.
    """
    if response_model is None or response.tool_calls_made:
        return
    content = response.content
    if isinstance(content, BaseModel):
        return
    if allow_empty and isinstance(content, str) and not content.strip():
        return
    # The content itself stays out of the message: a refusal can quote the
    # conversation.
    raise StructuredOutputError(
        f"Final answer is {type(content).__name__}, not {response_model.__name__}"
    )


def format_assistant_tool_message(
    provider: ModelTransport,
    content: Any,
    tool_calls: list[dict[str, Any]],
    thinking_blocks: list[dict[str, Any]] | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
    thinking_content: str | None = None,
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
        thinking_content=thinking_content,
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
    streaming call against the same pinned model for transient errors, and
    stop on a non-retryable client error, since no other model is reachable.
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
            retry=llm_retry_condition(
                can_fall_back=False,
                retry_attempts=retry_attempts,
                model=f"{winning_plan.provider}/{winning_plan.model}",
            ),
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
            # Surface the provider's own error once the budget is spent.
            # Without this tenacity raises RetryError, which erases the cause
            # and lands every outage on the generic 500 handler.
            reraise=True,
        )(_setup_stream)
        stream = await wrapped()
    else:
        stream = await _setup_stream()

    try:
        async for chunk in stream:
            yield chunk
    finally:
        if isinstance(stream, AsyncGenerator):
            await stream.aclose()


async def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    tool_executor: Callable[[str, dict[str, Any]], Any],
    all_tool_calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Run one response's tool calls in order.

    Successful calls are appended to `all_tool_calls`. Returns the results to
    send back to the model and the names of the tools that succeeded.
    """
    tool_results: list[dict[str, Any]] = []
    succeeded_tools: set[str] = set()
    for seq, tool_call in enumerate(tool_calls):
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
        set_last_tool_error(False)

        try:
            tool_result = await tool_executor(tool_name, tool_input)
            # Stash ToolResult.metadata on all_tool_calls so
            # specialist rollups can read created/deleted observation
            # counts without round-tripping through the event store.
            tool_result_metadata = get_last_tool_metadata()
            # The executor reports handler failures as returned
            # strings, so a normal return is not proof of success.
            if not get_last_tool_error():
                succeeded_tools.add(tool_name)
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
    return tool_results, succeeded_tools


def _append_budget_line(
    conversation_messages: list[dict[str, Any]],
    budget_line: BudgetLine | None,
    rounds_used: int,
    round_limit: int,
) -> None:
    """Append the caller's budget line, if any, as the newest user turn.

    It goes after the tool results rather than into the system prompt, so the
    cached prefix of the conversation stays stable.
    """
    if budget_line is None:
        return
    line = budget_line(rounds_used, round_limit)
    if line:
        conversation_messages.append({"role": "user", "content": line})


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
    retry_condition: Callable[[RetryCallState], bool],
    stream_final: bool = False,
    iteration_callback: IterationCallback | None = None,
    telemetry: LLMTelemetryContext | None = None,
    run_span: CapturedAgentSpan | None = None,
    force_tools_until: Collection[str] | None = None,
    max_forced_iterations: int = 3,
    budget_line: BudgetLine | None = None,
    final_call_tools: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[Any] | StreamingResponseWithMetadata:
    """Run the iterative tool calling loop for agentic LLM interactions.

    Loop per iteration:
      1. Make an LLM call with tools available
      2. Execute any tool calls the LLM requests
      3. Append tool results to the conversation
      4. Optionally append a budget line (see `budget_line`)
      5. Repeat until the LLM stops calling tools or max iterations reached

    When the iteration limit is reached, one final call is made. By default
    it offers no tools, so the model must answer. With `final_call_tools`, it
    offers those tools instead (for an agent whose output is its writes), and
    any calls it makes are executed; no further call follows them.

    Each iteration and the final call retry under `retry_condition`, which
    the caller builds with `llm_retry_condition` for its model config.

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
    if final_call_tools and stream_final:
        raise ValidationException(
            "final_call_tools cannot be combined with stream_final: the final "
            + "call would have to both stream an answer and run tools"
        )

    conversation_messages: list[dict[str, Any]] = (
        messages.copy() if messages else [{"role": "user", "content": prompt}]
    )

    # Seed one hash memo for the whole span. dataclasses.replace copies the dict reference into
    # every per-iteration telemetry copy, so each appended message is content-hashed exactly once
    # across the span.
    if telemetry is not None and telemetry.hash_memo is None:
        telemetry = dataclasses.replace(telemetry, hash_memo={})

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
    # A forced tool_choice ("required"/"any") relaxes to "auto" once a tool in
    # `force_tools_until` has run, or after `max_forced_iterations` rounds so a
    # model that keeps dodging still gets to answer. With no gate, the first
    # successful round is enough.
    effective_tool_choice = tool_choice
    # An empty gate would be unsatisfiable, so it means no gate.
    gate = frozenset(force_tools_until or ()) or None
    forced_rounds = 0

    while iteration < max_tool_iterations:
        step = start_captured_span(
            "step",
            _telemetry_for_iteration(telemetry, iteration + 1, step_seq=iteration + 1),
        )
        try:
            # Reset attempt counter so each iteration starts with the primary provider.
            current_attempt.set(1)
            logger.debug(
                f"Tool execution iteration {iteration + 1}/{max_tool_iterations}"
            )

            if max_input_tokens is not None:
                if count_message_tokens(conversation_messages) > max_input_tokens:
                    hit_input_token_cap = True
                conversation_messages = truncate_messages_to_fit(
                    conversation_messages, max_input_tokens
                )

            # A provider can ignore a forced tool_choice and answer outright.
            # While the gate is unmet, a no-tool answer is discarded for a
            # nudge, so it isn't held to `response_model`. Nor is it under
            # stream_final, which streams the answer afresh and doesn't
            # validate it.
            gate_nudge_due = (
                gate is not None
                and effective_tool_choice in ("required", "any")
                and forced_rounds + 1 < max_forced_iterations
                and iteration < max_tool_iterations - 1
            )
            empty_nudge_due = (
                empty_response_retries < 1 and iteration < max_tool_iterations - 1
            )

            async def _call_with_messages(
                tool_choice_for_call: str
                | dict[str, Any]
                | None = effective_tool_choice,
                captured_messages: list[dict[str, Any]] = conversation_messages,
                iteration_for_call: int = iteration + 1,
                check_answer: bool = not gate_nudge_due,
                allow_empty_answer: bool = empty_nudge_due,
            ) -> HonchoLLMCallResponse[Any]:
                plan = get_attempt_plan()
                result = await honcho_llm_call_inner(
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
                    tool_choice=tool_choice_for_call,
                    messages=captured_messages,
                    selected_config=plan.selected_config,
                    plan=plan,
                    telemetry=_telemetry_for_iteration(
                        telemetry, iteration_for_call, step_seq=iteration_for_call
                    ),
                )
                if check_answer and not stream_final:
                    _require_structured_answer(
                        result, response_model, allow_empty=allow_empty_answer
                    )
                return result

            call_func: Callable[[], Awaitable[HonchoLLMCallResponse[Any]]]
            if enable_retry:
                call_func = retry(
                    retry=retry_condition,
                    stop=stop_after_attempt(retry_attempts),
                    wait=wait_exponential(multiplier=1, min=4, max=10),
                    before_sleep=before_retry_callback,
                    # Surface the provider's own error once the budget is
                    # spent. Without this tenacity raises RetryError, which
                    # erases the cause and lands every outage on the generic
                    # 500 handler.
                    reraise=True,
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

                # While the gate is unmet, spend a forced round on a nudge
                # instead of returning that answer; at the cap, keep it.
                if gate_nudge_due and gate is not None:
                    forced_rounds += 1
                    conversation_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your last response called no tool. Call one of "
                                + ", ".join(sorted(gate))
                                + " to read the relevant memory before answering."
                            ),
                        }
                    )
                    iteration += 1
                    continue

                if (
                    isinstance(response.content, str)
                    and not response.content.strip()
                    and empty_nudge_due
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
                    # +2 (not +1): the in-loop call we just made used iteration+1,
                    # so the streamed tail needs the next ordinal — otherwise its
                    # trace resource id collides with that call's. Mirrors the
                    # synthesis path's distinct-next-value behavior.
                    stream_telemetry = _telemetry_for_iteration(
                        telemetry, iteration + 2, step_seq=iteration + 2
                    )
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
                        telemetry=stream_telemetry,
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
                        run_span=run_span,
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
                response.thinking_content,
            )
            conversation_messages.append(assistant_message)

            # Telemetry context — 1-indexed iteration.
            set_current_iteration(iteration + 1)

            tool_results, succeeded_tools = await _execute_tool_calls(
                response.tool_calls_made, tool_executor, all_tool_calls
            )

            append_tool_results(current_provider, tool_results, conversation_messages)

            _append_budget_line(
                conversation_messages, budget_line, iteration + 1, max_tool_iterations
            )
        finally:
            if step is not None:
                step.end()

        # Between-turn bookkeeping lives outside the step span — the span
        # scopes the LLM call + its tools, not the iteration accounting.
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

        if effective_tool_choice in ("required", "any"):
            forced_rounds += 1
            gate_met = gate is None or bool(succeeded_tools & gate)
            if gate_met or forced_rounds >= max_forced_iterations:
                effective_tool_choice = "auto"
                logger.debug(
                    "Relaxed tool_choice to 'auto' after %d forced round(s) (gate_met=%s)",
                    forced_rounds,
                    gate_met,
                )

        iteration += 1

    logger.warning(
        f"Tool execution loop reached max iterations ({max_tool_iterations})"
    )

    # The max-iteration synthesis call gets iteration N+1 in telemetry so 's
    # AgentIterationEvent and this LLMCallCompletedEvent line up sequentially.
    synthesis_iteration = iteration + 1

    if final_call_tools:
        final_tool_names = ", ".join(f"`{t['name']}`" for t in final_call_tools)
        synthesis_prompt = (
            "You have reached the maximum number of tool calls. "
            f"You may make one final call to {final_tool_names} to save what you "
            "have decided on; no other tools are available. If there is nothing "
            "to save, respond without calling a tool."
        )
    else:
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
        stream_telemetry = _telemetry_for_iteration(
            telemetry, synthesis_iteration, step_seq=synthesis_iteration
        )
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
            telemetry=stream_telemetry,
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
            run_span=run_span,
            hit_iteration_limit=True,
        )

    current_attempt.set(1)

    async def _final_call() -> HonchoLLMCallResponse[Any]:
        plan = get_attempt_plan()
        response = await honcho_llm_call_inner(
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
            tools=final_call_tools or None,
            tool_choice=None,
            messages=conversation_messages,
            selected_config=plan.selected_config,
            plan=plan,
            telemetry=_telemetry_for_iteration(
                telemetry, synthesis_iteration, step_seq=synthesis_iteration
            ),
        )
        # No nudge follows synthesis, so an empty answer is a miss too.
        _require_structured_answer(response, response_model, allow_empty=False)
        return response

    if enable_retry:
        final_call_func = retry(
            retry=retry_condition,
            stop=stop_after_attempt(retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            before_sleep=before_retry_callback,
            # Surface the provider's own error once the budget is spent.
            # Without this tenacity raises RetryError, which erases the cause
            # and lands every outage on the generic 500 handler.
            reraise=True,
        )(_final_call)
    else:
        final_call_func = _final_call

    # Step span around the synthesis call, same shape as in-loop iterations.
    synthesis_step = start_captured_span(
        "step",
        _telemetry_for_iteration(
            telemetry, synthesis_iteration, step_seq=synthesis_iteration
        ),
    )
    try:
        final_response = await final_call_func()
        if final_call_tools and final_response.tool_calls_made:
            # Only the offered tools may run; anything else the model names is
            # dropped rather than executed past the limit.
            allowed = {t["name"] for t in final_call_tools}
            set_current_iteration(synthesis_iteration)
            await _execute_tool_calls(
                [tc for tc in final_response.tool_calls_made if tc["name"] in allowed],
                tool_executor,
                all_tool_calls,
            )
    finally:
        if synthesis_step is not None:
            synthesis_step.end()

    # emit the synthesis-call iteration event BEFORE merging cumulative
    # totals onto final_response below — otherwise the event's per-iteration
    # token counts would double-count the running totals.
    _emit_agent_iteration(
        _telemetry_for_iteration(
            telemetry, synthesis_iteration, step_seq=synthesis_iteration
        ),
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
    final_response.hit_iteration_limit = True
    return final_response


__all__ = [
    "MAX_TOOL_ITERATIONS",
    "MIN_TOOL_ITERATIONS",
    "append_tool_results",
    "execute_tool_loop",
    "format_assistant_tool_message",
    "stream_final_response",
]
