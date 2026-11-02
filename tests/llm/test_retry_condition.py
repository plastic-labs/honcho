# pyright: reportArgumentType=false
"""The retry condition shared by every LLM retry site (DEV-3108).

A provider rejection of the request itself is not retried when no fallback can
take a later attempt; everything else keeps tenacity's retry-on-Exception
behavior, including the response checks that consume attempts by design.
"""

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors
from tenacity import retry, stop_after_attempt, wait_none

from src.config import (
    ConfiguredModelSettings,
    FallbackModelSettings,
    ModelConfig,
    ModelOverrideSettings,
    ResolvedFallbackConfig,
)
from src.exceptions import (
    ConflictException,
    DegradedLLMResponseError,
    EmptySummaryError,
    UpstreamLLMError,
    ValidationException,
)
from src.llm import HonchoLLMCallResponse, honcho_llm_call
from src.llm import tool_loop as tool_loop_module
from src.llm.errors import is_non_retryable_client_error, llm_retry_condition
from src.llm.runtime import AttemptPlan
from src.llm.structured_output import StructuredOutputError

_REQUEST = httpx.Request("POST", "http://tentacruel-litellm.internal/v1/chat")


def _response(status: int) -> httpx.Response:
    """An httpx response with `status`, as the provider SDKs wrap one."""
    return httpx.Response(status, request=_REQUEST, text="rejected")


def _openai(status: int) -> openai.APIStatusError:
    """An OpenAI SDK status error carrying `status`."""
    return openai.APIStatusError(
        f"Error code: {status}", response=_response(status), body=None
    )


def _anthropic(status: int) -> anthropic.APIStatusError:
    """An Anthropic SDK status error carrying `status`."""
    return anthropic.APIStatusError(
        f"Error code: {status}", response=_response(status), body=None
    )


def _degraded() -> DegradedLLMResponseError:
    """What the summarizer's `validate_response` raises on an empty summary."""
    return EmptySummaryError.for_prompt(
        "hi", parse_class="truncated", finish_reasons=["length"], output_tokens=0
    )


# --- classifier -------------------------------------------------------------

NON_RETRYABLE = [
    pytest.param(_openai(400), id="openai-400"),
    pytest.param(_openai(401), id="openai-401"),
    pytest.param(_openai(403), id="openai-403"),
    pytest.param(_openai(404), id="openai-404"),
    pytest.param(_openai(413), id="openai-413"),
    pytest.param(_openai(422), id="openai-422"),
    pytest.param(_anthropic(400), id="anthropic-400"),
    pytest.param(_anthropic(401), id="anthropic-401"),
    pytest.param(genai_errors.ClientError(400, {}), id="google-400"),
    pytest.param(genai_errors.ClientError(403, {}), id="google-403"),
]

RETRYABLE = [
    # The 4xx a provider sends about its own transient state.
    pytest.param(_openai(408), id="openai-408"),
    pytest.param(_openai(409), id="openai-409"),
    pytest.param(_openai(425), id="openai-425"),
    pytest.param(_openai(429), id="openai-429"),
    pytest.param(_anthropic(429), id="anthropic-429"),
    pytest.param(genai_errors.ClientError(429, {}), id="google-429"),
    # Outages. Completions reach the retry already translated to
    # UpstreamLLMError; a raw 5xx (the streaming path) must stay retryable too.
    pytest.param(_openai(500), id="openai-500"),
    pytest.param(genai_errors.ServerError(503, {}), id="google-503"),
    pytest.param(UpstreamLLMError("Model provider returned HTTP 502"), id="upstream"),
    pytest.param(openai.APIConnectionError(request=_REQUEST), id="openai-connect"),
    pytest.param(TimeoutError(), id="timeout"),
    # Honcho's own exceptions carry 4xx status codes but say nothing about the
    # provider request, so status alone must not decide.
    pytest.param(ValidationException("bad config"), id="honcho-422"),
    pytest.param(ConflictException("conflict"), id="honcho-409"),
    # Response checks that consume attempts by design.
    pytest.param(StructuredOutputError("schema miss"), id="structured-output"),
    pytest.param(_degraded(), id="degraded-response"),
    pytest.param(ValueError("anything else"), id="unrelated"),
]


@pytest.mark.parametrize("exc", NON_RETRYABLE)
def test_provider_rejections_are_not_retryable(exc: BaseException) -> None:
    """A 4xx that rejects the request itself is not retryable, from any SDK."""
    assert is_non_retryable_client_error(exc)


@pytest.mark.parametrize("exc", RETRYABLE)
def test_everything_else_stays_retryable(exc: BaseException) -> None:
    """Transient statuses, outages, Honcho's own errors and response checks stay
    retryable.
    """
    assert not is_non_retryable_client_error(exc)


# --- condition, driven through tenacity --------------------------------------


async def _run_with_condition(
    outcomes: list[BaseException | str],
    *,
    can_fall_back: bool,
    retry_attempts: int = 3,
) -> tuple[int, str | BaseException]:
    """Call through a tenacity retry built like the production sites.

    Each call consumes the next outcome: an exception is raised, a string is
    returned. Returns the number of calls and the result or raised exception.
    """
    calls = 0

    async def _call() -> str:
        """Return or raise the next scripted outcome."""
        nonlocal calls
        outcome = outcomes[min(calls, len(outcomes) - 1)]
        calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    wrapped = retry(
        retry=llm_retry_condition(
            can_fall_back=can_fall_back,
            retry_attempts=retry_attempts,
            model="openai/primary-model",
        ),
        stop=stop_after_attempt(retry_attempts),
        wait=wait_none(),
        reraise=True,
    )(_call)
    try:
        result: str | BaseException = await wrapped()
    except BaseException as exc:  # noqa: BLE001 - the test inspects it
        result = exc
    return calls, result


async def test_success_is_not_retried() -> None:
    """A successful call returns after one attempt."""
    calls, result = await _run_with_condition(["ok"], can_fall_back=False)

    assert (calls, result) == (1, "ok")


async def test_rejection_without_fallback_stops_and_reraises_the_original() -> None:
    """Without a fallback, a rejection stops after one call and surfaces the provider's
    own exception.
    """
    rejection = _openai(400)

    calls, raised = await _run_with_condition([rejection], can_fall_back=False)

    assert calls == 1
    assert raised is rejection


async def test_rejection_with_fallback_keeps_the_full_budget() -> None:
    """With a fallback, a rejection keeps the full budget, so the fallback still gets
    its attempt.
    """
    calls, raised = await _run_with_condition([_openai(400)], can_fall_back=True)

    assert calls == 3
    assert isinstance(raised, openai.APIStatusError)


async def test_rejection_after_a_transient_failure_stops_there() -> None:
    """A rejection stops the retries even after an earlier transient failure was
    retried.
    """
    calls, raised = await _run_with_condition(
        [UpstreamLLMError("blip"), _openai(400), "ok"], can_fall_back=False
    )

    assert calls == 2
    assert isinstance(raised, openai.APIStatusError)


async def test_transient_failures_still_use_the_budget() -> None:
    """Rate limits and outages are retried until a later attempt succeeds."""
    calls, result = await _run_with_condition(
        [_openai(429), UpstreamLLMError("blip"), "ok"], can_fall_back=False
    )

    assert (calls, result) == (3, "ok")


async def test_cancellation_is_not_retried() -> None:
    """Cancellation propagates on the first attempt, as with tenacity's default."""
    cancelled = asyncio.CancelledError()

    calls, raised = await _run_with_condition([cancelled], can_fall_back=True)

    assert calls == 1
    assert raised is cancelled


async def test_early_stop_is_logged_once(caplog: pytest.LogCaptureFixture) -> None:
    """An early stop logs one warning naming the model, the attempt and the status."""
    with caplog.at_level(logging.WARNING, logger="src.llm.errors"):
        await _run_with_condition([_openai(400)], can_fall_back=False)

    [record] = caplog.records
    assert "Not retrying openai/primary-model after attempt 1/3" in record.message
    assert "HTTP 400" in record.message


async def test_rejection_on_the_last_attempt_is_not_an_early_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """tenacity asks the condition before checking the budget, so a one-attempt
    budget must not log a stop that the budget made anyway."""
    with caplog.at_level(logging.WARNING, logger="src.llm.errors"):
        calls, _ = await _run_with_condition(
            [_openai(400)], can_fall_back=False, retry_attempts=1
        )

    assert calls == 1
    assert caplog.records == []


# --- honcho_llm_call: the two tool-less sites in api.py ----------------------

_PRIMARY = "primary-model"
_FALLBACK = "fallback-model"

_NO_FALLBACK = ModelConfig(model=_PRIMARY, transport="openai", api_key="test-key")
_WITH_FALLBACK = ModelConfig(
    model=_PRIMARY,
    transport="openai",
    api_key="test-key",
    fallback=ResolvedFallbackConfig(
        model=_FALLBACK, transport="openai", api_key="test-key"
    ),
)

# Each kwargs set lands on a different retry site in api.py: prompt-only uses
# the decorated call; explicit messages or an input-token cap use the
# message-carrying `_toolless_call` the deriver goes through.
TOOLLESS_PATHS = [
    pytest.param({}, id="prompt-only"),
    pytest.param({"messages": [{"role": "user", "content": "hi"}]}, id="messages"),
    pytest.param({"max_input_tokens": 1000}, id="input-token-cap"),
]


class _Recorder:
    """Stands in for `honcho_llm_call_inner`, recording the model per call."""

    def __init__(self, outcome: Any) -> None:
        """Answer every call with `outcome`, raising it if it is an exception."""
        self.outcome: Any = outcome
        self.models: list[str] = []

    async def __call__(self, _provider: str, model: str, *_a: Any, **_kw: Any) -> Any:
        """Record the model, then return or raise the configured outcome."""
        self.models.append(model)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.mark.parametrize("path_kwargs", TOOLLESS_PATHS)
async def test_toolless_rejection_without_fallback_is_one_call(
    path_kwargs: dict[str, Any],
) -> None:
    """Each tool-less retry site stops after one call, without a backoff sleep, when
    there is no fallback.
    """
    inner = _Recorder(_openai(400))
    sleep = AsyncMock()

    with (
        patch("src.llm.api.honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=sleep),
        pytest.raises(openai.APIStatusError),
    ):
        await honcho_llm_call(
            model_config=_NO_FALLBACK,
            prompt="hi",
            max_tokens=10,
            retry_attempts=3,
            **path_kwargs,
        )

    assert inner.models == [_PRIMARY]
    sleep.assert_not_awaited()


@pytest.mark.parametrize("path_kwargs", TOOLLESS_PATHS)
async def test_toolless_rejection_with_fallback_still_reaches_it(
    path_kwargs: dict[str, Any],
) -> None:
    """Each tool-less retry site keeps today's attempt sequence when a fallback is
    configured.
    """
    inner = _Recorder(_openai(400))

    with (
        patch("src.llm.api.honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(openai.APIStatusError),
    ):
        await honcho_llm_call(
            model_config=_WITH_FALLBACK,
            prompt="hi",
            max_tokens=10,
            retry_attempts=3,
            **path_kwargs,
        )

    assert inner.models == [_PRIMARY, _PRIMARY, _FALLBACK]


async def test_settings_shaped_config_with_a_fallback_reaches_it() -> None:
    """Fallback presence is read from the resolved config, so operator-style
    settings behave the same as a runtime ModelConfig."""
    settings = ConfiguredModelSettings(
        model=_PRIMARY,
        transport="openai",
        overrides=ModelOverrideSettings(api_key="test-key"),
        fallback=FallbackModelSettings(
            model=_FALLBACK,
            transport="openai",
            overrides=ModelOverrideSettings(api_key="test-key"),
        ),
    )
    inner = _Recorder(_openai(400))

    with (
        patch("src.llm.api.honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(openai.APIStatusError),
    ):
        await honcho_llm_call(
            model_config=settings, prompt="hi", max_tokens=10, retry_attempts=3
        )

    assert inner.models == [_PRIMARY, _PRIMARY, _FALLBACK]


@pytest.mark.parametrize(
    ("config", "retry_attempts", "enable_retry", "expected_models"),
    [
        # A one-attempt budget makes that attempt the fallback's, as before.
        pytest.param(_WITH_FALLBACK, 1, True, [_FALLBACK], id="budget-1"),
        pytest.param(_WITH_FALLBACK, 1, False, [_FALLBACK], id="budget-1-no-retry"),
        # Retries off with the default budget: one call, on the primary.
        pytest.param(_WITH_FALLBACK, 3, False, [_PRIMARY], id="no-retry"),
        pytest.param(_NO_FALLBACK, 3, False, [_PRIMARY], id="no-retry-no-fallback"),
    ],
)
async def test_budget_and_retry_flag_edges_are_unchanged(
    config: ModelConfig,
    retry_attempts: int,
    enable_retry: bool,
    expected_models: list[str],
) -> None:
    """`retry_attempts=1` and `enable_retry=False` select models exactly as before."""
    inner = _Recorder(_openai(400))

    with (
        patch("src.llm.api.honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(openai.APIStatusError),
    ):
        await honcho_llm_call(
            model_config=config,
            prompt="hi",
            max_tokens=10,
            retry_attempts=retry_attempts,
            enable_retry=enable_retry,
        )

    assert inner.models == expected_models


async def test_validate_response_refusals_still_use_the_budget() -> None:
    """The deny-list leaves response checks alone, even without a fallback."""
    inner = _Recorder(
        HonchoLLMCallResponse(content="", output_tokens=1, finish_reasons=["stop"])
    )

    def refuse(_response: HonchoLLMCallResponse[Any]) -> None:
        """A `validate_response` hook that refuses every response, as the summarizer
        does with an empty summary.
        """
        raise _degraded()

    with (
        patch("src.llm.api.honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(DegradedLLMResponseError),
    ):
        await honcho_llm_call(
            model_config=_NO_FALLBACK,
            prompt="hi",
            max_tokens=10,
            retry_attempts=3,
            validate_response=refuse,
        )

    assert inner.models == [_PRIMARY, _PRIMARY, _PRIMARY]


# --- honcho_llm_call: the tool-loop iteration and synthesis sites -------------

_TOOLS: list[dict[str, Any]] = [
    {"name": "recall", "description": "recall", "input_schema": {}}
]


async def _recall(_name: str, _input: dict[str, Any]) -> str:
    """Tool executor stub: every tool call succeeds."""
    return "recalled"


def _tool_call_response() -> HonchoLLMCallResponse[Any]:
    """A model response that calls the `recall` tool, so the loop continues."""
    return HonchoLLMCallResponse(
        content="",
        output_tokens=1,
        finish_reasons=["tool_use"],
        tool_calls_made=[{"id": "t0", "name": "recall", "input": {}}],
    )


class _ToolLoopRecorder:
    """Answers the first call from `first`, then raises `rejection`."""

    def __init__(self, first: Any, rejection: BaseException) -> None:
        """Answer the first call with `first`, if given, and every later call with
        `rejection`.
        """
        self.first: Any = first
        self.rejection: BaseException = rejection
        self.models: list[str] = []

    async def __call__(self, *_args: Any, **kwargs: Any) -> Any:
        """Record the attempt's model, then answer from the script."""
        self.models.append(kwargs["plan"].model)
        if len(self.models) == 1 and self.first is not None:
            return self.first
        raise self.rejection


@pytest.mark.parametrize(
    ("config", "expected_models"),
    [
        pytest.param(_NO_FALLBACK, [_PRIMARY], id="no-fallback"),
        pytest.param(_WITH_FALLBACK, [_PRIMARY, _PRIMARY, _FALLBACK], id="fallback"),
    ],
)
async def test_tool_loop_iteration_rejection(
    config: ModelConfig, expected_models: list[str]
) -> None:
    """The per-iteration retry site stops on a rejection only when there is no fallback."""
    inner = _ToolLoopRecorder(first=None, rejection=_anthropic(400))

    with (
        patch.object(tool_loop_module, "honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(anthropic.APIStatusError),
    ):
        await honcho_llm_call(
            model_config=config,
            prompt="hi",
            max_tokens=10,
            retry_attempts=3,
            tools=_TOOLS,
            tool_executor=_recall,
        )

    assert inner.models == expected_models


@pytest.mark.parametrize(
    ("config", "expected_synthesis_models"),
    [
        pytest.param(_NO_FALLBACK, [_PRIMARY], id="no-fallback"),
        pytest.param(_WITH_FALLBACK, [_PRIMARY, _PRIMARY, _FALLBACK], id="fallback"),
    ],
)
async def test_tool_loop_synthesis_rejection(
    config: ModelConfig, expected_synthesis_models: list[str]
) -> None:
    # One iteration that calls a tool, so the loop hits its limit and the
    # rejection lands on the synthesis call.
    """The final-answer retry site stops on a rejection only when there is no fallback."""
    inner = _ToolLoopRecorder(first=_tool_call_response(), rejection=_openai(400))

    with (
        patch.object(tool_loop_module, "honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        pytest.raises(openai.APIStatusError),
    ):
        await honcho_llm_call(
            model_config=config,
            prompt="hi",
            max_tokens=10,
            retry_attempts=3,
            tools=_TOOLS,
            tool_executor=_recall,
            max_tool_iterations=1,
        )

    assert inner.models == [_PRIMARY, *expected_synthesis_models]


# --- stream setup: pinned to one model, so a rejection always stops ----------


@pytest.mark.parametrize(
    ("pinned_model", "is_fallback"),
    [
        pytest.param(_PRIMARY, False, id="pinned-primary"),
        pytest.param(_FALLBACK, True, id="pinned-fallback"),
    ],
)
async def test_stream_setup_rejection_stops_and_names_the_pinned_model(
    pinned_model: str, is_fallback: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Covers the retried setup call only: the provider request itself starts
    when the stream is first read, outside the retry (a separate follow-up)."""
    inner = _Recorder(_openai(400))
    winning_plan = AttemptPlan(
        provider="openai",
        model=pinned_model,
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=None,
        attempt=1,
        retry_attempts=3,
        is_fallback=is_fallback,
    )

    with (
        patch.object(tool_loop_module, "honcho_llm_call_inner", new=inner),
        patch("asyncio.sleep", new=AsyncMock()),
        caplog.at_level(logging.WARNING, logger="src.llm.errors"),
        pytest.raises(openai.APIStatusError),
    ):
        stream = tool_loop_module.stream_final_response(
            winning_plan=winning_plan,
            prompt="hi",
            max_tokens=64,
            conversation_messages=[{"role": "user", "content": "x"}],
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=True,
            retry_attempts=3,
            before_retry_callback=lambda _r: None,
        )
        async for _chunk in stream:
            pass

    assert inner.models == [pinned_model]
    [record] = caplog.records
    assert f"Not retrying openai/{pinned_model}" in record.message
