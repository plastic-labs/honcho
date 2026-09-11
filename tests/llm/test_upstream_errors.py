# pyright: reportArgumentType=false
from typing import Any
from unittest.mock import patch

import anthropic
import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from src.config import ModelConfig
from src.exceptions import UpstreamLLMError
from src.llm import tool_loop as tool_loop_module
from src.llm.backend import CompletionResult
from src.llm.errors import as_upstream_error
from src.llm.request_builder import execute_completion
from src.llm.runtime import AttemptPlan
from src.llm.tool_loop import execute_tool_loop
from tests.llm.conftest import FakeBackend

_REQUEST = httpx.Request("POST", "http://tentacruel-litellm.internal/v1/messages")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=_REQUEST, text="upstream failure")


def _anthropic_status(status: int) -> anthropic.APIStatusError:
    return anthropic.APIStatusError(
        f"Error code: {status}", response=_response(status), body=None
    )


# Each SDK reports an outage its own way; the mapping has to cover all three.
TRANSLATED = [
    pytest.param(_anthropic_status(502), id="anthropic-502"),
    pytest.param(_anthropic_status(500), id="anthropic-500"),
    pytest.param(
        openai.APIStatusError("boom", response=_response(503), body=None),
        id="openai-503",
    ),
    pytest.param(genai_errors.ServerError(502, {}), id="google-502"),
    pytest.param(
        anthropic.APIConnectionError(request=_REQUEST), id="anthropic-connect"
    ),
    pytest.param(openai.APIConnectionError(request=_REQUEST), id="openai-connect"),
    pytest.param(anthropic.APITimeoutError(request=_REQUEST), id="anthropic-timeout"),
    pytest.param(httpx.ConnectError("connection refused"), id="httpx-connect"),
    pytest.param(httpx.ReadTimeout("timed out"), id="httpx-timeout"),
]

# 4xx is our request being wrong, and 429 is a throttle the caller handles
# differently -- neither should be laundered into a 503.
PASSED_THROUGH = [
    pytest.param(_anthropic_status(400), id="anthropic-400"),
    pytest.param(_anthropic_status(429), id="anthropic-429"),
    pytest.param(_anthropic_status(404), id="anthropic-404"),
    pytest.param(genai_errors.ClientError(400, {}), id="google-400"),
    pytest.param(ValueError("not a provider error"), id="unrelated"),
]


@pytest.mark.parametrize("exc", TRANSLATED)
def test_provider_outages_become_upstream_errors(exc: BaseException) -> None:
    translated = as_upstream_error(exc)

    assert translated is not None
    assert translated.status_code == 503


@pytest.mark.parametrize("exc", PASSED_THROUGH)
def test_client_errors_are_left_alone(exc: BaseException) -> None:
    assert as_upstream_error(exc) is None


class RaisingBackend(FakeBackend):
    """Backend whose `complete` fails the way a provider outage does."""

    _error: BaseException

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self._error = error

    async def complete(self, **kwargs: Any) -> CompletionResult:
        raise self._error


async def _complete_with(error: BaseException) -> None:
    await execute_completion(
        RaisingBackend(error),
        ModelConfig(model="claude-haiku-4-5", transport="anthropic"),
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )


async def test_execute_completion_translates_a_proxy_502() -> None:
    """The shape of the tentacruel outage: litellm's proxy returns 502."""
    original = _anthropic_status(502)

    with pytest.raises(UpstreamLLMError) as caught:
        await _complete_with(original)

    assert caught.value.status_code == 503
    # The provider error stays reachable for Sentry and for local debugging.
    assert caught.value.__cause__ is original


async def test_execute_completion_leaves_a_bad_request_alone() -> None:
    original = _anthropic_status(400)

    with pytest.raises(anthropic.APIStatusError) as caught:
        await _complete_with(original)

    assert caught.value is original


async def test_exhausted_retries_surface_the_provider_error_not_retry_error() -> None:
    """`reraise=True` is what keeps the tool loop's failure legible.

    Without it tenacity raises `RetryError`, which carries no status and no
    `__cause__`, so every provider outage lands on the generic 500 handler
    instead of the 503 the caller can act on.
    """

    async def _always_502(*_args: object, **_kwargs: object) -> object:
        raise UpstreamLLMError("Model provider returned HTTP 502")

    # `client`/`selected_config` are never touched: honcho_llm_call_inner is
    # patched out, so the plan only has to carry the retry budget.
    plan = AttemptPlan(
        provider="anthropic",
        model="claude-haiku-4-5",
        client=object(),
        thinking_budget_tokens=None,
        reasoning_effort=None,
        selected_config=None,
        attempt=1,
        # One attempt: enough to exhaust the budget without a backoff sleep.
        retry_attempts=1,
        is_fallback=False,
    )

    with (
        patch.object(tool_loop_module, "honcho_llm_call_inner", new=_always_502),
        pytest.raises(UpstreamLLMError) as caught,
    ):
        await execute_tool_loop(
            prompt="hi",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            tools=[
                {
                    "name": "noop",
                    "description": "no-op",
                    "input_schema": {"type": "object"},
                }
            ],
            tool_choice="auto",
            tool_executor=lambda _name, _input: "",
            max_tool_iterations=5,
            response_model=None,
            json_mode=False,
            temperature=None,
            stop_seqs=None,
            verbosity=None,
            enable_retry=True,
            retry_attempts=1,
            max_input_tokens=None,
            get_attempt_plan=lambda: plan,
            before_retry_callback=lambda _r: None,
            stream_final=False,
            telemetry=None,
        )

    assert caught.value.status_code == 503


async def test_toolless_path_also_surfaces_the_provider_error() -> None:
    """The deriver's path, from Rootly 3WyMAJ.

    `minimal_deriver_batch` calls `honcho_llm_call` with no tools, so it goes
    through `_toolless_call` in `api.py` rather than the tool loop -- a
    separate pair of `retry(...)` sites that the tool-loop fix does not cover.
    """
    from src.llm import api as api_module

    async def _always_502(*_args: object, **_kwargs: object) -> object:
        raise UpstreamLLMError("Model provider returned HTTP 502")

    with (
        patch.object(api_module, "honcho_llm_call_inner", new=_always_502),
        pytest.raises(UpstreamLLMError) as caught,
    ):
        await api_module.honcho_llm_call(
            model_config=ModelConfig(
                model="claude-haiku-4-5",
                transport="anthropic",
                # Enough for the registry to build a client; it is never used,
                # since honcho_llm_call_inner is patched out.
                api_key="test-key",
            ),
            prompt="hi",
            max_tokens=64,
            enable_retry=True,
            # One attempt: exhausts the budget without a backoff sleep.
            retry_attempts=1,
        )

    assert caught.value.status_code == 503


def test_every_llm_retry_site_sets_reraise() -> None:
    """Structural guard for the gap 3WyMAJ exposed.

    The first pass at this fix patched the three `retry(...)` sites in
    `tool_loop.py` and missed the two in `api.py`, so the deriver kept raising
    bare `RetryError`. Any new site has to opt in too.
    """
    import ast
    from pathlib import Path

    offenders: list[str] = []
    for path in sorted(Path("src/llm").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "retry"):
                continue
            if not any(kw.arg == "reraise" for kw in node.keywords):
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == [], (
        f"tenacity retry() without reraise=True: {offenders}. "
        "Without it an exhausted budget raises RetryError, erasing the cause."
    )
