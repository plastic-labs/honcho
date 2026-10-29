from typing import Any
from unittest.mock import AsyncMock, patch

from src.config import ModelConfig
from src.llm import executor
from src.llm.backend import CompletionResult
from src.llm.runtime import with_opencode_session_header
from src.llm.types import LLMTelemetryContext


def _config(
    *,
    base_url: str = "https://opencode.ai/zen/go/v1",
    provider_params: dict[str, Any] | None = None,
) -> ModelConfig:
    return ModelConfig(
        model="mimo-v2.5",
        transport="openai",
        base_url=base_url,
        provider_params=provider_params or {},
    )


def test_uses_honcho_session_id_and_preserves_other_headers() -> None:
    config = _config(
        provider_params={
            "extra_headers": {
                "X-OpenCode-Session": "static-workload-id",
                "HTTP-Referer": "https://honcho.dev",
            }
        }
    )

    result = with_opencode_session_header(
        config, LLMTelemetryContext(session_id="honcho-session-123", run_id="run-456")
    )

    assert result.provider_params["extra_headers"] == {
        "HTTP-Referer": "https://honcho.dev",
        "x-opencode-session": "honcho-session-123",
    }
    assert config.provider_params["extra_headers"]["X-OpenCode-Session"] == (
        "static-workload-id"
    )


def test_uses_run_id_for_sessionless_calls() -> None:
    result = with_opencode_session_header(
        _config(), LLMTelemetryContext(run_id="dialectic-run-789")
    )

    assert result.provider_params["extra_headers"]["x-opencode-session"] == (
        "dialectic-run-789"
    )


def test_does_not_match_similarly_prefixed_opencode_paths() -> None:
    config = _config(base_url="https://opencode.ai/zen/gopher/v1")
    telemetry = LLMTelemetryContext(session_id="honcho-session-123")

    assert with_opencode_session_header(config, telemetry) is config


def test_leaves_other_providers_and_missing_ids_unchanged() -> None:
    config = _config(base_url="https://api.openai.com/v1")
    telemetry = LLMTelemetryContext(session_id="honcho-session-123")

    assert with_opencode_session_header(config, telemetry) is config
    opencode_config = _config()
    assert with_opencode_session_header(opencode_config, None) is opencode_config
    assert (
        with_opencode_session_header(opencode_config, LLMTelemetryContext())
        is opencode_config
    )


async def test_executor_sends_conversation_id_to_provider() -> None:
    config = _config(
        provider_params={"extra_headers": {"x-opencode-session": "static-workload-id"}}
    )
    completion = CompletionResult(
        content="ok", input_tokens=1, output_tokens=1, finish_reason="stop"
    )
    execute = AsyncMock(return_value=completion)

    with (
        patch.object(executor, "default_client", return_value=object()),
        patch.object(executor, "backend_for_provider", return_value=object()),
        patch.object(executor, "execute_completion", new=execute),
        patch("src.telemetry.events.emit"),
    ):
        await executor.honcho_llm_call_inner(
            "openai",
            "mimo-v2.5",
            "hello",
            max_tokens=32,
            selected_config=config,
            telemetry=LLMTelemetryContext(session_id="real-honcho-session"),
        )

    assert execute.await_args is not None
    sent_config = execute.await_args.args[1]
    assert sent_config.provider_params["extra_headers"]["x-opencode-session"] == (
        "real-honcho-session"
    )
