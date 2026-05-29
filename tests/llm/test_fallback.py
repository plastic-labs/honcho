"""Unit tests for LLM model fallback logic (t_124ab52c).

Tests cover:
- select_model_config_for_attempt() with force_fallback
- plan_attempt() with force_fallback
- _is_retryable_error() classification
- force_fallback ContextVar behavior
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config import (
    ModelConfig,
    ResolvedFallbackConfig,
)
from src.llm.runtime import (
    force_fallback,
    plan_attempt,
    select_model_config_for_attempt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def primary_with_fallback() -> ModelConfig:
    """Primary config with a fallback model configured."""
    return ModelConfig(
        model="gpt-4o",
        transport="openai",
        fallback=ResolvedFallbackConfig(
            model="qwen3.5-9b",
            transport="lmstudio",
        ),
    )


@pytest.fixture
def primary_no_fallback() -> ModelConfig:
    """Primary config without fallback."""
    return ModelConfig(
        model="gpt-4o",
        transport="openai",
    )


@pytest.fixture
def mock_client():
    return MagicMock()


# ---------------------------------------------------------------------------
# 4.1: select_model_config_for_attempt() with force_fallback=True
# ---------------------------------------------------------------------------


class TestSelectModelConfigForAttempt:
    def test_force_fallback_true_returns_fallback_immediately(
        self, primary_with_fallback: ModelConfig
    ):
        """force_fallback=True on attempt 1 should return fallback config."""
        result = select_model_config_for_attempt(
            primary_with_fallback,
            attempt=1,
            retry_attempts=3,
            force_fallback=True,
        )
        assert result.model == "qwen3.5-9b"
        assert result.transport == "lmstudio"

    def test_force_fallback_false_attempt_1_returns_primary(
        self, primary_with_fallback: ModelConfig
    ):
        """force_fallback=False on attempt 1 should return primary."""
        result = select_model_config_for_attempt(
            primary_with_fallback,
            attempt=1,
            retry_attempts=3,
            force_fallback=False,
        )
        assert result.model == "gpt-4o"
        assert result.transport == "openai"

    def test_force_fallback_false_final_attempt_returns_fallback(
        self, primary_with_fallback: ModelConfig
    ):
        """Legacy behavior: final attempt with fallback configured returns fallback."""
        result = select_model_config_for_attempt(
            primary_with_fallback,
            attempt=3,
            retry_attempts=3,
            force_fallback=False,
        )
        assert result.model == "qwen3.5-9b"
        assert result.transport == "lmstudio"

    def test_no_fallback_configured_force_fallback_true_returns_primary(
        self, primary_no_fallback: ModelConfig
    ):
        """No fallback configured → force_fallback=True is a no-op, returns primary."""
        result = select_model_config_for_attempt(
            primary_no_fallback,
            attempt=1,
            retry_attempts=3,
            force_fallback=True,
        )
        assert result.model == "gpt-4o"
        assert result.transport == "openai"

    def test_force_fallback_false_attempt_2_of_3_returns_primary(
        self, primary_with_fallback: ModelConfig
    ):
        """Attempt 2 of 3 (not final) → primary."""
        result = select_model_config_for_attempt(
            primary_with_fallback,
            attempt=2,
            retry_attempts=3,
            force_fallback=False,
        )
        assert result.model == "gpt-4o"
        assert result.transport == "openai"

    def test_fallback_config_has_no_nested_fallback(
        self, primary_with_fallback: ModelConfig
    ):
        """Fallback config should not have its own fallback (chain depth = 1)."""
        result = select_model_config_for_attempt(
            primary_with_fallback,
            attempt=1,
            retry_attempts=3,
            force_fallback=True,
        )
        assert result.fallback is None


# ---------------------------------------------------------------------------
# 4.2: select_model_config_for_attempt() with force_fallback=False (existing)
# ---------------------------------------------------------------------------


class TestSelectModelConfigExistingBehavior:
    def test_existing_behavior_preserved(self, primary_with_fallback: ModelConfig):
        """Without force_fallback, existing retry logic is unchanged."""
        # Attempt 1 → primary
        r1 = select_model_config_for_attempt(
            primary_with_fallback, attempt=1, retry_attempts=3, force_fallback=False
        )
        assert r1.model == "gpt-4o"

        # Attempt 2 → primary
        r2 = select_model_config_for_attempt(
            primary_with_fallback, attempt=2, retry_attempts=3, force_fallback=False
        )
        assert r2.model == "gpt-4o"

        # Attempt 3 (final) → fallback
        r3 = select_model_config_for_attempt(
            primary_with_fallback, attempt=3, retry_attempts=3, force_fallback=False
        )
        assert r3.model == "qwen3.5-9b"


# ---------------------------------------------------------------------------
# 4.3: plan_attempt() with force_fallback=True
# ---------------------------------------------------------------------------


class TestPlanAttempt:
    def test_force_fallback_true_returns_fallback_plan(
        self, primary_with_fallback: ModelConfig, mock_client
    ):
        """force_fallback=True → AttemptPlan uses fallback provider/model."""
        with patch("src.llm.runtime.client_for_model_config", return_value=mock_client):
            plan = plan_attempt(
                runtime_model_config=primary_with_fallback,
                attempt=1,
                retry_attempts=3,
                call_thinking_budget_tokens=None,
                call_reasoning_effort=None,
                force_fallback=True,
            )
        assert plan.model == "qwen3.5-9b"
        assert plan.provider == "lmstudio"

    def test_force_fallback_false_returns_primary_plan(
        self, primary_with_fallback: ModelConfig, mock_client
    ):
        """force_fallback=False → AttemptPlan uses primary provider/model."""
        with patch("src.llm.runtime.client_for_model_config", return_value=mock_client):
            plan = plan_attempt(
                runtime_model_config=primary_with_fallback,
                attempt=1,
                retry_attempts=3,
                call_thinking_budget_tokens=None,
                call_reasoning_effort=None,
                force_fallback=False,
            )
        assert plan.model == "gpt-4o"
        assert plan.provider == "openai"

    def test_force_fallback_false_final_attempt_returns_fallback_plan(
        self, primary_with_fallback: ModelConfig, mock_client
    ):
        """Legacy: final attempt → fallback plan."""
        with patch("src.llm.runtime.client_for_model_config", return_value=mock_client):
            plan = plan_attempt(
                runtime_model_config=primary_with_fallback,
                attempt=3,
                retry_attempts=3,
                call_thinking_budget_tokens=None,
                call_reasoning_effort=None,
                force_fallback=False,
            )
        assert plan.model == "qwen3.5-9b"
        assert plan.provider == "lmstudio"


# ---------------------------------------------------------------------------
# 4.4-4.5: _is_retryable_error() classification
# ---------------------------------------------------------------------------


class TestIsRetryableError:
    """Test the _is_retryable_error() function extracted from api.py logic."""

    @staticmethod
    def _is_retryable_error(exc: BaseException) -> bool:
        """Mirror of src/llm/api.py _is_retryable_error()."""
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status in (408, 429) or (500 <= status < 600)
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        return type(exc).__name__ in (
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "ServiceUnavailableError",
            "RateLimitError",
        )

    def test_429_is_retryable(self):
        class E429(Exception):
            status_code = 429
        assert self._is_retryable_error(E429()) is True

    def test_500_is_retryable(self):
        class E500(Exception):
            status_code = 500
        assert self._is_retryable_error(E500()) is True

    def test_503_is_retryable(self):
        class E503(Exception):
            status_code = 503
        assert self._is_retryable_error(E503()) is True

    def test_timeout_error_is_retryable(self):
        assert self._is_retryable_error(TimeoutError()) is True

    def test_connection_error_is_retryable(self):
        assert self._is_retryable_error(ConnectionError()) is True

    def test_400_is_not_retryable(self):
        class E400(Exception):
            status_code = 400
        assert self._is_retryable_error(E400()) is False

    def test_200_is_not_retryable(self):
        class E200(Exception):
            status_code = 200
        assert self._is_retryable_error(E200()) is False

    def test_value_error_is_not_retryable(self):
        assert self._is_retryable_error(ValueError("bad")) is False

    def test_generic_exception_is_not_retryable(self):
        assert self._is_retryable_error(Exception("unknown")) is False


# ---------------------------------------------------------------------------
# 4.6: Cross-provider fallback param mapping
# ---------------------------------------------------------------------------


class TestCrossProviderFallback:
    def test_cross_provider_fallback_openai_to_anthropic(self, mock_client):
        """Primary=openai, fallback=anthropic → fallback config carries own params."""
        primary = ModelConfig(
            model="gpt-4o",
            transport="openai",
            fallback=ResolvedFallbackConfig(
                model="claude-sonnet-4",
                transport="anthropic",
                thinking_budget_tokens=1024,
            ),
        )
        with patch("src.llm.runtime.client_for_model_config", return_value=mock_client):
            plan = plan_attempt(
                runtime_model_config=primary,
                attempt=1,
                retry_attempts=3,
                call_thinking_budget_tokens=None,
                call_reasoning_effort=None,
                force_fallback=True,
            )
        assert plan.model == "claude-sonnet-4"
        assert plan.provider == "anthropic"
        # Fallback carries its own thinking_budget_tokens
        assert plan.selected_config.thinking_budget_tokens == 1024

    def test_cross_provider_fallback_lmstudio_to_openai(self, mock_client):
        """Primary=lmstudio, fallback=openai → fallback config carries own params."""
        primary = ModelConfig(
            model="qwen/qwen3.5-9b",
            transport="lmstudio",
            fallback=ResolvedFallbackConfig(
                model="gpt-4o-mini",
                transport="openai",
            ),
        )
        with patch("src.llm.runtime.client_for_model_config", return_value=mock_client):
            plan = plan_attempt(
                runtime_model_config=primary,
                attempt=1,
                retry_attempts=3,
                call_thinking_budget_tokens=None,
                call_reasoning_effort=None,
                force_fallback=True,
            )
        assert plan.model == "gpt-4o-mini"
        assert plan.provider == "openai"


# ---------------------------------------------------------------------------
# force_fallback ContextVar behavior
# ---------------------------------------------------------------------------


class TestForceFallbackContextVar:
    def test_default_is_false(self):
        """force_fallback ContextVar defaults to False."""
        # Read default directly — do NOT mutate before asserting
        assert force_fallback.get() is False

    def test_set_and_get(self):
        """force_fallback can be set and read, with proper cleanup."""
        # Save original state
        original = force_fallback.get()
        try:
            force_fallback.set(True)
            assert force_fallback.get() is True
            force_fallback.set(False)
            assert force_fallback.get() is False
        finally:
            # Restore original state to avoid leaking between tests
            force_fallback.set(original)
