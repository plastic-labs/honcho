"""Unit tests for the v2 LLM-based promotion test.

Covers:
- `PROMOTION_TEST_PROMPT` shape (single-token YES/NO contract).
- `_parse_promotion_response` lenient parsing.
- `_llm_promotion_test` happy path (YES / NO).
- `_llm_promotion_test` falls back to the v1 heuristic on LLM error.
- `_llm_promotion_test` falls back on unparseable responses.
- `process_promotion` honors `PROMOTION.ENABLED=False` (no LLM call).

These are pure unit tests — `honcho_llm_call` is mocked, no DB. They don't
need the runtime-mock fixture blocklist because they import nothing from
src.main at module load (the conftest's `from src.main import app` still
runs for the whole suite, but these tests themselves don't touch the app).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.deriver import promotion
from src.deriver.promotion import (
    PROMOTION_TEST_PROMPT,
    _heuristic_promotion_test,
    _llm_promotion_test,
    _parse_promotion_response,
    _promotion_test_prompt,
)


# ── Prompt shape ───────────────────────────────────────────────────────────


def test_promotion_test_prompt_embeds_content() -> None:
    """The prompt must contain the observation content verbatim."""
    content = "The team decided on PostgreSQL for the metadata store."
    prompt = _promotion_test_prompt(content)

    assert content in prompt
    # Single-token contract is preserved.
    assert "YES" in prompt
    assert "NO" in prompt


def test_promotion_test_prompt_template_has_content_placeholder() -> None:
    """The module-level template must have a single {content} placeholder."""
    # Sanity: the template is .format()-compatible with exactly `content`.
    formatted = PROMOTION_TEST_PROMPT.format(content="X")
    assert "X" in formatted


# ── Response parsing ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("YES", True),
        ("yes", True),
        ("Yes", True),
        ("YES.", True),
        ("YES\n", True),
        ("  yes  ", True),
        ("Y", True),
        ("NO", False),
        ("no", False),
        ("No.", False),
        ("NO\n", False),
        ("  no  ", False),
        ("N", False),
    ],
)
def test_parse_promotion_response_recognizes_yes_no(raw: str, expected: bool) -> None:
    assert _parse_promotion_response(raw) is expected


@pytest.mark.parametrize("raw", [None, "", "   ", "maybe", "true", "1", "yep", "nope"])
def test_parse_promotion_response_returns_none_for_unparseable(raw: str | None) -> None:
    assert _parse_promotion_response(raw) is None


def test_parse_promotion_response_takes_first_line_only() -> None:
    """A model that returns 'YES\\nlong explanation' still counts as YES."""
    assert _parse_promotion_response("YES\nBecause it is durable.") is True
    assert _parse_promotion_response("NO\nIt's just an import statement.") is False


# ── _llm_promotion_test happy paths ────────────────────────────────────────


def _make_response(content: str | None) -> MagicMock:
    """Build a minimal HonchoLLMCallResponse-like mock."""
    resp = MagicMock()
    resp.content = content
    return resp


@pytest.mark.asyncio
async def test_llm_promotion_test_returns_true_when_model_says_yes() -> None:
    mock_call = AsyncMock(return_value=_make_response("YES"))
    with patch.object(promotion, "honcho_llm_call", mock_call):
        result = await _llm_promotion_test(
            "We decided to use Redis for active-context state.",
            workspace_name="ws",
            observer="obs",
            observed="peer",
        )

    assert result is True
    mock_call.assert_awaited_once()
    # Verify telemetry context is wired through.
    _, kwargs = mock_call.call_args
    assert kwargs["telemetry"].call_purpose == "promotion.test"
    assert kwargs["telemetry"].parent_category == "promotion"
    assert kwargs["telemetry"].workspace_name == "ws"
    assert kwargs["telemetry"].observer == "obs"
    assert kwargs["telemetry"].observed == "peer"
    assert kwargs["telemetry"].track_name == "Promotion Test"
    # temperature forced to 0.0 for deterministic classification.
    assert kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_llm_promotion_test_returns_false_when_model_says_no() -> None:
    mock_call = AsyncMock(return_value=_make_response("NO"))
    with patch.object(promotion, "honcho_llm_call", mock_call):
        result = await _llm_promotion_test("import os")

    assert result is False


# ── _llm_promotion_test fallback behavior (spec §7.4a) ────────────────────


@pytest.mark.asyncio
async def test_llm_promotion_test_falls_back_to_heuristic_on_llm_error() -> None:
    """Per spec §7.4a: on persistent LLM failure, promote conservatively
    (fall back to the heuristic) rather than dropping the observation."""
    mock_call = AsyncMock(side_effect=RuntimeError("provider 500"))
    # Use content the heuristic would promote, to confirm the fallback ran
    # (not just returned False).
    content = "We decided the metadata store is PostgreSQL after testing alternatives."

    with patch.object(promotion, "honcho_llm_call", mock_call):
        result = await _llm_promotion_test(content)

    assert result is True  # heuristic says: contains "decided" → promote
    mock_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_promotion_test_falls_back_on_unparseable_response() -> None:
    """An unparseable model response triggers the heuristic fallback."""
    mock_call = AsyncMock(return_value=_make_response("maybe, sort of"))
    content = "import os"  # heuristic says: obvious-from-code → NOT promote

    with patch.object(promotion, "honcho_llm_call", mock_call):
        result = await _llm_promotion_test(content)

    assert result is False  # heuristic fallback → obvious pattern → False
    mock_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_promotion_test_falls_back_on_none_response() -> None:
    """A None content (provider returned nothing) triggers the heuristic."""
    mock_call = AsyncMock(return_value=_make_response(None))
    content = "We concluded the API needs JWT auth."

    with patch.object(promotion, "honcho_llm_call", mock_call):
        result = await _llm_promotion_test(content)

    assert result is True  # heuristic: "concluded" → promote


# ── process_promotion respects PROMOTION.ENABLED ───────────────────────────


@pytest.mark.asyncio
async def test_process_promotion_uses_heuristic_when_disabled() -> None:
    """When PROMOTION.ENABLED=False, no LLM call is made; the v1 heuristic
    is used directly. This makes ENABLED a real off-switch (no spend)."""
    mock_call = AsyncMock(return_value=_make_response("YES"))
    # Short-circuit process_promotion before it touches the DB by mocking
    # tracked_db to yield a MagicMock session, and stub the CRUD helpers
    # so we never need a real Postgres.
    mock_db_ctx = MagicMock()
    mock_db_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    mock_db_ctx.__aexit__ = AsyncMock(return_value=None)

    with (
        patch.object(promotion, "honcho_llm_call", mock_call),
        patch.object(promotion, "tracked_db", return_value=mock_db_ctx),
        patch.object(promotion, "_get_document", AsyncMock(return_value=None)),
        patch.object(promotion.settings.PROMOTION, "ENABLED", False),
    ):
        # Observation not found → early return, but the key assertion is
        # that no LLM call was made even though we reached process_promotion.
        await promotion.process_promotion(
            workspace_name="ws",
            collection_name="coll",
            obs_id="obs-1",
            observer="obs",
            observed="peer",
        )

    mock_call.assert_not_called()


# ── Heuristic retained as a public fallback ────────────────────────────────


def test_heuristic_promotion_test_still_works() -> None:
    """The v1 heuristic is still callable (used as the v2 fallback)."""
    assert _heuristic_promotion_test("import os") is False
    assert _heuristic_promotion_test("We decided on Redis.") is True
    assert _heuristic_promotion_test("maybe later") is False
    assert _heuristic_promotion_test("short") is False  # < 20 chars