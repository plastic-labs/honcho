from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.dreamer.specialists import DeductionSpecialist, InductionSpecialist
from src.llm import HonchoLLMCallResponse
from src.utils.agent_tools import TOOLS


def test_induction_specialist_prompt_discourages_cross_domain_trait_merges() -> None:
    specialist = InductionSpecialist()
    prompt = specialist.build_system_prompt("alice")

    assert "Preserve scope and applicability conditions" in prompt
    assert "Do NOT merge unrelated examples into a single personality trait" in prompt
    assert (
        "PREFERENCE: For reading nonfiction, prefers annotated print books over ebooks"
        in prompt
    )
    assert (
        "TRAIT: Meticulous planner (e.g., vacation itinerary optimization, desk cable management by length)"
        in prompt
    )


def test_update_peer_card_tool_description_mentions_scope_and_example_bundles() -> None:
    tool = TOOLS["update_peer_card"]
    description = tool["description"]
    content_description = tool["input_schema"]["properties"]["content"]["description"]

    assert "Preserve applicability conditions" in description
    assert "Do not merge unrelated examples into one trait" in description
    assert "temporary events" in description
    assert "duplicate entries" in description
    assert "not an evidence list or example bundle" in content_description


@pytest.mark.asyncio
async def test_deduction_specialist_uses_nested_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.METRICS, "ENABLED", False)
    specialist = DeductionSpecialist()
    mock_response = HonchoLLMCallResponse(
        content="done",
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["stop"],
    )

    with (
        patch(
            "src.dreamer.specialists.crud.get_peer",
            new=AsyncMock(),
        ),
        patch(
            "src.dreamer.specialists.crud.get_peer_card",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "src.dreamer.specialists.create_tool_executor",
            new=AsyncMock(return_value=AsyncMock()),
        ),
        patch(
            "src.dreamer.specialists.honcho_llm_call",
            new=AsyncMock(return_value=mock_response),
        ) as mock_llm_call,
    ):
        result = await specialist.run(
            workspace_name="workspace",
            observer="alice",
            observed="alice",
            session_name="session",
        )

    await_args = mock_llm_call.await_args
    if await_args is None:
        raise AssertionError("Expected dreamer LLM call")
    kwargs = await_args.kwargs
    expected_config = settings.DREAM.DEDUCTION_MODEL_CONFIG

    assert result.content == "done"
    assert kwargs["model_config"] == expected_config
    assert "llm_settings" not in kwargs
