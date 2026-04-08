from unittest.mock import AsyncMock, patch

import pytest

from src.config import settings
from src.dreamer.specialists import DeductionSpecialist
from src.utils.clients import HonchoLLMCallResponse


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
