from unittest.mock import MagicMock, patch

import pytest

from src.deriver.tom import long_term, single_prompt


@pytest.mark.asyncio
async def test_extract_facts_long_term_function_exists():
    """Test that extract_facts_long_term function exists and can be mocked"""
    with patch("src.deriver.tom.long_term.extract_facts_long_term") as mock_extract:
        mock_extract.return_value = MagicMock(facts=["fact1", "fact2"])

        result = await long_term.extract_facts_long_term("test chat history")

        assert mock_extract.called
        assert result.facts == ["fact1", "fact2"]


@pytest.mark.asyncio
async def test_get_user_representation_long_term_function_exists():
    """Test that get_user_representation_long_term function exists and can be mocked"""
    with patch(
        "src.deriver.tom.long_term.get_user_representation_long_term"
    ) as mock_rep:
        mock_rep.return_value = MagicMock(
            current_state="test state",
            tentative_patterns=["pattern1"],
            knowledge_gaps=["gap1"],
            expectation_violations=[],
            updates=["update1"],
        )

        result = await long_term.get_user_representation_long_term(
            chat_history="test history",
            user_representation="test rep",
            tom_inference="test inference",
            facts=["fact1"],
        )

        assert mock_rep.called
        assert result.current_state == "test state"


@pytest.mark.asyncio
async def test_get_tom_inference_single_prompt_function_exists():
    """Test that get_tom_inference_single_prompt function exists and can be mocked"""
    with patch(
        "src.deriver.tom.single_prompt.get_tom_inference_single_prompt"
    ) as mock_tom:
        mock_tom.return_value = MagicMock(inference="test inference")

        result = await single_prompt.get_tom_inference_single_prompt("test history")

        assert mock_tom.called
        assert result.inference == "test inference"


@pytest.mark.asyncio
async def test_get_user_representation_single_prompt_function_exists():
    """Test that get_user_representation_single_prompt function exists and can be mocked"""
    with patch(
        "src.deriver.tom.single_prompt.get_user_representation_single_prompt"
    ) as mock_rep:
        mock_rep.return_value = "test representation"

        result = await single_prompt.get_user_representation_single_prompt(
            chat_history="test history", user_representation="existing rep"
        )

        assert mock_rep.called
        assert result == "test representation"
