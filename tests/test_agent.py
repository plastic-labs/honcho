from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src import agent


@pytest.mark.asyncio
async def test_dialectic_call_function_exists():
    """Test that dialectic_call function exists and can be mocked"""
    with patch("src.agent.dialectic_call") as mock_call:
        mock_call.return_value = MagicMock(content="test response")

        # This would normally make an LLM call, but it's mocked
        result = await agent.dialectic_call(
            query="test query",
            working_representation="test representation",
            additional_context="test context",
        )

        assert mock_call.called
        assert result.content == "test response"


@pytest.mark.asyncio
async def test_dialectic_stream_function_exists():
    """Test that dialectic_stream function exists and can be mocked"""
    with patch("src.agent.dialectic_stream") as mock_stream:
        mock_stream.return_value = AsyncMock()

        # This would normally make a streaming LLM call, but it's mocked
        result = await agent.dialectic_stream(
            query="test query",
            working_representation="test representation",
            additional_context="test context",
        )

        assert mock_stream.called
        assert result is not None


@pytest.mark.asyncio
async def test_generate_semantic_queries_llm_function_exists():
    """Test that generate_semantic_queries_llm function exists and can be mocked"""
    with patch("src.agent.generate_semantic_queries_llm") as mock_queries:
        mock_queries.return_value = ["query1", "query2", "query3"]

        # This would normally make an LLM call, but it's mocked
        result = await agent.generate_semantic_queries_llm("test query")

        assert mock_queries.called
        assert result == ["query1", "query2", "query3"]


@pytest.mark.asyncio
async def test_run_tom_inference_function():
    """Test that run_tom_inference function works with new Pydantic objects"""
    with patch("src.agent.get_tom_inference_single_prompt") as mock_tom:
        from src.deriver.tom.single_prompt import (
            CurrentState,
            TentativeInference,
            TomInferenceOutput,
        )

        # Mock the function to return a proper Pydantic object
        mock_tom_response = TomInferenceOutput(
            current_state=CurrentState(
                immediate_context="test context",
                active_goals="test goals",
                present_mood="test mood",
            ),
            tentative_inferences=[
                TentativeInference(interpretation="test inference", basis="test basis")
            ],
            knowledge_gaps=[],
            expectation_violations=[],
        )
        mock_tom.return_value = mock_tom_response

        # Test the function
        result = await agent.run_tom_inference("test chat history")

        # Verify it extracted the right information from the Pydantic object
        assert "test context" in result
        assert "test inference" in result
        assert mock_tom.called
