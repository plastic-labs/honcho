"""Fixtures for mocking LLM calls in tests.

This module provides pytest fixtures that mock honcho_llm_call to avoid real API calls
during testing while still executing tool calls to properly test agent behavior.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from src.utils.clients import HonchoLLMCallResponse


class MockLLMResponse:
    """Helper class for creating mock LLM responses."""

    @staticmethod
    def create_response(
        content: str | BaseModel = "Mock LLM response",
        input_tokens: int = 100,
        output_tokens: int = 50,
        tool_calls_made: list[dict[str, Any]] | None = None,
    ) -> HonchoLLMCallResponse[Any]:
        """Create a mock HonchoLLMCallResponse.

        Args:
            content: Response content (string or pydantic model)
            input_tokens: Mock input token count
            output_tokens: Mock output token count
            tool_calls_made: List of tool calls that were made

        Returns:
            HonchoLLMCallResponse with mock data
        """
        return HonchoLLMCallResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            finish_reasons=["end_turn"],
            tool_calls_made=tool_calls_made or [],
        )

    @staticmethod
    def create_tool_response(
        tool_results: list[dict[str, Any]],
        final_content: str = "Task completed successfully",
    ) -> HonchoLLMCallResponse[str]:
        """Create a mock response that simulates tool execution.

        Args:
            tool_results: List of tool execution results
            final_content: Final response after tool execution

        Returns:
            HonchoLLMCallResponse with tool execution results
        """
        return HonchoLLMCallResponse(
            content=final_content,
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            finish_reasons=["end_turn"],
            tool_calls_made=tool_results,
        )


async def mock_honcho_llm_call_with_tools(
    tool_executor: Any = None,
    tools: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> HonchoLLMCallResponse[Any]:
    """Mock implementation of honcho_llm_call that executes tools.

    This mock will:
    1. Execute tool calls if tool_executor is provided
    2. Return a mock response quickly without real LLM API calls
    3. Simulate realistic tool execution flow

    Args:
        tool_executor: Callable for executing tools (synchronous)
        tools: Available tool definitions
        **kwargs: Other arguments (ignored in mock)

    Returns:
        Mock HonchoLLMCallResponse
    """
    # If tool_executor is provided and tools are available, simulate tool calls
    if tool_executor and tools:
        # Extract tool names to understand what operations are being requested
        tool_names = [tool.get("name") for tool in tools]

        # Simulate tool execution based on available tools
        if "create_hypothesis" in tool_names:
            # Extract premise IDs from messages context if available
            source_premise_ids = []
            messages = kwargs.get("messages", [])

            for message in messages:
                content = message.get("content", "") if isinstance(message, dict) else str(message)
                # Look for premise ID patterns (nanoid format)
                import re
                premise_ids = re.findall(r'\b[A-Za-z0-9_-]{21}\b', content)
                source_premise_ids.extend(premise_ids)

            # Use fallback if no IDs found
            if not source_premise_ids:
                source_premise_ids = ["mock_premise_fallback"]

            # Limit to first 3 premise IDs
            source_premise_ids = source_premise_ids[:3]

            # Simulate hypothesis creation
            tool_executor(
                "create_hypothesis",
                {
                    "content": "Mock hypothesis: User prefers feature X",
                    "confidence": 0.75,
                    "tier": 1,
                    "source_premise_ids": source_premise_ids,  # Use extracted IDs
                },
            )
            return MockLLMResponse.create_tool_response(
                tool_results=[{"tool": "create_hypothesis", "count": 1}],
                final_content="Generated 1 hypothesis based on observations",
            )

        if "create_prediction" in tool_names:
            # Simulate prediction creation
            tool_executor(
                "create_prediction",
                {
                    "content": "Mock prediction: User will request feature X",
                    "tier": 1,
                    "confidence": 0.7,
                    "specificity": 0.7,
                },
            )
            return MockLLMResponse.create_tool_response(
                tool_results=[{"tool": "create_prediction", "count": 1}],
                final_content="Generated 1 prediction from hypothesis",
            )

        if "generate_search_query" in tool_names:
            # Simulate falsification with proper tool sequence
            tool_executor(
                "generate_search_query",
                {
                    "query": "User prefers feature X",
                    "strategy": "direct_contradiction",
                },
            )
            tool_executor(
                "evaluate_prediction",
                {
                    "evidence_summary": "No contradicting evidence found",
                    "confidence": 0.9,  # Above unfalsified threshold
                    "determination": "unfalsified",
                },
            )
            return MockLLMResponse.create_tool_response(
                tool_results=[{"tool": "evaluate_prediction", "determination": "unfalsified"}],
                final_content="Falsification complete: prediction unfalsified",
            )

        if "create_induction" in tool_names:
            # Simulate induction creation
            tool_executor(
                "create_induction",
                {
                    "content": "Mock pattern: User consistently prefers X over Y",
                    "pattern_type": "preferential",
                    "confidence": 0.85,
                    "pattern_strength": 0.9,
                    "generalization_scope": "domain-specific",
                },
            )
            return MockLLMResponse.create_tool_response(
                tool_results=[{"tool": "create_induction", "count": 1}],
                final_content="Extracted 1 pattern from unfalsified predictions",
            )

    # Default response if no tool execution needed
    return MockLLMResponse.create_response()


@pytest.fixture
def mock_llm_call():
    """Fixture that mocks honcho_llm_call for all agents.

    This fixture patches honcho_llm_call to use the mock implementation,
    allowing tests to run quickly without real API calls while still
    executing tool calls that write to the database.

    Usage:
        @pytest.mark.asyncio
        async def test_something(mock_llm_call):
            # honcho_llm_call is automatically mocked
            result = await agent.execute(...)
            # Tool calls will have executed, creating DB records
            assert result is not None
    """
    with patch(
        "src.utils.clients.honcho_llm_call",
        new=AsyncMock(side_effect=mock_honcho_llm_call_with_tools),
    ) as mock:
        yield mock


@pytest.fixture
def mock_llm_call_no_tools():
    """Fixture that mocks honcho_llm_call without tool execution.

    Useful for testing scenarios where you want to prevent any LLM calls
    but don't need tool execution.

    Usage:
        @pytest.mark.asyncio
        async def test_something(mock_llm_call_no_tools):
            # honcho_llm_call returns immediately with mock response
            result = await some_function_that_calls_llm()
            assert result is not None
    """
    with patch(
        "src.utils.clients.honcho_llm_call",
        new=AsyncMock(return_value=MockLLMResponse.create_response()),
    ) as mock:
        yield mock


@pytest.fixture
def mock_abducer_llm():
    """Mock specifically for Abducer agent LLM calls."""

    async def mock_abducer_call(**kwargs: Any) -> HonchoLLMCallResponse[str]:
        tool_executor = kwargs.get("tool_executor")
        tool_calls_made = []
        if tool_executor:
            # Extract premise IDs from messages context if available
            # The Abducer passes premises in the messages, so we can extract their IDs
            source_premise_ids = []
            messages = kwargs.get("messages", [])

            # Try to extract premise IDs from the message content
            # Messages should contain premise information from the agent
            for message in messages:
                content = message.get("content", "") if isinstance(message, dict) else str(message)
                # Look for premise ID patterns (nanoid format: alphanumeric with hyphens)
                import re
                # Match nanoid-style IDs in the content
                premise_ids = re.findall(r'\b[A-Za-z0-9_-]{21}\b', content)
                source_premise_ids.extend(premise_ids)

            # If we couldn't extract any IDs, use a placeholder
            # (This shouldn't happen in real tests with proper setup)
            if not source_premise_ids:
                source_premise_ids = ["mock_premise_fallback"]

            # Use only the first few premise IDs (limit to 3 for realism)
            source_premise_ids = source_premise_ids[:3]

            # Simulate creating one hypothesis using the actual tool name from agent
            # Note: source_premise_ids must be non-empty (agent validation requirement)
            tool_call_input = {
                "content": "User demonstrates preference for concise responses",
                "confidence": 0.8,
                "tier": 1,
                "source_premise_ids": source_premise_ids,  # Use extracted or fallback IDs
            }
            result = tool_executor("create_hypothesis", tool_call_input)
            tool_calls_made.append({
                "name": "create_hypothesis",
                "input": tool_call_input,
                "result": result,
            })
        return MockLLMResponse.create_response(
            "Generated hypothesis from premises",
            tool_calls_made=tool_calls_made,
        )

    with patch(
        "src.agents.abducer.agent.honcho_llm_call",
        new=AsyncMock(side_effect=mock_abducer_call),
    ) as mock:
        yield mock


@pytest.fixture
def mock_predictor_llm():
    """Mock specifically for Predictor agent LLM calls."""

    async def mock_predictor_call(**kwargs: Any) -> HonchoLLMCallResponse[str]:
        tool_executor = kwargs.get("tool_executor")
        if tool_executor:
            # Simulate creating one prediction
            tool_executor(
                "create_prediction",
                {
                    "content": "User will prefer brief explanations in future queries",
                    "tier": 1,
                    "confidence": 0.75,
                    "specificity": 0.7,
                },
            )
        return MockLLMResponse.create_response("Generated prediction from hypothesis")

    with patch(
        "src.agents.predictor.agent.honcho_llm_call",
        new=AsyncMock(side_effect=mock_predictor_call),
    ) as mock:
        yield mock


@pytest.fixture
def mock_falsifier_llm():
    """Mock specifically for Falsifier agent LLM calls."""

    async def mock_falsifier_call(**kwargs: Any) -> HonchoLLMCallResponse[str]:
        tool_executor = kwargs.get("tool_executor")
        tool_calls_made = []
        if tool_executor:
            # Simulate falsifier tool calls: generate search query and evaluate prediction
            result1 = tool_executor(
                "generate_search_query",
                {
                    "query": "User demonstrates preference for concise responses",
                    "strategy": "direct_contradiction",
                },
            )
            tool_calls_made.append({
                "name": "generate_search_query",
                "input": {"query": "User demonstrates preference for concise responses", "strategy": "direct_contradiction"},
                "result": result1,
            })

            # Evaluate prediction as unfalsified with high confidence
            result2 = tool_executor(
                "evaluate_prediction",
                {
                    "evidence_summary": "No contradicting evidence found in observations",
                    "confidence": 0.9,  # Above unfalsified_confidence_threshold (0.7)
                    "determination": "unfalsified",
                },
            )
            tool_calls_made.append({
                "name": "evaluate_prediction",
                "input": {
                    "evidence_summary": "No contradicting evidence found in observations",
                    "confidence": 0.9,
                    "determination": "unfalsified",
                },
                "result": result2,
            })
        return MockLLMResponse.create_response(
            "Falsification complete",
            tool_calls_made=tool_calls_made,
        )

    with patch(
        "src.agents.falsifier.agent.honcho_llm_call",
        new=AsyncMock(side_effect=mock_falsifier_call),
    ) as mock:
        yield mock


@pytest.fixture
def mock_inductor_llm():
    """Mock specifically for Inductor agent LLM calls."""

    async def mock_inductor_call(**kwargs: Any) -> HonchoLLMCallResponse[str]:
        tool_executor = kwargs.get("tool_executor")
        if tool_executor:
            # Simulate creating induction
            tool_executor(
                "create_induction",
                {
                    "content": "User consistently prefers concise, actionable responses",
                    "pattern_type": "preferential",
                    "confidence": 0.9,
                    "pattern_strength": 0.85,
                    "generalization_scope": "general",
                },
            )
        return MockLLMResponse.create_response("Extracted behavioral pattern")

    with patch(
        "src.agents.inductor.agent.honcho_llm_call",
        new=AsyncMock(side_effect=mock_inductor_call),
    ) as mock:
        yield mock


@pytest.fixture
def mock_all_reasoning_agents(
    mock_abducer_llm,
    mock_predictor_llm,
    mock_falsifier_llm,
    mock_inductor_llm,
):
    """Composite fixture that mocks all reasoning agent LLM calls.

    This is the most convenient fixture for integration tests that exercise
    the full reasoning dream workflow.

    Usage:
        @pytest.mark.asyncio
        async def test_reasoning_workflow(db_session, mock_all_reasoning_agents):
            # All agent LLM calls are mocked
            metrics = await process_reasoning_dream(...)
            # Verify metrics and database state
            assert metrics.hypotheses_generated > 0
    """
    return {
        "abducer": mock_abducer_llm,
        "predictor": mock_predictor_llm,
        "falsifier": mock_falsifier_llm,
        "inductor": mock_inductor_llm,
    }
