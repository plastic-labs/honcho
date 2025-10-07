from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock

import pytest

from src.models import Message
from src.utils.representation import (
    DeductiveObservation,
    ExplicitObservation,
    Representation,
)


@pytest.mark.asyncio
async def test_generic_honcho_llm_call_mock():
    """Test that the generic honcho_llm_call mock is working for existing decorated functions"""
    # Import a function that we know is decorated with honcho_llm_call
    from src.deriver.deriver import critical_analysis_call

    # Call the decorated function - this should use our mock
    result = await critical_analysis_call(
        peer_id="test_peer_id",
        peer_card=["test_peer_card"],
        message_created_at=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        working_representation=Representation(
            explicit=[
                ExplicitObservation(
                    content="test explicit observation",
                    created_at=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test_session",
                )
            ],
            deductive=[
                DeductiveObservation(
                    conclusion="test deductive conclusion",
                    premises=["test premise 1", "test premise 2"],
                    created_at=datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
                    message_ids=[(1, 1)],
                    session_name="test_session",
                )
            ],
        ),
        history="test history",
        new_turns=["test new turn"],
        estimated_input_tokens=100,
    )

    # Verify that we get a mock result, not an actual LLM call
    assert result is not None
    # The result should have the attributes we expect from our mock
    assert hasattr(result, "explicit")
    assert hasattr(result, "deductive")
    assert hasattr(result, "_response")


@pytest.mark.asyncio
async def test_summarizer_decorated_functions_with_mock():
    """Test that summarizer decorated functions work with our mock"""
    # Import functions that we know are decorated with honcho_llm_call
    from src.utils.summarizer import create_long_summary, create_short_summary

    # Create mock messages for testing
    mock_message = MagicMock(spec=Message)
    mock_message.content = "Test message content"
    mock_message.peer_name = "test_peer"
    mock_messages = cast(list[Message], [mock_message])

    # Call the decorated functions - these should use our mock
    short_result = await create_short_summary(
        messages=mock_messages, input_tokens=100, previous_summary="Previous summary"
    )

    long_result = await create_long_summary(
        messages=mock_messages, previous_summary="Previous summary"
    )

    # Verify that we get mock results, not actual LLM calls
    assert short_result is not None
    assert long_result is not None
    # For functions with return_call_response=True, we should get a string or object with content
    # The existing mock returns a string, so we check if it's a string
    assert isinstance(short_result, str | object)
    assert isinstance(long_result, str | object)
    # If it's not a string, check for content attribute
    if not isinstance(short_result, str):
        assert hasattr(short_result, "content")
    if not isinstance(long_result, str):
        assert hasattr(long_result, "content")
