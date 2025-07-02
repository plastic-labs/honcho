from unittest.mock import AsyncMock, patch

import pytest

from src.deriver.tom import get_tom_inference, get_user_representation
from src.deriver.tom.single_prompt import (
    CertaintyLevel,
    CurrentState,
    InfoType,
    PersistentInfo,
    SourcedInfo,
    TentativeInference,
    TentativePattern,
    TomInferenceOutput,
    UpdateSection,
    UserCurrentState,
    UserExpectationViolation,
    UserKnowledgeGap,
    UserRepresentationOutput,
)


@pytest.mark.asyncio
async def test_get_tom_inference_conversational_method():
    """Test that get_tom_inference calls tom_inference_conversational when method='conversational'"""
    # Create a mock TomInferenceOutput to return
    mock_output = TomInferenceOutput(
        current_state=CurrentState(
            immediate_context="test context",
            active_goals="test goals",
            present_mood="test mood",
        ),
        tentative_inferences=[
            TentativeInference(interpretation="test interpretation", basis="test basis")
        ],
        knowledge_gaps=[],
        expectation_violations=[],
    )

    # Mock the tom_inference_conversational function
    with patch(
        "src.deriver.tom.tom_inference_conversational", new_callable=AsyncMock
    ) as mock_conversational:
        mock_conversational.return_value = mock_output

        # Call get_tom_inference with method='conversational'
        result = await get_tom_inference(
            chat_history="test chat history",
            user_representation="test representation",
            method="conversational",
        )

        # Verify that tom_inference_conversational was called
        mock_conversational.assert_called_once_with(
            "test chat history", "test representation"
        )

        # Verify the result is what we expected
        assert result == mock_output
        assert result.current_state.immediate_context == "test context"
        assert result.tentative_inferences[0].interpretation == "test interpretation"


@pytest.mark.asyncio
async def test_get_tom_inference_conversational_method_default():
    """Test that get_tom_inference defaults to conversational method when no method specified"""
    # Create a mock TomInferenceOutput to return
    mock_output = TomInferenceOutput(
        current_state=CurrentState(
            immediate_context="default context",
            active_goals="default goals",
            present_mood="default mood",
        ),
        tentative_inferences=[],
        knowledge_gaps=[],
        expectation_violations=[],
    )

    # Mock the tom_inference_conversational function
    with patch(
        "src.deriver.tom.tom_inference_conversational", new_callable=AsyncMock
    ) as mock_conversational:
        mock_conversational.return_value = mock_output

        # Call get_tom_inference without specifying method (should default to 'conversational')
        result = await get_tom_inference(
            chat_history="test chat history", user_representation="test representation"
        )

        # Verify that tom_inference_conversational was called (line 24 coverage)
        mock_conversational.assert_called_once_with(
            "test chat history", "test representation"
        )

        # Verify the result
        assert result == mock_output


@pytest.mark.asyncio
async def test_get_tom_inference_conversational_method_with_defaults():
    """Test that get_tom_inference works with default parameters for conversational method"""
    # Create a mock TomInferenceOutput to return
    mock_output = TomInferenceOutput(
        current_state=CurrentState(
            immediate_context="minimal context",
            active_goals="minimal goals",
            present_mood="minimal mood",
        ),
        tentative_inferences=[],
        knowledge_gaps=[],
        expectation_violations=[],
    )

    # Mock the tom_inference_conversational function
    with patch(
        "src.deriver.tom.tom_inference_conversational", new_callable=AsyncMock
    ) as mock_conversational:
        mock_conversational.return_value = mock_output

        # Call get_tom_inference with minimal parameters, should use conversational method
        result = await get_tom_inference(chat_history="minimal chat")

        # Verify that tom_inference_conversational was called with default user_representation
        mock_conversational.assert_called_once_with("minimal chat", "None")

        # Verify the result
        assert result == mock_output


@pytest.mark.asyncio
async def test_get_tom_inference_invalid_method():
    """Test that get_tom_inference raises ValueError for invalid method"""
    with pytest.raises(ValueError, match="Invalid method: invalid_method"):
        await get_tom_inference(
            chat_history="test chat history",
            user_representation="test representation",
            method="invalid_method",
        )


@pytest.mark.asyncio
async def test_get_user_representation_conversational_method():
    """Test that get_user_representation calls user_representation_conversational when method='conversational'"""
    # Create a mock UserRepresentationOutput to return
    mock_output = UserRepresentationOutput(
        current_state=UserCurrentState(
            active_context=SourcedInfo(detail="test context", source="message 1"),
            temporary_conditions=SourcedInfo(
                detail="test conditions", source="message 2"
            ),
            present_mood_activity=SourcedInfo(detail="test mood", source="message 3"),
        ),
        persistent_information=[
            PersistentInfo(
                detail="test persistent info",
                source="conversation",
                info_type=InfoType.STATEMENT,
            )
        ],
        tentative_patterns=[
            TentativePattern(
                pattern="test pattern",
                source="conversation",
                certainty_level=CertaintyLevel.LIKELY,
            )
        ],
        knowledge_gaps=[UserKnowledgeGap(missing_info="test missing info")],
        expectation_violations=[
            UserExpectationViolation(
                potential_surprise="test surprise",
                reason="test reason",
                confidence_level=0.8,
            )
        ],
        updates=UpdateSection(
            new_information=[SourcedInfo(detail="new info", source="message 4")],
            changes=[SourcedInfo(detail="changed info", source="message 5")],
            removals=[SourcedInfo(detail="removed info", source="message 6")],
        ),
    )

    # Mock the user_representation_conversational function
    with patch(
        "src.deriver.tom.user_representation_conversational", new_callable=AsyncMock
    ) as mock_conversational:
        mock_conversational.return_value = mock_output

        # Call get_user_representation with method='conversational'
        result = await get_user_representation(
            chat_history="test chat history",
            user_representation="test representation",
            tom_inference="test inference",
            method="conversational",
        )

        # Verify that user_representation_conversational was called with correct parameters
        mock_conversational.assert_called_once_with(
            "test chat history", "test representation", "test inference"
        )

        # Verify the result is what we expected
        assert result == mock_output
        assert result.current_state.active_context.detail == "test context"
        assert result.persistent_information[0].detail == "test persistent info"
        assert result.tentative_patterns[0].pattern == "test pattern"


@pytest.mark.asyncio
async def test_get_user_representation_conversational_method_default():
    """Test that get_user_representation defaults to conversational method when no method specified"""
    # Create a mock UserRepresentationOutput to return
    mock_output = UserRepresentationOutput(
        current_state=UserCurrentState(
            active_context=SourcedInfo(detail="default context", source="default"),
            temporary_conditions=SourcedInfo(
                detail="default conditions", source="default"
            ),
            present_mood_activity=SourcedInfo(detail="default mood", source="default"),
        ),
        persistent_information=[],
        tentative_patterns=[],
        knowledge_gaps=[],
        expectation_violations=[],
        updates=UpdateSection(new_information=[], changes=[], removals=[]),
    )

    # Mock the user_representation_conversational function
    with patch(
        "src.deriver.tom.user_representation_conversational", new_callable=AsyncMock
    ) as mock_conversational:
        mock_conversational.return_value = mock_output

        # Call get_user_representation without specifying method (should default to 'conversational')
        result = await get_user_representation(
            chat_history="test chat history",
            user_representation="test representation",
            tom_inference="test inference",
        )

        # Verify that user_representation_conversational was called with correct parameters
        mock_conversational.assert_called_once_with(
            "test chat history", "test representation", "test inference"
        )

        # Verify the result
        assert result == mock_output


@pytest.mark.asyncio
async def test_get_user_representation_conversational_method_with_defaults():
    """Test that get_user_representation works with default parameters for conversational method"""
    # Create a mock UserRepresentationOutput to return
    mock_output = UserRepresentationOutput(
        current_state=UserCurrentState(
            active_context=SourcedInfo(detail="minimal context", source="minimal"),
            temporary_conditions=SourcedInfo(
                detail="minimal conditions", source="minimal"
            ),
            present_mood_activity=SourcedInfo(detail="minimal mood", source="minimal"),
        ),
        persistent_information=[],
        tentative_patterns=[],
        knowledge_gaps=[],
        expectation_violations=[],
        updates=UpdateSection(new_information=[], changes=[], removals=[]),
    )

    # Mock the user_representation_conversational function
    with patch(
        "src.deriver.tom.user_representation_conversational", new_callable=AsyncMock
    ) as mock_conversational:
        mock_conversational.return_value = mock_output

        # Call get_user_representation with minimal parameters, should use conversational method and default values
        result = await get_user_representation(chat_history="minimal chat")

        # Verify that user_representation_conversational was called with default parameters
        mock_conversational.assert_called_once_with("minimal chat", "None", "None")

        # Verify the result
        assert result == mock_output


@pytest.mark.asyncio
async def test_get_user_representation_single_prompt_method():
    """Test that get_user_representation calls user_representation_single_prompt when method='single_prompt'"""
    # Create a mock UserRepresentationOutput to return
    mock_output = UserRepresentationOutput(
        current_state=UserCurrentState(
            active_context=SourcedInfo(detail="single prompt context", source="single"),
            temporary_conditions=SourcedInfo(
                detail="single prompt conditions", source="single"
            ),
            present_mood_activity=SourcedInfo(
                detail="single prompt mood", source="single"
            ),
        ),
        persistent_information=[
            PersistentInfo(
                detail="single prompt persistent info",
                source="single prompt",
                info_type=InfoType.STATEMENT,
            )
        ],
        tentative_patterns=[
            TentativePattern(
                pattern="single prompt pattern",
                source="single prompt",
                certainty_level=CertaintyLevel.LIKELY,
            )
        ],
        knowledge_gaps=[],
        expectation_violations=[],
        updates=UpdateSection(new_information=[], changes=[], removals=[]),
    )

    # Mock the user_representation_single_prompt function
    with patch(
        "src.deriver.tom.user_representation_single_prompt", new_callable=AsyncMock
    ) as mock_single_prompt:
        mock_single_prompt.return_value = mock_output

        # Call get_user_representation with method='single_prompt'
        result = await get_user_representation(
            chat_history="test chat history",
            user_representation="test representation",
            tom_inference="test inference",
            method="single_prompt",
        )

        # Verify that user_representation_single_prompt was called with correct parameters (lines 42-43)
        mock_single_prompt.assert_called_once_with(
            "test chat history", "test representation", "test inference"
        )

        # Verify the result is what we expected
        assert result == mock_output
        assert result.current_state.active_context.detail == "single prompt context"
        assert (
            result.persistent_information[0].detail == "single prompt persistent info"
        )
        assert result.tentative_patterns[0].pattern == "single prompt pattern"


@pytest.mark.asyncio
async def test_get_user_representation_long_term_method():
    """Test that get_user_representation calls get_user_representation_long_term when method='long_term'"""
    # Create a mock UserRepresentationOutput to return
    mock_output = UserRepresentationOutput(
        current_state=UserCurrentState(
            active_context=SourcedInfo(detail="long term context", source="long term"),
            temporary_conditions=SourcedInfo(
                detail="long term conditions", source="long term"
            ),
            present_mood_activity=SourcedInfo(
                detail="long term mood", source="long term"
            ),
        ),
        persistent_information=[
            PersistentInfo(
                detail="long term persistent info",
                source="long term",
                info_type=InfoType.STATEMENT,
            )
        ],
        tentative_patterns=[
            TentativePattern(
                pattern="long term pattern",
                source="long term",
                certainty_level=CertaintyLevel.LIKELY,
            )
        ],
        knowledge_gaps=[],
        expectation_violations=[],
        updates=UpdateSection(new_information=[], changes=[], removals=[]),
    )

    # Mock the get_user_representation_long_term function
    with patch(
        "src.deriver.tom.get_user_representation_long_term", new_callable=AsyncMock
    ) as mock_long_term:
        mock_long_term.return_value = mock_output

        # Call get_user_representation with method='long_term'
        result = await get_user_representation(
            chat_history="test chat history",
            user_representation="test representation",
            tom_inference="test inference",
            method="long_term",
        )

        # Verify that get_user_representation_long_term was called with correct parameters (lines 46-47)
        mock_long_term.assert_called_once_with(
            "test chat history", "test representation", "test inference"
        )

        # Verify the result is what we expected
        assert result == mock_output
        assert result.current_state.active_context.detail == "long term context"
        assert result.persistent_information[0].detail == "long term persistent info"
        assert result.tentative_patterns[0].pattern == "long term pattern"


@pytest.mark.asyncio
async def test_get_user_representation_invalid_method():
    """Test that get_user_representation raises ValueError for invalid method"""
    with pytest.raises(ValueError, match="Invalid method: invalid_method"):
        await get_user_representation(
            chat_history="test chat history",
            user_representation="test representation",
            tom_inference="test inference",
            method="invalid_method",
        )


@pytest.mark.asyncio
async def test_get_tom_inference_single_prompt_method():
    """Test that get_tom_inference calls tom_inference_single_prompt when method='single_prompt'"""
    # Create a mock TomInferenceOutput to return
    mock_output = TomInferenceOutput(
        current_state=CurrentState(
            immediate_context="single prompt context",
            active_goals="single prompt goals",
            present_mood="single prompt mood",
        ),
        tentative_inferences=[
            TentativeInference(
                interpretation="single prompt interpretation",
                basis="single prompt basis",
            )
        ],
        knowledge_gaps=[],
        expectation_violations=[],
    )

    # Mock the tom_inference_single_prompt function
    with patch(
        "src.deriver.tom.tom_inference_single_prompt", new_callable=AsyncMock
    ) as mock_single_prompt:
        mock_single_prompt.return_value = mock_output

        # Call get_tom_inference with method='single_prompt'
        result = await get_tom_inference(
            chat_history="test chat history",
            user_representation="test representation",
            method="single_prompt",
        )

        # Verify that tom_inference_single_prompt was called with correct parameters (lines 25-26)
        mock_single_prompt.assert_called_once_with(
            "test chat history", "test representation"
        )

        # Verify the result is what we expected
        assert result == mock_output
        assert result.current_state.immediate_context == "single prompt context"
        assert (
            result.tentative_inferences[0].interpretation
            == "single prompt interpretation"
        )
