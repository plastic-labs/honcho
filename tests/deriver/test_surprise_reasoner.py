from datetime import datetime
from inspect import cleandoc as c

import pytest
from mirascope import llm
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.deriver.models import ReasoningResponse, StructuredObservation
from src.deriver.surprise_reasoner import SurpriseReasoner
from src.deriver.tom.embeddings import CollectionEmbeddingStore


# Fixture for the database session
@pytest.fixture
async def db_session():
    # Create an in-memory SQLite database for testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        yield session


# Fixture for the embedding store
@pytest.fixture
async def embedding_store(db_session):
    # Initialize with real embedding store
    store = CollectionEmbeddingStore(
        db=db_session,
        app_id="test_app",
        user_id="test_user",
        collection_id="test_collection",
    )
    return store


# Fixture for the surprise reasoner
@pytest.fixture
async def surprise_reasoner(embedding_store):
    return SurpriseReasoner(embedding_store)


@pytest.mark.asyncio
async def test_basic_reasoning_with_real_llm(surprise_reasoner, db_session):
    """Test basic reasoning with the real LLM."""

    # Initial context with some basic observations
    initial_context = ReasoningResponse(
        thinking="Initial analysis",
        explicit=["User said: I am a software engineer"],
        deductive=[
            StructuredObservation(
                conclusion="User works in technology",
                premises=["User is a software engineer"],
            )
        ],
        inductive=[],
        abductive=[],
    )

    # Sample conversation history
    history = """
    User: I'm working on a new project
    Assistant: What kind of project?
    User: It's a web application using Python
    """

    # New turn to analyze
    new_turn = "I'm using FastAPI for the backend and React for the frontend"

    # Current time
    current_time = datetime.now().isoformat()

    # Run the recursive reasoning
    final_observations, trace = await surprise_reasoner.recursive_reason_with_trace(
        db=db_session,
        context=initial_context,
        history=history,
        new_turn=new_turn,
        message_id="test_message_1",
        session_id="test_session_1",
        current_time=current_time,
    )

    # Basic assertions to verify the reasoning worked
    assert final_observations is not None
    assert trace is not None

    # Verify we got some observations back
    assert len(final_observations.explicit) > 0
    assert len(final_observations.deductive) > 0

    # Verify the trace captured the reasoning process
    assert "reasoning_iterations" in trace
    assert len(trace["reasoning_iterations"]) > 0

    # Print the results for inspection
    print("\nFinal Observations:")
    print(f"Explicit: {final_observations.explicit}")
    print(f"Deductive: {[obs.conclusion for obs in final_observations.deductive]}")
    print(f"Inductive: {[obs.conclusion for obs in final_observations.inductive]}")
    print(f"Abductive: {[obs.conclusion for obs in final_observations.abductive]}")

    validation_report = await verify_reasoning(
        context=initial_context,
        history=history,
        new_turn=new_turn,
        final_observations=final_observations,
    )

    print("\nValidation Report:")
    print(validation_report.model_dump_json(indent=2))

    assert validation_report.is_explicit_fine
    assert validation_report.is_deductive_fine
    assert validation_report.is_inductive_fine
    assert validation_report.is_abductive_fine

    print("\nTrace Summary:")
    print(f"Total iterations: {trace['summary']['total_iterations']}")
    print(f"Total duration: {trace['summary']['total_duration_ms']}ms")


@pytest.mark.asyncio
async def test_validation_report_catches_invalid_explicit(
    surprise_reasoner, db_session
):
    """Test that the validation report correctly identifies invalid explicit observations."""

    # Initial context with some basic observations
    initial_context = ReasoningResponse(
        thinking="Initial analysis",
        explicit=["User said: I am a software engineer"],
        deductive=[
            StructuredObservation(
                conclusion="User works in technology",
                premises=["User is a software engineer"],
            )
        ],
        inductive=[],
        abductive=[],
    )

    # Sample conversation history
    history = """
    User: I'm working on a new project
    Assistant: What kind of project?
    User: It's a web application using Python
    """

    # New turn to analyze
    new_turn = "I'm using FastAPI for the backend and React for the frontend"

    # Create fake final observations with an invalid explicit observation
    # This observation is NOT literally stated in the conversation
    fake_final_observations = ReasoningResponse(
        thinking="Analysis complete",
        explicit=[
            "User said: I am a software engineer",
            "User is using FastAPI for the backend",
            "User is using React for the frontend",
            "User is an expert developer",  # This is NOT explicitly stated - it's an inference
        ],
        deductive=[
            StructuredObservation(
                conclusion="User works in technology",
                premises=["User is a software engineer"],
            )
        ],
        inductive=[],
        abductive=[],
    )

    # Run validation on the fake observations
    validation_report = await verify_reasoning(
        context=initial_context,
        history=history,
        new_turn=new_turn,
        final_observations=fake_final_observations,
    )

    print("\nValidation Report for Invalid Explicit:")
    print(validation_report.model_dump_json(indent=2))

    # The validation should catch that "User is an expert developer" is not explicitly stated
    assert (
        not validation_report.is_explicit_fine
    ), "Validation should have caught the invalid explicit observation"


class ValidationReport(BaseModel):
    thinking: str
    is_explicit_fine: bool = Field(
        description="Do the explicit observations match the requirements provided?"
    )
    is_deductive_fine: bool = Field(
        description="Do the deductive observations follow from the explicit observations?"
    )
    is_inductive_fine: bool = Field(
        description="Do the inductive observations follow from the deductive observations?"
    )
    is_abductive_fine: bool = Field(
        description="Do the abductive observations follow from the inductive observations?"
    )


@llm.call(
    provider="google",
    model="gemini-2.0-flash-lite",
    call_params={"temperature": 0},
    response_model=ValidationReport,
)
async def verify_reasoning(
    context: ReasoningResponse,
    history: str,
    new_turn: str,
    final_observations: ReasoningResponse,
):
    return c(
        f"""
        You are an expert reasoning validator. Your task is to verify that the final observations are logically sound and properly derived from the given context and conversation.

        **INITIAL CONTEXT:**
        Explicit: {[obs for obs in context.explicit]}
        Deductive: {[f"{obs.conclusion} (from: {obs.premises})" for obs in context.deductive]}
        Inductive: {[f"{obs.conclusion} (from: {obs.premises})" for obs in context.inductive]}
        Abductive: {[f"{obs.conclusion} (from: {obs.premises})" for obs in context.abductive]}

        **CONVERSATION HISTORY:**
        {history}

        **NEW TURN:**
        {new_turn}

        **FINAL OBSERVATIONS TO VALIDATE:**
        Explicit: {[obs for obs in final_observations.explicit]}
        Deductive: {[f"{obs.conclusion} (from: {obs.premises})" for obs in final_observations.deductive]}
        Inductive: {[f"{obs.conclusion} (from: {obs.premises})" for obs in final_observations.inductive]}
        Abductive: {[f"{obs.conclusion} (from: {obs.premises})" for obs in final_observations.abductive]}

        **VALIDATION CRITERIA:**

        1. **EXPLICIT OBSERVATIONS**: Are these literally stated in the conversation? No interpretation allowed.

        2. **DEDUCTIVE OBSERVATIONS**: Do these conclusions MUST follow from the premises? Check logical necessity.

        3. **INDUCTIVE OBSERVATIONS**: Are these reasonable generalizations based on patterns? Check if there's sufficient evidence.

        4. **ABDUCTIVE OBSERVATIONS**: Do these provide plausible explanations for the overall pattern? Check coherence.

        **VALIDATION QUESTIONS:**
        - Are any observations unsupported by the evidence?
        - Are the premises correctly identified for each structured observation?
        - Do the reasoning levels follow proper hierarchy (explicit → deductive → inductive → abductive)?
        - Are there any logical fallacies or unsupported leaps?

        **VALIDATION REPORT:**
        Please provide a detailed validation report identifying any issues with the reasoning process and final observations.
        """
    )
