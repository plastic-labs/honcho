"""Test utilities for reasoning dream tests.

This module provides helper functions and utilities for testing the reasoning
dream workflow, including fixture creation, mock setup, and assertion helpers.
"""

from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas


async def create_test_workspace(db: AsyncSession, workspace_name: str | None = None) -> str:
    """Create a test workspace.

    Args:
        db: Database session
        workspace_name: Optional workspace name (generated if not provided)

    Returns:
        Workspace name
    """
    if workspace_name is None:
        workspace_name = f"test_workspace_{uuid4().hex[:8]}"

    workspace_create = schemas.WorkspaceCreate(name=workspace_name)
    result = await crud.workspace.get_or_create_workspace(db, workspace=workspace_create)
    return result.resource.name


async def create_test_peer(
    db: AsyncSession, workspace_name: str, peer_name: str | None = None
) -> str:
    """Create a test peer.

    Args:
        db: Database session
        workspace_name: Workspace to create peer in
        peer_name: Optional peer name (generated if not provided)

    Returns:
        Peer name
    """
    if peer_name is None:
        peer_name = f"test_peer_{uuid4().hex[:8]}"

    peer_create = schemas.PeerCreate(name=peer_name)
    result = await crud.peer.get_or_create_peers(
        db, workspace_name=workspace_name, peers=[peer_create]
    )
    return result.resource[0].name


async def create_test_observations(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    count: int = 5,
    content_prefix: str = "User prefers",
) -> list[models.Document]:
    """Create test observations (documents).

    Args:
        db: Database session
        workspace_name: Workspace name
        observer: Observer peer name
        observed: Observed peer name
        count: Number of observations to create
        content_prefix: Prefix for observation content

    Returns:
        List of created Document models
    """
    # Create a test session for the observations
    session_name = f"test_session_{uuid4().hex[:8]}"
    session_create = schemas.SessionCreate(name=session_name)
    session_result = await crud.session.get_or_create_session(
        db, workspace_name=workspace_name, session=session_create
    )
    session_id = session_result.resource.name

    # Build observation schemas
    observations_data = []
    for i in range(count):
        content = f"{content_prefix} {i+1}"
        observations_data.append(
            schemas.ConclusionCreate(
                content=content,
                observer_id=observer,
                observed_id=observed,
                session_id=session_id,
            )
        )

    observations = await crud.document.create_observations(
        db,
        observations=observations_data,
        workspace_name=workspace_name,
    )

    return observations


async def create_test_hypothesis(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    content: str | None = None,
    confidence_score: float = 0.7,
    tier: int = 1,
    status: str = "active",
    source_premise_ids: list[str] | None = None,
) -> models.Hypothesis:
    """Create a test hypothesis.

    Args:
        db: Database session
        workspace_name: Workspace name
        observer: Observer peer name
        observed: Observed peer name
        content: Hypothesis content (generated if not provided)
        confidence_score: Confidence score (0.0-1.0)
        tier: Tier level (1-3)
        status: Status (active, superseded, falsified)
        source_premise_ids: List of source observation IDs

    Returns:
        Created Hypothesis model
    """
    if content is None:
        content = f"Test hypothesis: {observed} has consistent preferences"

    if source_premise_ids is None:
        source_premise_ids = []

    hypothesis_data = schemas.HypothesisCreate(
        content=content,
        observer=observer,
        observed=observed,
        confidence=confidence_score,
        tier=tier,
        status=status,
        source_premise_ids=source_premise_ids,
        reasoning_metadata={},
    )

    hypothesis = await crud.hypothesis.create_hypothesis(
        db,
        hypothesis=hypothesis_data,
        workspace_name=workspace_name,
    )
    return hypothesis


async def create_test_prediction(
    db: AsyncSession,
    workspace_name: str,
    hypothesis_id: str,
    content: str | None = None,
    status: str = "untested",
    is_blind: bool = True,
) -> models.Prediction:
    """Create a test prediction.

    Args:
        db: Database session
        workspace_name: Workspace name
        hypothesis_id: Parent hypothesis ID
        content: Prediction content (generated if not provided)
        status: Status (untested, unfalsified, falsified, superseded)
        is_blind: Whether prediction was made blindly

    Returns:
        Created Prediction model
    """
    if content is None:
        content = f"Test prediction based on hypothesis {hypothesis_id[:8]}"

    prediction_data = schemas.PredictionCreate(
        hypothesis_id=hypothesis_id,
        content=content,
        status=status,
        is_blind=is_blind,
    )

    prediction = await crud.prediction.create_prediction(
        db,
        prediction_data,
        workspace_name,
    )
    return prediction


async def create_test_induction(
    db: AsyncSession,
    workspace_name: str,
    observer: str,
    observed: str,
    content: str | None = None,
    pattern_type: str = "preferential",
    confidence: str = "medium",
    stability_score: float = 0.75,
    source_prediction_ids: list[str] | None = None,
    source_premise_ids: list[str] | None = None,
) -> models.Induction:
    """Create a test induction.

    Args:
        db: Database session
        workspace_name: Workspace name
        observer: Observer peer name
        observed: Observed peer name
        content: Induction content (generated if not provided)
        pattern_type: Pattern type
        confidence: Confidence level (low, medium, high)
        stability_score: Stability score (0.0-1.0)
        source_prediction_ids: Source prediction IDs
        source_premise_ids: Source premise IDs

    Returns:
        Created Induction model
    """
    if content is None:
        content = f"Test pattern: {observed} consistently exhibits behavior"

    if source_prediction_ids is None:
        source_prediction_ids = []

    if source_premise_ids is None:
        source_premise_ids = []

    induction_data = schemas.InductionCreate(
        observer=observer,
        observed=observed,
        content=content,
        pattern_type=pattern_type,
        confidence=confidence,
        stability_score=stability_score,
        source_prediction_ids=source_prediction_ids,
        source_premise_ids=source_premise_ids,
    )

    induction = await crud.induction.create_induction(
        db,
        induction_data,
        workspace_name,
    )
    return induction


def assert_hypothesis_valid(hypothesis: models.Hypothesis, expected_observer: str, expected_observed: str):
    """Assert hypothesis has expected properties.

    Args:
        hypothesis: Hypothesis to validate
        expected_observer: Expected observer peer
        expected_observed: Expected observed peer
    """
    assert hypothesis is not None
    assert hypothesis.observer == expected_observer
    assert hypothesis.observed == expected_observed
    assert hypothesis.confidence >= 0.0
    assert hypothesis.confidence <= 1.0
    assert hypothesis.tier >= 0  # tier can be 0, 1, 2, 3...
    assert hypothesis.status in ["active", "superseded", "falsified"]
    assert isinstance(hypothesis.source_premise_ids, (list, type(None)))
    assert isinstance(hypothesis.reasoning_metadata, dict)


def assert_prediction_valid(prediction: models.Prediction, expected_hypothesis_id: str):
    """Assert prediction has expected properties.

    Args:
        prediction: Prediction to validate
        expected_hypothesis_id: Expected parent hypothesis ID
    """
    assert prediction is not None
    assert prediction.hypothesis_id == expected_hypothesis_id
    assert prediction.status in ["untested", "unfalsified", "falsified", "superseded"]
    assert isinstance(prediction.is_blind, bool)
    assert prediction.content is not None
    assert len(prediction.content) > 0


def assert_trace_valid(trace: models.FalsificationTrace, expected_prediction_id: str):
    """Assert falsification trace has expected properties.

    Args:
        trace: Trace to validate
        expected_prediction_id: Expected prediction ID
    """
    assert trace is not None
    assert trace.prediction_id == expected_prediction_id
    assert isinstance(trace.search_queries, list)
    assert isinstance(trace.contradicting_premise_ids, list)
    assert trace.reasoning_chain is not None
    assert trace.final_status in ["unfalsified", "falsified", "untested"]
    assert trace.search_count >= 0
    if trace.search_efficiency_score is not None:
        assert trace.search_efficiency_score >= 0.0
        assert trace.search_efficiency_score <= 1.0


def assert_induction_valid(induction: models.Induction, expected_observer: str, expected_observed: str):
    """Assert induction has expected properties.

    Args:
        induction: Induction to validate
        expected_observer: Expected observer peer
        expected_observed: Expected observed peer
    """
    assert induction is not None
    assert induction.observer == expected_observer
    assert induction.observed == expected_observed
    assert induction.pattern_type in [
        "preferential",
        "behavioral",
        "personality",
        "tendency",
        "temporal",
        "conditional",
        "structural",
        "correlational",
    ]
    assert induction.confidence in ["low", "medium", "high"]
    if induction.stability_score is not None:
        assert induction.stability_score >= 0.0
        assert induction.stability_score <= 1.0
    assert isinstance(induction.source_prediction_ids, list)
    assert isinstance(induction.source_premise_ids, list)
