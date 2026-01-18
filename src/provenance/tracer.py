"""Provenance tracer for tracking agent execution.

Provides a context manager for capturing detailed execution traces of
Top-Down reasoning agents (Abducer, Predictor, Falsifier, Inductor).
"""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """A single step in the agent's reasoning process."""

    step_number: int = Field(..., description="Sequential step number")
    action: str = Field(..., description="Action taken (e.g., 'search', 'evaluate', 'generate')")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for this step"
    )
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Output data from this step"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Step execution timestamp"
    )
    duration_ms: float | None = Field(
        default=None, description="Step execution duration in milliseconds"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional step metadata"
    )


class ProvenanceTrace(BaseModel):
    """Complete provenance trace for an agent execution."""

    trace_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique trace ID"
    )
    agent_type: str = Field(
        ...,
        description="Agent type (abducer | predictor | falsifier | inductor | extractor)",
    )
    workspace_name: str = Field(..., description="Workspace name")
    collection_id: str | None = Field(default=None, description="Collection ID")

    # Execution context
    observer: str | None = Field(default=None, description="Observer peer name")
    observed: str | None = Field(default=None, description="Observed peer name")

    # Input/output
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Agent input parameters"
    )
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="Agent output results"
    )

    # Execution tracking
    steps: list[AgentStep] = Field(
        default_factory=list, description="Reasoning steps"
    )
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Execution start time"
    )
    end_time: datetime | None = Field(default=None, description="Execution end time")
    duration_ms: float | None = Field(
        default=None, description="Total execution duration in milliseconds"
    )

    # Results
    success: bool = Field(default=True, description="Whether execution succeeded")
    error_message: str | None = Field(
        default=None, description="Error message if execution failed"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional execution metadata"
    )


class ProvenanceTracer:
    """
    Context manager for tracking agent execution with detailed provenance.

    Usage:
        async with ProvenanceTracer(
            agent_type="abducer",
            workspace_name="my_workspace",
            observer="user1",
            observed="user1",
            input_data={"premises": [...]},
        ) as tracer:
            # Agent execution
            tracer.add_step(
                action="search_premises",
                input_data={"query": "..."},
                output_data={"results": [...]}
            )
            tracer.set_output({"hypotheses": [...]})

        # After context exit, tracer.trace contains complete provenance
    """

    def __init__(
        self,
        agent_type: str,
        workspace_name: str,
        observer: str | None = None,
        observed: str | None = None,
        collection_id: str | None = None,
        input_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize provenance tracer.

        Args:
            agent_type: Agent type (abducer | predictor | falsifier | inductor | extractor)
            workspace_name: Workspace name
            observer: Observer peer name (optional)
            observed: Observed peer name (optional)
            collection_id: Collection ID (optional)
            input_data: Agent input parameters
            metadata: Additional metadata
        """
        self.trace: ProvenanceTrace = ProvenanceTrace(
            agent_type=agent_type,
            workspace_name=workspace_name,
            observer=observer,
            observed=observed,
            collection_id=collection_id,
            input_data=input_data or {},
            metadata=metadata or {},
        )
        self._step_counter: int = 0
        self._step_start_time: float | None = None
        self._start_perf_time: float = 0.0

    def add_step(
        self,
        action: str,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentStep:
        """
        Add a reasoning step to the trace.

        Args:
            action: Action name (e.g., 'search', 'evaluate', 'generate')
            input_data: Input data for this step
            output_data: Output data from this step
            metadata: Additional step metadata

        Returns:
            The created AgentStep
        """
        self._step_counter += 1

        # Calculate duration if step timing was tracked
        duration_ms = None
        if self._step_start_time is not None:
            duration_ms = (time.perf_counter() - self._step_start_time) * 1000
            self._step_start_time = None

        step = AgentStep(
            step_number=self._step_counter,
            action=action,
            input_data=input_data or {},
            output_data=output_data or {},
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self.trace.steps.append(step)
        return step

    def start_step(self) -> None:
        """Start timing for the next step."""
        self._step_start_time = time.perf_counter()

    def set_output(self, output_data: dict[str, Any]) -> None:
        """
        Set the final output data for the agent execution.

        Args:
            output_data: Agent output results
        """
        self.trace.output_data = output_data

    def set_error(self, error_message: str) -> None:
        """
        Mark the execution as failed with an error message.

        Args:
            error_message: Error message describing the failure
        """
        self.trace.success = False
        self.trace.error_message = error_message

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """
        Update the trace metadata.

        Args:
            metadata: Metadata to merge with existing metadata
        """
        self.trace.metadata.update(metadata)

    async def __aenter__(self) -> "ProvenanceTracer":
        """Enter the async context manager."""
        self.trace.start_time = datetime.now(timezone.utc)
        self._start_perf_time = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager and finalize the trace."""
        self.trace.end_time = datetime.now(timezone.utc)
        self.trace.duration_ms = (
            time.perf_counter() - self._start_perf_time
        ) * 1000

        # If there was an exception, record it
        if exc_type is not None:
            self.set_error(f"{exc_type.__name__}: {exc_val}")


@asynccontextmanager
async def trace_agent_execution(
    agent_type: str,
    workspace_name: str,
    observer: str | None = None,
    observed: str | None = None,
    collection_id: str | None = None,
    input_data: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[ProvenanceTracer, None]:
    """
    Async context manager factory for tracing agent execution.

    This is a convenience function that creates and manages a ProvenanceTracer.

    Args:
        agent_type: Agent type (abducer | predictor | falsifier | inductor | extractor)
        workspace_name: Workspace name
        observer: Observer peer name (optional)
        observed: Observed peer name (optional)
        collection_id: Collection ID (optional)
        input_data: Agent input parameters
        metadata: Additional metadata

    Yields:
        ProvenanceTracer instance

    Example:
        async with trace_agent_execution(
            agent_type="abducer",
            workspace_name="my_workspace",
            input_data={"premises": [...]},
        ) as tracer:
            # Agent execution
            tracer.add_step("search_premises", output_data={"count": 10})
            tracer.set_output({"hypotheses": [...]})
    """
    tracer = ProvenanceTracer(
        agent_type=agent_type,
        workspace_name=workspace_name,
        observer=observer,
        observed=observed,
        collection_id=collection_id,
        input_data=input_data,
        metadata=metadata,
    )

    async with tracer:
        yield tracer
