"""
Base Agent class for Honcho agents.

This module defines the abstract base class that all Honcho agents should inherit from,
providing a consistent interface and common functionality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all Honcho agents.

    All agents should inherit from this class and implement the required abstract methods.
    This ensures a consistent interface across all agents and provides common functionality
    for logging, error handling, and provenance tracking.

    Attributes:
        db: Database session for agent operations
        config: Agent-specific configuration
        agent_type: String identifier for the agent type (e.g., "abducer", "predictor")
    """

    def __init__(self, db: AsyncSession, config: Any = None, **kwargs):
        """
        Initialize the base agent.

        Args:
            db: SQLAlchemy async database session
            config: Agent-specific configuration object
            **kwargs: Additional agent-specific parameters
        """
        self.db = db
        self.config = config
        self.agent_type = self.__class__.__name__.lower()

        # Store additional kwargs for agent-specific parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

        logger.debug(f"Initialized {self.agent_type} agent")

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.

        This is the primary method that performs the agent's work. All agents
        must implement this method with their specific logic.

        Args:
            input_data: Dictionary containing the input data for the agent.
                       The structure depends on the specific agent type.

        Returns:
            Dictionary containing the agent's output. The structure depends
            on the specific agent type.

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If agent execution fails
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate the input data before execution.

        This method should check that the input_data dictionary contains
        all required fields and that values are of the correct type.

        Args:
            input_data: Dictionary containing input data to validate

        Returns:
            True if input is valid, False otherwise

        Raises:
            ValueError: If input validation fails with a descriptive error message
        """
        pass

    async def trace_execution(
        self,
        input_data: Dict[str, Any],
        output: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """
        Record execution trace for provenance tracking.

        This method records the agent's execution for training data generation
        and debugging purposes. The default implementation logs basic information,
        but agents can override this to provide more detailed tracing.

        Args:
            input_data: The input provided to the agent
            output: The output produced by the agent
            metadata: Optional additional metadata about the execution
                     (e.g., execution time, model used, tokens consumed)

        Note:
            This is optional and has a default implementation that logs basic info.
            Agents can override this for more detailed provenance tracking.
        """
        logger.info(
            f"[{self.agent_type}] Execution trace",
            extra={
                "agent_type": self.agent_type,
                "input_keys": list(input_data.keys()),
                "output_keys": list(output.keys()),
                "metadata": metadata or {},
            },
        )

    async def pre_execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called before execute(). Can be used for setup, validation, etc.

        Args:
            input_data: The input data that will be passed to execute()

        Returns:
            Modified input_data (or original if no modifications needed)

        Note:
            This is optional and has a default implementation that validates input.
            Agents can override this for additional pre-processing.
        """
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input for {self.agent_type} agent")
        return input_data

    async def post_execute(
        self, input_data: Dict[str, Any], output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook called after execute(). Can be used for cleanup, logging, etc.

        Args:
            input_data: The input data that was passed to execute()
            output: The output produced by execute()

        Returns:
            Modified output (or original if no modifications needed)

        Note:
            This is optional and has a default implementation that traces execution.
            Agents can override this for additional post-processing.
        """
        await self.trace_execution(input_data, output)
        return output

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full agent execution pipeline with hooks.

        This method orchestrates the full execution flow:
        1. pre_execute (validation, setup)
        2. execute (main agent logic)
        3. post_execute (cleanup, tracing)

        Args:
            input_data: Dictionary containing input data

        Returns:
            Dictionary containing agent output

        Raises:
            ValueError: If input validation fails
            RuntimeError: If execution fails
        """
        try:
            # Pre-execution hook
            validated_input = await self.pre_execute(input_data)

            # Main execution
            output = await self.execute(validated_input)

            # Post-execution hook
            final_output = await self.post_execute(validated_input, output)

            return final_output

        except Exception as e:
            logger.error(
                f"[{self.agent_type}] Execution failed: {str(e)}",
                exc_info=True,
            )
            raise

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<{self.__class__.__name__} type={self.agent_type}>"
