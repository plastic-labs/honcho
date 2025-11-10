"""Base abstract class for reasoners."""

from abc import ABC, abstractmethod
from typing import Any


class BaseReasoner(ABC):
    """Abstract base class for reasoner implementations.

    Reasoners are responsible for processing messages and deriving insights,
    representations, or other cognitive outputs for the agent system.
    """

    @abstractmethod
    async def reason(self, *args: Any, **kwargs: Any) -> Any:
        """Process input and derive reasoning outputs.

        Args:
            *args: Positional arguments for reasoning
            **kwargs: Keyword arguments for reasoning

        Returns:
            The result of the reasoning process

        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        pass
