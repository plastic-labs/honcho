from abc import ABC, abstractmethod
from typing import Union

from metacognition_sdk.messages import ConversationHistory


class LlmAdapter(ABC):
    """
    Abstract base class that defines the interface for language model adapters.
    """

    @abstractmethod
    async def inference(self, input: str) -> str:
        """
        Perform inference using the language model.

        Args:
            input (str): The input string for the language model.

        Returns:
            str: The inference result as a string.
        """
        pass

    @abstractmethod
    async def chat_inference(
        self,
        conversation_history: ConversationHistory,
        system_prompt: Union[str, None] = None,
    ) -> str:
        """
        Perform chat-based inference using the language model.

        This method should take the conversation history as input and return the language model's
        response as a string.

        Args:
            conversation_history (ConversationHistory): An object containing the history of the conversation.

        Returns:
            str: The language model's response as a string.
        """
        pass


class UserModelStorageAdapter(ABC):
    """
    Abstract base class that defines the interface for user model storage adapters that define how and where a user model is stored
    """

    default_user_model = "none, new user"

    @abstractmethod
    async def get_user_model(self) -> str:
        """
        Retrieve the user model from storage.

        Returns:
            str: The user model as a string.
        """
        pass

    @abstractmethod
    async def set_user_model(self, new_user_model: str) -> None:
        """
        Update the user model in storage.

        Args:
            new_user_model (str): The new user model as a string.
        """
        pass
