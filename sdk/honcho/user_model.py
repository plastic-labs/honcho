from langchain.chat_models.base import BaseChatModel

from abc import ABC, abstractmethod


class UserModel(ABC):
    """
    Abstract class representing a user model. This class should be inherited by
    any specific user model implementations that store and manage user context
    and goals within a conversation.
    """

    @abstractmethod
    def __init__(
        self,
        llm: BaseChatModel,
        user_id: str,
    ):
        """
        Initializes the UserModel

        Args:
            llm (LLMChain): The language chain to use for inference.
            user_id (str): The ID of the user for this model.
        """

    @abstractmethod
    async def revise(self, insight: str) -> None:
        """
        Abstract method to revise the user's context based on new insights.

        Args:
            insight (str): Insight about the user that may affect their context.
        """
        raise NotImplementedError("Must implement revise_context method.")

    @abstractmethod
    async def query(self, query: str) -> str:
        """
        Abstract method to respond to queries about the user's context.

        Args:
            query (str): Query about the user's context.

        Returns:
            str: Response to the query using the user's context.
        """
        raise NotImplementedError("Must implement respond_to_query method.")
