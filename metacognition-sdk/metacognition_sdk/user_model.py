from metacognition_sdk.interfaces import UserModelStorageAdapter, LlmAdapter


class UserRewardModel:
    """
    A model for managing and updating a user's context and goals within a conversation.

    This model uses a language model to revise and query the user's context description
    based on insights or queries provided to the system. The goal is to maintain an
    accurate and helpful representation of the user's intentions and needs.
    """

    # Template for revising the user's context based on new insights
    reward_revision_prompt = """You are trying to optimize the behaviour of an AI agent that is interacting with a user.
You are trying to do so by maintaining a paragraph of context describing the user's goals in their interaction with the agent, in order to provide the agent the context it needs to most effectively help the user.

Here is your current user context description:
"{user_context}"

Here is an insight about the user:
"{insight}"

Respond with a revised paragraph describing the user's goals and context based on this insight. Try not to remove content when possible."""

    # Template for responding to queries about the user's context
    reward_query_prompt = """You are trying to optimize the behaviour of an AI agent that is interacting with a user.
You are trying to do so by maintaining a paragraph of context describing the user's goals in their interaction with the agent, in order to provide the agent the context it needs to most effectively help the user.

Here is your current user context description:
"{user_context}"

You have been given this query about the user:
"{query}"

(If the current user context paragraph doesn't contain the information you need, stating that it might be helpful to ask the user a particular question to gain that information would be acceptable!)

Now write a concise response to the query using the information in the user context description.
"""

    def __init__(
        self, llm: LlmAdapter, user_model_storage_adapter: UserModelStorageAdapter
    ):
        """
        Initializes the UserRewardModel with a language model adapter and a storage adapter.

        Args:
            llm (LlmAdapter): The language model adapter to use for inference.
            user_model_storage_adapter (UserModelStorageAdapter): The storage adapter for persisting the user model.
        """
        self.llm = llm
        self.user_model_storage_adapter = user_model_storage_adapter

    async def revise(self, insight: str):
        """
        Revises the user's context description based on a provided insight.

        Args:
            insight (str): An insight about the user that may affect their context description.

        Returns:
            None
        """
        revised_context_message = await self.llm.inference(
            self.reward_revision_prompt.format(
                user_context=await self.user_model_storage_adapter.get_user_model(),
                insight=insight,
            )
        )

        await self.user_model_storage_adapter.set_user_model(revised_context_message)

    async def query(self, query: str):
        """
        Responds to a query about the user's context using the current context description.

        Args:
            query (str): A query regarding the user's context or goals.

        Returns:
            str: The response to the query, based on the current user context description.
        """
        response = await self.llm.inference(
            self.reward_query_prompt.format(
                user_context=await self.user_model_storage_adapter.get_user_model(),
                query=query,
            )
        )
        return response
