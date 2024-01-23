from langchain.prompts import SystemMessagePromptTemplate
from langchain.chat_models.base import BaseChatModel


class UserRewardModel:
    reward_revision_prompt = SystemMessagePromptTemplate.from_template(
        """You are trying to optimize the behaviour of an AI agent that is interacting with a user.
You are trying to do so by maintaining a paragraph of context describing the user's goals in their interaction with the agent, in order to provide the agent the context it needs to most effectively help the user.

Here is your current user context description:
"{user_context}"

Here is an insight about the user:
"{insight}"

Respond with a revised paragraph describing the user's goals and context based on this insight. Try not to remove content when possible."""
    )
    reward_query_prompt = SystemMessagePromptTemplate.from_template(
        """You are trying to optimize the behaviour of an AI agent that is interacting with a user.
You are trying to do so by maintaining a paragraph of context describing the user's goals in their interaction with the agent, in order to provide the agent the context it needs to most effectively help the user.

Here is your current user context description:
"{user_context}"

You have been given this query about the user:
"{query}"

(If the current user context paragraph doesn't contain the information you need, stating that it might be helpful to ask the user a particular question to gain that information would be acceptable!)

Now write a concise response to the query using the information in the user context description.
"""
    )

    def __init__(self, llm: BaseChatModel, user_context=None):
        self.llm = llm
        self.user_context = "none, new user" if user_context is None else user_context

    async def revise(self, insight: str):
        revised_context_message = await self.llm.ainvoke(
            [
                self.reward_revision_prompt.format(
                    user_context=self.user_context, insight=insight
                )
            ]
        )
        self.user_context = revised_context_message.content

    async def query(self, query: str):
        reward_query_message = await self.llm.ainvoke(
            [
                self.reward_query_prompt.format(
                    user_context=self.user_context, query=query
                )
            ]
        )
        return reward_query_message.content
