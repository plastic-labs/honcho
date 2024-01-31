from typing import Union

from ..interfaces import LlmAdapter
from ..messages import ConversationHistory

from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage


class LangchainAdapter(LlmAdapter):
    """
    Adapter class to interface with Langchain's language models.

    This adapter allows for sending prompts to the Langchain model and receiving the generated responses.
    It implements the LlmAdapter interface defined in the metacognition SDK.
    """

    def __init__(self, model: BaseChatModel):
        """
        Initializes a new instance of the LangchainAdapter.

        Args:
            model (BaseChatModel): An instance of a Langchain BaseChatModel.
        """
        self.model = model

    async def inference(self, input: str) -> str:
        """
        Performs an inference using the specified Langchain model.

        Sends a prompt to the Langchain model and returns the generated response.

        Args:
            input (str): The input string to send as a prompt to the language model.

        Returns:
            str: The content of the message from the language model's response.
        """
        next_message = await self.model.ainvoke([SystemMessage(content=input)])

        return next_message.content

    async def chat_inference(
        self, input: ConversationHistory, system_prompt: Union[str, None] = None
    ) -> str:
        """
        Perform chat-based inference using the Langchain model.

        This method processes the conversation history, sends it to the Langchain model,
        and returns the language model's response as a string.

        Args:
            input (ConversationHistory): An object containing the history of the conversation.
            system_prompt (Union[str, None]): Prompt to include in the system message. A system message is only added if this argument is specified.

        Returns:
            str: The language model's response as a string.
        """
        messages = [
            AIMessage(content=message["content"])
            if message["role"] == "AI"
            else HumanMessage(content=message["content"])
            for message in input.messages
        ]

        # Add system message if a system prompt was specified
        if system_prompt:
            messages.insert(0, SystemMessage(content=system_prompt))

        # Send the conversation history to the Langchain model and wait for the response
        next_message = await self.model.ainvoke(messages)

        return next_message.content
