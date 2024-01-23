from typing import Union

from metacognition_sdk.interfaces import LlmAdapter
from metacognition_sdk.messages import ConversationHistory

from openai import AsyncOpenAI


class ChatOpenAILlmAdapter(LlmAdapter):
    """
    Adapter class to interface with OpenAI's Chat language models.

    This adapter allows for sending prompts to the OpenAI API and receiving the generated responses.
    It implements the LlmAdapter interface defined in the metacognition SDK.
    """

    def __init__(self, openai_client: AsyncOpenAI, model="gpt-4"):
        """
        Initializes a new instance of the ChatOpenAILlmAdapter.

        Args:
            openai_client (AsyncOpenAI): An instance of the AsyncOpenAI client.
            model (str): The model identifier to use for completions. Defaults to 'gpt-4'.
        """
        self.openai_client = openai_client
        self.model = model

    async def inference(self, input: str) -> str:
        """
        Performs an inference using the specified OpenAI Chat model.

        Sends a prompt to the OpenAI API and returns the generated response.

        Args:
            input (str): The input string to send as a prompt to the language model.

        Returns:
            str: The content of the message from the language model's response.
        """
        chat_completion = await self.openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": input,
                }
            ],
            model=self.model,
        )

        # Extract the content of the message from the first choice in the response
        return chat_completion.choices[0].message.content

    async def chat_inference(
        self, input: ConversationHistory, system_prompt: Union[str, None] = None
    ) -> str:
        """
        Perform chat-based inference using the OpenAI Chat model.

        This method processes the conversation history, sends it to the OpenAI API,
        and returns the language model's response as a string.

        Args:
            input (ConversationHistory): An object containing the history of the conversation.
            system_prompt: Prompt to include in the system message. A system message is only added if this argument is specified.

        Returns:
            str: The language model's response as a string.
        """
        # Convert the conversation history into the format expected by the OpenAI API
        messages = [
            {"role": message["role"].lower(), "content": message["content"]}
            for message in input.messages
        ]

        # Add system message if a system prompt was specified
        messages = (
            [{"role": "system", "content": system_prompt}] + messages
            if system_prompt
            else messages
        )

        # Send the conversation history to the OpenAI API and wait for the response
        chat_completion = await self.openai_client.chat.completions.create(
            messages=messages,
            model=self.model,
        )

        # Extract the content of the message from the first choice in the response
        return chat_completion.choices[0].message.content
