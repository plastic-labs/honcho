import logging
import json
from typing import Any
import datetime

import sentry_sdk
from sentry_sdk.ai.monitoring import ai_track
from langfuse.decorators import langfuse_context, observe
from rich.console import Console
from ..utils import history

from src.utils.model_client import ModelClient, ModelProvider


logger = logging.getLogger(__name__)
logging.getLogger("sqlalchemy.engine.Engine").disabled = True

console = Console(markup=False)

PROVIDER = ModelProvider.ANTHROPIC
MODEL = "claude-3-7-sonnet-20250219"

SURPRISE_SYSTEM_PROMPT = """
You are an expert at reasoning about a user based on abductive, inductive, and deductive context along with conversation history. Given the new turn of conversation, you must assess whether or not you're surprised and output a boolean value along with the level(s) upon which you're surprised. If the new turn of conversation introduces new explicit or deductive facts, you'd output surprise: true, levels: ["deductive"]. If the new turn of conversation impacts inductive understanding as well, output surprise: true, levels: ["deductive", "inductive"]. If the new turn of conversation changes our high level understanding of the user, output surprise: true, levels: ["abductive"]. Your job is to let us know if your expectation has been violated on any ofthose 3 reasoning levels.
"""

SURPRISE_USER_PROMPT = """
Here's the existing user context:
<context>
{context}
</context>

Here's the conversation history:
<history>
{history}
</history>

Here's the new turn of conversation:
<new_turn>
{new_turn}
</new_turn>

Format your response as follows:
<response>
{{
    "surprise": true,
    "levels": ["deductive", "inductive", "abductive"]
}}
</response>
Output your response only, no other text.
"""

REASONING_SYSTEM_PROMPT = """
You are an expert at reasoning about a user based on abductive, inductive, and deductive context along with conversation history. Another instance of yourself determined the levels upon which it was surprised based on the user context, conversation history, and new turn of conversation. Your job is to output update the user context for each of the reasoning levels that have been denoted as surprising.
"""

REASONING_USER_PROMPT = """
Here's the existing user context:
<context>
{context}
</context>

Here's the conversation history:
<history>
{history}
</history>

Here's the new turn of conversation:
<new_turn>
{{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}} user: {new_turn}
</new_turn>

The other instance of you felt surprised on the following reasoning levels:
<levels>
{levels}
</levels>

Format your response with one dictionary per level:
<response>
{{
    "level": "reasoning level",
    "facts": ["fact1", "fact2", ...]
}},
</response>
Feel free to think inside <think></think> tags before your response.
"""


class SurpriseReasoner:
    def __init__(self, embedding_store=None):
        self.max_recursion_depth = 5  # Prevent infinite recursion
        self.current_depth = 0
        self.embedding_store = embedding_store
    

    @observe()
    @ai_track("Surprise Check")
    async def check_for_surprise(self, context, history, new_turn) -> dict[str, bool | list[str]]:
        """
        Function that checks if there are any surprises in the reasoning result.
        Returns a tuple of (surprise_info).
        """

        system_prompt = SURPRISE_SYSTEM_PROMPT
        user_prompt = SURPRISE_USER_PROMPT.format(context=context, history=history, new_turn=new_turn)

        client = ModelClient(provider=PROVIDER, model=MODEL)

        # prepare the messages
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]

        langfuse_context.update_current_observation(input=messages, model=MODEL)

        try:
            # generate the response
            response = await client.generate(
                messages=messages,
                system=system_prompt,
                max_tokens=1000,
                temperature=1,
                use_caching=True,
            )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"Error generating surprise check response: {e}")
            raise e

        # Parse the response to extract surprise information
        try:
            # Find the JSON response between <response> tags
            response_start = response.find("<response>") + len("<response>")
            response_end = response.find("</response>")
            json_str = response[response_start:response_end].strip()
            
            # Parse the JSON response
            surprise_info = json.loads(json_str)

            return surprise_info
            
        except (ValueError, json.JSONDecodeError) as e:
            console.print(f"Failed to parse surprise check response: {e}")
            return {"surprise": False, "levels": []}

    @observe()
    @ai_track("Reason About Surprises")
    async def reason_about_surprises(self, context, history, new_turn, surprise_info, message_id: str) -> dict[str, list[str]]:
        """
        Function that reasons about the surprise values found.
        Returns a new reasoning that incorporates the surprises.
        """

        if not surprise_info["surprise"]:
            console.print("No surprises found")
            return {}

        levels = surprise_info["levels"]

        system_prompt = REASONING_SYSTEM_PROMPT
        user_prompt = REASONING_USER_PROMPT.format(context=context, history=history, new_turn=new_turn, levels=levels)

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_prompt},
        ]
        
        client = ModelClient(provider=PROVIDER, model=MODEL)

        langfuse_context.update_current_observation(input=messages, model=MODEL)

        console.print(f"REASONING ABOUT SURPRISES...")
        
        try:
            response = await client.generate(
                messages=messages,
                system=system_prompt,
                max_tokens=1000,
                temperature=1,
                use_caching=True,
            )
            console.print(response)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            console.print(f"Error generating reasoning about surprises: {e}")
            raise e
        
        # Parse the response to extract the reasoning
        try:
            # Find the JSON response between <response> tags
            response_start = response.find("<response>") + len("<response>")
            response_end = response.find("</response>")
            json_str = response[response_start:response_end].strip()
            
            # Wrap the sequence of objects in an array if it's not already
            if not json_str.startswith('['):
                json_str = '[' + json_str + ']'

            console.print(f"JSON STRING: {json_str}")
            
            reasoning_info = json.loads(json_str)
            console.print(f"PARSED OUT REASONING INFO: {reasoning_info}")

            # If we have an embedding store, save the new facts
            if self.embedding_store and reasoning_info:
                console.print("ENTERING FOR LOOP TO WRITE FACTS TO COLLECTION")
                for reasoning in reasoning_info:
                    level = reasoning["level"]
                    facts = reasoning["facts"]
                    # Save each fact with its metadata
                    console.print(f"FACTS TO BE WRITTEN: {facts}")
                    await self.embedding_store.save_facts(
                        facts,
                        message_id=message_id,
                        level=level
                    )
            return reasoning_info
        
        except Exception as e:
            console.print(f"Failed to parse reasoning about surprises: {e}")
            return {}
        
    
    async def recursive_reason(self, context: dict, history: str, new_turn: str, message_id: str) -> dict:
        """
        Main recursive reasoning function that checks for surprises
        and calls itself again if surprises are found.
        """
        # Check if we've hit the maximum recursion depth
        if self.current_depth >= self.max_recursion_depth:
            logger.warning("Maximum recursion depth reached")
            return context

        # Increment the recursion depth counter
        self.current_depth += 1

        try:
            # Check for surprises in the current context
            surprise_info = await self.check_for_surprise(context, history, new_turn)

            console.print(f"RESULT FROM CHECKING SURPRISE: {surprise_info}")
            
            # If no surprises found, return the current context
            if not surprise_info["surprise"]:
                return context

            # Reason about the surprises and get updated context
            reasoning_info: dict[str, list[str]] = await self.reason_about_surprises(
                context, history, new_turn, surprise_info, message_id
            )
            
            console.print(f"RESULT FROM REASONING ABOUT SURPRISE: {reasoning_info}")

            # Update the context with the new reasoning
            updated_context = {**context}
            console.print(f"COPIED LIST OF EXISTING FACTS: {updated_context}")
            # Handle both single dict and list of dicts
            reasoning_list = reasoning_info if isinstance(reasoning_info, list) else [reasoning_info]
            for reasoning in reasoning_list:
                console.print("ENTERED FOR LOOP OVER REASONING INFO")
                assert isinstance(reasoning, dict)
                level = reasoning["level"]
                facts = reasoning["facts"]
                console.print(f"LEVEL: {level}\n FACTS: {facts}")
                # if level not in updated_context:
                #     updated_context[level] = []
                updated_context[level].extend(facts)
                console.print(f"NEW FACTS APPENDED TO CONTEXT: {updated_context}")


            # Recursively check for more surprises with the updated context
            return await self.recursive_reason(updated_context, history, new_turn, message_id)

        finally:
            # Decrement the recursion depth counter when we're done
            self.current_depth -= 1

