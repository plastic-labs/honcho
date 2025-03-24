import os
from typing import Any, Dict, List, Literal, Optional

import sentry_sdk
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe

from src.utils.model_client import ModelClient, ModelProvider

# Load environment variables
load_dotenv()

# Define default provider and model
DEF_PROVIDER = ModelProvider.ANTHROPIC
DEF_ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"

# Constants for prompt templates
QUERY_GENERATION_TEMPLATE = """Given this query about a user, generate 3 focused search queries that would help retrieve relevant facts about the user.
Each query should focus on a specific aspect related to the original query, rephrased to maximize semantic search effectiveness.
For example, if the original query asks "what does the user like to eat?", generated queries might include "user's food preferences", "user's favorite cuisine", etc.

ORIGINAL QUERY:
{query}

Format your response as a JSON array of strings, with each string being a search query. 
Respond only in valid JSON, without markdown formatting or quotes, and nothing else.
Example:
["query about interests", "query about personality", "query about experiences"]
"""

@observe()
async def get_response(
    messages: List[Dict[str, Any]],
    provider: ModelProvider = DEF_PROVIDER,
    model: str = DEF_ANTHROPIC_MODEL,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000,
    use_caching: bool = False
) -> str:
    """
    Get a response from the specified model provider using the ModelClient.
    
    Args:
        messages: The messages to send to the model
        provider: The model provider to use
        model: The model to use
        system: Optional system prompt
        temperature: The temperature to use for sampling
        max_tokens: The maximum number of tokens to generate
        use_caching: Whether to use provider-side caching for the response
        
    Returns:
        The model's response as a string
    """
    print(f"[LLM] get_response called with provider={provider}, model={model}, use_caching={use_caching}")
    
    # Create a new model client with the specified provider and model
    client = ModelClient(provider=provider, model=model)
    
    # Use langfuse to track the observation
    langfuse_context.update_current_observation(
        input=messages, model=model
    )
    
    # We don't need to modify the messages as the caching is handled 
    # by the model provider based on request similarity, not explicit cache_control
    
    # Use the client's asynchronous method to get a response
    with sentry_sdk.start_transaction(op="llm-api", name=f"{provider} API Call") as transaction:
        print(f"[LLM] Starting API call to {provider} with {len(messages)} messages")
        api_start = os.times()[4]
        
        # Use await directly instead of asyncio.run()
        try:
            result = await client.generate(
                messages=messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens
            )
            api_time = os.times()[4] - api_start
            print(f"[LLM] API call completed in {api_time:.2f}s, response length: {len(result)}")
            return result
        except Exception as e:
            print(f"[LLM] Error during API call: {str(e)}")
            raise
