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

@observe()
async def get_response(
    messages: List[Dict[str, str]],
    provider: ModelProvider = DEF_PROVIDER,
    model: str = DEF_ANTHROPIC_MODEL,
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1000
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
        
    Returns:
        The model's response as a string
    """
    print(f"[LLM] get_response called with provider={provider}, model={model}")
    
    # Create a new model client with the specified provider and model
    client = ModelClient(provider=provider, model=model)
    
    # Use langfuse to track the observation
    langfuse_context.update_current_observation(
        input=messages, model=model
    )
    
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
