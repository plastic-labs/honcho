import os
import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Dict, Tuple, TypedDict, Union

import sentry_sdk
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe

from src.utils.model_client import ModelClient, ModelProvider

# Load environment variables
load_dotenv()

# Define default provider and model
DEF_PROVIDER = ModelProvider.ANTHROPIC
DEF_ANTHROPIC_MODEL = "claude-3-7-sonnet-20250219"

# Define message types
class BasicMessage(TypedDict):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class CachedMessage(BasicMessage):
    cache_control: Dict[str, str]

# Message type that can be either a basic message or a cached message
MessageType = Union[BasicMessage, CachedMessage]

# Cache storage
_RESPONSE_CACHE: Dict[str, str] = {}

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

def _generate_cache_key(provider: ModelProvider, model: str, system: Optional[str], messages: List[MessageType], temperature: float) -> str:
    """
    Generate a unique cache key based on the request parameters.
    
    Args:
        provider: The model provider
        model: The model name
        system: The system prompt (if any)
        messages: The messages to send
        temperature: The temperature setting
        
    Returns:
        A hash string to use as cache key
    """
    # Create a dictionary with all the parameters that affect the response
    cache_dict = {
        "provider": provider.value,
        "model": model,
        "system": system,
        "messages": messages,
        "temperature": temperature
    }
    
    # Convert to a stable JSON string and hash it
    cache_json = json.dumps(cache_dict, sort_keys=True)
    return hashlib.md5(cache_json.encode('utf-8')).hexdigest()

@observe()
async def get_response(
    messages: List[MessageType],
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
        use_caching: Whether to use caching for the response
        
    Returns:
        The model's response as a string
    """
    print(f"[LLM] get_response called with provider={provider}, model={model}, use_caching={use_caching}")
    
    # If caching is enabled, check if we have a cached response
    if use_caching:
        cache_key = _generate_cache_key(provider, model, system, messages, temperature)
        cache_key_short = cache_key[:8]  # First 8 chars for logging
        
        if cache_key in _RESPONSE_CACHE:
            print(f"[LLM] Cache HIT for key {cache_key_short}! Returning cached response")
            return _RESPONSE_CACHE[cache_key]
        else:
            print(f"[LLM] Cache MISS for key {cache_key_short}, will call API")
    
    # Create a new model client with the specified provider and model
    client = ModelClient(provider=provider, model=model)
    
    # Use langfuse to track the observation
    langfuse_context.update_current_observation(
        input=messages, model=model
    )
    
    # Prepare the extra headers for prompt caching if using Anthropic
    extra_params = {}
    cached_messages = messages.copy()  # Make a copy to avoid modifying the original
    
    if use_caching and provider == ModelProvider.ANTHROPIC:
        # Add prompt caching header for Anthropic
        extra_params["extra_headers"] = {"anthropic-beta": "prompt-caching-2024-07-31"}
        print(f"[LLM] Using Anthropic prompt caching with anthropic-beta header")
        
        # Add cache_control to the last message if it's from the user
        if cached_messages and cached_messages[-1]["role"] == "user":
            # Create a modified message with cache_control
            last_msg = cached_messages[-1]
            
            # Only add cache_control if not already present
            if "cache_control" not in last_msg:
                print(f"[LLM] Adding cache_control to user message")
                cached_msg: CachedMessage = {
                    "role": last_msg["role"],
                    "content": last_msg["content"],
                    "cache_control": {"type": "ephemeral"}
                }
                cached_messages = cached_messages[:-1] + [cached_msg]
            else:
                print(f"[LLM] Message already has cache_control, using as is")
    
    # Use the client's asynchronous method to get a response
    with sentry_sdk.start_transaction(op="llm-api", name=f"{provider} API Call") as transaction:
        print(f"[LLM] Starting API call to {provider} with {len(cached_messages)} messages")
        api_start = os.times()[4]
        
        # Use await directly instead of asyncio.run()
        try:
            result = await client.generate(
                messages=cached_messages,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra_params
            )
            api_time = os.times()[4] - api_start
            print(f"[LLM] API call completed in {api_time:.2f}s, response length: {len(result)}")
            
            # Cache the result if caching is enabled
            if use_caching:
                cache_key = _generate_cache_key(provider, model, system, messages, temperature)
                cache_key_short = cache_key[:8]
                _RESPONSE_CACHE[cache_key] = result
                print(f"[LLM] Response cached with key {cache_key_short}")
                
                # Log cache stats
                cache_size = len(_RESPONSE_CACHE)
                print(f"[LLM] Cache now contains {cache_size} entries")
                
            return result
        except Exception as e:
            print(f"[LLM] Error during API call: {str(e)}")
            raise
