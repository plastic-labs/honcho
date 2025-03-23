"""
Utility functions for interacting with various language model APIs.
"""
import os
import asyncio
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Protocol

import sentry_sdk
from anthropic import Anthropic
from anthropic.types import ContentBlock, MessageParam
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supported model providers
class ModelProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"
    # Add other providers as needed

# Default models for each provider
DEFAULT_MODELS = {
    ModelProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    ModelProvider.OPENAI: "gpt-4o",
    ModelProvider.OPENROUTER: "meta-llama/Llama-3.3-70B-Instruct",
}

class Message(Protocol):
    """Protocol for a message that works with any provider."""
    role: str
    content: str

class ModelClient:
    """A client for interacting with various language model APIs."""
    
    def __init__(
        self, 
        provider: ModelProvider = ModelProvider.ANTHROPIC,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the model client.
        
        Args:
            provider: The model provider to use
            model: The specific model to use, or None to use the default model for the provider
            api_key: The API key to use, or None to read from environment variables
            base_url: Custom base URL for the API endpoints (used for OpenRouter)
        """
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.base_url = base_url
        self.openai_client = None
        
        # Setup provider-specific clients
        if provider == ModelProvider.ANTHROPIC:
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("Anthropic API key is required")
            self.client = Anthropic(api_key=self.api_key)
        elif provider in [ModelProvider.OPENAI, ModelProvider.OPENROUTER, ModelProvider.CEREBRAS]:
            # Import OpenAI inside the method to avoid issues if the package is not installed
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("OpenAI package is not installed. Install it with 'pip install openai'.")
            
            if provider == ModelProvider.OPENAI:
                self.api_key = api_key or os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OpenAI API key is required")
                self.openai_client = AsyncOpenAI(api_key=self.api_key)
                
            elif provider == ModelProvider.OPENROUTER:
                self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
                if not self.api_key:
                    raise ValueError("OpenRouter API key is required")
                # Use the provided base_url or get from environment variables, or use the default
                self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
                self.openai_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            elif provider == ModelProvider.CEREBRAS:
                self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
                if not self.api_key:
                    raise ValueError("Cerebras API key is required")
                self.base_url = base_url or os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
                self.openai_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def create_message(self, role: str, content: str) -> Dict[str, str]:
        """
        Create a message that works with the current provider.
        
        Args:
            role: The role of the message (e.g., "user", "assistant")
            content: The message content
            
        Returns:
            A message compatible with the current provider
        """
        # For now, just return a dictionary that works with both Anthropic and OpenAI
        return {"role": role, "content": content}
    
    @observe(as_type="generation")
    async def generate(
        self, 
        messages: List[Dict[str, Any]], 
        system: Optional[str] = None, 
        max_tokens: int = 1000,
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a response using the configured model.
        
        Args:
            messages: The conversation history
            system: Optional system prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            extra_headers: Optional headers to add to the request
            
        Returns:
            The generated text
        """
        with sentry_sdk.start_transaction(op="llm-api", name=f"{self.provider} API Call"):
            # Log to langfuse
            langfuse_context.update_current_observation(
                input=messages, model=self.model
            )
            
            if self.provider == ModelProvider.ANTHROPIC:
                return await self._generate_anthropic(messages, system, max_tokens, temperature, extra_headers)
            elif self.provider in [ModelProvider.OPENAI, ModelProvider.OPENROUTER, ModelProvider.CEREBRAS]:
                return await self._generate_openai(messages, system, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _generate_anthropic(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate a response using the Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized.")
            
        # Convert messages to the format expected by Anthropic
        anthropic_messages = []
        for message in messages:
            # Handle cache_control if present
            msg_dict = {
                "role": message["role"],
                "content": message["content"]
            }
            
            # Add cache_control if present in the message
            if "cache_control" in message:
                msg_dict["cache_control"] = message["cache_control"]
                
            anthropic_messages.append(msg_dict)
            
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if system:
            params["system"] = system
            
        # Add headers if provided
        if extra_headers:
            params["headers"] = extra_headers
            
        # Use running loop for async execution
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.client.messages.create(**params)
        )
        
        # Extract the text from the response
        if response.content and len(response.content) > 0:
            content_block = response.content[0]
            # Check content_block by checking for attribute 'type' instead of using isinstance
            if hasattr(content_block, 'type') and content_block.type == "text":
                return content_block.text
            return str(content_block)
        return ""
    
    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> str:
        """Generate text using OpenAI or OpenRouter API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        # Prepare messages
        formatted_messages = []
        
        # Add system message if provided
        if system:
            formatted_messages.append({"role": "system", "content": system})
        
        # Add the rest of the messages
        formatted_messages.extend(messages)
        
        # Make the API call with the OpenAI client
        response = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract the generated text
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""
        return ""
    
    @observe(as_type="generation")
    async def stream(
        self, 
        messages: List[Dict[str, Any]], 
        system: Optional[str] = None, 
        max_tokens: int = 1000,
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Stream a response using the configured model.
        
        Args:
            messages: The conversation history
            system: Optional system prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
            extra_headers: Optional headers to add to the request
            
        Returns:
            A streaming response from the provider
        """
        with sentry_sdk.start_transaction(op="llm-api-stream", name=f"{self.provider} API Stream"):
            # Log to langfuse
            langfuse_context.update_current_observation(
                input=messages, model=self.model
            )
            
            if self.provider == ModelProvider.ANTHROPIC:
                return await self._stream_anthropic(messages, system, max_tokens, temperature, extra_headers)
            elif self.provider in [ModelProvider.OPENAI, ModelProvider.OPENROUTER]:
                return await self._stream_openai(messages, system, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _stream_anthropic(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """Stream text using Anthropic API."""
        if not self.client:
            raise ValueError("Anthropic client not initialized.")
            
        # Convert messages to the format expected by Anthropic
        anthropic_messages = []
        for message in messages:
            # Handle cache_control if present
            msg_dict = {
                "role": message["role"],
                "content": message["content"]
            }
            
            # Add cache_control if present in the message
            if "cache_control" in message:
                msg_dict["cache_control"] = message["cache_control"]
                
            anthropic_messages.append(msg_dict)
            
        params = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            params["system"] = system
        
        # Add headers if provided
        if extra_headers:
            params["headers"] = extra_headers
        
        # Use run_in_executor to run the synchronous Anthropic call in a thread
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.client.messages.stream(**params)
        )
    
    async def _stream_openai(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.0
    ) -> Any:
        """Stream text using OpenAI or OpenRouter API."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")
            
        # Prepare messages
        formatted_messages = []
        
        # Add system message if provided
        if system:
            formatted_messages.append({"role": "system", "content": system})
        
        # Add the rest of the messages
        formatted_messages.extend(messages)
        
        # Make the API call with the OpenAI client
        stream = await self.openai_client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        return stream 