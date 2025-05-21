import json

import pytest
import respx
from httpx import Response, TimeoutException

from src.agent import QUERY_GENERATION_SYSTEM
from src.deriver.tom.single_prompt import (
    TOM_SYSTEM_PROMPT,
    USER_REPRESENTATION_SYSTEM_PROMPT,
)


@pytest.fixture
def mock_model_client():
    """
    Fixture to create and return mocked model client responses using respx.
    This fixture mocks the underlying HTTP requests rather than the model client class.
    
    Example usage:
        def test_something(mock_model_client):
            # The mock_model_client fixture will automatically set up the mocked responses
            # You can customize the responses in your test:
            mock_model_client.tom_response = "<prediction>Custom prediction</prediction>"
            mock_model_client.update_routes()
    """
    class MockResponses:
        def __init__(self):
            self.tom_response = "<prediction>The user appears to be initiating a friendly conversation with a polite greeting.</prediction>"
            self.query_response = ["query about interests", "query about personality", "query about experiences"]
            self.default_response = "Default response"
            self.user_representation_response = "<representation>The user appears to be a friendly and outgoing person.</representation>"
            self.error_mode = False
            self.timeout_mode = False
            self.setup_routes()

        def setup_routes(self):
            # Mock Anthropic API
            self.anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
                side_effect=self.handle_request
            )
            
            # Mock OpenAI API
            self.openai_route = respx.post("https://api.openai.com/v1/chat/completions").mock(
                side_effect=self.handle_request
            )

            # Mock Groq API
            self.groq_route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
                side_effect=self.handle_request
            )

        def update_routes(self):
            """Call this after changing any response values to update the routes"""
            self.setup_routes()

        async def handle_request(self, request):
            """Handle all API requests with common error handling"""
            if self.error_mode:
                return Response(500, json={"error": {"message": "Mock API error"}})
            if self.timeout_mode:
                raise TimeoutException("Mock timeout error")

            # Parse the request body
            body = json.loads(request.content)
            response_content_str = self.get_response_for_request(body)

            # Format response based on the API endpoint
            if "anthropic.com" in str(request.url):
                return Response(200, json={
                    "content": [{"type": "text", "text": response_content_str}]
                })
            else:  # OpenAI-compatible format (OpenAI, Groq)
                return Response(200, json={
                    "choices": [{"message": {"content": response_content_str}}]
                })

        def get_response_for_request(self, body):
            """Determine which response to return based on the request body content and structure."""
            system_prompt_content = None

            # Check for Anthropic-style top-level system prompt
            if "system" in body and isinstance(body["system"], str):
                system_prompt_content = body["system"]
            # Check for OpenAI/Groq-style system prompt within the messages array
            elif "messages" in body and isinstance(body["messages"], list):
                for msg in body["messages"]:
                    if isinstance(msg, dict) and msg.get("role") == "system" and "content" in msg:
                        system_prompt_content = msg["content"]
                        break
            
            if system_prompt_content:
                if system_prompt_content == TOM_SYSTEM_PROMPT:
                    return self.tom_response
                elif system_prompt_content == QUERY_GENERATION_SYSTEM:
                    return json.dumps(self.query_response)
                elif system_prompt_content == USER_REPRESENTATION_SYSTEM_PROMPT:
                    return json.dumps(self.user_representation_response)
            
            return self.default_response

    return MockResponses()
