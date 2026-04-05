"""Tests for AWS Bedrock provider integration.

These tests mock boto3 to avoid needing real AWS credentials. Environment
variables are patched before any src imports to satisfy module-level
provider validation in src.utils.clients.
"""

import io
import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Patch environment before importing src modules so module-level validation passes
os.environ.setdefault(
    "DB_CONNECTION_URI", "postgresql+psycopg://user:pass@localhost/test"
)
os.environ.setdefault("LLM_ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("LLM_GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("SUMMARY_PROVIDER", "anthropic")
os.environ.setdefault("DERIVER_PROVIDER", "anthropic")

from src.utils.clients import (  # noqa: E402
    BedrockClient,
    _append_tool_results,
    _format_assistant_tool_message,
    convert_tools_for_provider,
)


class TestBedrockClient:
    """Tests for the BedrockClient wrapper."""

    def test_init_with_explicit_credentials(self) -> None:
        client = BedrockClient(
            region_name="us-west-2",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            aws_session_token="FwoGZXIvYXdzEBAaDH...",
        )
        assert client.region_name == "us-west-2"
        assert client.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"
        assert (
            client.aws_secret_access_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        )
        assert client.aws_session_token == "FwoGZXIvYXdzEBAaDH..."
        assert client._client is None

    def test_init_without_credentials(self) -> None:
        """Client can be created without explicit credentials (for IAM roles)."""
        client = BedrockClient(region_name="us-east-1")
        assert client.aws_access_key_id is None
        assert client.aws_secret_access_key is None
        assert client.aws_session_token is None

    @patch("boto3.client")
    def test_client_lazy_initialization(self, mock_boto3_client: MagicMock) -> None:
        mock_boto3_client.return_value = MagicMock()
        client = BedrockClient(
            region_name="us-east-1",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
        )
        assert client._client is None
        _ = client.client
        mock_boto3_client.assert_called_once_with(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
        )
        assert client._client is not None

    @patch("boto3.client")
    def test_client_reuses_instance(self, mock_boto3_client: MagicMock) -> None:
        mock_boto3_client.return_value = MagicMock()
        client = BedrockClient(region_name="us-east-1")
        first = client.client
        second = client.client
        assert first is second
        mock_boto3_client.assert_called_once()

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_converse(self, mock_boto3_client: MagicMock) -> None:
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello!"}],
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "stopReason": "end_turn",
        }
        mock_boto3_client.return_value = mock_bedrock

        client = BedrockClient(region_name="us-east-1")
        result = await client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"maxTokens": 100},
        )

        assert result["output"]["message"]["content"][0]["text"] == "Hello!"
        assert result["usage"]["inputTokens"] == 10
        mock_bedrock.converse.assert_called_once()

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_converse_with_tools(self, mock_boto3_client: MagicMock) -> None:
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"text": "Let me search for that."},
                        {
                            "toolUse": {
                                "toolUseId": "call_123",
                                "name": "search_memory",
                                "input": {"query": "test"},
                            }
                        },
                    ],
                }
            },
            "usage": {"inputTokens": 20, "outputTokens": 15},
            "stopReason": "tool_use",
        }
        mock_boto3_client.return_value = mock_bedrock

        client = BedrockClient(region_name="us-east-1")
        result = await client.converse(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": "Search for test"}]}],
            inferenceConfig={"maxTokens": 200},
            toolConfig={
                "tools": [
                    {
                        "toolSpec": {
                            "name": "search_memory",
                            "description": "Search",
                            "inputSchema": {"json": {}},
                        }
                    }
                ]
            },
        )

        content = result["output"]["message"]["content"]
        assert content[0]["text"] == "Let me search for that."
        assert content[1]["toolUse"]["name"] == "search_memory"
        assert result["stopReason"] == "tool_use"

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_converse_stream(self, mock_boto3_client: MagicMock) -> None:
        mock_bedrock = MagicMock()
        mock_bedrock.converse_stream.return_value = {
            "stream": [
                {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": ""}}},
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockDelta": {"delta": {"text": " world"}}},
                {"messageStop": {"stopReason": "end_turn"}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
            ]
        }
        mock_boto3_client.return_value = mock_bedrock

        client = BedrockClient(region_name="us-east-1")
        result = await client.converse_stream(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"maxTokens": 100},
        )

        events = result["stream"]
        assert len(events) == 5
        assert "contentBlockDelta" in events[1]
        assert events[1]["contentBlockDelta"]["delta"]["text"] == "Hello"

    @pytest.mark.asyncio
    @patch("boto3.client")
    async def test_invoke_model(self, mock_boto3_client: MagicMock) -> None:
        embedding = [0.1, 0.2, 0.3]
        response_body = json.dumps({"embedding": embedding}).encode()
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = {
            "body": io.BytesIO(response_body),
        }
        mock_boto3_client.return_value = mock_bedrock

        client = BedrockClient(region_name="us-east-1")
        result = await client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(
                {"inputText": "test", "dimensions": 1024, "normalize": True}
            ),
            contentType="application/json",
            accept="application/json",
        )

        body = json.loads(result["body"].read())
        assert body["embedding"] == embedding
        mock_bedrock.invoke_model.assert_called_once()


class TestConvertToolsForBedrock:
    """Tests for Bedrock tool format conversion."""

    def test_convert_tools_for_bedrock(self) -> None:
        tools: list[dict[str, Any]] = [
            {
                "name": "search_memory",
                "description": "Search memory for relevant info",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        result = convert_tools_for_provider(tools, "bedrock")
        assert len(result) == 1
        assert "toolSpec" in result[0]
        assert result[0]["toolSpec"]["name"] == "search_memory"
        assert result[0]["toolSpec"]["inputSchema"]["json"] == tools[0]["input_schema"]


class TestBedrockMessageFormatting:
    """Tests for Bedrock message formatting helpers."""

    def test_format_assistant_tool_message(self) -> None:
        tool_calls: list[dict[str, Any]] = [
            {"id": "call_1", "name": "search", "input": {"q": "test"}},
        ]
        result = _format_assistant_tool_message(
            content="Searching...",
            tool_calls=tool_calls,
            provider="bedrock",
        )

        assert result["role"] == "assistant"
        assert result["content"][0] == {"text": "Searching..."}
        assert result["content"][1]["toolUse"]["toolUseId"] == "call_1"
        assert result["content"][1]["toolUse"]["name"] == "search"

    def test_append_tool_results(self) -> None:
        messages: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = [
            {"tool_id": "call_1", "result": "found 3 results"},
            {"tool_id": "call_2", "result": "error occurred", "is_error": True},
        ]
        _append_tool_results("bedrock", tool_results, messages)

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["toolResult"]["toolUseId"] == "call_1"
        assert content[0]["toolResult"]["content"][0]["text"] == "found 3 results"
        assert "status" not in content[0]["toolResult"]
        assert content[1]["toolResult"]["status"] == "error"
