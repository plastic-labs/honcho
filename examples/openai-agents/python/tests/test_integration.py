"""Integration tests for the OpenAI Agents SDK + Honcho memory integration.

These tests run against the live Honcho API and require ``HONCHO_API_KEY``
to be set. They are skipped automatically when the key is absent.
"""

import os
import sys
import time
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.client import HonchoContext
from tools.get_context import get_context
from tools.save_memory import save_memory

pytestmark = pytest.mark.skipif(
    not os.getenv("HONCHO_API_KEY"),
    reason="HONCHO_API_KEY not set — skipping integration tests",
)


@pytest.fixture(autouse=True)
def rate_limit_delay():
    """Pause between tests to stay under the Honcho API rate limit."""
    yield
    time.sleep(0.5)


def unique_id(prefix: str) -> str:
    """Return a unique prefixed ID to avoid cross-test state leakage."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestSaveMemory:
    """Tests for the save_memory helper."""

    def test_returns_confirmation_string(self):
        result = save_memory(unique_id("user"), "Hello!", "user", unique_id("session"))
        assert isinstance(result, str) and len(result) > 0

    def test_saves_user_message(self):
        user_id = unique_id("user")
        result = save_memory(user_id, "I enjoy hiking", "user", unique_id("session"))
        assert user_id in result or "user" in result.lower()

    def test_saves_assistant_message(self):
        result = save_memory(
            unique_id("user"), "Great to hear!", "assistant", unique_id("session")
        )
        assert isinstance(result, str) and len(result) > 0

    def test_saves_multiple_turns_same_session(self):
        user_id = unique_id("user")
        session_id = unique_id("session")
        r1 = save_memory(user_id, "I love jazz music", "user", session_id)
        r2 = save_memory(user_id, "Jazz is wonderful!", "assistant", session_id)
        assert len(r1) > 0 and len(r2) > 0

    def test_non_assistant_role_treated_as_user(self):
        result = save_memory(
            unique_id("user"), "Testing role fallback", "human", unique_id("session")
        )
        assert isinstance(result, str) and len(result) > 0

    def test_custom_assistant_id(self):
        result = save_memory(
            unique_id("user"),
            "Hello!",
            "assistant",
            unique_id("session"),
            assistant_id="my-bot",
        )
        assert isinstance(result, str) and len(result) > 0


class TestGetContext:
    """Tests for the get_context helper."""

    def test_returns_list(self):
        user_id = unique_id("user")
        session_id = unique_id("session")
        save_memory(user_id, "Hello there!", "user", session_id)

        ctx = HonchoContext(user_id=user_id, session_id=session_id)
        result = get_context(ctx)
        assert isinstance(result, list)

    def test_returns_openai_format(self):
        user_id = unique_id("user")
        session_id = unique_id("session")
        save_memory(user_id, "My name is Alex", "user", session_id)
        save_memory(user_id, "Nice to meet you, Alex!", "assistant", session_id)

        ctx = HonchoContext(user_id=user_id, session_id=session_id)
        result = get_context(ctx)

        assert isinstance(result, list)
        for msg in result:
            assert "role" in msg and "content" in msg
            assert msg["role"] in ("user", "assistant", "system")
            assert isinstance(msg["content"], str)

    def test_empty_session_returns_list(self):
        ctx = HonchoContext(user_id=unique_id("user"), session_id=unique_id("session"))
        result = get_context(ctx)
        assert isinstance(result, list)

    def test_respects_token_limit(self):
        user_id = unique_id("user")
        session_id = unique_id("session")
        for i in range(5):
            save_memory(user_id, f"Message {i} with some longer content here", "user", session_id)

        ctx = HonchoContext(user_id=user_id, session_id=session_id)
        small = get_context(ctx, tokens=50)
        large = get_context(ctx, tokens=8000)

        assert isinstance(small, list) and isinstance(large, list)
        assert len(large) >= len(small)


class TestSaveGetRoundtrip:
    """End-to-end tests combining save_memory and get_context."""

    def test_saved_messages_appear_in_context(self):
        user_id = unique_id("user")
        session_id = unique_id("session")
        user_content = "Hello from the integration test!"
        assistant_content = "Hi there, integration test!"

        save_memory(user_id, user_content, "user", session_id)
        save_memory(user_id, assistant_content, "assistant", session_id)

        ctx = HonchoContext(user_id=user_id, session_id=session_id)

        # Retry briefly — Honcho processes messages asynchronously
        messages = []
        for _ in range(5):
            messages = get_context(ctx)
            contents = [m["content"] for m in messages]
            if user_content in contents and assistant_content in contents:
                break
            time.sleep(1)

        contents = [m["content"] for m in messages]
        assert user_content in contents, "User message not found in context"
        assert assistant_content in contents, "Assistant message not found in context"
