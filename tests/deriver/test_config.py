"""Configuration for deriver tests - handles real vs mocked LLM API calls.

This module provides configuration utilities for deriver tests to either:
1. Use real LLM API calls when environment variables are set
2. Use mocked responses when API keys are not available
3. Skip tests that require real API calls if not configured

Environment variables:
- ENABLE_REAL_LLM_TESTS: Set to 'true' to enable real API calls
- ANTHROPIC_API_KEY: Required for real Anthropic API calls
- GOOGLE_AI_API_KEY: Required for real Google AI API calls
"""

import asyncio
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Configuration flags
ENABLE_REAL_LLM_TESTS = os.getenv("ENABLE_REAL_LLM_TESTS", "false").lower() == "true"
HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_GOOGLE_KEY = bool(os.getenv("GOOGLE_AI_API_KEY"))
HAS_GROQ_KEY = bool(os.getenv("GROQ_API_KEY"))

# Rate limiting for real API calls
API_CALL_DELAY = float(os.getenv("LLM_TEST_DELAY", "1.0"))  # Seconds between calls


class LLMTestConfig:
    """Configuration class for LLM testing behavior."""

    def __init__(self):
        self.use_real_apis = ENABLE_REAL_LLM_TESTS
        self.has_anthropic = HAS_ANTHROPIC_KEY
        self.has_anthropic_api_key = HAS_ANTHROPIC_KEY
        self.has_google = HAS_GOOGLE_KEY
        self.has_groq_api_key = HAS_GROQ_KEY
        self.call_delay = API_CALL_DELAY
        self._call_count = 0

    @property
    def can_use_real_apis(self) -> bool:
        """Check if we can use real API calls."""
        return self.use_real_apis and (
            self.has_anthropic or self.has_google or self.has_groq_api_key
        )

    async def rate_limit(self):
        """Apply rate limiting between API calls."""
        if self.use_real_apis and self._call_count > 0:
            await asyncio.sleep(self.call_delay)
        self._call_count += 1


# Global config instance
test_config = LLMTestConfig()


def requires_real_llm_apis(test_func):
    """Decorator to skip tests that require real LLM API access."""
    return pytest.mark.skipif(
        not test_config.can_use_real_apis,
        reason="Real LLM API access not configured. Set ENABLE_REAL_LLM_TESTS=true and provide API keys.",
    )(test_func)


class MockResponseGenerator:
    """Generates realistic mock responses for LLM APIs."""

    @staticmethod
    def create_tom_inference_response() -> dict[str, Any]:
        """Create a realistic TOM inference response."""
        return {
            "current_state": {
                "immediate_context": "User is sharing information about their background",
                "active_goals": "Providing personal/professional details",
                "present_mood": "Conversational and informative",
            },
            "tentative_inferences": [
                {
                    "interpretation": "User has technical expertise",
                    "basis": "Mentions specific technologies and experience",
                }
            ],
            "knowledge_gaps": [
                {"topic": "Specific project details"},
                {"topic": "Career goals and aspirations"},
            ],
            "expectation_violations": [
                {
                    "possible_surprise": "User reveals they're just starting their career",
                    "reason": "Would contradict mentioned experience level",
                    "confidence_level": 0.1,
                }
            ],
        }

    @staticmethod
    def create_user_representation_response() -> dict[str, Any]:
        """Create a realistic user representation response."""
        return {
            "current_state": {
                "active_context": {
                    "detail": "User discussing professional background",
                    "source": "Recent conversation messages",
                },
                "temporary_conditions": {
                    "detail": "Sharing career information",
                    "source": "Current conversation context",
                },
                "present_mood_activity": {
                    "detail": "Engaged in professional discussion",
                    "source": "Tone and content of messages",
                },
            },
            "persistent_information": [
                {
                    "detail": "User works in technology field",
                    "source": "Professional background discussion",
                    "info_type": "STATEMENT",
                }
            ],
            "tentative_patterns": [
                {
                    "pattern": "Professional and detail-oriented communication",
                    "source": "Communication style",
                    "certainty_level": "LIKELY",
                }
            ],
            "knowledge_gaps": [
                {"missing_info": "Specific technical skills"},
                {"missing_info": "Years of experience"},
            ],
            "expectation_violations": [
                {
                    "potential_surprise": "User switches to completely different topic",
                    "reason": "Current focus is on professional background",
                    "confidence_level": 0.2,
                }
            ],
            "updates": {
                "new_information": [
                    {
                        "detail": "User provided professional context",
                        "source": "Current conversation",
                    }
                ],
                "changes": [],
                "removals": [],
            },
        }

    @staticmethod
    def create_fact_extraction_response(content: str) -> list[str]:
        """Create realistic facts based on message content."""
        facts = []

        # Extract basic patterns
        if "python" in content.lower():
            facts.append("User has Python programming experience")
        if "engineer" in content.lower():
            facts.append("User works as an engineer")
        if "machine learning" in content.lower() or "ml" in content.lower():
            facts.append("User has machine learning experience")
        if "years" in content.lower():
            facts.append("User has professional experience")
        if "seattle" in content.lower():
            facts.append("User is based in Seattle")
        if "san francisco" in content.lower():
            facts.append("User is located in San Francisco")
        if "remote" in content.lower():
            facts.append("User works remotely")

        # Default facts if nothing specific found
        if not facts:
            facts = ["User shared information about their background"]

        return facts


async def setup_llm_mocking():
    """Set up LLM API mocking for tests."""
    if test_config.use_real_apis:
        # Don't mock anything - use real APIs
        return None

    # Set up comprehensive mocking
    mock_generators = MockResponseGenerator()

    patches = []

    # Mock Anthropic API
    anthropic_patch = patch("src.deriver.tom.conversational.anthropic.messages.create")
    mock_anthropic = anthropic_patch.__enter__()
    mock_message = MagicMock()
    mock_message.content = [MagicMock()]
    mock_message.content[
        0
    ].text = "<prediction>User seems engaged and providing professional information</prediction>"
    mock_anthropic.return_value = mock_message
    patches.append(anthropic_patch)

    # Mock Mirascope functions
    tom_patch = patch("src.deriver.tom.single_prompt.tom_inference")
    mock_tom = tom_patch.__enter__()

    def mock_tom_inference_func(*args, **kwargs):
        response = mock_generators.create_tom_inference_response()
        # Return a mock object that has the expected structure
        mock_obj = MagicMock()
        mock_obj.current_state = response["current_state"]
        mock_obj.tentative_inferences = response["tentative_inferences"]
        mock_obj.knowledge_gaps = response["knowledge_gaps"]
        mock_obj.expectation_violations = response["expectation_violations"]
        return mock_obj

    mock_tom.side_effect = mock_tom_inference_func
    patches.append(tom_patch)

    # Mock user representation inference
    user_rep_patch = patch(
        "src.deriver.tom.single_prompt.user_representation_inference"
    )
    mock_user_rep = user_rep_patch.__enter__()

    def mock_user_rep_func(*args, **kwargs):
        response = mock_generators.create_user_representation_response()
        mock_obj = MagicMock()
        mock_obj.current_state = response["current_state"]
        mock_obj.persistent_information = response["persistent_information"]
        mock_obj.tentative_patterns = response["tentative_patterns"]
        mock_obj.knowledge_gaps = response["knowledge_gaps"]
        mock_obj.expectation_violations = response["expectation_violations"]
        mock_obj.updates = response["updates"]
        return mock_obj

    mock_user_rep.side_effect = mock_user_rep_func
    patches.append(user_rep_patch)

    # Mock fact extraction
    fact_patch = patch("src.deriver.tom.long_term.extract_facts_long_term")
    mock_fact = fact_patch.__enter__()

    def mock_fact_extraction_func(chat_history):
        facts = mock_generators.create_fact_extraction_response(chat_history)
        mock_obj = MagicMock()
        mock_obj.facts = facts
        mock_obj.information_extraction = {
            "pieces": [],
            "challenge": "Extracting relevant facts from conversation",
        }
        return mock_obj

    mock_fact.side_effect = mock_fact_extraction_func
    patches.append(fact_patch)

    return patches


def cleanup_llm_mocking(patches):
    """Clean up LLM API mocking."""
    if patches:
        for patch_obj in patches:
            patch_obj.__exit__(None, None, None)


class RealLLMRateLimiter:
    """Rate limiter for real LLM API calls during testing."""

    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.last_call = 0.0

    async def wait_if_needed(self):
        """Wait if needed to respect rate limits."""
        import time

        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last_call = time.time()


# Global rate limiter
rate_limiter = RealLLMRateLimiter(API_CALL_DELAY)


def conditional_mock_llm():
    """Conditionally mock LLM based on configuration."""
    return not test_config.use_real_apis


def with_retry_and_rate_limit(func):
    """Decorator to add retry and rate limiting to LLM calls."""

    async def wrapper(*args, **kwargs):
        if test_config.use_real_apis:
            await rate_limiter.wait_if_needed()
        return await func(*args, **kwargs)

    return wrapper
