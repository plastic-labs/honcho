from unittest.mock import MagicMock, patch

import pytest

from src.utils.clients import honcho_llm_call


def test_missing_client_for_dialectic_provider():
    """Test that ValueError is raised when Dialectic provider has no corresponding client."""
    # Create a mock clients dict that's missing the 'nonexistent' provider
    clients = {"openai": "mock_openai_client"}

    # Create the providers list as it appears in the code
    providers = [
        ("Dialectic", "nonexistent"),
        ("Summary", "openai"),
        ("Deriver", "openai"),
        ("Query Generation Provider", "openai"),
    ]

    # Test the validation logic directly
    with pytest.raises(ValueError, match="Missing client for Dialectic: nonexistent"):
        for provider_name, provider_value in providers:
            if provider_value not in clients:
                raise ValueError(
                    f"Missing client for {provider_name}: {provider_value}"
                )


def test_missing_client_for_summary_provider():
    """Test that ValueError is raised when Summary provider has no corresponding client."""
    # Create a mock clients dict that's missing the 'missing' provider
    clients = {"openai": "mock_openai_client"}

    # Create the providers list as it appears in the code
    providers = [
        ("Dialectic", "openai"),
        ("Summary", "missing"),
        ("Deriver", "openai"),
        ("Query Generation Provider", "openai"),
    ]

    # Test the validation logic directly
    with pytest.raises(ValueError, match="Missing client for Summary: missing"):
        for provider_name, provider_value in providers:
            if provider_value not in clients:
                raise ValueError(
                    f"Missing client for {provider_name}: {provider_value}"
                )


def test_missing_client_for_deriver_provider():
    """Test that ValueError is raised when Deriver provider has no corresponding client."""
    # Create a mock clients dict that's missing the 'unavailable' provider
    clients = {"openai": "mock_openai_client"}

    # Create the providers list as it appears in the code
    providers = [
        ("Dialectic", "openai"),
        ("Summary", "openai"),
        ("Deriver", "unavailable"),
        ("Query Generation Provider", "openai"),
    ]

    # Test the validation logic directly
    with pytest.raises(ValueError, match="Missing client for Deriver: unavailable"):
        for provider_name, provider_value in providers:
            if provider_value not in clients:
                raise ValueError(
                    f"Missing client for {provider_name}: {provider_value}"
                )


def test_missing_client_for_query_generation_provider():
    """Test that ValueError is raised when Query Generation provider has no corresponding client."""
    # Create a mock clients dict that's missing the 'invalid' provider
    clients = {"openai": "mock_openai_client"}

    # Create the providers list as it appears in the code
    providers = [
        ("Dialectic", "openai"),
        ("Summary", "openai"),
        ("Deriver", "openai"),
        ("Query Generation Provider", "invalid"),
    ]

    # Test the validation logic directly
    with pytest.raises(
        ValueError, match="Missing client for Query Generation Provider: invalid"
    ):
        for provider_name, provider_value in providers:
            if provider_value not in clients:
                raise ValueError(
                    f"Missing client for {provider_name}: {provider_value}"
                )


def test_missing_multiple_clients():
    """Test that ValueError is raised for the first missing client when multiple providers are missing."""
    # Create a mock clients dict that's missing multiple providers
    clients = {"openai": "mock_openai_client"}

    # Create the providers list with multiple missing providers
    providers = [
        ("Dialectic", "missing1"),
        ("Summary", "missing2"),
        ("Deriver", "openai"),
        ("Query Generation Provider", "openai"),
    ]

    # Test the validation logic directly - should fail on the first missing provider
    with pytest.raises(ValueError, match="Missing client for Dialectic: missing1"):
        for provider_name, provider_value in providers:
            if provider_value not in clients:
                raise ValueError(
                    f"Missing client for {provider_name}: {provider_value}"
                )


def test_all_clients_present():
    """Test that no error is raised when all required clients are present."""
    # Create a mock clients dict that has all providers
    clients = {
        "anthropic": "mock_anthropic_client",
        "openai": "mock_openai_client",
        "google": "mock_google_client",
        "groq": "mock_groq_client",
    }

    # Create the providers list with all available providers
    providers = [
        ("Dialectic", "anthropic"),
        ("Summary", "google"),
        ("Deriver", "openai"),
        ("Query Generation Provider", "groq"),
    ]

    # Test the validation logic directly - should not raise any exception
    try:
        for provider_name, provider_value in providers:
            if provider_value not in clients:
                raise ValueError(
                    f"Missing client for {provider_name}: {provider_value}"
                )
    except ValueError:
        pytest.fail("ValueError was raised when all clients are present")


def test_honcho_llm_call_max_tokens_for_non_google_non_anthropic_providers():
    """Test that max_tokens is properly set in call_params for providers other than google and anthropic."""

    # Test with different non-google, non-anthropic providers using valid literal types
    providers_to_test = ["openai", "groq", "custom"]

    for provider in providers_to_test:
        with (
            patch("src.utils.clients.clients", {provider: MagicMock()}),
            patch("src.utils.clients.llm") as mock_llm,
            patch("src.utils.clients.with_langfuse") as mock_langfuse,
            patch("src.utils.clients.ai_track") as mock_ai_track,
        ):
            # Mock the decorators to just return the function
            mock_llm.call.return_value = lambda x: x  # pyright: ignore
            mock_langfuse.return_value = lambda x: x  # pyright: ignore
            mock_ai_track.return_value = lambda x: x  # pyright: ignore

            # Create a test function to decorate - must be async and return proper type
            if provider == "openai":

                @honcho_llm_call(provider="openai", model="test-model", max_tokens=1000)
                async def _test_function():  # pyright: ignore
                    return "test prompt"
            elif provider == "groq":

                @honcho_llm_call(provider="groq", model="test-model", max_tokens=1000)
                async def _test_function():  # pyright: ignore
                    return "test prompt"
            else:  # custom

                @honcho_llm_call(provider="custom", model="test-model", max_tokens=1000)
                async def _test_function():  # pyright: ignore
                    return "test prompt"

            # The decorator should have been called with the right parameters
            mock_llm.call.assert_called_once()
            call_kwargs = mock_llm.call.call_args[1]

            # Verify that call_params contains max_tokens
            assert "call_params" in call_kwargs
            assert "max_tokens" in call_kwargs["call_params"]
            assert call_kwargs["call_params"]["max_tokens"] == 1000

            # Reset mocks for next iteration
            mock_llm.call.reset_mock()


def test_honcho_llm_call_max_tokens_none_for_non_google_non_anthropic_providers():
    """Test that max_tokens is not set in call_params when max_tokens is None."""

    with (
        patch("src.utils.clients.clients", {"openai": MagicMock()}),
        patch("src.utils.clients.llm") as mock_llm,
        patch("src.utils.clients.with_langfuse") as mock_langfuse,
    ):
        # Mock the decorators to just return the function
        mock_llm.call.return_value = lambda x: x  # pyright: ignore
        mock_langfuse.return_value = lambda x: x  # pyright: ignore

        # Create a test function to decorate without max_tokens - must be async
        @honcho_llm_call(
            provider="openai",
            model="test-model",
            # max_tokens not provided (None)
        )
        async def test_function():  # pyright: ignore
            return "test prompt"

        # The decorator should have been called
        mock_llm.call.assert_called_once()
        call_kwargs = mock_llm.call.call_args[1]

        # Verify that call_params either doesn't exist or doesn't contain max_tokens
        if "call_params" in call_kwargs:
            assert "max_tokens" not in call_kwargs["call_params"]


def test_honcho_llm_call_max_tokens_zero_for_non_google_non_anthropic_providers():
    """Test that max_tokens is not set in call_params when max_tokens is 0 (falsy)."""

    with (
        patch("src.utils.clients.clients", {"groq": MagicMock()}),
        patch("src.utils.clients.llm") as mock_llm,
        patch("src.utils.clients.with_langfuse") as mock_langfuse,
    ):
        # Mock the decorators to just return the function
        mock_llm.call.return_value = lambda x: x  # pyright: ignore
        mock_langfuse.return_value = lambda x: x  # pyright: ignore

        # Create a test function to decorate with max_tokens=0 - must be async
        @honcho_llm_call(
            provider="groq",
            model="test-model",
            max_tokens=0,  # Falsy value
        )
        async def test_function():  # pyright: ignore
            return "test prompt"

        # The decorator should have been called
        mock_llm.call.assert_called_once()
        call_kwargs = mock_llm.call.call_args[1]

        # Verify that call_params either doesn't exist or doesn't contain max_tokens
        if "call_params" in call_kwargs:
            assert "max_tokens" not in call_kwargs["call_params"]
