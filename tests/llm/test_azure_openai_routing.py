import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.config import ModelConfig
from src.llm import registry
from src.llm.backends.openai import OpenAIBackend


def test_client_for_model_config_builds_azure_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``azure_openai`` transport must instantiate AsyncAzureOpenAI with
    azure_endpoint + api_version from the ModelConfig overrides."""
    registry.get_azure_openai_override_client.cache_clear()
    captured: dict[str, Any] = {}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(registry, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    config = ModelConfig(
        model="gpt-4o-mini-deployment",
        transport="azure_openai",
        api_key="az-test-key",
        base_url="https://gateway.example/azure-openai",
        api_version="2024-10-21",
    )

    try:
        client = registry.client_for_model_config("azure_openai", config)

        assert isinstance(client, FakeAsyncAzureOpenAI)
        assert captured == {
            "api_key": "az-test-key",
            "azure_endpoint": "https://gateway.example/azure-openai",
            "api_version": "2024-10-21",
        }
    finally:
        # Evict the FakeAsyncAzureOpenAI instance so later tests that hit
        # the same (endpoint, key, version) cache key get the real class
        # back (restored by monkeypatch teardown).
        registry.get_azure_openai_override_client.cache_clear()


def test_client_for_model_config_builds_azure_client_with_entra_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``use_entra_id=True``, the factory must pass
    ``azure_ad_token_provider`` instead of ``api_key``."""
    registry.get_azure_openai_override_client.cache_clear()
    captured: dict[str, Any] = {}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(registry, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    mock_credential = MagicMock()
    mock_token_provider = MagicMock(return_value="fake-token")

    mock_azure_identity = MagicMock(
        DefaultAzureCredential=MagicMock(return_value=mock_credential),
        get_bearer_token_provider=MagicMock(return_value=mock_token_provider),
    )
    monkeypatch.setitem(sys.modules, "azure.identity", mock_azure_identity)

    config = ModelConfig(
        model="gpt-4o-mini-deployment",
        transport="azure_openai",
        base_url="https://gateway.example/azure-openai",
        api_version="2024-10-21",
        use_entra_id=True,
    )

    try:
        client = registry.client_for_model_config("azure_openai", config)

        assert isinstance(client, FakeAsyncAzureOpenAI)
        assert captured["azure_ad_token_provider"] is mock_token_provider
        assert captured["azure_endpoint"] == "https://gateway.example/azure-openai"
        assert captured["api_version"] == "2024-10-21"
        assert "api_key" not in captured
    finally:
        registry.get_azure_openai_override_client.cache_clear()
        sys.modules.pop("azure.identity", None)


def test_entra_id_does_not_require_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``use_entra_id=True`` must not raise for a missing API key."""
    registry.get_azure_openai_override_client.cache_clear()

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(registry, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    mock_azure_identity = MagicMock(
        DefaultAzureCredential=MagicMock(return_value=MagicMock()),
        get_bearer_token_provider=MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setitem(sys.modules, "azure.identity", mock_azure_identity)

    config = ModelConfig(
        model="gpt-4o-mini-deployment",
        transport="azure_openai",
        base_url="https://gateway.example/azure-openai",
        api_version="2024-10-21",
        use_entra_id=True,
        # no api_key
    )

    try:
        # Should NOT raise ValidationException
        registry.client_for_model_config("azure_openai", config)
    finally:
        registry.get_azure_openai_override_client.cache_clear()
        sys.modules.pop("azure.identity", None)


def test_use_entra_id_false_uses_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default ``use_entra_id=False`` must use API key auth."""
    registry.get_azure_openai_override_client.cache_clear()
    captured: dict[str, Any] = {}

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(registry, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    config = ModelConfig(
        model="gpt-4o-mini-deployment",
        transport="azure_openai",
        api_key="az-test-key",
        base_url="https://gateway.example/azure-openai",
        api_version="2024-10-21",
        use_entra_id=False,
    )

    try:
        client = registry.client_for_model_config("azure_openai", config)

        assert isinstance(client, FakeAsyncAzureOpenAI)
        assert captured == {
            "api_key": "az-test-key",
            "azure_endpoint": "https://gateway.example/azure-openai",
            "api_version": "2024-10-21",
        }
        assert "azure_ad_token_provider" not in captured
    finally:
        registry.get_azure_openai_override_client.cache_clear()


def test_entra_id_without_azure_identity_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When use_entra_id=True but azure-identity is not installed, the
    factory must raise ValidationException with a ``pip install 'honcho[azure]'``
    install hint — not bubble up a bare ModuleNotFoundError."""
    from src.exceptions import ValidationException

    registry.get_azure_openai_override_client.cache_clear()

    class FakeAsyncAzureOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(registry, "AsyncAzureOpenAI", FakeAsyncAzureOpenAI)

    # Force the lazy ``from azure.identity import ...`` to raise ImportError
    # even if the package happens to be installed in the test environment.
    import builtins

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "azure.identity" or name.startswith("azure.identity."):
            raise ImportError(f"No module named {name!r}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    config = ModelConfig(
        model="gpt-4o-mini-deployment",
        transport="azure_openai",
        base_url="https://gateway.example/azure-openai",
        api_version="2024-10-21",
        use_entra_id=True,
    )

    try:
        with pytest.raises(ValidationException, match=r"honcho\[azure\]"):
            registry.client_for_model_config("azure_openai", config)
    finally:
        registry.get_azure_openai_override_client.cache_clear()


def test_backend_for_provider_wraps_azure_client_in_openai_backend() -> None:
    """Same backend serves both ``openai`` and ``azure_openai``."""
    sentinel_client = object()

    backend = registry.backend_for_provider("azure_openai", sentinel_client)

    assert isinstance(backend, OpenAIBackend)
    assert backend._client is sentinel_client  # pyright: ignore[reportPrivateUsage]
