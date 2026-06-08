import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config import (
    ConfiguredEmbeddingModelSettings,
    ConfiguredModelSettings,
    ModelOverrideSettings,
    settings,
)
from src.routers.runtime import router


def _runtime_client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/v3")
    return TestClient(app)


def test_llm_runtime_exposes_current_model_provider_and_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", False)
    monkeypatch.setattr(
        settings.DIALECTIC.LEVELS["low"],
        "MODEL_CONFIG",
        ConfiguredModelSettings(
            transport="openai",
            model="anthropic/claude-3.5-sonnet",
            overrides=ModelOverrideSettings(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
            ),
        ),
    )

    response = _runtime_client().get("/v3/runtime/llm")

    assert response.status_code == 200
    body = response.json()
    assert body["current_source"] == "dialectic.low"
    assert body["current"] == {
        "model": "anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "transport": "openai",
        "auth_mechanism": "api_key",
        "base_url": "https://openrouter.ai/api/v1",
    }
    assert body["models"]["dialectic.low"] == body["current"]
    assert "deriver" in body["models"]


def test_embedding_runtime_exposes_model_provider_auth_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", False)
    monkeypatch.setattr(settings.LLM, "GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setattr(
        settings.EMBEDDING,
        "MODEL_CONFIG",
        ConfiguredEmbeddingModelSettings(transport="gemini"),
    )
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 768)

    response = _runtime_client().get("/v3/runtime/embeddings")

    assert response.status_code == 200
    assert response.json() == {
        "model": "gemini-embedding-001",
        "provider": "gemini",
        "transport": "gemini",
        "auth_mechanism": "api_key",
        "base_url": None,
        "vector_dimensions": 768,
        "dimensions_mode": "auto",
    }


def test_runtime_endpoints_use_global_provider_credentials_and_base_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", False)
    monkeypatch.setattr(settings.LLM, "OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(settings.LLM, "OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(
        settings.DIALECTIC.LEVELS["low"],
        "MODEL_CONFIG",
        ConfiguredModelSettings(
            transport="openai",
            model="anthropic/claude-3.5-sonnet",
        ),
    )
    monkeypatch.setattr(
        settings.EMBEDDING,
        "MODEL_CONFIG",
        ConfiguredEmbeddingModelSettings(
            transport="openai",
            model="text-embedding-3-small",
        ),
    )
    monkeypatch.setattr(settings.EMBEDDING, "VECTOR_DIMENSIONS", 1536)

    llm_response = _runtime_client().get("/v3/runtime/llm")
    embeddings_response = _runtime_client().get("/v3/runtime/embeddings")
    auth_response = _runtime_client().get("/v3/runtime/auth")

    assert llm_response.status_code == 200
    assert llm_response.json()["current"] == {
        "model": "anthropic/claude-3.5-sonnet",
        "provider": "openrouter",
        "transport": "openai",
        "auth_mechanism": "api_key",
        "base_url": "https://openrouter.ai/api/v1",
    }

    assert embeddings_response.status_code == 200
    assert embeddings_response.json() == {
        "model": "text-embedding-3-small",
        "provider": "openai",
        "transport": "openai",
        "auth_mechanism": "api_key",
        "base_url": None,
        "vector_dimensions": 1536,
        "dimensions_mode": "auto",
    }

    assert auth_response.status_code == 200
    assert auth_response.json() == {
        "llm_source": "dialectic.low",
        "llm": llm_response.json()["current"],
        "embeddings": {
            "model": "text-embedding-3-small",
            "provider": "openai",
            "transport": "openai",
            "auth_mechanism": "api_key",
            "base_url": None,
        },
    }


def test_runtime_auth_exposes_current_llm_and_embedding_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings.AUTH, "USE_AUTH", False)
    monkeypatch.setattr(
        settings.DIALECTIC.LEVELS["low"],
        "MODEL_CONFIG",
        ConfiguredModelSettings(
            transport="openai",
            model="gpt-5.4-mini",
            overrides=ModelOverrideSettings(api_key="test-openai-key"),
        ),
    )
    monkeypatch.setattr(
        settings.EMBEDDING,
        "MODEL_CONFIG",
        ConfiguredEmbeddingModelSettings(
            transport="openai",
            model="text-embedding-3-small",
        ),
    )
    monkeypatch.setattr(settings.LLM, "OPENAI_API_KEY", "test-openai-key")

    response = _runtime_client().get("/v3/runtime/auth")

    assert response.status_code == 200
    body = response.json()
    assert body["llm_source"] == "dialectic.low"
    assert body["llm"]["provider"] == "openai"
    assert body["llm"]["auth_mechanism"] == "api_key"
    assert body["embeddings"]["provider"] == "openai"
    assert body["embeddings"]["auth_mechanism"] == "api_key"
