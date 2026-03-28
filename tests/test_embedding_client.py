"""
Tests for src/embedding_client.py

Tests cover:
- Provider initialization (openai, gemini, voyage, openrouter)
- Dimension validation against provider constraints
- Configurable output dimensions via VECTOR_STORE.DIMENSIONS
- Single embed, batch embed, and chunked embed for Voyage provider
- EmbeddingClient singleton API key resolution
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import voyageai

from src.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(provider: str, api_key: str = "test-key", dimensions: int = 1536):
    """Create an _EmbeddingClient with patched settings."""
    with patch.object(settings.VECTOR_STORE, "DIMENSIONS", dimensions):
        from src.embedding_client import _EmbeddingClient
        return _EmbeddingClient(api_key=api_key, provider=provider)


# ---------------------------------------------------------------------------
# Initialization & dimension validation
# ---------------------------------------------------------------------------

class TestProviderInit:
    """Test that each provider initializes correctly."""

    def test_openai_default_dimensions(self):
        client = _make_client("openai")
        assert client.provider == "openai"
        assert client.output_dimensions == 1536
        assert client.model == "text-embedding-3-small"

    def test_gemini_default_dimensions(self):
        client = _make_client("gemini")
        assert client.provider == "gemini"
        assert client.output_dimensions == 1536

    def test_voyage_with_supported_dimensions(self):
        for dim in [256, 512, 1024, 2048]:
            client = _make_client("voyage", dimensions=dim)
            assert client.provider == "voyage"
            assert client.output_dimensions == dim
            assert client.model == "voyage-4"
            assert client.max_batch_size == 128

    def test_openrouter_default_dimensions(self):
        with patch.object(settings.LLM, "OPENAI_COMPATIBLE_BASE_URL", None):
            client = _make_client("openrouter")
            assert client.provider == "openrouter"
            assert client.output_dimensions == 1536

    def test_openai_custom_dimensions(self):
        client = _make_client("openai", dimensions=512)
        assert client.output_dimensions == 512

    def test_gemini_custom_dimensions(self):
        client = _make_client("gemini", dimensions=768)
        assert client.output_dimensions == 768


class TestDimensionValidation:
    """Test that unsupported dimensions are rejected at init time."""

    def test_voyage_rejects_1536(self):
        with pytest.raises(ValueError, match="not supported"):
            _make_client("voyage", dimensions=1536)

    def test_voyage_rejects_arbitrary_value(self):
        with pytest.raises(ValueError, match="not supported"):
            _make_client("voyage", dimensions=768)

    def test_voyage_error_message_includes_supported_values(self):
        with pytest.raises(ValueError, match=r"\[256, 512, 1024, 2048\]"):
            _make_client("voyage", dimensions=1536)

    def test_openai_accepts_any_dimension(self):
        # OpenAI text-embedding-3 supports Matryoshka — any dimension works
        client = _make_client("openai", dimensions=42)
        assert client.output_dimensions == 42


class TestMissingApiKey:
    """Test that missing API keys raise clear errors."""

    def test_voyage_no_key(self):
        with patch.object(settings.LLM, "VOYAGE_API_KEY", None), \
             pytest.raises(ValueError, match="LLM_VOYAGE_API_KEY"):
            _make_client("voyage", api_key=None, dimensions=1024)

    def test_openai_no_key(self):
        with patch.object(settings.LLM, "OPENAI_API_KEY", None), \
             pytest.raises(ValueError, match="OpenAI API key"):
            _make_client("openai", api_key=None)

    def test_gemini_no_key(self):
        with patch.object(settings.LLM, "GEMINI_API_KEY", None), \
             pytest.raises(ValueError, match="Gemini API key"):
            _make_client("gemini", api_key=None)


# ---------------------------------------------------------------------------
# Voyage embed methods
# ---------------------------------------------------------------------------

class TestVoyageEmbed:
    """Test Voyage AI embed paths with mocked client."""

    @pytest.fixture
    def voyage_client(self):
        client = _make_client("voyage", dimensions=1024)
        client.client = AsyncMock(spec=voyageai.AsyncClient)
        return client

    @pytest.mark.asyncio
    async def test_single_embed(self, voyage_client):
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024]
        voyage_client.client.embed = AsyncMock(return_value=mock_response)

        result = await voyage_client.embed("test query")
        assert len(result) == 1024

        voyage_client.client.embed.assert_called_once_with(
            texts=["test query"],
            model="voyage-4",
            output_dimension=1024,
        )

    @pytest.mark.asyncio
    async def test_single_embed_empty_response_raises(self, voyage_client):
        mock_response = MagicMock()
        mock_response.embeddings = []
        voyage_client.client.embed = AsyncMock(return_value=mock_response)

        with pytest.raises(ValueError, match="No embedding returned"):
            await voyage_client.embed("test")

    @pytest.mark.asyncio
    async def test_batch_embed(self, voyage_client):
        texts = ["hello", "world"]
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * 1024, [0.2] * 1024]
        voyage_client.client.embed = AsyncMock(return_value=mock_response)

        results = await voyage_client.simple_batch_embed(texts)
        assert len(results) == 2
        assert all(len(e) == 1024 for e in results)


# ---------------------------------------------------------------------------
# Singleton API key resolution
# ---------------------------------------------------------------------------

class TestSingletonKeyResolution:
    """Test EmbeddingClient singleton resolves the right API key per provider."""

    def test_voyage_key_resolution(self):
        from src.embedding_client import EmbeddingClient

        with patch.object(settings.LLM, "EMBEDDING_PROVIDER", "voyage"), \
             patch.object(settings.LLM, "VOYAGE_API_KEY", "voy-test-key"), \
             patch.object(settings.VECTOR_STORE, "DIMENSIONS", 1024):
            # Reset singleton
            singleton = EmbeddingClient()
            singleton._instance = None

            client = singleton._get_client()
            assert client.provider == "voyage"

    def test_openai_key_resolution(self):
        from src.embedding_client import EmbeddingClient

        with patch.object(settings.LLM, "EMBEDDING_PROVIDER", "openai"), \
             patch.object(settings.LLM, "OPENAI_API_KEY", "sk-test"):
            singleton = EmbeddingClient()
            singleton._instance = None

            client = singleton._get_client()
            assert client.provider == "openai"
