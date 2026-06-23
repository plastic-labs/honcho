"""Tests for src.llm.registry helpers."""

from __future__ import annotations

from src.llm.registry import _default_headers_for  # pyright: ignore[reportPrivateUsage]


def test_default_headers_for_openrouter_base_url() -> None:
    """OpenRouter base URLs get the app-attribution headers."""
    headers = _default_headers_for("https://openrouter.ai/api/v1")
    assert headers["HTTP-Referer"] == "https://honcho.dev"
    assert headers["X-Openrouter-Title"] == "Honcho"


def test_default_headers_for_non_openrouter_base_url() -> None:
    """Other OpenAI-compatible providers get no extra headers."""
    assert _default_headers_for("https://api.openai.com/v1") == {}


def test_default_headers_for_none_base_url() -> None:
    """A missing base URL (default OpenAI) gets no extra headers."""
    assert _default_headers_for(None) == {}
