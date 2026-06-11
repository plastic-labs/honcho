"""OpenRouter-specific client helpers."""

from __future__ import annotations

from collections.abc import Mapping

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def is_openrouter_base_url(base_url: str | None) -> bool:
    """Return true when a configured OpenAI-compatible URL points at OpenRouter."""

    if not base_url:
        return False
    return base_url.rstrip("/") == OPENROUTER_BASE_URL


def attribution_headers(
    *,
    base_url: str | None,
    app_url: str | None,
    app_title: str | None,
) -> Mapping[str, str] | None:
    """Build OpenRouter attribution headers for OpenAI-compatible clients.

    OpenRouter shows requests as "Unknown" in account activity unless callers send
    app attribution headers. Only attach these headers when the base URL is
    OpenRouter so other OpenAI-compatible providers do not receive irrelevant
    metadata.
    """

    if not is_openrouter_base_url(base_url):
        return None

    headers: dict[str, str] = {}
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-OpenRouter-Title"] = app_title
        # Backward-compatible alias still recognized by OpenRouter and older SDKs.
        headers["X-Title"] = app_title

    return headers or None
