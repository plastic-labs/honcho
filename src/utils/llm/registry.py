"""
Provider client initialization and validation.

This module owns the global `CLIENTS` registry that maps Honcho provider names to
initialized SDK clients. It also validates that all configured providers and
backup providers are available at import time.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic
from google import genai
from groq import AsyncGroq
from openai import AsyncOpenAI

from src.config import settings
from src.utils.types import SupportedProviders

# Global mapping of provider identifiers to initialized SDK clients.
CLIENTS: dict[
    SupportedProviders,
    AsyncAnthropic | AsyncOpenAI | genai.Client | AsyncGroq,
] = {}

if settings.LLM.ANTHROPIC_API_KEY:
    anthropic = AsyncAnthropic(
        api_key=settings.LLM.ANTHROPIC_API_KEY,
        timeout=600.0,
    )
    CLIENTS["anthropic"] = anthropic

if settings.LLM.OPENAI_API_KEY:
    openai_client = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_API_KEY,
    )
    CLIENTS["openai"] = openai_client

if settings.LLM.OPENAI_COMPATIBLE_API_KEY:
    CLIENTS["openrouter"] = AsyncOpenAI(
        api_key=settings.LLM.OPENAI_COMPATIBLE_API_KEY,
        base_url=settings.LLM.OPENAI_COMPATIBLE_BASE_URL
        or "https://openrouter.ai/api/v1",
    )

if settings.LLM.VLLM_API_KEY and settings.LLM.VLLM_BASE_URL:
    CLIENTS["vllm"] = AsyncOpenAI(
        api_key=settings.LLM.VLLM_API_KEY,
        base_url=settings.LLM.VLLM_BASE_URL,
    )

if settings.LLM.GEMINI_API_KEY:
    google = genai.client.Client(api_key=settings.LLM.GEMINI_API_KEY)
    CLIENTS["google"] = google

if settings.LLM.GROQ_API_KEY:
    groq = AsyncGroq(api_key=settings.LLM.GROQ_API_KEY)
    CLIENTS["groq"] = groq

SELECTED_PROVIDERS: list[tuple[str, Any]] = [
    ("Summary", settings.SUMMARY.PROVIDER),
    ("Deriver", settings.DERIVER.PROVIDER),
]

for level, level_settings in settings.DIALECTIC.LEVELS.items():
    SELECTED_PROVIDERS.append((f"Dialectic ({level})", level_settings.PROVIDER))
    if level_settings.SYNTHESIS is not None:
        SELECTED_PROVIDERS.append(
            (f"Dialectic ({level}) Synthesis", level_settings.SYNTHESIS.PROVIDER)
        )

for provider_name, provider_value in SELECTED_PROVIDERS:
    if provider_value not in CLIENTS:
        raise ValueError(f"Missing client for {provider_name}: {provider_value}")

BACKUP_PROVIDERS: list[tuple[str, SupportedProviders | None]] = [
    ("Deriver", settings.DERIVER.BACKUP_PROVIDER),
    ("Summary", settings.SUMMARY.BACKUP_PROVIDER),
    ("Dream", settings.DREAM.BACKUP_PROVIDER),
]

for level, level_settings in settings.DIALECTIC.LEVELS.items():
    BACKUP_PROVIDERS.append((f"Dialectic ({level})", level_settings.BACKUP_PROVIDER))
    if level_settings.SYNTHESIS is not None:
        BACKUP_PROVIDERS.append(
            (f"Dialectic ({level}) Synthesis", level_settings.SYNTHESIS.BACKUP_PROVIDER)
        )

for component_name, backup_provider in BACKUP_PROVIDERS:
    if backup_provider is not None and backup_provider not in CLIENTS:
        raise ValueError(
            f"Backup provider for {component_name} is set to {backup_provider}, "
            + "but this provider is not initialized. Please set the required API key/URL environment "
            + "variables or remove the backup configuration."
        )
