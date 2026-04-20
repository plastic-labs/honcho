"""Honcho LLM orchestration package — stable public surface.

Application code should import from `src.llm` (or specific submodules like
`src.llm.api` / `src.llm.types`). The old `src/utils/clients.py` entrypoint
is gone; everything lives here now.
"""

from __future__ import annotations

from .api import honcho_llm_call
from .backend import CompletionResult, ProviderBackend, StreamChunk, ToolCallResult
from .credentials import default_transport_api_key, resolve_credentials
from .executor import honcho_llm_call_inner
from .registry import (
    CLIENTS,
    backend_for_provider,
    client_for_model_config,
    get_anthropic_client,
    get_anthropic_override_client,
    get_backend,
    get_gemini_client,
    get_gemini_override_client,
    get_openai_client,
    get_openai_override_client,
    history_adapter_for_provider,
)
from .types import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    IterationData,
    ProviderClient,
    ReasoningEffortType,
    StreamingResponseWithMetadata,
    VerbosityType,
)

__all__ = [
    "CLIENTS",
    "CompletionResult",
    "HonchoLLMCallResponse",
    "HonchoLLMCallStreamChunk",
    "IterationCallback",
    "IterationData",
    "ProviderBackend",
    "ProviderClient",
    "ReasoningEffortType",
    "StreamChunk",
    "StreamingResponseWithMetadata",
    "ToolCallResult",
    "VerbosityType",
    "backend_for_provider",
    "client_for_model_config",
    "default_transport_api_key",
    "get_anthropic_client",
    "get_anthropic_override_client",
    "get_backend",
    "get_gemini_client",
    "get_gemini_override_client",
    "get_openai_client",
    "get_openai_override_client",
    "history_adapter_for_provider",
    "honcho_llm_call",
    "honcho_llm_call_inner",
    "resolve_credentials",
]
