"""
Honcho LLM client layer.

This package provides a provider-agnostic API (`honcho_llm_call`) backed by
provider adapters that encapsulate the idiosyncrasies of each upstream SDK.
"""

from src.utils.llm.core import (
    handle_streaming_response,
    honcho_llm_call,
    honcho_llm_call_inner,
)
from src.utils.llm.history import count_message_tokens, truncate_messages_to_fit
from src.utils.llm.models import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    IterationCallback,
    IterationData,
    StreamingResponseWithMetadata,
)
from src.utils.llm.registry import CLIENTS

__all__ = [
    "CLIENTS",
    "IterationCallback",
    "IterationData",
    "HonchoLLMCallResponse",
    "HonchoLLMCallStreamChunk",
    "StreamingResponseWithMetadata",
    "count_message_tokens",
    "truncate_messages_to_fit",
    "honcho_llm_call",
    "honcho_llm_call_inner",
    "handle_streaming_response",
]
