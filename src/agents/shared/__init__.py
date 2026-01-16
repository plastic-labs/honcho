"""
Shared agent infrastructure for Honcho.

This module provides base classes, utilities, and common functionality
shared across all agents in the Honcho system.
"""

from .base_agent import BaseAgent
from .config import (
    AbducerConfig,
    AgentConfig,
    DialecticConfig,
    DreamerConfig,
    ExtractorConfig,
    FalsifierConfig,
    InductorConfig,
    PredictorConfig,
    create_config_from_dict,
)
from .prompts import (
    format_context_section,
    format_peer_info,
    format_provenance_chain,
    format_system_prompt,
    truncate_text,
)
from .tools import (
    create_tool_definition,
    extract_tool_arguments,
    format_tool_result,
    validate_tool_call,
)

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentConfig",
    # Agent configs
    "ExtractorConfig",
    "AbducerConfig",
    "PredictorConfig",
    "FalsifierConfig",
    "InductorConfig",
    "DialecticConfig",
    "DreamerConfig",
    # Config utilities
    "create_config_from_dict",
    # Prompt utilities
    "format_system_prompt",
    "format_context_section",
    "format_provenance_chain",
    "format_peer_info",
    "truncate_text",
    # Tool utilities
    "create_tool_definition",
    "validate_tool_call",
    "extract_tool_arguments",
    "format_tool_result",
]
