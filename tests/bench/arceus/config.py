"""Configuration management for Arceus system."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Model pricing per 1K tokens (input, output) in USD
# Pricing as of December 2024
MODEL_PRICING = {
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-opus-4-5-20251101": {"input": 0.015, "output": 0.075},
    "claude-opus-4-5": {"input": 0.015, "output": 0.075},
    "claude-sonnet-3-5-20240620": {"input": 0.003, "output": 0.015},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
}


@dataclass
class ArceusConfig:
    """Configuration for the Arceus ARC-AGI-2 solver."""

    # Honcho Configuration
    honcho_url: str = "http://localhost:8000"
    workspace_id: str = "arc-agi-2-solver"

    # LLM Configuration
    llm_provider: Literal["anthropic", "openai", "gemini"] = "anthropic"
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5-20250929"

    # ARC Data Paths (relative to project root)
    # Path: config.py -> arceus -> bench -> tests -> honcho -> arceus (project root)
    arc_data_path: Path = Path(__file__).parent.parent.parent.parent.parent / "ARC-AGI-2/data"
    training_path: Path = Path(__file__).parent.parent.parent.parent.parent / "ARC-AGI-2/data/training"
    evaluation_path: Path = Path(__file__).parent.parent.parent.parent.parent / "ARC-AGI-2/data/evaluation"

    # Solver Configuration
    max_iterations: int = 10
    timeout_seconds: int = 180
    enable_memory: bool = True

    # TUI Configuration
    enable_tui: bool = True
    tui_refresh_rate: float = 0.1  # seconds

    # Debugging Configuration
    enable_json_trace: bool = True
    trace_output_dir: Path = Path("traces")
    enable_verbose_logging: bool = True

    # Metrics Configuration
    track_token_usage: bool = True
    track_memory_operations: bool = True

    def __post_init__(self):
        """Post-initialization to load from environment variables."""
        # Load from environment variables
        self.llm_api_key = os.getenv("ANTHROPIC_API_KEY", self.llm_api_key)
        if not self.llm_api_key:
            self.llm_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.llm_api_key:
            self.llm_api_key = os.getenv("GEMINI_API_KEY", "")

        # Create trace output directory
        if self.enable_json_trace:
            self.trace_output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "ArceusConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        if os.getenv("HONCHO_URL"):
            config.honcho_url = os.getenv("HONCHO_URL")
        if os.getenv("WORKSPACE_ID"):
            config.workspace_id = os.getenv("WORKSPACE_ID")
        if os.getenv("LLM_PROVIDER"):
            config.llm_provider = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            config.llm_model = os.getenv("LLM_MODEL")
        if os.getenv("MAX_ITERATIONS"):
            config.max_iterations = int(os.getenv("MAX_ITERATIONS"))
        if os.getenv("ENABLE_TUI"):
            config.enable_tui = os.getenv("ENABLE_TUI").lower() == "true"
        if os.getenv("ENABLE_JSON_TRACE"):
            config.enable_json_trace = os.getenv("ENABLE_JSON_TRACE").lower() == "true"

        return config
