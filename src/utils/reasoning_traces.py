"""
Utility for logging traces from LLM calls.

This module provides structured JSONL logging of LLM inputs/outputs.
"""

import fcntl
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import LLMComponentSettings, settings


def get_reasoning_traces_file_path() -> Path | None:
    """Get the traces file path from settings."""
    if settings.REASONING_TRACES_FILE:
        return Path(settings.REASONING_TRACES_FILE)
    return None


def log_reasoning_trace(
    task_type: str,
    llm_settings: LLMComponentSettings,
    prompt: str,
    response: Any,
    *,
    max_tokens: int | None = None,
    thinking_budget_tokens: int | None = None,
    reasoning_effort: str | None = None,
    json_mode: bool = False,
    stop_seqs: list[str] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> None:
    """
    Log a trace to the configured JSONL file.

    Args:
        task_type: Type of task (e.g., "minimal_deriver", "dialectic_chat")
        llm_settings: LLM settings used for the call
        prompt: The full prompt text sent to the LLM (used if messages is None)
        response: HonchoLLMCallResponse object with the LLM response
        max_tokens: Max output tokens setting
        thinking_budget_tokens: Anthropic thinking budget (if used)
        reasoning_effort: OpenAI reasoning effort (if used)
        json_mode: Whether JSON mode was enabled
        stop_seqs: Stop sequences used (if any)
        messages: Full conversation history for multi-turn/agentic calls
    """
    traces_file = get_reasoning_traces_file_path()
    if not traces_file:
        return

    # Serialize response content - handle Pydantic models
    content = response.content
    if isinstance(content, BaseModel):
        content = content.model_dump()

    trace_entry: dict[str, Any] = {
        "timestamp": time.time(),
        "task_type": task_type,
        "provider": llm_settings.PROVIDER,
        "model": llm_settings.MODEL,
        "settings": {
            "max_tokens": max_tokens,
            "thinking_budget_tokens": thinking_budget_tokens,
            "reasoning_effort": reasoning_effort,
            "json_mode": json_mode,
            "stop_seqs": stop_seqs,
        },
        "input": {
            "tokens": response.input_tokens,
        },
        "output": {
            "content": content,
            "tokens": response.output_tokens,
            "finish_reasons": response.finish_reasons,
            "thinking_content": response.thinking_content,
        },
    }

    # Use messages for multi-turn/agentic calls, otherwise use prompt
    if messages is not None:
        trace_entry["input"]["messages"] = messages
    else:
        trace_entry["input"]["prompt"] = prompt

    # Include tool calls if present
    if hasattr(response, "tool_calls_made") and response.tool_calls_made:
        trace_entry["output"]["tool_calls"] = response.tool_calls_made

    # Use file locking to handle concurrent writes from multiple processes
    with open(traces_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(trace_entry) + "\n")
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
