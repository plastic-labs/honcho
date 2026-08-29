"""
Utility for logging traces from LLM calls.

This module provides structured JSONL logging of LLM inputs/outputs.
"""

import contextlib
import json
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

fcntl: Any = None
msvcrt: Any = None
if sys.platform == "win32":
    import msvcrt as _msvcrt

    msvcrt = _msvcrt
else:
    import fcntl as _fcntl

    fcntl = _fcntl


from pydantic import BaseModel

from src.config import (
    ConfiguredModelSettings,
    ModelConfig,
    settings,
)


@contextmanager
def _locked(f: IO[str]) -> Generator[None, None, None]:
    """Exclusively lock an open file for the duration of the block.

    Multiple processes (API server and deriver) append to the same traces file, so
    writes must be serialized. POSIX uses fcntl.flock; Windows uses msvcrt.locking,
    which locks a byte range from the current offset. If neither is available the
    write still proceeds unlocked rather than losing the trace.
    """
    if sys.platform != "win32":
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    elif sys.platform == "win32":
        f.seek(0, os.SEEK_END)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:  # lock unavailable after retries — do not drop the trace
            yield
            return
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - no locking primitive available
        yield


def get_reasoning_traces_file_path() -> Path | None:
    """Get the traces file path from settings."""
    if settings.REASONING_TRACES_FILE:
        return Path(settings.REASONING_TRACES_FILE)
    return None


def log_reasoning_trace(
    task_type: str,
    model_config: ModelConfig | ConfiguredModelSettings,
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
        model_config: Model configuration used for the call
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
        "provider": model_config.transport,
        "model": model_config.model,
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
    with open(traces_file, "a") as f, _locked(f):
        f.write(json.dumps(trace_entry) + "\n")
        f.flush()
