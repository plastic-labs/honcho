"""
Utility for logging traces from LLM calls.

This module provides structured JSONL logging of LLM inputs/outputs.
"""

import contextlib
import json
import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel

from src.config import (
    ConfiguredModelSettings,
    ModelConfig,
    settings,
)

logger = logging.getLogger(__name__)

locking_module: Any = import_module("msvcrt" if sys.platform == "win32" else "fcntl")

# Windows has no blocking whole-file lock, so acquisition is retried explicitly
# rather than relying on msvcrt's implicit LK_LOCK retry policy.
_LOCK_RETRIES = 10
_LOCK_RETRY_DELAY_SECONDS = 0.1


@contextmanager
def _locked(f: IO[str]) -> Generator[bool, None, None]:
    """Exclusively lock an open file for the duration of the block.

    Multiple processes (API server and deriver) append to the same traces file, so
    writes must be serialized. POSIX uses fcntl.flock; Windows uses msvcrt.locking
    on a fixed byte range, retried under an explicit policy.

    Yields True when the lock is held. If Windows cannot acquire it within the
    retry budget the block is entered with False and the caller must skip the
    write: an unlocked append can interleave with another process and corrupt the
    JSONL file, so a dropped trace is preferable. Tracing is an opt-in debugging
    aid, so failure is logged rather than raised into the LLM call path.
    """
    if sys.platform != "win32":
        locking_module.flock(f.fileno(), locking_module.LOCK_EX)
        try:
            yield True
        finally:
            locking_module.flock(f.fileno(), locking_module.LOCK_UN)
        return

    # Every writer must coordinate on the same byte range. Keep the lock offset
    # so it can be restored before LK_UNLCK after the append.
    lock_offset = 0
    for attempt in range(_LOCK_RETRIES):
        f.seek(lock_offset)
        try:
            locking_module.locking(f.fileno(), locking_module.LK_NBLCK, 1)
        except OSError:
            if attempt < _LOCK_RETRIES - 1:
                time.sleep(_LOCK_RETRY_DELAY_SECONDS)
            continue
        try:
            yield True
        finally:
            f.seek(lock_offset)
            with contextlib.suppress(OSError):
                locking_module.locking(f.fileno(), locking_module.LK_UNLCK, 1)
        return

    logger.warning(
        "Could not lock reasoning traces file after %d attempts; dropping trace "
        "rather than appending without a lock.",
        _LOCK_RETRIES,
    )
    yield False


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
    with open(traces_file, "a") as f, _locked(f) as acquired:
        if acquired:
            f.write(json.dumps(trace_entry) + "\n")
            f.flush()
