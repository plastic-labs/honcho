"""
external_telemetry.py -- optional async telemetry hook for Honcho components.

Posts LLM usage metadata to an operator-configured HTTP endpoint. Never raises;
all errors are logged and swallowed so a telemetry collector outage cannot block
the deriver or any other caller.

Usage (inside an async function):
    from src.external_telemetry import log_llm_call

    await log_llm_call(
        component="honcho-deriver",
        model="example-model",
        prompt_tokens=1234,
        completion_tokens=56,
        latency_s=1.83,
        session_id=session_name,
    )
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Optional endpoint for operators who want to collect LLM usage outside Honcho.
_TELEMETRY_URL = os.environ.get("HONCHO_EXTERNAL_TELEMETRY_URL")
_TIMEOUT_S = 2.0  # short timeout so we never block the event loop for long


async def log_llm_call(
    *,
    component: str,
    model: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    cost: Optional[float] = None,
    latency_s: Optional[float] = None,
    session_id: Optional[str] = None,
) -> None:
    """POST LLM usage metadata to an external telemetry endpoint. Never raises."""
    if not _TELEMETRY_URL:
        return

    payload: dict = {"component": component}
    if model is not None:
        payload["model"] = model
    if prompt_tokens is not None:
        payload["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        payload["completion_tokens"] = completion_tokens
    if reasoning_tokens is not None:
        payload["reasoning_tokens"] = reasoning_tokens
    if cost is not None:
        payload["cost"] = cost
    if latency_s is not None:
        payload["latency_s"] = latency_s
    if session_id is not None:
        payload["session_id"] = session_id

    try:
        import httpx

        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.post(_TELEMETRY_URL, json=payload)
            if resp.status_code not in (200, 201):
                logger.debug(
                    "external telemetry: unexpected status %s from collector: %s",
                    resp.status_code,
                    resp.text[:200],
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Swallow all errors at DEBUG level so collector outages are invisible.
        logger.debug("external telemetry: post failed (ignored): %s", exc)
