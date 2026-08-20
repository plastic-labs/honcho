"""Netra SDK initialization for Honcho.

Instruments the LLM providers Honcho calls through src/llm/backends/
(Anthropic, OpenAI, Google GenAI) plus FastAPI for request-level spans.

Controlled by environment variables:
    NETRA_ENABLED       - set to "true" to activate (default: false)
    NETRA_API_KEY       - Netra project API key
    NETRA_OTLP_ENDPOINT - Netra OTLP collector endpoint
"""

import logging
import os

logger = logging.getLogger(__name__)


def init_netra(app_name: str = "honcho") -> None:
    if os.getenv("NETRA_ENABLED", "false").lower() != "true":
        return

    api_key = os.getenv("NETRA_API_KEY")
    if not api_key:
        logger.warning("NETRA_ENABLED=true but NETRA_API_KEY is not set — skipping")
        return

    try:
        from netra import Netra
        from netra.instrumentation.instruments import NetraInstruments
    except ImportError:
        logger.warning("netra-sdk not installed — skipping Netra initialization")
        return

    Netra.init(
        app_name=app_name,
        headers=f"x-api-key={api_key}",
        disable_batch=True,
        instruments={
            NetraInstruments.ANTHROPIC,
            NetraInstruments.OPENAI,
            NetraInstruments.GOOGLE_GENERATIVEAI,
            NetraInstruments.FASTAPI,
            NetraInstruments.HTTPX,
            NetraInstruments.SQLALCHEMY,
            NetraInstruments.REDIS,
            NetraInstruments.PSYCOPG,
        },
    )
    logger.info("Netra initialized for %s", app_name)
