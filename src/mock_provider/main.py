"""A deterministic, OpenAI-compatible provider for local and CI use.

Lets Honcho run with no model provider, no API key, and no spend. It answers
``/v1/chat/completions`` and ``/v1/embeddings`` with obviously-synthetic content
derived from the request, so the same request always produces the same response.

Runs as its own service from the standard Honcho image:

    fastapi run --host 0.0.0.0 src/mock_provider/main.py

Point Honcho at it with three variables — all three are required:

    LLM_OPENAI_API_KEY=sandbox            # gates client construction, not just auth
    LLM_OPENAI_BASE_URL=http://mock-provider:8000/v1
    EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=http://mock-provider:8000/v1

Embeddings resolve through a separate client that reads the base URL only from
the per-module override, so without the third variable embedding calls go to
api.openai.com for real. Do not set any per-module credential override
(``..._OVERRIDES__API_KEY`` / ``API_KEY_ENV``) — that makes the module ignore the
global base URL.

Embeddings are hash-derived and carry no semantic similarity. Recall assertions
against this provider must use lexical/full-text search, not vector ranking.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from src.mock_provider import chat, embeddings

app = FastAPI(
    title="Honcho Mock Provider",
    description="Deterministic OpenAI-compatible endpoint for local and CI use.",
    version="1.0.0",
)

# Mounted at both prefixes so the base URL works with or without /v1.
for _router in (chat.router, embeddings.router):
    app.include_router(_router, prefix="/v1")
    app.include_router(_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "provider": "mock"}


@app.get("/{path:path}")
async def catch_all(path: str) -> dict[str, Any]:
    """Answer any other GET, so a bare ``/`` works as a container healthcheck.

    Deliberately GET-only: an unimplemented POST returns 405 rather than a
    plausible-looking 200, so a missing endpoint fails loudly.
    """
    return {"object": "mock", "path": path, "detail": "mock provider placeholder"}
