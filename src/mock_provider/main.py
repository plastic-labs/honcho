"""A deterministic, OpenAI-compatible provider for local and CI use.

Lets Honcho run with no model provider, no API key, and no spend. It answers
``/v1/chat/completions`` and ``/v1/embeddings`` with obviously-synthetic content
derived from the request, so the same request always produces the same response.

Runs as its own service from the standard Honcho image:

    fastapi run --host 0.0.0.0 src/mock_provider/main.py

Point Honcho at it with three variables — all three are required:

    LLM_OPENAI_API_KEY=any-non-empty-string   # only truthiness is checked; value ignored
    LLM_OPENAI_BASE_URL=http://mock-provider:8000/v1
    EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL=http://mock-provider:8000/v1

The key's *value* is never checked — not by this mock, which reads no
Authorization header, and not by Honcho, which only tests it for truthiness
before constructing the client (``src/llm/registry.py``). Set the base URL
without it and the client is never built, so the base URL is silently ignored.
Keep the value obviously fake: if a module ever escapes the base-URL override it
then 401s against the real provider instead of spending.

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

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.mock_provider import chat, embeddings

app = FastAPI(
    title="Honcho Mock Provider",
    description="Deterministic OpenAI-compatible endpoint for local and CI use.",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def openai_error_response(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Answer a malformed request the way the real API does.

    FastAPI's default is a 422 carrying its own error shape. Mid-run that reads
    as a Honcho bug rather than a bad request, and it is not what an OpenAI
    client expects — the real API returns 400 with an ``error`` envelope, so
    that is what a faithful mock returns.
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": f"Invalid request: {exc.errors()}",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
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
