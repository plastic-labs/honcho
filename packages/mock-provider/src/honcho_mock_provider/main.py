"""Standalone app factory; importing it reads no server settings or credentials."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from honcho_mock_provider import chat, embeddings


async def openai_error_response(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise exc
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


async def health() -> dict[str, str]:
    return {"status": "ok", "provider": "mock"}


async def catch_all(path: str) -> dict[str, Any]:
    return {"object": "mock", "path": path, "detail": "mock provider placeholder"}


def create_app() -> FastAPI:
    app = FastAPI(title="Honcho Mock Provider", version="1.0.0")
    app.add_exception_handler(RequestValidationError, openai_error_response)
    for router in (chat.router, embeddings.router):
        app.include_router(router, prefix="/v1")
        app.include_router(router)
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/{path:path}", catch_all, methods=["GET"])
    return app


app = create_app()
