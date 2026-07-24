# Local Codex Responses relay

`src.codex_relay` provides a narrow OpenAI Chat Completions compatibility layer for
Honcho's local development setup. It reads the `openai-codex` credential from the
Hermes auth store on each request, refreshes an expiring OAuth token through the
standard Codex OAuth token endpoint, and atomically writes rotated credentials
back without logging token values.

The upstream request is always sent as a streaming Responses request to:

`https://chatgpt.com/backend-api/codex/responses`

The relay converts system/developer prompts, user and assistant messages,
function calls and function results, reasoning effort, tools, and JSON response
formats. Non-streaming Chat Completions calls aggregate the Codex SSE stream;
streaming calls receive translated Chat Completions SSE. Upstream HTTP error
status and bodies are returned unchanged. Provider errors emitted inside SSE are
returned as structured errors with a classification-friendly status.

## Start

From the repository root, the default listener is loopback-only:

```bash
uv run python -m src.codex_relay --host 127.0.0.1 --port 8787
```

For a Honcho container on the same private Docker network, bind explicitly to
that network interface/address rather than exposing the port publicly:

```bash
uv run python -m src.codex_relay --host 0.0.0.0 --port 8787
```

The relay does not modify `.env`, start/reload Honcho containers, or claim to be
deployed. Point an OpenAI-compatible Honcho model configuration at the relay's
OpenAI base URL (`http://127.0.0.1:8787/v1` for a host-local process).

## Checks

```bash
uv run pytest tests/llm/test_codex_relay.py -q
uv run ruff check src/codex_relay.py tests/llm/test_codex_relay.py
uv run basedpyright src/codex_relay.py tests/llm/test_codex_relay.py
```

The test suite uses a mocked SSE transport and never reads or prints live
credentials.
