---
name: verify
description: Build, launch, and drive a local Honcho stack to verify a change at its runtime surface (the /v3 HTTP API and the deriver queue). Use when verifying a diff or confirming a change works in the running app.
---

# Verifying changes in Honcho

## Prerequisites

Docker Compose is the preferred way to run the stack — `docker-compose.yml` at
the repo root brings up Postgres (pgvector), Redis, the API server, and the
deriver worker together. If Docker Compose isn't available, the stack can also
run directly on the host (see Launch below); you'll need Postgres with pgvector
and Redis reachable, plus a `.env` with connection strings and an LLM provider
key for any flow that hits a model (deriver, dialectic, dreamer).

## Launch

First check whether the stack is already running via Docker Compose — that's
the preferred setup, and reusing it beats starting a second one:

```bash
docker compose ps          # look for api / deriver / database / redis
docker compose up -d       # start it if not
```

Without Docker, run the two processes directly (shared Postgres + Redis):

```bash
uv run fastapi dev src/main.py      # API server (default :8000)
uv run python -m src.deriver        # deriver worker (queue consumer)
```

## Drive it

Prefer driving through the `honcho-cli` skill; fall back to the SDKs
(`sdks/python`, `sdks/typescript`), then raw REST against `/v3`, in that order
when a method isn't available at the higher level. The `honcho-integration`
skill covers how to use the SDKs.

When a change updates endpoints or makes schema changes, verify across all
three surfaces — CLI, SDKs, and REST — since they can drift independently.

## Configuration

Configuration is central to both driving the app and running tests: it's how
API keys reach the server and how a config-related change gets exercised at
all. Settings come from environment variables or files, with precedence
env > `.env` > `config.toml` > defaults. To verify a configuration change,
set the relevant option through one of these layers, restart the affected
process, and observe the behavioral difference at the surface — the same
mechanism lets you point provider base URLs, model choices, and timeouts at
credentials and proxies you actually have.

## Test suites

Three test types matter here. All run in CI, but they're also runnable locally
with whatever API keys and configuration you have — Honcho's config surface is
large, so options like per-agent timeouts and provider base URLs can be pointed
at your own keys/proxies to exercise a change:

```bash
# Unit tests (pytest; spins up its own infra via fixtures)
uv run pytest tests/

# Unified tests — step-based end-to-end flows defined in JSON
# (config hierarchy, multi-turn interactions, LLM-as-judge assertions)
uv run python -m tests.unified.run
uv run python -m tests.unified.run --test-dir tests/unified/test_cases

# Live LLM tests — real provider calls, for testing specific backends.
# Needs LLM_ANTHROPIC_API_KEY / LLM_OPENAI_API_KEY / LLM_GEMINI_API_KEY and
# LIVE_LLM_*_MODELS env vars; see tests/live_llm/README.md.
uv run pytest tests/live_llm -n 0 --live-llm --no-header -q
```
