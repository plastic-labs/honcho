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

Working in a worktree? It carries neither `.env` nor `node_modules`. Copy
`.env` from the main checkout (that's where the provider keys live), and run
`bun install` in `sdks/typescript` if you'll run the full test suite —
otherwise the pre-push gate fails on a phantom `Cannot find package 'zod'`.

## Launch

First check whether the stack is already running via Docker Compose:

```bash
docker compose ps          # look for api / deriver / database / redis
docker compose up -d       # start it if not
```

A running stack is not the same as *your branch's code* running — check the
CREATED column; the images may be weeks old. For verifying a diff, the cheap
path is to reuse the stack's Postgres/Redis containers but run the branch's
API as a host process on a spare port:

```bash
uv run uvicorn src.main:app --port 8901   # branch code, stack's DB/Redis
uv run python -m src.deriver              # if the diff touches the worker
```

This avoids both an image rebuild and the port/project-name conflicts a second
compose stack in a worktree would cause. Without Docker at all, run the two
processes the same way against host Postgres + Redis (API default port 8000
via `uv run fastapi dev src/main.py`).

When reading server logs, note that with the main `.env` the telemetry emitter
spams connection warnings at an unreachable endpoint — `grep -v
telemetry.emitter` before looking for the real error.

## Drive it

Prefer driving through the `honcho-cli` skill; fall back to the SDKs
(`sdks/python`, `sdks/typescript`), then raw REST against `/v3`, in that order
when a method isn't available at the higher level. The `honcho-integration`
skill covers how to use the SDKs.

When a change updates endpoints or makes schema changes, verify across all
three surfaces — CLI, SDKs, and REST — since they can drift independently.

For LLM-path changes, the fastest synchronous surface is dialectic at the
`minimal` reasoning level: send a couple of messages, then hit
`/v3/.../peers/{peer_id}/chat` and observe the response.

## Configuration

Configuration is central to both driving the app and running tests: it's how
API keys reach the server and how a config-related change gets exercised at
all. Settings come from environment variables or files, with precedence
env > `.env` > `config.toml` > defaults. To verify a configuration change,
set the relevant option through one of these layers, restart the affected
process, and observe the behavioral difference at the surface — the same
mechanism lets you point provider base URLs, model choices, and timeouts at
credentials and proxies you actually have.

Two concrete levers: deep-nested settings (e.g.
`[dialectic.levels.minimal.model_config.overrides.provider_params]`) are
miserable as env vars — drop a partial `config.toml` in the repo root instead.
And for load-time config behavior, `uv run python -c "import src.config; ..."`
is faster than booting the server.

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
# Needs provider API keys AND model vars — without LIVE_LLM_*_MODELS the
# tests silently deselect that provider. See tests/live_llm/README.md.
export LLM_ANTHROPIC_API_KEY=...
export LIVE_LLM_ANTHROPIC_45_PLUS_MODELS=claude-sonnet-4-5
uv run pytest tests/live_llm -n 0 --live-llm --no-header -q
```
