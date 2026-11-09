---
name: verify
description: Build, launch, and drive a local Honcho stack to verify a change at its runtime surface (the /v3 HTTP API and the deriver queue), including SQL/search/index performance. Use when verifying a diff or confirming a change works in the running app.
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

## Performance-sensitive verification

If the diff changes SQL predicates, joins, parent-liveness checks, search/vector
ordering or limits, indexes, or partitioning, verify query plans as well as
results. Small functional fixtures and green CI can miss workspace-wide parent
scans and loss of pgvector HNSW plans.

1. **Compare exact code and data.** Record baseline/candidate commit SHAs and
   exercise their actual query paths, not hand-written approximations. Use
   isolated databases with equivalent seeded data, statistics, and settings;
   apply each revision's migrations and record intentional schema differences.
   Do not benchmark against shared development or production data.
2. **Vary parent and result cardinality separately.** Include a large workspace
   with many peers/sessions but a small queried collection, and a large
   conclusion collection. Cover zero, sparse, and dense parent/row tombstones,
   selective and broad filters, and colliding workspace/peer/session names in
   different tenants. When message search is affected, include large-session
   message/MessageEmbedding cases and their vector and text query plans too.
3. **Record the environment.** Save PostgreSQL/pgvector versions, relevant
   planner/HNSW settings, application DB role and RLS/tenant context, migration
   and index state, data/seed provenance, parent/collection/message counts per
   tenant/workspace, vector dimensions/distribution, and statistics freshness.
   Record warm-up and cache state; do not label a run cold without evidence.
4. **Capture the real plan repeatedly.** Save ORM-emitted SQL and bound
   parameters after tenant hooks, then run natural
   `EXPLAIN (ANALYZE, BUFFERS)` under the same role/settings for both revisions.
   Preserve predicates, vector distance ordering, filters, and limits. Compare
   repeated samples, not one best timing: inspect subplan loops, rows removed
   by filters, estimates versus actual rows, buffer hits/reads, and temporary
   I/O. Measure the full read path too, including any fast-path prechecks.
   Never force planner flags as proof of performance; diagnostic overrides
   must be disclosed and followed by natural-plan evidence.
5. **Check index use and pruning.** For representative vector sizes, inspect
   whether distance-ordered HNSW scans survive or become scans plus sorts;
   verify the distance operator/opclass and filter placement. Do not universally
   demand HNSW on tiny tables where a sequential scan may be cheaper. Check
   bound tenant predicates and partition pruning, including planning/partition
   lock evidence when relevant, not only which partitions execute. Supporting
   partial indexes need both migrations and matching `src/models.py`
   declarations; use `migrations.utils.create_index_online`/`drop_index_online`
   for large or partitioned tables. Verify applied predicates, valid indexes,
   and attached indexes on every relevant partition, not just the parent.
6. **Keep correctness gates.** Any fast path must preserve row/parent
   visibility, authorization, tenant isolation, ordering/pagination, and
   history semantics. Exercise concurrent deletion under the actual transaction
   isolation: a cached liveness result or precheck followed by a read in a
   different snapshot is not correctness proof.

Keep commands, SQL/plans, sample results, and provenance with the verification
report. Disclose synthetic vectors, reduced dimensions, mocks, and local
hardware/timings; distinguish plan evidence from real user-path validation.
One fixture is not a production benchmark or proof of recall/LLM quality.

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
