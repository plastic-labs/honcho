# Rust Full Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the Honcho core runtime from Python FastAPI plus Python worker processes to Rust while preserving public API behavior, SDK compatibility, database compatibility, and safe rollback at every cutover point.

**Architecture:** The migration is incremental. Python remains authoritative until each Rust domain has contract parity, shadow validation, operational observability, and a rollback path. Rust services read and write the existing Postgres schema through explicit SQL first; Alembic and Python keep database ownership until the migration reaches the final cutover phase.

**Tech Stack:** Rust, Axum, Tokio, serde, sqlx, tracing, Postgres, pgvector, existing Alembic migrations, existing FastAPI routes as behavioral reference, pytest contract tests, Rust unit/integration tests, Docker Compose sidecars.

---

## Current Status

**Date:** 2026-06-16

**Overall migration status:** In progress. The project is not fully migrated to Rust.

**Current Rust milestone:** A standalone API sidecar exists under `api-rs/`. It is mostly read-only, with the first guarded write-shadow routes disabled by default behind `RUST_API_ENABLE_WRITES=false`.

**Current branch status:** Work is committed on branch `codex/rust-api-readonly-sidecar`. The sidecar milestone (read-only parity + guarded write shadows) was committed as `feat: add rust api sidecar with read-only parity and guarded write shadows`; the session-clone slice is the latest addition on top.

**Verified in the previous implementation pass:**

- `rtk cargo test --manifest-path mcp-rs/Cargo.toml` passed with 10 tests.
- `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 11 tests.
- `rtk uv run pytest tests/rust_api/ tests/routes/test_workspaces.py tests/routes/test_peers.py tests/routes/test_sessions.py` passed with 122 tests.
- `rtk uv run pytest tests/rust_api/ -q` passed with 3 Rust API contract tests after empty/null filter coverage was added.
- `rtk uv run ruff check tests/rust_api` passed.
- `docker build -t honcho-api-rs-test api-rs` passed.
- Current focused sidecar check: `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 29 tests.
- Current focused Python/Rust contract check: `rtk uv run pytest tests/rust_api/ -q` passed with 8 tests.
- Current webhook-list sidecar check: `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 31 tests.
- Current webhook-list Python/Rust contract check: `DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres rtk uv run pytest tests/rust_api/ -q` passed with 9 tests.
- Current key-route sidecar slice: `POST /v3/keys` implemented in `api-rs` as admin-only, DB-free JWT creation with Python-compatible disabled, validation, empty-scope, scope-claim, and expiration formatting behavior. `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 36 tests; `DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres rtk uv run pytest tests/rust_api/ -q` passed with 10 tests.
- Current guarded write-shadow slice: workspace mutations now include `POST /v3/workspaces` and `PUT /v3/workspaces/{workspace_id}` in `api-rs` behind `RUST_API_ENABLE_WRITES=false` by default. `POST` supports Python-compatible `name`/`id` aliasing, create-vs-existing status codes, metadata/configuration persistence, metadata NUL stripping and key/depth validation, selected workspace configuration validation/coercion, invalid-name validation shape, and mismatched or missing workspace-scope rejection. `PUT` mirrors Python get-or-create update behavior, including `200` on create/update, metadata replace/no-op semantics, top-level configuration merge, nested config normalization, ignored body `name`/`id`, path-name validation before DB access, and validation parity for selected invalid payloads. The Rust update SQL now avoids clobbering untouched columns by using `COALESCE` and JSONB top-level merge in the update statement.
- Current guarded peer write-shadow slice: peer mutations now include `POST /v3/workspaces/{workspace_id}/peers` and `PUT /v3/workspaces/{workspace_id}/peers/{peer_id}` in `api-rs` behind `RUST_API_ENABLE_WRITES=false` by default. `POST` mirrors Python get-or-create behavior, including workspace auto-create, `name`/`id` aliasing, `201` for new rows, `200` for existing rows, whole-object metadata/configuration replacement when provided, null/omitted no-op for existing rows, default `{}` values for new rows, metadata NUL stripping, validation parity for selected invalid payloads, and workspace/peer JWT scope checks. `PUT` mirrors Python update-or-create behavior, including missing workspace/peer creation, `200` status, ignored body `name`/`id`, metadata/configuration replace/no-op semantics, and persisted row parity. Route-level contract coverage now includes peer auth success/failure, peer POST default/no-op behavior, metadata key/depth validation, and Python/Rust persisted-row comparison. `api-rs` also parses Python-compatible `CACHE_ENABLED`, `CACHE_URL`, `CACHE_NAMESPACE`, and `NAMESPACE` settings and best-effort deletes Python's peer value cache key (`{CACHE_NAMESPACE}:v2:workspace:{workspace_name}:peer:{peer_name}`) after successful Rust peer writes. The Redis deletion path is now covered by an opt-in integration test using `HONCHO_API_RS_REDIS_TEST_URL`; it was verified against a real `redis:8.2` container at `redis://127.0.0.1:6379/2?suppress=true`. `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 55 tests; `HONCHO_API_RS_REDIS_TEST_URL='redis://127.0.0.1:6379/2?suppress=true' rtk cargo test --manifest-path api-rs/Cargo.toml peer_cache_invalidation_deletes_real_redis_value_key_when_configured` passed with 1 test; focused peer write contract tests passed with 6 tests; `DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres rtk uv run pytest tests/rust_api/ tests/routes/test_workspaces.py tests/routes/test_peers.py tests/routes/test_sessions.py -q` passed with 141 tests.
- Current guarded session write-shadow slice: session mutations now include no-peer `POST /v3/workspaces/{workspace_id}/sessions` and `PUT /v3/workspaces/{workspace_id}/sessions/{session_id}` in `api-rs` behind `RUST_API_ENABLE_WRITES=false` by default. `POST` mirrors Python get-or-create behavior for no-peer payloads, including workspace auto-create, `name`/`id` aliasing, `201` for new rows, `200` for existing rows, metadata replacement, typed `SessionConfiguration` validation/coercion, top-level configuration merge for existing rows, null/omitted no-op behavior, default `{}` values for new rows, inactive-session `404`, and workspace/session JWT scope checks. `PUT` mirrors Python update-or-create behavior, including missing workspace/session creation, `200` status, ignored body `name`/`id`, metadata replace/no-op/clear semantics, top-level configuration merge, inactive-session `404`, and persisted row parity. Embedded peer membership payloads on session create/update are still explicitly rejected in the Rust shadow; membership writes now start with the separate add-peers route below. Rust best-effort deletes Python's session value cache key (`{CACHE_NAMESPACE}:v2:workspace:{workspace_name}:session:{session_name}`) after successful Rust session writes; unlike Python create, it does not warm/set the cached value yet. `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 61 tests; focused session write contract tests passed with 4 tests; `DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres rtk uv run pytest tests/rust_api/ tests/routes/test_workspaces.py tests/routes/test_peers.py tests/routes/test_sessions.py` passed with 145 tests.
- Current guarded session membership slice: `POST`, `PUT`, and `DELETE /v3/workspaces/{workspace_id}/sessions/{session_id}/peers` plus `PUT /v3/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config` are implemented behind `RUST_API_ENABLE_WRITES=false` by default. Add/set routes accept Python's direct `dict[peer_id, SessionPeerConfig]` body shape and return `200` session JSON without embedding peers. Add-peers creates missing workspaces/sessions/peers through existing get-or-create paths, stores `observe_me`/`observe_others` session-peer configuration, preserves active membership configuration on re-add, rejoins previously-left peers by clearing `left_at` and replacing configuration, and enforces the observer limit before writes. Set-peers requires an existing active session, soft-deletes all active memberships by setting `left_at`, auto-creates requested peers, re-adds only the requested peer set with incoming configuration, and returns Python-compatible `404` for missing sessions. Remove-peers accepts Python's bare `list[str]` body, collapses duplicate peer IDs, verifies the session is active, soft-deletes only matching active memberships, ignores missing/non-member/already-left peers, and returns Python-compatible `404` for missing or inactive sessions. Peer-config writes require an existing active session and existing peer, insert a missing membership, partially merge submitted non-null config keys, preserve omitted/null keys, do not rejoin previously-left memberships, enforce the active-observer limit, and return Python-compatible `204` empty responses plus `404` errors. Membership writes best-effort invalidate the session value cache key and affected peer value cache keys; they do not warm Python cache values. `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 65 tests; focused add/set/remove/config-peers contract tests passed with 10 tests total; `DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres rtk uv run pytest tests/rust_api/ tests/routes/test_workspaces.py tests/routes/test_peers.py tests/routes/test_sessions.py` passed with 155 tests.
- Current guarded session clone slice: `POST /v3/workspaces/{workspace_id}/sessions/{session_id}/clone` is implemented behind `RUST_API_ENABLE_WRITES=false` by default. The Rust route mirrors Python `crud.clone_session`: it requires an active original session (`404 {"detail":"Original session not found"}` otherwise), validates an optional `message_id` cutoff against the source session (`404 {"detail":"Session not found"}` on a bad cutoff), creates a new session with a fresh nanoid name copying the original metadata/configuration, copies messages (optionally truncated at and including the cutoff id) with fresh public ids and the Python-default `token_count=0` plus default `internal_metadata`, and copies session-peer memberships with their configuration only when at least one message is cloned (matching Python's early return for empty sources). The Rust copy adds explicit workspace scoping to the message/peer reads that Python omits. No cache invalidation is needed (the clone is a brand-new session, read-through like Python). Known inherited quirk not yet contract-tested: Python's no-message clone returns before committing and skips peer copy; the Rust shadow always commits the new empty session. `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 67 tests; focused clone contract tests passed with 3 tests; `DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres rtk uv run pytest tests/rust_api/ -q` passed with 41 tests.
- Current guarded session delete slice: `DELETE /v3/workspaces/{workspace_id}/sessions/{session_id}` is implemented behind `RUST_API_ENABLE_WRITES=false` by default. The Rust route mirrors the Python HTTP route's fast path, not the worker hard-delete path: it requires an active session, sets `sessions.is_active=false`, inserts a `queue` row with `task_type='deletion'`, `work_unit_key='deletion:{workspace}:session:{session}'`, `session_id=NULL`, `message_id=NULL`, and payload including `task_type`, `deletion_type`, and `resource_id`, returns `202 {"message":"Session deleted successfully"}`, invalidates the Python session value cache key, and returns Python-compatible `404` for missing or inactive sessions. The asynchronous hard delete of messages, message embeddings, documents, session peers, and the session row remains Python-worker owned until the Rust worker migration phase. `rtk cargo test --manifest-path api-rs/Cargo.toml` passed with 65 tests; focused delete-session contract tests passed with 2 tests; `DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres rtk uv run pytest tests/rust_api/ tests/routes/test_workspaces.py tests/routes/test_peers.py tests/routes/test_sessions.py` passed with 157 tests.
- Current guarded-write blocker: broader Python cache parity is not complete for all cached resources. Peer writes, session create/update, and session membership writes now have best-effort Rust invalidation, but workspace cache behavior, cache warming, collection cache behavior, and all future write routes still need cache-specific parity before mixed Python/Rust traffic.

**Resolved (2026-06-16):** `WorkspaceConfiguration.reasoning.custom_instructions`
token-budget validation now has exact Rust/Python parity. The Rust sidecar uses
`tiktoken-rs`' bundled `o200k_base` ranks via `api-rs/src/tokens.rs`
(`estimate_tokens`, mirroring Python `src/utils/tokens.py`), replacing the old
`max(words, chars/4)` heuristic in `validate_custom_instructions`. Parity is
contract-tested at the 2000-token boundary, including the two cases the old
heuristic got wrong: a high-character/low-token string the char heuristic
over-counted (accepted by both now) and a single-"word"/high-token string the
whitespace heuristic under-counted (rejected by both now). The token budget
itself is still hardcoded to the Python default (2000); making it read
`DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS` is a separate, smaller change. This same
`tokens::estimate_tokens` is the prerequisite for message `token_count` parity in
the deferred message-write slice.

## Status Matrix

| Area | Python Status | Rust Status | Migration Status | Cutover State |
| --- | --- | --- | --- | --- |
| `mcp-rs` | Python/MCP behavior remains separate | Existing Rust crate | Baseline complete | No API cutover involved |
| API process bootstrap | FastAPI authoritative | `api-rs` sidecar exists | Partial | Sidecar only |
| Health endpoint | FastAPI authoritative | `GET /health` implemented | Complete for sidecar | Not routed publicly |
| Config parsing | Python authoritative | `RUST_API_BIND_ADDRESS`, `DB_CONNECTION_URI`, `AUTH_USE_AUTH`, `AUTH_JWT_SECRET`, `DB_SCHEMA`, `RUST_API_ENABLE_WRITES`, `CACHE_ENABLED`, `CACHE_URL`, `CACHE_NAMESPACE`, and `NAMESPACE` implemented | Partial | Sidecar only |
| JWT auth | `src/security.py` authoritative | HS256 validation and scope checks implemented | Partial | Sidecar only |
| Workspace list | FastAPI authoritative | Implemented | Contract-tested | Not routed publicly |
| Peer list | FastAPI authoritative | Implemented | Contract-tested | Not routed publicly |
| Session list | FastAPI authoritative | Implemented | Contract-tested | Not routed publicly |
| Peer sessions list | FastAPI authoritative | Implemented | Contract-tested | Not routed publicly |
| Session peer reads/writes | FastAPI authoritative | Peer list/config reads plus add, set, remove peers, and peer config writes implemented | Partial / in progress | Sidecar only; no public cutover |
| Session summaries | FastAPI authoritative | Implemented | Contract-tested | Sidecar only; summary generation remains Python worker only |
| Queue status | FastAPI authoritative | Implemented | Contract-tested | Not routed publicly |
| Filtering | Python utility authoritative | Compatible SQL builder exists for current read endpoints, including JSONB metadata, datetime, `in`, and escaped contains cases | Partial | Needs broad endpoint coverage |
| Pagination | FastAPI/Pydantic authoritative | Matching page shape exists | Partial | Needs broad endpoint coverage |
| Create/update/delete routes | FastAPI authoritative | Workspace, peer, no-peer session get-or-create/update, add/set/remove session membership, peer config membership writes, session soft-delete/enqueue, and session clone implemented behind `RUST_API_ENABLE_WRITES`; peer/session writes best-effort invalidate Python value cache keys when cache is enabled; worker hard-delete and remaining relationship mutations not implemented | Partial / in progress | Sidecar shadow only; Python remains public write path |
| Message routes | FastAPI authoritative | Read-only list/get implemented | Partial / in progress | Sidecar only; writes remain Python only |
| File upload routes | FastAPI authoritative | Not implemented | Not started | Python only |
| Search routes | FastAPI authoritative | Not implemented | Not started | Python only |
| Conclusion routes | FastAPI authoritative | Read-only list implemented | Partial / in progress | Sidecar only; query and writes remain Python only |
| Representation routes | FastAPI authoritative | Not implemented | Not started | Python only |
| Peer card routes | FastAPI authoritative | Read-only get implemented | Partial / in progress | Sidecar only; peer-card writes remain Python only |
| Key routes | FastAPI authoritative | `POST /v3/keys` implemented | Rust contract-tested; Python/Rust focused test added and DB-gated | Sidecar only |
| Webhook routes | FastAPI authoritative | Read-only list implemented | Partial / in progress | Sidecar only; create/delete/test remain Python only |
| Queue enqueue paths | Python authoritative | Not implemented | Not started | Python only |
| Deriver worker | Python authoritative | Not implemented | Not started | Python only |
| Reconciler worker | Python authoritative | Not implemented | Not started | Python only |
| Summarizer | Python authoritative | Not implemented | Not started | Python only |
| Dreamer | Python authoritative | Not implemented | Not started | Python only |
| Dialectic chat | Python authoritative | Not implemented | Not started | Python only |
| LLM provider layer | Python authoritative | Not implemented | Not started | Python only |
| Embeddings | Python authoritative | Not implemented | Not started | Python only |
| pgvector paths | Python authoritative | Not implemented | Not started | Python only |
| External vector stores | Python authoritative | Not implemented | Not started | Python only |
| Telemetry events | Python authoritative | Not implemented | Not started | Python only |
| SDK compatibility | Python API authoritative | Contract checks only | Partial | Python remains public surface |
| Deployment | Python API authoritative | Optional Compose sidecar | Partial | No proxying |
| Database migrations | Alembic authoritative | Not owned by Rust | Deferred | Python owns schema |

## Non-Negotiable Cutover Gates

- [ ] Every Rust endpoint has Python parity contract tests for status codes, JSON shape, aliases, pagination, null handling, empty filters, auth scopes, error payloads, and ordering.
- [ ] Rust write paths are shadow-tested against disposable Postgres fixtures before any production traffic reaches them.
- [ ] No Rust route becomes the public route until it has an explicit rollback switch to Python.
- [ ] Database schema changes remain in Alembic until a separate migration-ownership plan is approved.
- [ ] Rust code never calls external LLM, embedding, or webhook providers while holding a database transaction.
- [ ] Rust read-only paths use read-only database access where practical.
- [ ] Rust write paths that update Python-cached resources invalidate the matching Python cache keys before any mixed Python/Rust shadow traffic.
- [ ] All cutover phases include route-level metrics, structured logs, and error-rate dashboards.
- [ ] SDK tests pass unchanged against Python and Rust for any route proposed for cutover.

## Repository Layout Target

Use this layout only when the number of Rust crates makes the current standalone structure hard to maintain:

```text
honcho-rs/
├── api-rs/                 # Axum HTTP service; already exists
├── mcp-rs/                 # Existing Rust MCP crate; already exists
├── worker-rs/              # Future queue consumer, reconciler, deriver, summarizer, dreamer
├── crates/
│   ├── honcho-config/      # Shared env/config parsing
│   ├── honcho-auth/        # Shared JWT and auth scope logic
│   ├── honcho-db/          # Shared sqlx pool, query helpers, JSON models
│   ├── honcho-domain/      # Shared public API/domain structs
│   ├── honcho-llm/         # Provider-agnostic LLM layer
│   └── honcho-telemetry/   # Shared tracing and CloudEvents emitters
└── Cargo.toml              # Root workspace only after api-rs and worker-rs both need shared crates
```

Root workspace conversion is intentionally deferred. The current `api-rs/` crate should remain standalone until shared Rust crates reduce real duplication.

## Phase 0: Stabilize The Existing Rust Sidecar

**Status:** In progress.

**Files:**

- Existing: `api-rs/Cargo.toml`
- Existing: `api-rs/src/app.rs`
- Existing: `api-rs/src/auth.rs`
- Existing: `api-rs/src/config.rs`
- Existing: `api-rs/src/db.rs`
- Existing: `api-rs/src/error.rs`
- Existing: `api-rs/src/filters.rs`
- Existing: `api-rs/src/main.rs`
- Existing: `api-rs/src/pagination.rs`
- Existing: `api-rs/src/queue_status.rs`
- Existing: `api-rs/tests/contracts.rs`
- Existing: `api-rs/tests/http.rs`
- Existing: `tests/rust_api/test_readonly_contract.py`
- Existing: `docker-compose-rs.yml.example`

- [x] **Step 1: Implement the standalone Rust API sidecar**

  The sidecar exists under `api-rs/` and runs independently from the Python FastAPI application.

- [x] **Step 2: Implement the first read-only endpoint set**

  Implemented routes:

  ```text
  GET  /health
  POST /v3/workspaces/list
  POST /v3/workspaces/{workspace_id}/peers/list
  POST /v3/workspaces/{workspace_id}/sessions/list
  POST /v3/workspaces/{workspace_id}/peers/{peer_id}/sessions
  GET  /v3/workspaces/{workspace_id}/queue/status
  ```

- [x] **Step 3: Add Rust unit and integration tests**

  Existing checks cover config parsing, auth decisions, pagination defaults, filter SQL construction, queue status shaping, and health routing.

- [x] **Step 4: Add Python contract tests**

  Existing tests under `tests/rust_api/` start the Rust sidecar, seed the existing pytest database fixture, call Python and Rust, and compare selected responses.

- [x] **Step 5: Add Docker sidecar entry**

  `docker-compose-rs.yml.example` includes optional `api-rs` sidecar configuration on a separate local port.

- [x] **Step 6: Commit the sidecar milestone**

  Committed as `feat: add rust api sidecar with read-only parity and guarded write shadows`
  (commit `3bf8d96`), containing the `api-rs/` crate, `tests/rust_api/` contract
  suite, the Compose example, and this migration tracker. The commit message
  reflects the actual scope, which had already grown past read-only into guarded
  write shadows by the time of commit.

## Phase 1: Expand Read-Only API Parity

**Status:** In progress.

**2026-06-15 update:** Rust read-only parity now includes message list/get,
session peer list/config, session summaries, peer-card get, conclusion list,
and webhook list endpoints. The focused Python/Rust contract suite can run
locally with a temporary pgvector Postgres and explicit `DB_CONNECTION_URI`.
This is not full message, session, peer-card, conclusion, or webhook route
migration; message writes, uploads, updates, search, session mutations,
context, summary generation, peer-card writes, conclusion query, conclusion
writes, and webhook create/delete/test remain Python-only.

Known inherited behavior: session peer list pagination currently follows
Python's unordered query. Define and contract-test an explicit ordering policy
before any public cutover of this route.

Known inherited behavior: webhook list pagination currently follows Python's
unordered query. Keep the sidecar contract-compatible for now, but define and
contract-test an explicit ordering policy before any public cutover.

Known inherited behavior: conclusion list pagination orders only by
`created_at`, matching Python's current `get_documents_with_filters` query.
Do not add a Rust-only tie-breaker before cutover; define an explicit public
tie-breaker policy and update Python/Rust contracts together.

**Goal:** Cover all read-only API behavior before adding writes.

**Reference files:**

- `src/routers/workspaces.py`
- `src/routers/peers.py`
- `src/routers/sessions.py`
- `src/routers/messages.py`
- `src/routers/conclusions.py`
- `src/routers/keys.py`
- `src/routers/webhooks.py`
- `src/schemas/api.py`
- `src/utils/filter.py`
- `tests/routes/`

**Rust files:**

- Modify: `api-rs/src/app.rs`
- Modify: `api-rs/src/db.rs`
- Modify: `api-rs/src/filters.rs`
- Modify: `api-rs/src/error.rs`
- Modify: `api-rs/tests/contracts.rs`
- Modify: `tests/rust_api/test_readonly_contract.py`

- [ ] **Step 1: Inventory every read-only v3 route**

  Run:

  ```bash
  rg '@router\\.(get|post)' src/routers
  rg 'def test_.*(list|get|search)' tests/routes
  ```

  Expected: a route inventory grouped by workspace, peer, session, message, conclusion, key, and webhook domains.

- [ ] **Step 2: Add contract tests before each Rust route**

  For each route, add a Python test that calls the existing FastAPI `TestClient` and the Rust sidecar with identical seed data.

  Required assertions:

  ```python
  assert rust_response.status_code == python_response.status_code
  assert rust_response.json() == python_response.json()
  ```

- [ ] **Step 3: Implement missing read-only SQL in Rust**

  Use explicit `sqlx` queries. Keep SQL local to the domain function and include workspace scoping in every query predicate.

- [ ] **Step 4: Run read-only parity checks**

  Run:

  ```bash
  rtk cargo test --manifest-path api-rs/Cargo.toml
  rtk uv run pytest tests/rust_api/ tests/routes/test_workspaces.py tests/routes/test_peers.py tests/routes/test_sessions.py tests/routes/test_messages.py tests/routes/test_conclusions.py
  ```

  Expected: all selected Rust and Python tests pass without changing public response shapes.

- [ ] **Step 5: Commit read-only parity**

  Run:

  ```bash
  git add api-rs tests/rust_api
  git commit -m "feat: expand rust read-only api parity"
  ```

## Phase 2: Shared Rust Foundations

**Status:** In progress. Webhook list is implemented; key creation route is implemented in the sidecar with Rust verification and a DB-gated Python/Rust parity test.

**Goal:** Extract shared Rust logic only after `api-rs` grows enough duplication to justify it.

**Files:**

- Create later: `crates/honcho-config/`
- Create later: `crates/honcho-auth/`
- Create later: `crates/honcho-db/`
- Create later: `crates/honcho-domain/`
- Modify later: `api-rs/Cargo.toml`
- Modify later: root `Cargo.toml` if a workspace becomes justified

- [ ] **Step 1: Measure duplication**

  Run:

  ```bash
  rg 'AUTH_|DB_|Pagination|Claims|Pool|PgPool|Workspace' api-rs/src
  ```

  Expected: enough repeated config/auth/domain code to justify shared crates.

- [ ] **Step 2: Create shared crates only when needed**

  Extract stable modules in this order:

  ```text
  honcho-config
  honcho-auth
  honcho-domain
  honcho-db
  honcho-telemetry
  ```

- [ ] **Step 3: Keep public behavior unchanged**

  Run:

  ```bash
  rtk cargo test --manifest-path api-rs/Cargo.toml
  rtk uv run pytest tests/rust_api/
  ```

  Expected: refactor-only change; no JSON or status-code differences.

- [ ] **Step 4: Commit shared foundations**

  Run:

  ```bash
  git add api-rs crates Cargo.toml Cargo.lock
  git commit -m "refactor: extract shared rust foundations"
  ```

## Phase 3: Write API Parity Behind A Shadow Gate

**Status:** Partial. Sidecar-only webhook list and key creation exist.
Workspace, peer, no-peer session get-or-create/update, session soft-delete/enqueue, session clone, add/set/remove-peers session membership routes, and peer config membership writes are implemented behind
`RUST_API_ENABLE_WRITES=false` by default and route-contract-tested against Python.
Workspace coverage includes create/existing behavior, `id` aliasing, metadata
sanitization, selected configuration validation/coercion, unknown nested
configuration key dropping, validation failures, persisted rows,
update-as-create, metadata no-op/replace, top-level configuration merge,
ignored update body `name`/`id`, and mismatched/missing workspace-scope
rejection. Peer coverage includes workspace auto-create, peer create/existing
status codes, `id` aliasing, whole-object metadata/configuration replacement,
null/omitted no-op behavior for existing rows, default `{}` values for new
rows, update-as-create, validation failures, persisted rows, ignored update
body `name`/`id`, workspace/peer scope rejection, and Python/Rust row parity
for public peer fields. Session coverage includes no-peer create/existing
status codes, workspace auto-create, update-as-create, metadata sanitization,
metadata replace/no-op/clear semantics, typed session configuration
validation/coercion, top-level configuration merge, inactive-session `404`,
workspace/session scope rejection, and Python/Rust row parity for public
session fields. Session delete coverage includes Python-compatible `202`
success response, `is_active=false` persistence, deletion queue payload shape,
missing/inactive-session `404`, and session cache invalidation. Session
membership coverage includes Python's direct peer map
body shape, missing peer creation, session-peer configuration persistence,
add-peers active membership config preservation, previously-left peer rejoin
behavior, set-peers soft replacement of the active membership set, set-peers
missing-session `404`, remove-peers soft deletion/idempotent missing-peer behavior,
peer-config `204` empty responses, peer-config missing-member insertion,
partial config merge with submitted nulls excluded, peer-config updates that do
not rejoin left memberships, missing peer/session `404` parity, and
observer-limit enforcement. Rust still rejects embedded session peer
membership payloads on session create/update until those code paths support the
same semantics. Exact
`WorkspaceConfiguration.reasoning.custom_instructions` token-budget parity is now
implemented via `tiktoken-rs` `o200k_base` in `api-rs/src/tokens.rs` and
boundary-contract-tested (see Resolved note above).

Message creation remains deferred (Phase 3 Step 3 / Phase 4): it is queue-coupled
(enqueue payloads + queue rows) and has two further parity dependencies — the now
available `tokens::estimate_tokens` for `token_count`, and `EMBED_MESSAGES=true`
(default) which writes `MessageEmbedding` pending rows via Python's chunking that
must be ported before persisted-row parity is possible. Peer write shadow now
best-effort invalidates Python's peer value cache key after successful Rust
create/update when cache is enabled. Session write shadow similarly deletes the
Python session value cache key after successful Rust session writes, but does
not yet warm/set the cached value like Python create can. Add-peers also
deletes the affected peer value cache keys. All other mutations, webhook
mutation/delivery parity, complete scoped-key contract coverage, telemetry
emitters, broader cache behavior, and operational cutover gates are not
started.

**Goal:** Implement create/update/delete endpoints in Rust while Python remains the public write path.

**Reference files:**

- `src/routers/workspaces.py`
- `src/routers/peers.py`
- `src/routers/sessions.py`
- `src/routers/messages.py`
- `src/routers/conclusions.py`
- `src/crud/workspace.py`
- `src/crud/peer.py`
- `src/crud/session.py`
- `src/crud/message.py`
- `src/crud/collection.py`
- `src/crud/document.py`
- `src/deriver/enqueue.py`
- `tests/routes/`

**Rust files:**

- Modify: `api-rs/src/app.rs`
- Modify: `api-rs/src/db.rs`
- Create or modify domain modules under `api-rs/src/`
- Modify: `tests/rust_api/`

- [ ] **Step 1: Add disposable database contract tests for mutations**

  Initial coverage exists for `POST /v3/workspaces`. Broader mutation routes
  still need the same pattern:

  ```text
  1. Seed a clean database.
  2. Execute the Python route.
  3. Snapshot relevant rows.
  4. Reset the database.
  5. Execute the Rust route.
  6. Snapshot relevant rows.
  7. Assert matching response JSON and matching persisted rows.
  ```

- [ ] **Step 2: Implement workspace, peer, and session mutations first**

  Start with lower-risk mutations:

  ```text
  get-or-create workspace [done behind RUST_API_ENABLE_WRITES]
  update workspace [done behind RUST_API_ENABLE_WRITES]
  get-or-create peer [route-contract done behind RUST_API_ENABLE_WRITES; peer value cache invalidation implemented]
  update peer [route-contract done behind RUST_API_ENABLE_WRITES; peer value cache invalidation implemented]
  get-or-create session [no-peer route-contract done behind RUST_API_ENABLE_WRITES; session value cache invalidation implemented]
  update session [no-peer route-contract done behind RUST_API_ENABLE_WRITES; session value cache invalidation implemented]
  add peers to session [route-contract done behind RUST_API_ENABLE_WRITES; session and affected peer value cache invalidation implemented]
  set session peers [route-contract done behind RUST_API_ENABLE_WRITES; session and affected peer value cache invalidation implemented]
  remove peers from session [route-contract done behind RUST_API_ENABLE_WRITES; session and affected peer value cache invalidation implemented]
  set peer config [route-contract done behind RUST_API_ENABLE_WRITES; session and affected peer value cache invalidation implemented]
  delete session [HTTP soft-delete/enqueue route-contract done behind RUST_API_ENABLE_WRITES; worker hard-delete remains Python-owned]
  clone session [route-contract done behind RUST_API_ENABLE_WRITES; copies session/messages/peers, no cache invalidation needed]
  ```

- [ ] **Step 3: Implement message and conclusion mutations after queue behavior is specified**

  Message creation must preserve enqueue behavior. Conclusion creation and deletion must preserve collection/document semantics.

- [x] **Step 4: Add a Rust write shadow mode**

  Implemented:

  ```text
  RUST_API_ENABLE_WRITES=false
  ```

  Default is disabled. A disabled write route returns a controlled 405 error instead of mutating data.

- [ ] **Step 5: Run mutation parity tests**

  Run:

  ```bash
  rtk cargo test --manifest-path api-rs/Cargo.toml
  rtk uv run pytest tests/rust_api/ tests/routes/
  ```

  Expected: mutation tests prove equivalent response bodies and equivalent persisted rows.

- [ ] **Step 6: Commit mutation parity**

  Run:

  ```bash
  git add api-rs tests/rust_api
  git commit -m "feat: add shadowed rust write api parity"
  ```

## Phase 4: Queue Producer And Worker Runtime

**Status:** Not started.

**Goal:** Move queue production and consumption to Rust without changing task ordering or retry semantics.

**Reference files:**

- `src/deriver/enqueue.py`
- `src/deriver/queue_manager.py`
- `src/deriver/consumer.py`
- `src/reconciler/scheduler.py`
- `src/reconciler/queue_cleanup.py`
- `src/reconciler/sync_vectors.py`
- `src/schemas/internal.py`
- `src/utils/queue_payload.py`
- `tests/deriver/`

**Rust files:**

- Create later: `worker-rs/Cargo.toml`
- Create later: `worker-rs/src/main.rs`
- Create later: `worker-rs/src/queue.rs`
- Create later: `worker-rs/src/consumer.rs`
- Create later: `worker-rs/src/reconciler.rs`
- Create later: `worker-rs/tests/`
- Modify later: `docker-compose-rs.yml.example`

- [ ] **Step 1: Document queue item schema and state transitions**

  Extract exact payload fields from `src/schemas/internal.py` and queue state changes from `src/deriver/queue_manager.py`.

- [ ] **Step 2: Add Rust worker tests for claim, retry, completion, and stale cleanup**

  Tests must cover:

  ```text
  pending -> processing
  processing -> completed
  processing -> failed with retry
  stale processing -> pending
  workspace/session ordering
  ```

- [ ] **Step 3: Implement Rust queue consumer without LLM work**

  First worker milestone must claim and complete no-op tasks in fixture data.

- [ ] **Step 4: Implement queue producer compatibility**

  Rust API message routes must enqueue the same payloads as Python before public write cutover.

- [ ] **Step 5: Run worker tests**

  Run:

  ```bash
  rtk cargo test --manifest-path worker-rs/Cargo.toml
  rtk uv run pytest tests/deriver/
  ```

  Expected: Rust queue behavior matches Python queue behavior for database state transitions.

- [ ] **Step 6: Commit queue runtime**

  Run:

  ```bash
  git add worker-rs docker-compose-rs.yml.example tests/rust_api
  git commit -m "feat: add rust queue worker runtime"
  ```

## Phase 5: Deriver, Summarizer, And Reconciler

**Status:** Not started.

**Goal:** Port background memory processing while keeping Python as the rollback worker.

**Reference files:**

- `src/deriver/deriver.py`
- `src/deriver/prompts.py`
- `src/utils/summarizer.py`
- `src/reconciler/sync_vectors.py`
- `src/embedding_client.py`
- `src/models.py`
- `tests/deriver/`
- `tests/reconciler/` if added before this phase

**Rust files:**

- Modify later: `worker-rs/src/consumer.rs`
- Create later: `worker-rs/src/deriver.rs`
- Create later: `worker-rs/src/summarizer.rs`
- Create later: `worker-rs/src/embeddings.rs`
- Create later: `worker-rs/src/reconciler.rs`

- [ ] **Step 1: Port prompt assembly exactly**

  Keep the Rust prompt strings byte-for-byte compatible unless a separate prompt migration is approved.

- [ ] **Step 2: Add golden tests for derivation input and structured output parsing**

  Golden fixtures must include:

  ```text
  direct conclusion extraction
  deductive conclusion extraction
  malformed structured output
  custom instructions token budget
  empty message batch
  ```

- [ ] **Step 3: Implement summarizer parity**

  Preserve short and long summary thresholds from Python config.

- [ ] **Step 4: Implement reconciler embedding sync**

  Preserve `MessageEmbedding.sync_state` transitions and retry handling.

- [ ] **Step 5: Run background parity checks**

  Run:

  ```bash
  rtk cargo test --manifest-path worker-rs/Cargo.toml
  rtk uv run pytest tests/deriver/ tests/reconciler/
  ```

  Expected: Rust worker produces equivalent database side effects for fixture batches.

- [ ] **Step 6: Commit background processors**

  Run:

  ```bash
  git add worker-rs tests
  git commit -m "feat: port deriver summarizer and reconciler to rust"
  ```

## Phase 6: LLM Provider Layer

**Status:** Not started.

**Goal:** Port provider-agnostic LLM calls, fallback behavior, tool-loop plumbing, structured output, and prompt caching behavior.

**Reference files:**

- `src/llm/api.py`
- `src/llm/backend.py`
- `src/llm/executor.py`
- `src/llm/runtime.py`
- `src/llm/request_builder.py`
- `src/llm/registry.py`
- `src/llm/structured_output.py`
- `src/llm/tool_loop.py`
- `src/llm/backends/anthropic.py`
- `src/llm/backends/gemini.py`
- `src/llm/backends/openai.py`
- `src/config.py`
- `tests/llm/` if present or added before this phase

**Rust files:**

- Create later: `crates/honcho-llm/`
- Modify later: `worker-rs/src/deriver.rs`
- Modify later: `api-rs` chat paths after Phase 7

- [ ] **Step 1: Define Rust LLM trait boundaries**

  Required interfaces:

  ```text
  build request
  send provider request
  parse structured response
  stream response chunks
  execute fallback attempt plan
  record usage and cost attribution
  ```

- [ ] **Step 2: Add provider request golden tests**

  For each provider backend, assert generated HTTP payloads match Python fixtures.

- [ ] **Step 3: Add fallback-chain tests**

  Cover primary success, primary retry, primary failure to fallback, stream-final retry, and pinned attempt-plan behavior.

- [ ] **Step 4: Implement providers one at a time**

  Order:

  ```text
  OpenAI
  Anthropic
  Gemini
  ```

- [ ] **Step 5: Run LLM tests without live provider calls**

  Run:

  ```bash
  rtk cargo test --manifest-path crates/honcho-llm/Cargo.toml
  rtk uv run pytest tests/llm/
  ```

  Expected: mocked provider tests pass and no live credentials are required.

- [ ] **Step 6: Commit Rust LLM layer**

  Run:

  ```bash
  git add crates/honcho-llm worker-rs tests
  git commit -m "feat: add rust llm provider layer"
  ```

## Phase 7: Search, Representation, Conclusions, And Vector Stores

**Status:** Not started.

**Goal:** Port the memory retrieval and representation features that depend on Postgres FTS, pgvector, collections, documents, and optional external vector stores.

**Reference files:**

- `src/crud/collection.py`
- `src/crud/document.py`
- `src/crud/message.py`
- `src/crud/representation.py`
- `src/routers/conclusions.py`
- `src/routers/peers.py`
- `src/utils/search.py`
- `src/vector_store/lancedb.py`
- `src/vector_store/turbopuffer.py`
- `tests/routes/test_conclusions.py`
- `tests/routes/test_peers.py`

**Rust files:**

- Modify later: `api-rs/src/app.rs`
- Modify later: `api-rs/src/db.rs`
- Create later: `api-rs/src/search.rs`
- Create later: `api-rs/src/representation.rs`
- Create later: `api-rs/src/vector_store.rs`

- [ ] **Step 1: Add contract tests for search and representation routes**

  Include tests for:

  ```text
  text search
  vector search
  hybrid search
  observer/observed collection scoping
  local representation
  global representation
  reasoning configuration
  ```

- [ ] **Step 2: Port Postgres FTS and pgvector SQL**

  Keep ranking, limits, workspace scoping, and metadata filters compatible with Python behavior.

- [ ] **Step 3: Port conclusion CRUD**

  Preserve public "conclusion" naming even when internal Python code uses "observation" symbols.

- [ ] **Step 4: Port external vector store adapters after pgvector parity**

  Add Turbopuffer and LanceDB only after pgvector contract tests pass.

- [ ] **Step 5: Run retrieval parity checks**

  Run:

  ```bash
  rtk cargo test --manifest-path api-rs/Cargo.toml
  rtk uv run pytest tests/rust_api/ tests/routes/test_conclusions.py tests/routes/test_peers.py tests/routes/test_messages.py
  ```

  Expected: Python and Rust return equivalent retrieval results for deterministic fixtures.

- [ ] **Step 6: Commit retrieval and representation parity**

  Run:

  ```bash
  git add api-rs tests/rust_api
  git commit -m "feat: port rust search and representation routes"
  ```

## Phase 8: Dialectic Chat And Streaming

**Status:** Not started.

**Goal:** Port synchronous chat, tool-loop execution, context retrieval, and SSE streaming.

**Reference files:**

- `src/dialectic/chat.py`
- `src/dialectic/core.py`
- `src/dialectic/prompts.py`
- `src/utils/agent_tools.py`
- `src/routers/peers.py`
- `tests/dialectic/`
- `tests/routes/test_peers.py`

**Rust files:**

- Modify later: `api-rs/src/app.rs`
- Create later: `api-rs/src/dialectic.rs`
- Create later: `api-rs/src/streaming.rs`
- Modify later: `crates/honcho-llm/`

- [ ] **Step 1: Add non-streaming chat contract tests**

  Mock the LLM layer and assert tool call order, request shape, response shape, auth behavior, and database access.

- [ ] **Step 2: Add SSE contract tests**

  Assert event names, chunk ordering, final event behavior, retry behavior, and error event shape.

- [ ] **Step 3: Port the minimal reasoning tool set first**

  Implement:

  ```text
  search_memory
  search_messages
  ```

- [ ] **Step 4: Port the full Dialectic tool set**

  Implement:

  ```text
  get_observation_context
  grep_messages
  get_messages_by_date_range
  search_messages_temporal
  get_reasoning_chain
  ```

- [ ] **Step 5: Run chat parity checks**

  Run:

  ```bash
  rtk cargo test --manifest-path api-rs/Cargo.toml
  rtk uv run pytest tests/dialectic/ tests/routes/test_peers.py tests/rust_api/
  ```

  Expected: mocked chat behavior and streaming shape match Python.

- [ ] **Step 6: Commit Dialectic parity**

  Run:

  ```bash
  git add api-rs crates tests
  git commit -m "feat: port dialectic chat to rust"
  ```

## Phase 9: Dreamer And Advanced Memory Consolidation

**Status:** Not started.

**Goal:** Port scheduled dreams, specialist agents, reasoning trees, surprisal scoring, and peer-card updates.

**Reference files:**

- `src/dreamer/orchestrator.py`
- `src/dreamer/dream_scheduler.py`
- `src/dreamer/specialists.py`
- `src/dreamer/surprisal.py`
- `src/dreamer/trees/`
- `src/crud/peer_card.py`
- `tests/dreamer/` if present or added before this phase

**Rust files:**

- Modify later: `worker-rs/src/consumer.rs`
- Create later: `worker-rs/src/dreamer.rs`
- Create later: `worker-rs/src/reasoning_tree.rs`
- Create later: `worker-rs/src/peer_card.rs`

- [ ] **Step 1: Add reasoning-tree fixture tests**

  Cover premise links, downstream links, tree traversal, and reasoning-chain retrieval.

- [ ] **Step 2: Add surprisal scoring tests**

  Use deterministic fixtures and assert exact selected conclusion IDs.

- [ ] **Step 3: Port specialist tool execution**

  Preserve explicit, deductive, and inductive conclusion creation semantics.

- [ ] **Step 4: Port peer-card updates**

  Preserve peer-card JSON shape and update timing.

- [ ] **Step 5: Run dreamer parity checks**

  Run:

  ```bash
  rtk cargo test --manifest-path worker-rs/Cargo.toml
  rtk uv run pytest tests/dreamer/ tests/rust_api/
  ```

  Expected: deterministic dream fixtures produce equivalent database side effects.

- [ ] **Step 6: Commit Dreamer parity**

  Run:

  ```bash
  git add worker-rs tests
  git commit -m "feat: port dreamer to rust"
  ```

## Phase 10: Webhooks, Keys, Telemetry, And Operational Parity

**Status:** Not started.

**Goal:** Port remaining operational surfaces and production observability before any public cutover.

**Reference files:**

- `src/routers/webhooks.py`
- `src/routers/keys.py`
- `src/webhooks/`
- `src/telemetry/`
- `src/cache/`
- `tests/routes/test_webhooks.py` if present or added before this phase
- `tests/routes/test_keys.py` if present or added before this phase

**Rust files:**

- Existing: `api-rs/src/app.rs`
- Create later: `api-rs/src/webhooks.rs`
- Create later: `api-rs/src/keys.rs`
- Create later: `crates/honcho-telemetry/`

- [ ] **Step 1: Add webhook contract tests**

  Cover endpoint registration, list, delete, test emit, delivery payload shape, retry behavior, and signature behavior.

- [ ] **Step 2: Add scoped key contract tests**

  Cover workspace, peer, and session scopes.

- [ ] **Step 3: Port telemetry emitters**

  Preserve CloudEvents fields and deterministic sampling behavior.

- [ ] **Step 4: Add operational health endpoints**

  Health checks must distinguish process health from database connectivity and queue readiness.

- [ ] **Step 5: Run operational parity checks**

  Run:

  ```bash
  rtk cargo test --manifest-path api-rs/Cargo.toml
  rtk uv run pytest tests/rust_api/ tests/routes/test_webhooks.py tests/routes/test_keys.py
  ```

  Expected: keys, webhooks, and telemetry tests pass with Python-compatible behavior.

- [ ] **Step 6: Commit operational parity**

  Run:

  ```bash
  git add api-rs crates tests
  git commit -m "feat: port rust operational api surfaces"
  ```

## Phase 11: Shadow Traffic And Route Cutover

**Status:** Not started.

**Goal:** Move public traffic route-by-route from Python to Rust with live comparison and rollback.

**Files:**

- Modify later: deployment manifests or Compose files used by the target environment
- Modify later: reverse proxy configuration if present
- Modify later: release documentation

- [ ] **Step 1: Add shadow comparison mode**

  For selected routes, production ingress sends the public response from Python and mirrors the request to Rust without exposing Rust response to users.

- [ ] **Step 2: Log structured diffs**

  Diff fields:

  ```text
  route template
  status code
  response hash
  body diff summary
  auth scope
  workspace id
  latency
  ```

- [ ] **Step 3: Define per-route promotion gates**

  A route can move to Rust only when:

  ```text
  contract tests pass
  SDK tests pass
  shadow diff rate is zero or explained
  p95 latency is acceptable
  error rate is not worse than Python
  rollback switch is tested
  ```

- [ ] **Step 4: Cut over read-only routes first**

  Order:

  ```text
  health
  list routes
  get routes
  search routes
  representation/context routes
  ```

- [ ] **Step 5: Cut over write routes after read-only stability**

  Order:

  ```text
  workspace writes
  peer writes
  session writes
  message writes
  conclusion writes
  webhook/key writes
  ```

- [ ] **Step 6: Commit deployment cutover configuration**

  Run:

  ```bash
  git add docker-compose-rs.yml.example docs
  git commit -m "chore: document rust api cutover path"
  ```

## Phase 12: Python Retirement

**Status:** Not started.

**Goal:** Remove Python runtime ownership only after Rust has public parity and operational stability.

**Files:**

- Modify later: `pyproject.toml`
- Modify later: deployment files
- Modify later: `docs/v3/`
- Modify later: `README.md`
- Keep later: Alembic migrations until a separate schema-ownership plan is complete

- [ ] **Step 1: Freeze Python API changes**

  Only bug fixes and migration support changes should land in Python after Rust is public-authoritative.

- [ ] **Step 2: Keep Python test suite as compatibility archive**

  Preserve contract fixtures until Rust owns the API long enough to remove Python safely.

- [ ] **Step 3: Move schema ownership only with a separate plan**

  Do not replace Alembic implicitly. Create a dedicated schema migration ownership plan before changing database migration tools.

- [ ] **Step 4: Remove Python services from deployment**

  Remove only after Rust API and Rust worker have handled production traffic through a full stability window.

- [ ] **Step 5: Archive Python implementation docs**

  Move legacy architecture notes into a clearly labeled historical section.

- [ ] **Step 6: Commit Python retirement**

  Run:

  ```bash
  git add pyproject.toml docs README.md
  git commit -m "chore: retire python runtime after rust cutover"
  ```

## Master Checklist

- [x] Rust read-only sidecar exists.
- [x] Initial sidecar contract tests exist.
- [x] Optional Docker sidecar entry exists.
- [x] Sidecar milestone committed.
- [ ] Full read-only route parity complete.
- [ ] Session peer read endpoints verified in Rust sidecar contracts.
- [ ] Shared Rust crates extracted only when justified.
- [ ] Write routes implemented behind a disabled-by-default gate.
- [ ] Queue producer parity complete.
- [ ] Rust worker runtime complete.
- [ ] Deriver parity complete.
- [ ] Summarizer parity complete.
- [ ] Reconciler parity complete.
- [ ] LLM provider layer parity complete.
- [ ] Search and representation parity complete.
- [ ] Dialectic chat parity complete.
- [ ] Dreamer parity complete.
- [ ] Webhook and key route parity complete.
- [ ] Telemetry parity complete.
- [ ] Shadow traffic mode complete.
- [ ] Read-only public route cutover complete.
- [ ] Write public route cutover complete.
- [ ] Worker public cutover complete.
- [ ] Python runtime retirement complete.

## Immediate Next Actions

1. Open a PR for review so the sidecar can land without claiming full migration.
2. Continue Phase 3 write shadows: the remaining un-started mutations are message
   creation (must preserve enqueue behavior) and conclusion create/delete (must
   preserve collection/document semantics) — both depend on queue/vector behavior
   being specified first.
3. Start Phase 1 by inventorying all remaining read-only v3 routes (search,
   representation, context, file upload remain Python-only).
4. Add contract tests for the next read-only group before writing more Rust endpoint code.
5. Keep Python authoritative until each phase satisfies its cutover gates.

**Local contract-test note:** the Python/Rust contract suite needs a host-reachable
pgvector Postgres on `127.0.0.1:5432`. SQLAlchemy 2.0 masks the password in
`str(url)` and conftest builds the async engine from `str(url)`, so the test DB must
use trust auth (`POSTGRES_HOST_AUTH_METHOD=trust`); a password-auth container fails
the async engine connection even though sync DB creation succeeds. Run with
`DB_CONNECTION_URI=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres
rtk uv run pytest tests/rust_api/ -q`.

## Completion Definition

The full migration is complete only when:

- Rust API handles all public v3 routes.
- Rust worker handles all queue, deriver, summarizer, reconciler, dreamer, and embedding work.
- Rust LLM layer handles chat, structured output, streaming, retries, fallback chains, and telemetry.
- SDK tests pass unchanged against Rust.
- Production deployment routes public API and worker traffic to Rust.
- Python services are removed from active deployment.
- Database migration ownership has either intentionally stayed with Alembic or moved under a separate approved schema-ownership plan.
