# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Honcho Overview

## What is Honcho?

Honcho is an infrastructure layer for building AI agents with memory and social cognition. Its primary purposes include:

- Imbuing agents with a sense of identity
- Personalizing user experiences through understanding user psychology
- Providing a Chat Endpoint (the Dialectic agent) that injects personal context just-in-time
- Supporting development of LLM-powered applications that adapt to end users
- Enabling multi-peer sessions where multiple participants (users or agents) can interact

Honcho leverages the inherent reasoning capabilities of LLMs to build coherent models of user psychology over time, enabling more personalized and effective AI interactions.

## Core Concepts

### Peer Paradigm

Honcho uses a peer-based model where both users and agents are represented as "peers". This unified approach enables:

- Multi-participant sessions with mixed human and AI agents
- Configurable observation settings (which peers observe which others)
- Flexible identity management for all participants

### Key Primitives

- **Workspace** (formerly App): The root organizational unit containing all resources
- **Peer** (formerly User): Any participant in the system (human or AI)
- **Session**: A conversation context that can involve multiple peers
- **Message**: Data units that can represent communication between peers OR arbitrary data ingested by a peer to enhance its global representation
- **Collections & Documents**: Internal vector storage for peer representations. Collections are keyed by `(observer, observed)` peer pairs. Collections/Documents are not directly exposed via API, but the observations stored within them are exposed as **Conclusions** (see `/v3/.../conclusions` endpoints).

## Architecture Overview

### API Structure

All API routes follow the pattern: `/v3/{resource}/{id}/{action}`. Most "list/search" endpoints are `POST` so they can accept rich filter bodies.

- **Workspaces**: Create, list, update, search
- **Peers**: Create, list, update, chat (dialectic), messages, representation
- **Sessions**: Create, list, update, delete, clone, manage peers, get context
- **Messages**: Create (batch up to 100), upload (file), list, get, update
- **Conclusions**: Create, list, query (semantic search), delete — the API-facing name for observations stored in `(observer, observed)` collections
- **Keys**: Create scoped JWTs
- **Webhooks**: Register endpoint, list, delete, test

### Key Features

#### Chat Endpoint (Dialectic agent) (`/peers/{peer_id}/chat`)

- Provides bespoke responses informed by the representation
- Integrates long-term facts from vector storage
- Supports streaming responses
- Configurable LLM providers

#### Message Processing Pipeline

1. Messages created via API (batch or single)
2. Enqueued for background processing:
   - `representation`: Update peer's context
   - `summary`: Create session summaries
3. Session-based queue processing ensures order
4. Results stored internally in vector DB

### Configuration

- Hierarchical config: config.toml + environment variables
- Database settings with connection pooling
- Multiple LLM provider support
- Background worker (deriver) settings
- Authentication can be toggled on/off

## Development Guide

### Commands

- Setup: `uv sync`
- Run server: `uv run fastapi dev src/main.py`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/path/to/test_file.py::test_function`
- Linting: `uv run ruff check src/`
- Typechecking: `uv run basedpyright`
- Format code: `uv run ruff format src/`

### SDK Testing

#### TypeScript SDK

**🚨 DO NOT RUN `bun test` DIRECTLY. IT WILL NOT WORK. 🚨**

The TypeScript SDK tests require a running Honcho server with database and Redis. Running `bun test` alone will fail immediately because there's no server. The tests are orchestrated via pytest which handles all the infrastructure setup.

**The ONLY way to run TypeScript SDK tests:**

```bash
# From the monorepo root (not from sdks/typescript/)
uv run pytest tests/ -k typescript
```

**To type-check the TypeScript SDK (this is fine to run directly):**

```bash
cd sdks/typescript && bun run tsc --noEmit
```

### Code Style

- Follow isort conventions with absolute imports preferred
- Use explicit type hints with SQLAlchemy mapped_column annotations
- snake_case for variables/functions; PascalCase for classes
- Line length: 88 chars (Black compatible)
- Explicit error handling with appropriate exception types
- Docstrings: Use Google style docstrings
- **Never hold a DB session during external calls** (LLM, embedding, HTTP). If a function needs both a DB session and an external call result, compute the external result first and pass it as a parameter. This avoids tying up DB connections during slow network I/O. Use `tracked_db` for short-lived, DB-only operations; pass a shared session when multiple DB-only calls can reuse one connection.

### Runtime Architecture

Honcho runs as two cooperating processes that share a Postgres database and Redis cache:

- **API server** (`uv run fastapi dev src/main.py`) — handles HTTP, enqueues background work, returns immediately. Hosts the **Dialectic** agent inline (synchronous tool loop during chat requests).
- **Deriver worker** (`uv run python -m src.deriver`) — long-running queue consumer (uvloop). Runs the **Deriver**, **Summarizer**, and **Dreamer** off the queue. Can run multiple instances (`DERIVER_WORKERS`). Also hosts an in-process **Reconciler scheduler** (`src/reconciler/`) that periodically embeds messages with `sync_state='pending'` in `MessageEmbedding` and cleans up stale queue items — embedding generation is decoupled from message creation by design.

### Agent Architecture

Honcho uses several specialized LLM agents. They share tool definitions and the LLM client abstraction in `src/utils/agent_tools.py` + `src/llm/`.

> **Terminology:** what users see as **conclusions** (the public API surface and the term we use in documentation) is called **observations** in code symbols — `create_observations`, `delete_observations`, `get_observation_context`, etc. Doc prose below uses "conclusions"; references to actual code symbols stay as "observations."

#### 1. Deriver (`src/deriver/`)

**Role**: Memory formation through content ingestion.

The Deriver processes batches of incoming messages and extracts conclusions about peers. The current architecture is "minimal deriver" — a **single LLM call** per batch using structured output, not an agentic tool loop. This trades flexibility for cost and predictability.

- **Trigger**: Messages enqueued by `src/deriver/enqueue.py` on message create; consumed by `src/deriver/queue_manager.py` → `consumer.process_item()` → `deriver.process_representation_tasks_batch()`.
- **Output**: Explicit conclusions (direct facts) and deductive conclusions (inferences) saved to `(observer, observed)` collections.
- **Entry point**: `src/deriver/__main__.py` → `queue_manager.main()`.
- **Prompts**: `src/deriver/prompts.py` (`minimal_deriver_prompt`).
- **Custom instructions**: per-workspace/peer guidance can be threaded into the prompt via reasoning configuration; `DERIVER__MAX_CUSTOM_INSTRUCTIONS_TOKENS` caps the addition (default 2000) and `DERIVER__MAX_INPUT_TOKENS` defaults to 25000 to make room.

#### 2. Dialectic (`src/dialectic/`)

**Role**: Analysis and recall for answering queries.

The Dialectic answers questions about peers by strategically gathering context from memory. It is the only tool-using agent on the synchronous request path — it loops over `DIALECTIC_TOOLS` until it has enough context to answer. (The Dreamer specialists also use tools, but run off the queue.)

- **Trigger**: API call to `POST /v3/.../peers/{peer_id}/chat`.
- **Tools** (see `DIALECTIC_TOOLS` in `src/utils/agent_tools.py`): `search_memory`, `search_messages`, `get_observation_context`, `grep_messages`, `get_messages_by_date_range`, `search_messages_temporal`, `get_reasoning_chain`. At the `minimal` reasoning level, a reduced set (`DIALECTIC_TOOLS_MINIMAL`) is used: just `search_memory` + `search_messages`.
- **Reasoning levels**: 5 tiers — `minimal`, `low`, `medium`, `high`, `max` — each with its own model config (see `DialecticLevelSettings` in `src/config.py`).
- **Output**: Natural language response grounded in gathered context. Supports SSE streaming.
- **Entry point**: `src/dialectic/chat.py` → `agentic_chat()` / `agentic_chat_stream()` → `DialecticAgent` (in `src/dialectic/core.py`).

#### 3. Dreamer (`src/dreamer/`)

**Role**: Consolidation and self-improvement of memory.

The Dreamer is an orchestrated multi-specialist system that runs during scheduled "dreams" to consolidate conclusions and build reasoning trees.

- **Trigger**: Scheduled via `DreamScheduler` (`src/dreamer/dream_scheduler.py`) or explicit dream task on the queue.
- **Strategy**: Surprisal-based prioritization (`src/dreamer/surprisal.py`) selects which conclusions to expand. The orchestrator (`orchestrator.run_dream`) runs two specialist phases:
  1. **DeductionSpecialist** (`specialists.py`) — produces deductive conclusions from explicit conclusions. Tools: `get_recent_observations`, `search_memory`, `search_messages`, `create_observations_deductive`, `delete_observations`, `update_peer_card`.
  2. **InductionSpecialist** — produces inductive conclusions from explicit + deductive conclusions. Tools: same discovery set + `create_observations_inductive`, `update_peer_card`.
- **Reasoning trees** (`src/dreamer/trees/`, migration `f1a2b3c4d5e6_add_reasoning_tree_columns`): each conclusion links to its premises and downstream conclusions, enabling `get_reasoning_chain` traversal at recall time.
- **Output**: Deductive/inductive conclusions, consolidated redundancies, updated peer cards.
- **Entry point**: `src/dreamer/orchestrator.py` → `process_dream()` (the package-level export from `src/dreamer/__init__.py`), which wraps `run_dream()`.

#### 4. Summarizer (`src/utils/summarizer.py`)

**Role**: Two-tier session summarization (direct LLM call — no agentic tools).

- **Trigger**: Runs as part of the queue pipeline alongside representation tasks.
- **Tiers**: short summary every `SUMMARY_MESSAGES_PER_SHORT_SUMMARY` messages (default 20); long summary every `SUMMARY_MESSAGES_PER_LONG_SUMMARY` (default 60). Token caps configurable via `SUMMARY_MAX_TOKENS_SHORT` / `SUMMARY_MAX_TOKENS_LONG`.

#### Shared Agent Infrastructure

- **Tool definitions** (`src/utils/agent_tools.py`): unified `TOOLS` dict; per-agent lists (`DIALECTIC_TOOLS`, `DIALECTIC_TOOLS_MINIMAL`, `DREAMER_TOOLS`, `DEDUCTION_SPECIALIST_TOOLS`, `INDUCTION_SPECIALIST_TOOLS`).
- **LLM subsystem** (`src/llm/`): provider-agnostic `honcho_llm_call()`. Backends in `src/llm/backends/` (`anthropic.py`, `gemini.py`, `openai.py`). Includes prompt caching (`caching.py`), structured output (`structured_output.py`), tool loop (`tool_loop.py`), history adapters for cross-provider message formats, and a model registry. Per-retry provider selection is pinned via an `AttemptPlan` so stream-final retries don't bounce back to primary after the tool loop has settled on fallback.
- **Per-agent model config**: each agent has its own `MODEL_CONFIG` in `src/config.py` with fallback chains (see `ConfiguredModelSettings`, `FallbackModelSettings`).
- **Telemetry**: cloudevents in `src/telemetry/events/` cover API routes, dialectic, dream, deletion, reconciliation, representation, and per-call LLM accounting (`llm.py` — `LLMCallCompletedEvent` fires once per provider hit with full cost-attribution context). High-volume events are sampled deterministically per `run_id` via `TelemetrySettings.HIGH_VOLUME_SAMPLE_RATE`.

### Project Structure

```
src/
├── main.py              # FastAPI app: middleware, routers, lifespan, exception handlers
├── models.py            # SQLAlchemy ORM models (Workspace/Peer/Session/Message/
│                        #   MessageEmbedding/Collection/Document/QueueItem/...)
├── config.py            # Pydantic-settings configuration (very large; see README)
├── db.py                # Engine + session/context management (request_context var)
├── dependencies.py      # FastAPI DI (tracked_db, etc.)
├── exceptions.py        # Custom exception types (HonchoException + subclasses)
├── security.py          # JWT authentication
├── embedding_client.py  # Embedding provider client (configurable dimensions
│                        #   via EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE)
├── schemas/             # Pydantic schemas
│   ├── api.py            # Public API request/response schemas
│   ├── configuration.py  # Per-resource configuration schemas
│   └── internal.py       # Internal-only schemas (queue payloads, etc.)
├── crud/                # Per-resource DB operations
│   ├── collection.py, deriver.py, document.py, message.py
│   ├── peer.py, peer_card.py, representation.py  (RepresentationManager)
│   ├── session.py, webhook.py, workspace.py
├── routers/             # FastAPI route handlers (all under /v3)
│   ├── workspaces.py, peers.py (dialectic /chat lives here), sessions.py
│   ├── messages.py, conclusions.py, keys.py, webhooks.py
├── dialectic/           # Dialectic agent — runs inline per chat request
│   ├── chat.py           # agentic_chat() / agentic_chat_stream()
│   ├── core.py           # DialecticAgent (the tool-loop driver)
│   └── prompts.py
├── deriver/             # Background queue consumer (separate process)
│   ├── __main__.py       # `python -m src.deriver` entry point
│   ├── queue_manager.py  # QueueManager + main() loop
│   ├── consumer.py       # process_item dispatcher (representation / deletion / reconciler)
│   ├── deriver.py        # "minimal deriver" — single-LLM-call batch processor
│   ├── enqueue.py        # API → queue producer
│   └── prompts.py
├── dreamer/             # Memory consolidation (runs off the queue)
│   ├── orchestrator.py   # run_dream() / process_dream()
│   ├── specialists.py    # DeductionSpecialist + InductionSpecialist
│   ├── dream_scheduler.py
│   ├── surprisal.py      # Surprisal-based conclusion prioritization
│   └── trees/            # Reasoning-tree primitives
├── reconciler/          # In-process scheduler hosted by the deriver worker
│   ├── scheduler.py      # ReconcilerScheduler (started from queue_manager.py)
│   ├── sync_vectors.py   # Embeds MessageEmbedding rows with sync_state='pending'
│   └── queue_cleanup.py  # Removes stale queue items
├── llm/                 # Provider-agnostic LLM client subsystem
│   ├── api.py, backend.py, executor.py, runtime.py, registry.py
│   ├── caching.py, structured_output.py, tool_loop.py, conversation.py
│   ├── history_adapters.py, request_builder.py, credentials.py, types.py
│   └── backends/         # anthropic.py, gemini.py, openai.py
├── cache/               # Redis cache abstraction (cashews-backed)
│   └── client.py
├── vector_store/        # Optional external vector stores (pgvector is default,
│   │                    #   implemented via MessageEmbedding/Document in models+crud)
│   ├── lancedb.py
│   └── turbopuffer.py
├── telemetry/           # Observability
│   ├── emitter.py        # CloudEvents emitter
│   ├── logging.py        # Logging helpers + route-template extraction
│   ├── metrics_collector.py, reasoning_traces.py, sentry.py
│   ├── events/           # Event type definitions
│   └── prometheus/       # Prometheus metric definitions
├── utils/               # Cross-cutting utilities
│   ├── agent_tools.py    # Tool definitions + per-agent tool lists
│   ├── summarizer.py     # Two-tier session summarizer
│   ├── representation.py # Representation formatting (distinct from crud/representation.py)
│   ├── search.py, filter.py, formatting.py
│   ├── tokens.py         # tiktoken-based counting
│   ├── work_unit.py, queue_payload.py
│   ├── config_helpers.py, json_parser.py, files.py
│   └── types.py
└── webhooks/            # Webhook delivery
    ├── events.py
    └── webhook_delivery.py
```

- Tests in pytest with fixtures in tests/conftest.py; subdirs mirror src/ (`tests/deriver/`, `tests/dialectic/`, etc.) plus `tests/bench/` (perf benchmarks), `tests/integration/`, `tests/live_llm/` (gated by `--live-llm`), and `tests/unified/` (the unified runner).
- Use environment variables via python-dotenv (.env). Config precedence: env > .env > config.toml > defaults.

### Database Design

- All tables use text IDs (nanoid format) as primary keys
- Composite foreign keys for multi-tenant relationships
- Feature flags on workspace, peer, and session levels
- Token counting on messages for usage tracking
- JSONB metadata fields for extensibility
- HNSW indexes for vector similarity search

### Key Architectural Decisions

1. **Peer Paradigm**: humans and AI agents are unified as "Peers"; many-to-many with Sessions. Internal vector storage (Collections/Documents) is keyed by `(observer, observed)` peer pairs — the same mechanism powers self-representation (`observer == observed`) and cross-peer modeling.
2. **Multi-Peer Sessions**: Sessions can have multiple participants with different observation settings.
3. **API server / worker split**: API enqueues, deriver worker process consumes. Never block HTTP on LLM work. The Reconciler runs as an in-process scheduler inside the deriver, handling async embedding sync and queue cleanup.
4. **"Minimal" deriver**: memory formation is a single structured-output LLM call per batch, not an agentic tool loop. Predictable cost, lower latency. The Dialectic is the one true tool-using agent.
5. **Provider-agnostic LLM layer** (`src/llm/`): all model calls go through `honcho_llm_call()`. Backends (`anthropic`, `gemini`, `openai`) sit behind a registry; per-agent `MODEL_CONFIG` with fallback chains is resolved at call time.
6. **Dialectic reasoning tiers**: 5 levels (`minimal` → `max`); each level has its own model config and tool set (`minimal` uses a reduced toolset).
7. **Hybrid search**: Postgres FTS (GIN index on `to_tsvector('english', content)`) + vector similarity (HNSW on `MessageEmbedding.embedding`). `MessageEmbedding` is a separate table from `Message` with its own `sync_state` so embedding is decoupled from message creation.
8. **Pluggable external vector stores**: defaults to pgvector inline; can swap to turbopuffer or lancedb (`VECTOR_STORE_*` config; `src/vector_store/`).
9. **Composite-FK multi-tenancy**: `workspace_name` participates in nearly every composite FK. Cross-workspace data leakage is structurally impossible at the schema level.
10. **Scoped Authentication**: JWTs can be scoped to workspace, peer, or session level.
11. **Batch Operations**: Bulk message creation up to 100 messages per request.
12. **Session History**: Two-tier summarization — short every `SUMMARY_MESSAGES_PER_SHORT_SUMMARY` (default 20), long every `SUMMARY_MESSAGES_PER_LONG_SUMMARY` (default 60).

### Error Handling

- Custom exceptions defined in src/exceptions.py
- Use specific exception types (ResourceNotFoundException, ValidationException, etc.)
- Proper logging with context instead of print statements
- Global exception handlers defined in main.py
- See docs/contributing/error-handling.mdx for details

### Notes

- Always use `uv run` or `uv` to prefix any commands related to python to ensure you use the virtual environment
