# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- Azure OpenAI transport (`transport = "azure_openai"`) with a new `overrides.api_version` field. Every model config (deriver, dialectic levels, summary, dream specialists, embedding, fallback) can now target a native Azure OpenAI resource or an Azure-compatible gateway. Uses `AsyncAzureOpenAI` with `azure_endpoint=` + `api_version=` so the SDK constructs the `/openai/deployments/{model}/...?api-version=` URLs Azure expects. Reuses `OpenAIBackend` because the two clients share the same completion/embedding surface.
- New `src/llm/` package as the single owner of provider runtime: clients, backends, history adapters, tool loop, request builder, credentials, and caching policy
- `AttemptPlan` dataclass captures per-retry provider selection (client, model, reasoning_effort, thinking_budget_tokens, selected_config) and pins it across stream-final retries so streaming doesn't bounce back to primary after the tool loop has settled on fallback
- Gemini JSON-schema sanitizer for `function_declarations` — strips keywords Gemini's validator rejects (`additionalProperties`, `allOf`, etc.) while preserving semantics for all other backends
- Dreamer specialists derive `effective_max_tokens` from `model_config.max_output_tokens` with a per-specialist default fallback
- Regression tests covering fallback-config thinking-param reach, provider_params → extra_params boundary, OpenAI reasoning-model parameter routing, Gemini blocked finish_reason handling, and fail-fast `max_tool_iterations` validation

### Changed

- All LLM orchestration moved out of `src/utils/clients.py` into `src/llm/` with modules split by responsibility (api, executor, tool_loop, runtime, registry, conversation, request_builder, credentials, caching, backends, history_adapters)
- Default `ModelConfig` factories (deriver, summary, dreamer specialists, dialectic levels) normalized to `openai/gpt-5.4-mini` with no extra parameters set by default; operators add transport/thinking overrides explicitly
- OpenAI reasoning-model routing widened via `_uses_max_completion_tokens` heuristic covering `gpt-5.x` and `o1/o3/o4` — these models receive `max_completion_tokens` instead of `max_tokens`
- Override client factories switched from unbounded `@cache` to `@lru_cache(maxsize=128)` for predictable memory growth on long-running processes
- `get_backend` now delegates to `client_for_model_config`, so the live-test path and production path share one missing-API-key validation
- Blocked Gemini responses (`SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `BLOCKLIST`) raise `LLMError` in the streaming path too (previously only the non-streaming path), ensuring retry/fallback logic fires uniformly
- Transport-change env overrides now strip transport-specific thinking params (thinking_budget_tokens vs. reasoning_effort) during config merge, including at the dialectic-level merge, so switching from Anthropic → OpenAI doesn't leave orphaned Anthropic-only params that the OpenAI backend would reject
- `max_tool_iterations` out-of-range inputs now raise `ValidationException` instead of being silently clamped
- Troubleshooting docs updated to reflect nested-env-var form for per-component thinking-budget overrides

### Fixed

- Fallback `ModelConfig` temperature and `thinking_budget_tokens` reach the backend on the final retry — previously the primary's values were pre-populated into caller kwargs early and clobbered fallback values via `effective_config_for_call(update=...)`
- Stream-final retries pin to the `AttemptPlan` that succeeded rather than re-running provider selection through the outer `current_attempt` ContextVar (which could roll streaming back to primary after the tool loop had already switched to fallback)
- OpenAI structured-output calls continue to use `chat.completions.parse()` with strict schema enforcement, while tool-calling paths use `chat.completions.create()` without `strict:True` for broader proxy compatibility (OpenRouter, vLLM, Ollama)
- Gemini `cached_content` reuse keys now include `system_instruction` and `tool_config` so cache hits don't cross configurations that differ only in those fields

### Removed

- `src/utils/clients.py` deleted; its responsibilities are split across `src/llm/registry.py`, `src/llm/credentials.py`, and the backend-specific modules

## [3.0.6] - 2026-04-10

### Changed

- Tightened transaction scopes across search, agent tools, queue manager, and webhook delivery to minimize DB connection hold time during external operations (#525)
- Search operations refactored to two-phase pattern — external work (embeddings, LLM calls) completes before opening a transaction (#525)
- Agent tool executor performs external operations before acquiring DB sessions (#525)
- Queue manager transaction scope reduced to only the critical section (#525)
- Webhook delivery no longer holds a DB session parameter (#525)

### Fixed

- Session leakage in non-session-scoped dialectic chat calls (#526)

### Added

- Health check endpoint (`/health`) for container orchestration and load balancer probes (#510)

## [3.0.5] - 2026-04-03

### Fixed

- explicit rollback on all transactions to force connection closed

## [3.0.4] - 2026-04-02

### Added

- JSONB metadata validation enforces 100 key limit and max depth of 5 (#419)

### Changed

- Schemas refactored from single `schemas.py` into `schemas/api.py`, `schemas/configuration.py`, and `schemas/internal.py` with backwards-compatible re-exports (#419)

### Fixed

- Missing `deleted_at` filter on `RepresentationManager._query_documents_recent()` and `._query_documents_most_derived()` allowed soft-deleted documents to leak into the deriver's working representation (#456)
- `CleanupStaleItemsCompletedEvent` emitted spuriously when no queue item was actually deleted (#454)
- Empty JSON file uploads caused unhandled errors; now returns normalized error responses (#434)
- Memory leak: `_observation_locks` switched to `WeakValueDictionary` to prevent unbounded growth (#419)
- SQL injection in `dependencies.py`: parameterized `set_config` calls to prevent injection via request context (#419)
- NUL byte crashes: string inputs (message content, queries, peer cards) now stripped at schema level (#419)
- Filter recursion depth capped at 5 to prevent stack overflow (#419)
- Dedup-skipped observations now correctly reflected in created counts (#477)
- External vector store support for message search — routes queries through configured external vector store with oversampling and
  deduplication to handle chunked embeddings (#479)
- Dialectic agent no longer holds a DB connection during LLM calls — embeddings are pre-computed before tool execution, DB sessions isolated in `extract_preferences`, `query_documents` no longer accepts a DB session parameter (#477)

## [3.0.3] - 2026-02-25

### Added

- Consolidated session context into a single DB session with 40/60 token budget allocation between summary and messages
- Observation validation via `ObservationInput` Pydantic schema with partial-success support and batch embedding with per-observation fallback
- Peer card hard cap of 40 facts with case-insensitive deduplication and whitespace normalization
- Safe integer coercion (`_safe_int`) for all LLM tool inputs to handle non-integer values like `"Infinity"`
- Embedding pre-computation and reuse across multiple search calls in dialectic and representation flows
- Peer existence validation in dialectic chat endpoints — raises ResourceNotFoundException instead of silently failing
- Logging filter to suppress noisy `GET /metrics` access logs
- Oolong long-context aggregation benchmark (synth and real variants, 1K–4M token context windows)
- MolecularBench fact quality evaluation (ambiguity, decontextuality, minimality scoring)
- CoverageBench information recall evaluation (gold fact extraction, coverage matching, QA verification)
- LoCoMo summary-as-context baseline evaluation
- Webhook delivery tests, dependency lifecycle tests, queue cleanup tests, summarizer fallback tests
- Parallel test execution via pytest-xdist with worker-specific databases
- `test_reasoning_levels.py` script for LOCOM dataset testing across reasoning levels

### Changed

- Workspace deletion is now async — returns 202 Accepted, validates no active sessions (409 Conflict), cascade-deletes in background
- Redis caching layer now stores plain-dict instead of ORM objects, with v2-prefixed keys, storage, resilient `safe_cache_set`/`safe_cache_delete` helpers, and deferred post-commit cache invalidation
- All `get_or_create_*` CRUD operations now use savepoints (`db.begin_nested()`) instead of commit/rollback for race condition prevention
- Reconciler vector sync uses direct ORM mutation instead of batch parameterized UPDATE statements
- Summarizer enforces hard word limit in prompt and creates fallback text for empty summaries with `summary_tokens = 0`
- Blocked Gemini responses (SAFETY, RECITATION, PROHIBITED_CONTENT, BLOCKLIST) now raise `LLMError` to trigger retry/backup-provider logic
- Gemini client explicitly sets `max_output_tokens` from `max_tokens` parameter
- All deriver and metrics collector logging replaced with structured `logging.getLogger(__name__)` calls
- Dreamer specialist prompts updated to enforce durable-facts-only peer cards with max 40 entries and deduplication
- `GetOrCreateResult` changed from `NamedTuple` to `dataclass` with `async post_commit()` method
- FastAPI upgraded from 0.111.0 to 0.131.0; added pyarrow dependency
- Queue status filtering to only show user-facing tasks (representation, summary, dream); excludes internal infrastructure tasks

### Fixed

- JWT timestamp bug — `JWTParams.t` was evaluated once at class definition time instead of per-instance
- Session cache invalidation on deletion was missing
- `get_peer_card()` now properly propagates `ResourceNotFoundException` instead of swallowing it
- `set_peer_card()` ensures peer exists via `get_or_create_peers()` before updating
- Backup provider failover with proper tool input type safety
- Removed `setup_admin_jwt()` from server startup
- Sentry coroutine detection switched from `asyncio.iscoroutinefunction` to `inspect.iscoroutinefunction`

### Removed

- `explicit.py` and `obex.py` benchmarks replaced by coverage.py and molecular.py
- Claude Code review automation workflow (`.github/workflows/claude.yml`)
- Coverage reporting from default pytest configuration

## [3.0.2] - 2026-01-27

### Added

- Documentation for reasoning_level and Claude Code plugin

### Changed

- Gave dreaming sub-agents better prompting around peer card creation, tweaked overall prompts

### Fixed

- Added message-search fallback for memory search tool, necessary in fresh sessions
- Made FLUSH_ENABLED a config value
- Removed N+1 query in search_messages

## [3.0.1] - 2026-01-27

### Fixed

- Token counting in Explicit Agent Loop
- Backwards compatibility of queue items

## [3.0.0] - 2026-01-19

### Added

- Agentic Dreamer for intelligent memory consolidation using LLM agents
- Agentic Dialectic for query answering using LLM agents with tool use
- Reasoning levels configuration for dialectic (`minimal`, `low`, `medium`, `high`, `max`)
- Prometheus token tracking for deriver and dialectic operations
- n8n integration
- Cloud Events for auditable telemetry
- External Vector Store support for turbopuffer and lancedb with reconciliation flow

### Changed

- API route renaming for consistency
- Dreamer and dialectic now respect peer card configuration settings
- Observations renamed to Conclusions across API and SDKs
- Deriver to buffer representation tasks to normalize workloads
- Local Representation tasks to create singular QueueItems
- getContext endpoint to use `search_query` rather than force `last_user_message`

### Fixed

- Dream scheduling bugs
- Summary creation when start_message_id > end_message_id
- Cashews upgrade to prevent NoScriptError
- Memory leak in `accumulate_metric` call

### Removed

- Peer card configuration from message configuration; peer cards no longer created/updated in deriver process

## [2.5.1] - 2025-12-15

### Fixed

- Backwards compatibility for `message_ids` field in documents to handle legacy tuple format

## [2.5.0] - 2025-12-03

### Added

- Message level configurations
- CRUD operations for observations
- Comprehensive test cases for harness
- Peer level get_context
- Set Peer Card Method
- Manual dreaming trigger endpoint

### Changed

- Configurations to support more flags for fine-grained control of the deriver, peer cards, summaries, etc.
- Working Representations to support more fine-grained parameters

### Fixed

- File uploads to match `MessageCreate` structure
- Cache invalidation strategy

## [2.4.3] - 2025-11-20

### Added

- Redis caching to improve DB IO
- Backup LLM provider to avoid failures when a provider is down

### Changed

- QueueItems to use standardized columns
- Improved Deduplication logic for Representation Tasks
- More finegrained metrics for representation, summary, and peer card tasks
- DB constraint to follow standard naming conventions

## [2.4.2] - 2025-11-03

### Fixed

- Langfuse tracing to have readable waterfalls
- Alembic Migrations to match models.py
- message_in_seq correctly included in webhook payload

### Changed

- Alembic to always use a session pooler
- Statement timeout during alembic operations to 5 min

## [2.4.1] - 2025-10-24

### Added

- Alembic migration validation test suite

### Fixed

- Alembic migrations to batch changes
- Batch message creation sequence number

### Changed

- Logging infrastructure to remove noisy messages
- Sentry integration is centralized

## [2.4.0] - 2025-10-09

### Added

- Unified `Representation` class
- vllm client support
- Periodic queue cleanup logic
- WIP Dreaming Feature
- LongMemEval to Test Bench
- Prometheus Client for better Metrics
- Performance metrics instrumentation
- Error reporting to deriver
- Workspace Delete Method
- Multi-db option in test harness

### Changed

- Working Representations are Queried on the fly rather than cached in metadata
- EmbeddingStore to RepresentationFactory
- Summary Response Model to use public_id of message for cutoff
- Semantic across codebase to reference resources based on `observer` and `observed`
- Prompts for Deriver & Dialectic to reference peer_id and add examples
- `Get Context` route returns peer card and representation in addition to messages and summaries
- Refactoring logger.info calls to logger.debug where applicable

### Fixed

- Gemini client to use async methods

## [2.3.3] — 2025-10-01

### Changed

- Deriver Rollup Queue processes interleaved messages for more context

### Fixed

- Dialectic Streaming to follow SSE conventions
- Sentry tracing in the deriver

## [2.3.2] — 2025-09-25

### Added

- Get peer cards endpoint (`GET /v2/peers/{peer_id}/card`) for retrieving targeted peer context information

### Changed

- Replaced Mirascope dependency with small client implementation for better control
- Optimized deriver performance by using joins on messages table instead of storing token count in queue payload
- Database scope optimization for various operations
- Batch representation task processing for ~10x speed improvement in practice

### Fixed

- Separated clean and claim work units in queue manager to prevent race conditions
- Skip locked ActiveQueueSession rows on delete operations
- Langfuse SDK integration updates for compatibility
- Added configurable maximum message size to prevent token overflow in deriver
- Various minor bugfixes

## [2.3.1] - 2025-09-18

### Fixed

- Added max message count to deriver in order to not overflow token limits

## [2.3.0] — 2025-08-14

### Added

- `getSummaries` endpoint to get all available summaries for a session directly
- Peer Card feature to improve context for deriver and dialectic

### Changed

- Session Peer limit to be based on observers instead, renamed config value to
  `SESSION_OBSERVERS_LIMIT`
- `Messages` can take a custom timestamp for the `created_at` field, defaulting
  to the current time
- `get_context` endpoint returns detailed `Summary` object rather than just
  summary content
- Working representations use a FIFO queue structure to maintain facts rather
  than a full rewrite
- Optimized deriver enqueue by prefetching message sequence numbers (eliminates N+1 queries)

### Fixed

- Deriver uses `get_context` internally to prevent context window limit errors
- Embedding store will truncate context when querying documents to prevent embedding
  token limit errors
- Queue manager to schedule work based on available works rather than total
  number of workers
- Queue manager to use atomic db transactions rather than long lived transaction
  for the worker lifecycle
- Timestamp formats unified to ISO 8601 across the codebase
- Internal get_context method's cutoff value is exclusive now

## [2.2.0] — 2025-08-07

### Added

- Arbitrary filters now available on all search endpoints
- Search combines full-text and semantic using reciprocal rank fusion
- Webhook support (currently only supports queue_empty and test events, more to come)
- Small test harness and custom test format for evaluating Honcho output quality
- Added MCP server and documentation for it

### Changed

- Search has 10 results by default, max 100 results
- Queue structure generalized to handle more event types
- Summarizer now exhaustive by default and tuned for performance

### Fixed

- Resolve race condition for peers that leave a session while sending messages
- Added explicit rollback to solve integrity error in queue
- Re-introduced Sentry tracing to deriver
- Better integrity logic in get_or_create API methods

## [2.1.2] — 2025-07-30

### Fixed

- Summarizer module to ignore empty summaries and pass appropriate one to get_context
- Structured Outputs calls with OpenAI provider to pass strict=True to Pydantic Schema

## [2.1.1] — 2025-07-23

### Added

- Test harness for custom Honcho evaluations
- Better support for session and peer aware dialectic queries
- Langfuse settings
- Added recent history to dialectic prompt, dynamic based on new context window size setting

### Fixed

- Summary queue logic
- Formatting of logs
- Filtering by session
- Peer targeting in queries

### Changed

- Made query expansion in dialectic off by default
- Overhauled logging
- Refactor summarization for performance and code clarity
- Refactor queue payloads for clarity

## [2.1.0] — 2025-07-17

### Added

- File uploads
- Brand new "ROTE" deriver system
- Updated dialectic system
- Local working representations
- Better logging for deriver/dialectic
- Endpoint for deriver queue status

### Fixed

- Document insertion
- Session-scoped and peer-targeted dialectic queries work now

### Removed

- Peer-level messages

### Changed

- Dialectic chat endpoint takes a single query
- Rearranged configuration values (LLM, Deriver, Dialectic, History->Summary)

## [2.0.5] - 2025-07-11

### Fixed

- Groq API client to use the Async library

## [2.0.4] - 2025-07-02

### Fixed

- Migration/provision scripts did not have correct database connection arguments, causing timeouts

## [2.0.3] - 2025-07-01

### Fixed

- Bug that causes runtime error when Sentry flags are enabled

## [2.0.2] - 2025-06-27

### Fixed

- Database initialization was misconfigured and led to provision_db script failing: switch to consistent working configuration with transaction pooler

## [2.0.1] - 2025-06-26

### Added

- Ergonomic SDKs for Python and TypeScript (uses Stainless underneath)
- Deriver Queue Status endpoint
- Complex arbitrary filters on workspace/session/peer/message
- Message embedding table for full semantic search

### Changed

- Overhauled documentation
- BasedPyright typing for entire project
- Resource filtering expanded to include logical operators

### Fixed

- Various bugs
- Use new config arrangement everywhere
- Remove hardcoded responses

## [2.0.0] - 2025-06-24

### Added

- Ability to get a peer's working representation
- Metadata to all data primitives (Workspaces, Peers, Sessions, Messages)
- Internal metadata to store Honcho's state no longer exposed in API
- Batch message operations and enhanced message querying with token and message count limits
- Search and summary functionalities scoped by workspace, peer, and session
- Session context retrieval with summaries and token allocation
- HNSW Index for Documents Table
- Centralized Configuration via Environment Variables or `config.toml` file

### Changed

- API route is now /v2/
- New architecture centered around the concept of a "peer" replaces the former
  "app"/"user"/"session" paradigm
- Workspaces replace "apps" as top-level namespace
- Peers replace "users"
- Sessions no longer nested beneath peers and no longer limited to a single
  user-assistant model. A session exists independently of any one peer and
  peers can be added to and removed from sessions.
- Dialectic API is now part of the Peer, not the Session
- Dialectic API now allows queries to be scoped to a session or "targeted"
  to a fellow peer
- Database schema migrated to adopt workspace/peer/session naming and structure
- Authentication and JWT scopes updated to workspace/peer/session hierarchy
- Queue processing now works on 'work units' instead of sessions
- Message token counting updated with tiktoken integration and fallback heuristic
- Queue and message processing updated to handle sender/target and task types for multi-peer scenarios

### Fixed

- Improved error handling and validation for batch message operations and metadata
- Database Sessions to be more atomic to reduce idle in transaction time

### Removed

- Metamessages removed in favor of metadata
- Collections and Documents no longer exposed in the API, solely internal
- Obsolete tests for apps, users, collections, documents, and metamessages

## [1.1.0] - 2025-05-15

### Added

- Normalize resources to remove joins and increase query performance
- Query tracing for debugging

### Changed

- `/list` endpoints to not require a request body
- `metamessage_type` to `label` with backwards compatibility
- Database Provisioning to rely on alembic
- Database Session Manager to explicitly rollback transactions before closing
  the connection

### Fixed

- Alembic Migrations to include initial database migrations
- Sentry Middleware to not report Honcho Exceptions

## [1.0.0] - 2025-04-10

### Added

- JWT based API authentication
- Configurable logging
- Consolidated LLM Inference via `ModelClient` class
- Dynamic logging configurable via environment variables

### Changed

- Deriver & Dialectic API to use Hybrid Memory Architecture
- Metamessages are not strictly tied to a message
- Database provisioning is a separate script instead of happening on startup
- Consolidated `session/chat` and `session/chat/stream` endpoints

## [0.0.16] - 2025-03-05

### Added

- Detailed custom exceptions for better error handling
- CLAUDE.md for claude code

### Changed

- Deriver to use a new cognitive architecture that only updates on user messages
  and updates user representation to apply more confidence scores to its known
  facts
- Dialectic API token cutoff from 150 tokens to 300
- Dialectic API uses Claude 3.7 Sonnet
- SQLAlchemy echo changed to false by default, can be enabled with SQL_DEBUG
  environment flag

### Fixed

- Self-hosting documentation and README to mention `uv` instead of `poetry`

## [0.0.15] - 2025-01-06

### Added

- Alembic for handling database migrations
- Additional indexes for reading Messages and Metamessages
- Langfuse for prompt tracing

### Changed

- API validation using Pydantic

### Fixed

- Dialectic Streaming Endpoint properly sends text in `StreamingResponse`
- Deriver Queue handles graceful shutdown

## [0.0.14] — 2024-11-14

### Changed

- Query Documents endpoint is a POST request for better DX
- `String` columns are now `TEXT` columns to match postgres best practices
- Docstrings to have better stainless generations

### Fixed

- Dialectic API to use most recent user representation
- Prepared Statements Transient Error with `psycopg`
- Queue parallel worker scheduling

## [0.0.13] — 2024-11-07

### Added

- Ability to clone session for a user to achieve more [loom-like](https://github.com/socketteer/loom/) behavior

## [0.0.12] — 2024-10-21

### Added

- GitHub Actions Testing
- Ability to disable derivations on a session using the `deriver_disabled` flag
  in a session's metadata
- `/v1/` prefix to all routes
- Environment variable to control deriver workers

### Changed

- public_ids to use [NanoID](https://github.com/ai/nanoid) and internal ID to
  use `BigInt`
- Dialectic Endpoint can take a list of queries
- Using `uv` for project management
- User Representations stored in a metamessage rather than using reserved
  collection
- Base model for Dialectic API and Deriver is now Claude 3.5 Sonnet
- Paginated GET requests now POST requests for better developer UX

### Removed

- Mirascope Dependency
- Slowapi Dependency
- Opentelemetry Dependencies and Setup

## [0.0.11] — 2024-08-01

### Added

- `session_id` column to `QueueItem` Table
- `ActiveQueueSession` Table to track, which sessions are being actively
  processed
- Queue can process multiple sessions at once

### Changed

- Sessions do not require a `location_id`
- Detailed printing using `rich`

## [0.0.10] — 2024-07-23

### Added

- Test cases for Storage API
- Sentry tracing and profiling
- Additional Error handling

### Changed

- Document API uses same embedding endpoint as deriver
- CRUD operations use one less database call by removing extra refresh
- Use database for timestampz rather than API
- Pydantic schemas to use modern syntax

### Fixed

- Deriver queue resolution

## [0.0.9] — 2024-05-16

### Added

- Deriver to docker compose
- Postgres based Queue for background jobs

### Changed

- Deriver to use a queue instead of supabase realtime
- Using mirascope instead of langchain

### Removed

- Legacy SDKs in preference for stainless SDKs

## [0.0.8] — 2024-05-09

### Added

- Documentation to OpenAPI
- Bearer token auth to OpenAPI routes
- Get by ID routes for users and collections
- [NodeJS](https://github.com/plastic-labs/honcho-node) SDK support

### Changed

- Authentication Middleware now implemented using built-in FastAPI Security
  module
- Get by name routes for users and collections now include "name" in slug
- Python SDK moved to separate [repository](https://github.com/plastic-labs/honcho-python)

### Fixed

- Error reporting for methods with integrity errors due to unique key
  constraints

## [0.0.7] — 2024-04-01

### Added

- Authentication Middleware Interface

## [0.0.6] — 2024-03-21

### Added

- Full docker-compose for API and Database

### Fixed

- API Response schema removed unnecessary fields
- OTEL logging to properly work with async database engine
- `fly.toml` default settings for deriver set `auto_stop=false`

### Changed

- Refactored API server into multiple route files

## [0.0.5] — 2024-03-14

### Added

- Metadata to all data primitives (Users, Sessions, Messages, etc.)
- Ability to filter paginated GET requests by JSON filter based on metadata
- Optional Sentry error monitoring
- Optional Opentelemetry logging
- Dialectic API to interact with honcho agent and get insights about users
- Automatic Fact Derivation Script for automatically generating simple memory

### Changed

- API Server now uses async methods to make use of benefits of FastAPI

## [0.0.4] — 2024-02-22

### Added

- apps table with a relationship to the users table
- users table with a relationship to the collections and sessions tables
- Reverse Pagination support to get recent messages, sessions, etc. more easily
- Linting Rules

### Changed

- Get sessions method returns all sessions including inactive
- using timestampz instead of timestamp

## [0.0.3] — 2024-02-15

### Added

- Collections table to reference a collection of embedding documents
- Documents table to hold vector embeddings for RAG workflows
- Local scripts for running a postgres database with pgvector installed
- OpenAI Dependency for embedding models
- PGvector dependency for vector db support

### Changed

- session_data is now metadata
- session_data is a JSON field used python `dict` for compatibility

## [0.0.2] — 2024-02-01

### Added

- Pagination for requests via `fastapi_pagination`
- Metamessages
- `get_message` routes
- `created_at` field added to each Table
- Message size limits

### Changed

- IDs are now UUIDs
- default rate limit now 100 requests per minute

### Removed

- Removed messages from session response model

## [0.0.1] — 2024-02-01

### Added

- Rate limiting of 10 requests for minute
- Application level scoping
