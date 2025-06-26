# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.0.1]

### Added

- Ergonomic SDKs for Python and TypeScript (uses Stainless underneath)
- Deriver Queue Status endpoint
- Complex arbitrary filters on workspace/session/peer/message
- Message embedding table for full semantic search

### Changed

- Overhaulted documentation
- BasedPyright typing for entire project

### Fixed

- Various bugs
- Use new config arrangement everywhere
- Remove hardcoded responses

## [2.0.0]

### Added

- Ability to get a peer's working representation
- Metadata to all data primitives (Workspaces, Peers, Sessions, Messages)
- Internal metadata to store Honcho's state no longer exposed in API
- Batch message operations and enhanced message querying with token and message count limits
- Search and summary functionalities scoped by workspace, peer, and session
- Ability to search message histories using semantic search with cosine
  similarity
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
- Resource filtering expanded to include logical operators

### Fixed

- Improved error handling and validation for batch message operations and metadata
- Database Sessions to be more atomic to reduce idle in transaction time

### Removed

- Metamessages removed in favor of metadata
- Collections and Documents no longer exposed in the API, solely internal
- Obsolete tests for apps, users, collections, documents, and metamessages

## [1.1.0]

### Added

- Normalize resources to remove joins and increase query performance
- Query tracing for debugging

### Changed

- `/list` endpoints to not require a request body
- `metamessage_type` to `label` with backwards compatability
- Database Provisioning to rely on alembic
- Database Session Manager to explicitly rollback transactions before closing
  the connection

### Fixed

- Alembic Migrations to include initial database migrations
- Sentry Middleware to not report Honcho Exceptions

## [1.0.0]

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

## [0.0.16]

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

## [0.0.15]

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
- Python SDK moved to separate [respository](https://github.com/plastic-labs/honcho-python)

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
- session_data is a JSON field used python `dict` for compatability

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
