# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Honcho Overview

## What is Honcho?

Honcho is an infrastructure layer for building AI agents with social cognition and theory of mind capabilities. Its primary purposes include:

- Imbuing agents with a sense of identity
- Personalizing user experiences through understanding user psychology
- Providing a Dialectic API that injects personal context just-in-time
- Supporting development of LLM-powered applications that adapt to end users
- Enabling multi-peer sessions where multiple participants (users or agents) can interact

Honcho leverages the inherent theory-of-mind capabilities of LLMs to build coherent models of user psychology over time, enabling more personalized and effective AI interactions.

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
- **Collections & Documents**: Internal vector storage for theory-of-mind representations (not exposed via API)

## Architecture Overview

### API Structure

All API routes follow the pattern: `/v1/{resource}/{id}/{action}`

- **Workspaces**: Create, list, update, search
- **Peers**: Create, list, update, chat (dialectic), messages, representation
- **Sessions**: Create, list, update, delete, clone, manage peers, get context
- **Messages**: Create (batch up to 100), list, get, update
- **Keys**: Create scoped JWTs

### Key Features

#### Dialectic API (`/peers/{peer_id}/chat`)

- Provides theory-of-mind informed responses
- Integrates long-term facts from vector storage
- Supports streaming responses
- Configurable LLM providers

#### Message Processing Pipeline

1. Messages created via API (batch or single)
2. Enqueued for background processing:
   - `representation`: Update peer's theory of mind
   - `summary`: Create session summaries
3. Session-based queue processing ensures order
4. Results stored internally in vector DB

#### Theory of Mind System

- Multiple implementation methods (conversational, single_prompt, long_term)
- Facts extracted from messages and stored in collections
- Representations combine short-term inference with long-term facts
- Configurable via peer and session feature flags

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

### Code Style

- Follow isort conventions with absolute imports preferred
- Use explicit type hints with SQLAlchemy mapped_column annotations
- snake_case for variables/functions; PascalCase for classes
- Line length: 88 chars (Black compatible)
- Explicit error handling with appropriate exception types
- Docstrings: Use Google style docstrings

### Agent Architecture

Honcho uses three specialized LLM agents that work together to form memories and answer queries:

#### 1. Deriver Agent (`src/deriver/agent/`)

**Role**: Memory formation through content ingestion

The Deriver processes incoming messages and extracts observations about peers.

- **Trigger**: Messages created via API are enqueued for background processing
- **Tools**: `create_observations`, `update_peer_card`, `get_recent_history`, `search_memory`, `get_observation_context`, `search_messages`
- **Output**: Explicit observations (direct facts) and deductive observations (inferences)
- **Entry point**: `src/deriver/agent/worker.py` → `Agent.run_loop()`

#### 2. Dialectic Agent (`src/dialectic/agent/`)

**Role**: Analysis and recall for answering queries

The Dialectic answers questions about peers by strategically gathering context from memory.

- **Trigger**: API call to `/peers/{peer_id}/chat` with `agentic=true`
- **Tools**: `search_memory`, `get_recent_history`, `get_observation_context`, `search_messages`, `get_recent_observations`, `get_most_derived_observations`, `get_session_summary`, `get_peer_card`, `create_observations` (deductive only)
- **Output**: Natural language response grounded in gathered context
- **Entry point**: `src/dialectic/chat.py` → `agentic_chat()` → `DialecticAgent.answer()`

#### 3. Dreamer Agent (`src/dreamer/agent.py`)

**Role**: Consolidation and self-improvement of memory

The Dreamer explores and consolidates observations to improve memory quality.

- **Trigger**: Scheduled or explicit dream task via queue
- **Tools**: `get_recent_observations`, `get_most_derived_observations`, `search_memory`, `create_observations`, `delete_observations`, `update_peer_card`
- **Strategy**: Random walk exploration - start from recent/high-value observations, search for related content, consolidate redundancies
- **Output**: Consolidated observations, deleted redundancies
- **Entry point**: `src/dreamer/agent.py` → `DreamerAgent.consolidate()`

#### Shared Agent Infrastructure

All agents share common infrastructure in `src/utils/agent_tools.py`:

- **Tool definitions**: Unified tool schemas used by all agents
- **Tool executor**: `create_tool_executor()` factory creates context-aware executors
- **LLM client**: `honcho_llm_call()` handles tool calling loops with configurable iterations

### Project Structure

```
src/
├── main.py              # FastAPI app setup with middleware and exception handlers
├── models.py            # SQLAlchemy ORM models with proper type annotations
├── schemas.py           # Pydantic validation schemas for API
├── config.py            # Configuration management
├── db.py                # Database connection and session management
├── dependencies.py      # Dependency injection (DB sessions)
├── exceptions.py        # Custom exception types
├── security.py          # JWT authentication
├── embedding_client.py  # Embedding service client
├── crud/                # Database operations
│   ├── __init__.py
│   ├── collection.py     # Collection CRUD operations
│   ├── deriver.py        # Deriver-related CRUD operations
│   ├── document.py       # Document CRUD operations
│   ├── message.py        # Message CRUD operations
│   ├── peer.py           # Peer CRUD operations
│   ├── peer_card.py      # Peer Card CRUD operations
│   ├── representation.py # RepresentationManager and representation operations
│   ├── session.py        # Session CRUD operations
│   ├── webhook.py        # Webhook CRUD operations
│   └── workspace.py      # Workspace CRUD operations
├── dialectic/            # Dialectic API implementation
│   ├── __init__.py
│   ├── chat.py           # Chat functionality (standard + agentic)
│   ├── prompts.py        # Prompt templates
│   └── agent/            # Agentic dialectic implementation
│       ├── __init__.py
│       ├── core.py       # DialecticAgent class
│       └── prompts.py    # Agent system prompts
├── routers/              # API endpoints
│   ├── workspaces.py
│   ├── peers.py
│   ├── sessions.py
│   ├── messages.py
│   ├── keys.py
│   └── webhooks.py      # Webhook endpoints
├── deriver/             # Background processing system
│   ├── __init__.py
│   ├── __main__.py      # Deriver entry point
│   ├── consumer.py      # Message consumer
│   ├── enqueue.py       # Queue operations
│   ├── queue_manager.py # Queue management
│   └── agent/           # Agentic deriver implementation
│       ├── __init__.py
│       ├── core.py      # Agent class
│       ├── worker.py    # Task processing
│       └── prompts.py   # Agent system prompts
├── dreamer/             # Memory consolidation system
│   ├── __init__.py
│   ├── agent.py         # DreamerAgent class + process_agent_dream
│   └── dreamer.py       # Legacy dreamer (scheduled)
├── utils/               # Utilities
│   ├── __init__.py
│   ├── agent_tools.py   # Shared agent tools and executor
│   ├── clients.py       # LLM client abstraction
│   ├── files.py         # File handling utilities
│   ├── filter.py        # Query filtering utilities
│   ├── formatting.py    # Message formatting utilities
│   ├── logging.py       # Logging and metrics (Rich console output)
│   ├── search.py        # Search functionality
│   ├── shared_models.py # Shared data models
│   ├── summarizer.py    # Session summarization
│   └── types.py         # Type definitions
└── webhooks/            # Webhook system
    ├── events.py        # Webhook event definitions
    ├── webhook_delivery.py # Webhook delivery logic
    └── README.md        # Webhook documentation
```

- Tests in pytest with fixtures in tests/conftest.py
- Use environment variables via python-dotenv (.env)

### Database Design

- All tables use text IDs (nanoid format) as primary keys
- Composite foreign keys for multi-tenant relationships
- Feature flags on workspace, peer, and session levels
- Token counting on messages for usage tracking
- JSONB metadata fields for extensibility
- HNSW indexes for vector similarity search

### Key Architectural Decisions

1. **Multi-Peer Sessions**: Sessions can have multiple participants with different observation settings
2. **Flexible Theory of Mind**: Pluggable ToM implementations (conversational, single_prompt, long_term)
3. **Background Processing**: Async queue system for expensive operations
4. **Provider Abstraction**: Model client supports multiple LLM providers
5. **Scoped Authentication**: JWTs can be scoped to workspace, peer, or session level
6. **Batch Operations**: Support for bulk message creation (up to 100 messages)
7. **Session History**: Two-tier summarization (short every 20 messages, long every 60)

### Error Handling

- Custom exceptions defined in src/exceptions.py
- Use specific exception types (ResourceNotFoundException, ValidationException, etc.)
- Proper logging with context instead of print statements
- Global exception handlers defined in main.py
- See docs/contributing/error-handling.mdx for details

### Notes

- Always use `uv run` or `uv` to prefix any commands related to python to ensure you use the virtual environment
