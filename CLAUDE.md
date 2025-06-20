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
- **Keys**: Create scoped JWT tokens

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
- Run server: `fastapi dev src/main.py`
- Run tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test_file.py::test_function`
- Linting: `ruff check src/`
- Format code: `ruff format src/`

### Code Style

- Follow isort conventions with absolute imports preferred
- Use explicit type hints with SQLAlchemy mapped_column annotations
- snake_case for variables/functions; PascalCase for classes
- Line length: 88 chars (Black compatible)
- Explicit error handling with appropriate exception types
- Docstrings: Use Google style docstrings

### Project Structure

```
src/
├── main.py          # FastAPI app setup with middleware and exception handlers
├── models.py        # SQLAlchemy ORM models with proper type annotations
├── schemas.py       # Pydantic validation schemas for API
├── crud.py          # Database operations
├── dependencies.py  # Dependency injection (DB sessions)
├── exceptions.py    # Custom exception types
├── security.py      # JWT authentication
├── agent.py         # Dialectic API implementation
├── routers/         # API endpoints
│   ├── workspaces.py
│   ├── peers.py
│   ├── sessions.py
│   ├── messages.py
│   └── keys.py
├── deriver/         # Background processing system
│   ├── consumer.py  # Message processing logic
│   ├── queue.py     # Queue management
│   └── tom/         # Theory of Mind implementations
└── utils/           # Utilities
    ├── history.py   # Session history management
    ├── cache.py     # Caching utilities
    └── model_client.py # LLM client abstraction
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
5. **Scoped Authentication**: JWT tokens can be scoped to workspace, peer, or session level
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
