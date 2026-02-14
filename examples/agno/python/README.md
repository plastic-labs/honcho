# Honcho Agno Integration

Give your [Agno](https://agno.com) agents persistent memory with [Honcho](https://honcho.dev).

## Installation

```bash
pip install honcho-agno
```

Or with uv:

```bash
uv add honcho-agno
```

## Quick Start

```python
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from honcho import Honcho
from honcho_agno import HonchoTools

# Initialize Honcho client
honcho = Honcho(workspace_id="my-app")

# Create Honcho tools for the agent
honcho_tools = HonchoTools(honcho_client=honcho)

# Create an agent with memory tools
agent = Agent(
    name="Memory Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[honcho_tools],
)

# Create peers and session for message persistence
user_peer = honcho.peer("user-123")
assistant_peer = honcho.peer("assistant")
session = honcho.session("session-123")

# Save user message (orchestration handles persistence)
session.add_messages([user_peer.message("I prefer Python over JavaScript")])

# Run the agent - user_id and session_id flow through RunContext to tools
response = agent.run(
    "What programming language does the user prefer?",
    user_id="user-123",
    session_id="session-123",
)

# Save assistant response
session.add_messages([assistant_peer.message(str(response.content))])
```

## How It Works

HonchoTools maps to Agno's user/assistant architecture:

| Agno Concept | Honcho Concept | Description |
|--------------|----------------|-------------|
| `user_id` (from RunContext) | Peer | The human user being queried about |
| `session_id` (from RunContext) | Session | The conversation context |

**Key insight**: Tools query Honcho about the **USER**, not the agent. When the agent asks "What does this user prefer?", Honcho returns insights about the human user identified by `run_context.user_id`.

### Message Persistence

This toolkit is **read-only** - it provides tools for querying Honcho's memory but does not automatically save messages. Your orchestration code handles message persistence using the Honcho client directly:

```python
# Save messages using the Honcho client (not the toolkit)
session.add_messages([
    user_peer.message("User's message"),
    assistant_peer.message("Assistant's response"),
])
```

This separation gives you explicit control over what gets saved to memory.

## Features

The `HonchoTools` toolkit provides three memory tools:

| Tool | Description |
|------|-------------|
| `honcho_get_context` | Retrieve conversation context within token limits |
| `honcho_search_messages` | Semantic search through past messages |
| `honcho_chat` | Query Honcho for synthesized insights about the user |

## Configuration

### Basic Configuration

```python
from honcho import Honcho
from honcho_agno import HonchoTools

# Create shared Honcho client
honcho = Honcho(workspace_id="my-app")

# Create toolkit
tools = HonchoTools(honcho_client=honcho)
```

### Without Pre-configured Client

```python
from honcho_agno import HonchoTools

# Creates its own Honcho client internally
tools = HonchoTools(workspace_id="my-app")
```

Note: When `honcho_client` is provided, `workspace_id` is ignored since the client already has its workspace configured.

### Environment Variables

**Honcho Settings:**

- `HONCHO_ENVIRONMENT`: `local` or `production` (default: production)
- `HONCHO_API_KEY`: API key for production environment
- `HONCHO_WORKSPACE_ID`: Default workspace ID

**OpenAI Settings (for examples):**

- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4o)

## Tool Details

### honcho_get_context

Retrieve recent conversation context. Uses `session_id` from RunContext.

```python
# Called by the agent automatically with RunContext
# Or call directly with a mock context for testing
```

### honcho_search_messages

Search through past messages semantically. Uses `session_id` from RunContext.

```python
# Called by the agent automatically with RunContext
# Query example: "programming preferences"
```

### honcho_chat

Ask questions about the user using Honcho's reasoning. Uses both `user_id` and `session_id` from RunContext.

```python
# Called by the agent automatically with RunContext
# Query example: "What programming languages does the user prefer?"
```

## Multi-Agent Systems (Teams)

Agno Teams share context within a run, but what about across runs? What if Agent A needs to remember what Agent B learned last week? That's where Honcho comes in.

```python
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from honcho import Honcho
from honcho_agno import HonchoTools

# Shared Honcho client - all agents share the same memory
honcho = Honcho(workspace_id="advisory-app")
honcho_tools = HonchoTools(honcho_client=honcho)

# Tech advisor with Honcho memory
tech_agent = Agent(
    name="Tech Advisor",
    model=OpenAIChat(id="gpt-4o"),
    tools=[honcho_tools],
)

# Business advisor with Honcho memory
biz_agent = Agent(
    name="Business Advisor",
    model=OpenAIChat(id="gpt-4o"),
    tools=[honcho_tools],
)

# Create team
team = Team(
    name="Advisory Team",
    agents=[tech_agent, biz_agent],
)

# Run with shared user_id and session_id
# Both agents query Honcho about the same user
response = team.run(
    "How should I scale my startup?",
    user_id="founder-123",
    session_id="strategy-session",
)
```

## Architecture Notes

- **Read-only toolkit**: `HonchoTools` provides read access to Honcho (context, search, chat)
- **Orchestration pattern**: Message saving is handled by your orchestration code using `honcho.session().add_messages()`
- **RunContext integration**: `user_id` and `session_id` flow through Agno's RunContext automatically
- **Cross-run memory**: Unlike Agno Teams (context within a run), Honcho persists memory across runs

## Examples

See the [examples](./examples) directory for complete working examples:

- `simple_example.py`: Basic usage with HonchoTools demonstrating memory persistence and context retrieval

## Development

### Setup

```bash
cd examples/agno/python
uv sync
```

### Run Tests

```bash
uv run pytest
```

### Run Examples

```bash
uv run python examples/simple_example.py
```

## License

AGPL-3.0-or-later
