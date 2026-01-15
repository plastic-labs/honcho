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
honcho_tools = HonchoTools(
    peer_id="assistant",
    session_id="session-123",
    honcho_client=honcho,
)

# Create user peer for orchestration
user_peer = honcho.peer("user")

# Create an agent with memory tools
agent = Agent(
    name="Memory Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[honcho_tools],
    description="An assistant with persistent memory powered by Honcho.",
)

# Add user message via orchestration
honcho_tools.session.add_messages([user_peer.message("I prefer Python over JavaScript")])

# Run the agent
response = agent.run("What programming language does the user prefer?")

# Save assistant response via orchestration
honcho_tools.session.add_messages([honcho_tools.peer.message(str(response.content))])
```

## Features

The `HonchoTools` toolkit provides three memory tools:

| Tool | Description |
|------|-------------|
| `get_context` | Retrieve conversation context within token limits |
| `search_messages` | Semantic search through past messages |
| `chat` | Query Honcho for synthesized insights about the conversation |

## Configuration

### Basic Configuration

```python
from honcho import Honcho
from honcho_agno import HonchoTools

# Create shared Honcho client
honcho = Honcho(workspace_id="my-app")

# Create toolkit for an agent
tools = HonchoTools(
    peer_id="assistant",          # Identity for this agent
    session_id="session-456",     # Optional: specific session ID
    honcho_client=honcho,         # Shared Honcho client
)
```

### Without Pre-configured Client

```python
from honcho_agno import HonchoTools

# Creates its own Honcho client internally
tools = HonchoTools(
    app_id="my-app",              # Workspace ID (used to create internal client)
    peer_id="assistant",          # Identity for this agent
    session_id="session-456",     # Optional: auto-generated if not provided
)
```

Note: When `honcho_client` is provided, `app_id` is ignored since the client already has its workspace configured.

### Environment Variables

Configure via `.env` file in the root honcho directory:

**Honcho Settings:**

- `HONCHO_ENVIRONMENT`: `local` or `production` (default: production)
- `HONCHO_API_KEY`: API key for production environment
- `HONCHO_WORKSPACE_ID`: Default workspace ID

**OpenAI Settings (for examples):**

- `OPENAI_API_KEY` or `LLM_OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4o)

## Tool Details

### get_context

Retrieve recent conversation context.

```python
context = honcho_tools.get_context(
    tokens=2000,           # Max tokens to include (optional)
    include_summary=True,  # Include session summary (default: True)
)
```

### search_messages

Search through past messages semantically.

```python
results = honcho_tools.search_messages(
    query="programming preferences",
    limit=10,  # Max results (default: 10)
)
```

### chat

Ask questions about the conversation using Honcho's reasoning.

```python
insights = honcho_tools.chat(
    query="What programming languages does the user prefer?"
)
```

## Multi-Peer Conversations

For multi-agent systems, create separate `HonchoTools` instances for each agent, sharing the same session:

```python
from honcho import Honcho
from honcho_agno import HonchoTools

# Shared Honcho client and session
honcho = Honcho(workspace_id="advisory-app")
session_id = "shared-session-123"

# Tech advisor agent
tech_tools = HonchoTools(
    peer_id="tech-advisor",
    session_id=session_id,
    honcho_client=honcho,
)

# Business advisor agent
biz_tools = HonchoTools(
    peer_id="biz-advisor",
    session_id=session_id,
    honcho_client=honcho,
)

# User peer for orchestration
user = honcho.peer("user")

# Add messages via orchestration (not toolkit methods)
tech_tools.session.add_messages([user.message("How should I scale my startup?")])
tech_tools.session.add_messages([tech_tools.peer.message("Consider microservices...")])
biz_tools.session.add_messages([biz_tools.peer.message("Focus on unit economics...")])
```

## Architecture Notes

- **Read-only toolkit**: `HonchoTools` provides read access to Honcho (context, search, chat)
- **Orchestration pattern**: Message saving is handled by your orchestration code, not the toolkit
- **One peer per toolkit**: Each `HonchoTools` instance represents one agent identity
- **Shared sessions**: Multiple toolkits can share a session for multi-agent conversations

## Examples

See the [examples](./examples) directory for complete working examples:

- `simple_example.py`: Basic usage with HonchoTools
- `multi_tool_example.py`: Using all tools together
- `multi_peer_example.py`: Multi-agent conversation with different perspectives

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

## License

AGPL-3.0-or-later
