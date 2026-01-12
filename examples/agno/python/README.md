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
from honcho_agno import HonchoTools

# Create Honcho tools with user context
honcho_tools = HonchoTools(
    app_id="my-app",
    user_id="user-123",
)

# Create an agent with memory
agent = Agent(
    name="Memory Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[honcho_tools],
    description="An assistant with persistent memory powered by Honcho.",
)

# The agent can now use memory tools
response = agent.run("Remember that I prefer Python over JavaScript")
```

## Features

The `HonchoTools` toolkit provides four memory tools:

| Tool | Description |
|------|-------------|
| `add_message` | Store messages in the session for future recall |
| `get_context` | Retrieve conversation context within token limits |
| `search_messages` | Semantic search through past messages |
| `query_user` | Query the dialectic API for user insights |

## Configuration

### Basic Configuration

```python
from honcho_agno import HonchoTools

tools = HonchoTools(
    app_id="my-app",           # Workspace/application identifier
    user_id="user-123",        # User identifier
    session_id="session-456",  # Optional: specific session ID
)
```

### With API Key

```python
tools = HonchoTools(
    app_id="my-app",
    user_id="user-123",
    api_key="your-api-key",
    base_url="https://api.honcho.dev",  # Optional: custom endpoint
)
```

### Environment Variables

Copy `.env.template` to `.env` and configure:

**Honcho Settings:**

- `HONCHO_ENVIRONMENT`: `local` or `production` (default: production)
- `HONCHO_API_KEY`: API key for production environment
- `HONCHO_WORKSPACE_ID`: Default workspace ID

**OpenAI Settings (for examples):**

- `OPENAI_API_KEY` or `LLM_OPENAI_API_KEY`: OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4o)

## Tool Details

### add_message

Store a message in the current session.

```python
# The agent can call this tool to save information
result = honcho_tools.add_message(
    content="User prefers dark mode",
    role="user"  # or "assistant"
)
```

### get_context

Retrieve recent conversation context.

```python
context = honcho_tools.get_context(
    tokens=2000,           # Max tokens to include
    include_summary=True,  # Include session summary
)
```

### search_messages

Search through past messages semantically.

```python
results = honcho_tools.search_messages(
    query="programming preferences",
    limit=10,
)
```

### query_user

Ask questions about the user via the dialectic API.

```python
insights = honcho_tools.query_user(
    query="What programming languages does the user prefer?"
)
```

## Examples

See the [examples](./examples) directory for complete working examples:

- `simple_example.py`: Basic usage with HonchoTools
- `multi_tool_example.py`: Using all tools together

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
