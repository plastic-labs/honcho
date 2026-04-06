# Honcho Memory Integration for the OpenAI Agents SDK

Give your [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) agents persistent memory using [Honcho](https://honcho.dev).

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's instructions on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and formatted for the LLM before every request via dynamic `instructions`.
- **Zero Boilerplate**: Pass a `HonchoContext` to `Runner.run()` — the tools and instructions handle the rest.

## Installation

```bash
pip install honcho-ai openai-agents python-dotenv
```

Or with uv:

```bash
uv add honcho-ai openai-agents python-dotenv
```

## Environment Variables

Create a `.env` file:

```env
HONCHO_API_KEY=your-honcho-api-key
HONCHO_WORKSPACE_ID=default
OPENAI_API_KEY=your-openai-api-key
```

Get your Honcho API key at [honcho.dev](https://honcho.dev).

## Quick Start

```python
import asyncio
from agents import Agent, RunContextWrapper, Runner
from tools.client import HonchoContext
from tools.get_context import get_context
from tools.query_memory import query_memory
from tools.save_memory import save_memory


def honcho_instructions(ctx: RunContextWrapper[HonchoContext], agent: Agent) -> str:
    base = "You are a helpful assistant with persistent memory powered by Honcho."
    history = get_context(ctx.context, tokens=2000)
    if not history:
        return base
    formatted = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
    return f"{base}\n\n## Conversation History\n{formatted}"


agent = Agent[HonchoContext](
    name="HonchoMemoryAgent",
    instructions=honcho_instructions,
    tools=[query_memory],
    model="gpt-4.1-mini",
)


async def chat(user_id: str, message: str, session_id: str) -> str:
    ctx = HonchoContext(user_id=user_id, session_id=session_id)
    save_memory(user_id, message, "user", session_id)
    result = await Runner.run(agent, message, context=ctx)
    response = str(result.final_output)
    save_memory(user_id, response, "assistant", session_id)
    return response


# Run a conversation turn
response = asyncio.run(chat("alice", "I love hiking in the mountains", "session-1"))
print(response)
```

Run the interactive demo:

```bash
python main.py
```

## How It Works

### 1. Dynamic Instructions

The agent uses a callable `instructions` function instead of a static string. Before every LLM call, the SDK invokes this function with the current `RunContextWrapper`. The function calls `get_context()` to fetch recent messages from Honcho and injects them into the system prompt:

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tools

The `query_memory` tool is exposed to the LLM via `@function_tool`. When the user asks "What do you remember about me?", the agent calls this tool to query Honcho's Dialectic API — a semantic memory layer that synthesizes observations about the user into a natural language answer.

### 3. Auto-Save

The `chat()` helper in `main.py` wraps `Runner.run()` to save the user message before the run and the assistant response after. This keeps Honcho in sync with every conversation turn.

## API Reference

### `HonchoContext`

```python
@dataclass
class HonchoContext:
    user_id: str       # Unique identifier for the human peer
    session_id: str    # Identifier for the current conversation session
    assistant_id: str  # Peer ID for the assistant (default: "assistant")
```

Pass this as the `context` argument to `Runner.run()`.

---

### `save_memory(user_id, content, role, session_id, assistant_id="assistant")`

Saves a message to Honcho. Creates the peer and session if they don't exist.

| Param | Type | Description |
|---|---|---|
| `user_id` | `str` | Unique user identifier |
| `content` | `str` | Message text |
| `role` | `str` | `"user"` or `"assistant"` |
| `session_id` | `str` | Session identifier |
| `assistant_id` | `str` | Peer ID for the assistant (default: `"assistant"`) |

---

### `get_context(ctx, tokens=2000)`

Returns recent conversation history from Honcho as OpenAI-format message dicts.

| Param | Type | Description |
|---|---|---|
| `ctx` | `HonchoContext` | Context with user, session, and assistant IDs |
| `tokens` | `int` | Max tokens to include (default: `2000`) |

Returns `list[dict[str, str]]` — suitable for direct use as LLM input.

---

### `query_memory` (agent tool)

A `@function_tool` decorated function the agent calls to query Honcho's Dialectic API.

| Param | Type | Description |
|---|---|---|
| `ctx` | `RunContextWrapper[HonchoContext]` | Injected automatically by the SDK |
| `query` | `str` | Natural language question about the user |

Returns a natural language answer from Honcho's memory.

## Concept Mapping

| OpenAI Agents SDK | Honcho |
|---|---|
| `context.user_id` | Peer (human) |
| `context.assistant_id` | Peer (agent) |
| `context.session_id` | Session |
| `Runner.run()` input | Message |

## Running Tests

```bash
# Structural tests (no API keys required)
pytest tests/test_basic.py -v

# Integration tests (requires HONCHO_API_KEY)
pytest tests/test_integration.py -v
```

## License

AGPL-3.0-or-later
