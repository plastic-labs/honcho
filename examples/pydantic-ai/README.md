# Honcho Memory Integration for Pydantic AI

Give your [Pydantic AI](https://ai.pydantic.dev) agents persistent memory using [Honcho](https://honcho.dev).

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's system prompt on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and appended to the system prompt via `@agent.system_prompt`.
- **In-Session Coherence**: Pydantic AI's `message_history` parameter keeps the agent coherent within a single session, complementing Honcho's cross-session memory.

## Structure

```
pydantic-ai/
├── README.md
└── python/
    ├── main.py
    ├── pyproject.toml
    └── tools/
        ├── client.py
        ├── save_memory.py
        └── get_context.py
```

## Environment Variables

Create a `.env` file in the `python/` directory:

```env
HONCHO_API_KEY=your-honcho-api-key
HONCHO_WORKSPACE_ID=default
OPENAI_API_KEY=your-openai-api-key
```

Get your Honcho API key at [honcho.dev](https://honcho.dev).

## Installation

```bash
pip install pydantic-ai honcho-ai python-dotenv
```

Or with uv:

```bash
uv add pydantic-ai honcho-ai python-dotenv
```

## Quick Start

```python
import asyncio
from main import chat

async def main():
    message_history = []
    # First turn
    response, message_history = await chat("alice", "I love hiking in the mountains", "session-1", message_history)
    print(response)
    # Second turn — history is threaded automatically
    response, message_history = await chat("alice", "What do you remember about me?", "session-1", message_history)
    print(response)

asyncio.run(main())
```

## Run the Demo

```bash
cd python
python main.py
```

## How It Works

### 1. Dynamic System Prompt

The `@agent.system_prompt` decorator registers `honcho_system_prompt()`, which is called by Pydantic AI before every LLM request. It fetches recent messages from Honcho and appends them to the system prompt:

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tool

The `@agent.tool` decorator registers `query_memory()`, which calls Honcho's Dialectic API. When the user asks "What do you remember about me?", the agent invokes this tool to query the semantic memory layer.

### 3. Message History Threading

`chat()` returns `(response, result.all_messages())`. Pass the returned history back on the next call to maintain in-session coherence. Honcho provides cross-session memory; `message_history` provides within-session context.

### 4. Auto-Save

The `chat()` function saves the user message before the agent runs and the assistant response after, keeping Honcho in sync with every turn.

## Concept Mapping

| Pydantic AI | Honcho |
|---|---|
| `deps.ctx.user_id` | Peer (human) |
| `deps.ctx.assistant_id` | Peer (agent) |
| `deps.ctx.session_id` | Session |
| `message_history` | In-session context |
| Agent input | Message |

## License

AGPL-3.0-or-later
