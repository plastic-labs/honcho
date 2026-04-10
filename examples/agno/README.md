# Honcho Memory Integration for Agno

Give your [Agno](https://docs.agno.com) agents persistent memory using [Honcho](https://honcho.dev).

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's description on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and formatted for the LLM before every request via dynamic `description`.
- **Zero Boilerplate**: A single `chat()` function handles context injection, agent creation, and memory persistence automatically.

## Structure

```
agno/
└── python/
    ├── main.py
    ├── pyproject.toml
    └── tools/
        ├── client.py
        ├── save_memory.py
        ├── query_memory.py
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
pip install agno honcho-ai python-dotenv openai
```

Or with uv:

```bash
uv add agno honcho-ai python-dotenv openai
```

## Quick Start

```python
from main import chat

response = chat("alice", "I love hiking in the mountains", "session-1")
print(response)
```

## Run the Demo

```bash
cd python
python main.py
```

## How It Works

### 1. Dynamic Description

Before every LLM call, `chat()` fetches recent messages from Honcho and injects them into the agent's `description` string:

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tools

The `query_memory` tool is exposed to the LLM via Agno's `@tool` decorator. When the user asks "What do you remember about me?", the agent calls this tool to query Honcho's Dialectic API — a semantic memory layer that synthesizes observations about the user into a natural language answer.

Because Agno tools are plain functions without run-context injection, `make_query_memory_tool(user_id)` is a factory that closes over the user ID at call time.

### 3. Auto-Save

The `chat()` function saves the user message before the agent runs and the assistant response after, keeping Honcho in sync with every turn.

## Concept Mapping

| Agno | Honcho |
|---|---|
| `user_id` | Peer (human) |
| `"assistant"` | Peer (agent) |
| `session_id` | Session |
| Agent input | Message |

## License

AGPL-3.0-or-later
