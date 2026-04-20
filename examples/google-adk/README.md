# Honcho Memory Integration for Google ADK

Give your [Google ADK](https://google.github.io/adk-docs/) agents persistent memory using [Honcho](https://honcho.dev).

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's instructions on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and formatted for the LLM before every request via dynamic `instruction`.
- **Zero Boilerplate**: A single `chat()` function handles context injection, agent creation, and memory persistence automatically.

## Structure

```
google-adk/
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
GOOGLE_API_KEY=your-google-ai-api-key
```

Get your Honcho API key at [honcho.dev](https://honcho.dev).
Get your Google AI API key at [aistudio.google.com](https://aistudio.google.com).

## Installation

```bash
pip install google-adk honcho-ai python-dotenv
```

Or with uv:

```bash
uv add google-adk honcho-ai python-dotenv
```

## Quick Start

```python
import asyncio
from main import chat

response = asyncio.run(chat("alice", "I love hiking in the mountains", "session-1"))
print(response)
```

## Run the Demo

```bash
cd python
python main.py
```

## How It Works

### 1. Dynamic Instructions

Before every LLM call, `build_instruction()` fetches recent messages from Honcho and injects them into the agent's `instruction` string:

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tools

The `query_memory` function is wrapped in a `FunctionTool` and exposed to Gemini. When the user asks "What do you remember about me?", the agent calls this tool to query Honcho's Dialectic API — a semantic memory layer that synthesizes observations about the user into a natural language answer.

### 3. Auto-Save

The `chat()` function saves the user message before the agent runs and the assistant response after, keeping Honcho in sync with every turn.

## Concept Mapping

| Google ADK | Honcho |
|---|---|
| `user_id` | Peer (human) |
| `"assistant"` | Peer (agent) |
| `session_id` | Session |
| Agent input | Message |

## License

AGPL-3.0-or-later
