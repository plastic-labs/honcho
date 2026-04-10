# Honcho Memory Integration for Smolagents

Give your [Smolagents](https://huggingface.co/docs/smolagents) agents persistent memory using [Honcho](https://honcho.dev).

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's task string on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and prepended to the agent's input message so the model always has an up-to-date view.
- **Zero Boilerplate**: A single `chat()` function handles context injection, agent creation, and memory persistence automatically.

## Structure

```
smolagents/
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
pip install "smolagents[litellm]" honcho-ai python-dotenv
```

Or with uv:

```bash
uv add "smolagents[litellm]" honcho-ai python-dotenv
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

### 1. Dynamic Context Injection

Before every agent call, `chat()` fetches recent messages from Honcho and prepends them to the user message:

```
## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?

User: What trail would you recommend for beginners?
```

### 2. Memory Tools

`QueryMemoryTool` subclasses `smolagents.Tool` and exposes Honcho's Dialectic API to the agent. When the user asks "What do you remember about me?", the agent calls `forward()` which queries Honcho's semantic memory layer.

### 3. Auto-Save

The `chat()` function saves the user message before the agent runs and the assistant response after, keeping Honcho in sync with every turn.

## Concept Mapping

| Smolagents | Honcho |
|---|---|
| `user_id` | Peer (human) |
| `"assistant"` | Peer (agent) |
| `session_id` | Session |
| Agent task | Message |

## License

AGPL-3.0-or-later
