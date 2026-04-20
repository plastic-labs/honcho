# Honcho Memory Integration for LlamaIndex

Give your [LlamaIndex](https://docs.llamaindex.ai) agents persistent memory using [Honcho](https://honcho.dev). Available in both Python and TypeScript.

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's system prompt on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and formatted for the LLM before every request via `prefix_messages`.
- **Zero Boilerplate**: A single `chat()` function handles context injection, agent creation, and memory persistence automatically.

## Structure

```
llamaindex/
├── README.md
├── python/
│   ├── main.py
│   ├── pyproject.toml
│   └── tools/
│       ├── client.py
│       ├── save_memory.py
│       ├── query_memory.py
│       └── get_context.py
└── typescript/
    ├── main.ts
    ├── package.json
    ├── tsconfig.json
    └── tools/
        ├── client.ts
        ├── saveMemory.ts
        ├── queryMemory.ts
        └── getContext.ts
```

## Environment Variables

Create a `.env` file in whichever subdirectory you run from:

```env
HONCHO_API_KEY=your-honcho-api-key
HONCHO_WORKSPACE_ID=default
OPENAI_API_KEY=your-openai-api-key
```

Get your Honcho API key at [honcho.dev](https://honcho.dev).

---

## Python

### Installation

```bash
pip install llama-index llama-index-llms-openai honcho-ai python-dotenv
```

Or with uv:

```bash
uv add llama-index llama-index-llms-openai honcho-ai python-dotenv
```

### Quick Start

```python
from main import chat

response = chat("alice", "I love hiking in the mountains", "session-1")
print(response)
```

### Run the demo

```bash
cd python
python main.py
```

---

## TypeScript

### Installation

```bash
cd typescript
bun install
# or: npm install
```

### Quick Start

```typescript
import { chat } from './main.js';

const response = await chat('alice', 'I love hiking in the mountains', 'session-1');
console.log(response);
```

### Run the demo

```bash
cd typescript
bun run main.ts
```

---

## How It Works

### 1. Dynamic System Prompt

Before every LLM call, `chat()` fetches recent messages from Honcho and injects them into the agent via `prefix_messages` (Python) or `chatHistory` (TypeScript):

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tools

The `query_memory` tool is exposed to the LLM via `FunctionTool`. When the user asks "What do you remember about me?", the agent calls this tool to query Honcho's Dialectic API — a semantic memory layer that synthesizes observations about the user into a natural language answer.

### 3. Auto-Save

The `chat()` function saves the user message before the agent runs and the assistant response after, keeping Honcho in sync with every turn.

## Concept Mapping

| LlamaIndex | Honcho |
|---|---|
| `user_id` | Peer (human) |
| `"assistant"` | Peer (agent) |
| `session_id` | Session |
| Agent input | Message |

## License

AGPL-3.0-or-later
