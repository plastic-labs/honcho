# Honcho Memory Integration for the OpenAI Agents SDK

Give your [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) agents persistent memory using [Honcho](https://honcho.dev). Available in both Python and TypeScript.

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's instructions on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and formatted for the LLM before every request via dynamic `instructions`.
- **Zero Boilerplate**: Pass a context object to `Runner.run()` — the tools and instructions handle the rest.

## Structure

```
openai-agents/
├── python/          # Python implementation (openai-agents package)
│   ├── main.py
│   ├── pyproject.toml
│   └── tools/
│       ├── client.py
│       ├── save_memory.py
│       ├── query_memory.py
│       └── get_context.py
└── typescript/      # TypeScript implementation (@openai/agents package)
    ├── main.ts
    ├── package.json
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
pip install honcho-ai openai-agents python-dotenv
```

Or with uv:

```bash
uv add honcho-ai openai-agents python-dotenv
```

### Quick Start

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


response = asyncio.run(chat("alice", "I love hiking in the mountains", "session-1"))
print(response)
```

### Run the demo

```bash
cd python
python main.py
```

### Tests

```bash
cd python
# Structural tests (no API keys required)
pytest tests/test_basic.py -v

# Integration tests (requires HONCHO_API_KEY)
pytest tests/test_integration.py -v
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
import { Agent, run } from '@openai/agents';
import type { RunContext } from '@openai/agents';
import { createContext, HonchoContext } from './tools/client.js';
import { getContext } from './tools/getContext.js';
import { queryMemory } from './tools/queryMemory.js';
import { saveMemory } from './tools/saveMemory.js';

async function honchoInstructions(runContext: RunContext<HonchoContext>): Promise<string> {
  const base = 'You are a helpful assistant with persistent memory powered by Honcho.';
  const history = await getContext(runContext.context, 2000);
  if (history.length === 0) return base;
  const formatted = history.map(m => `${m.role}: ${m.content}`).join('\n');
  return `${base}\n\n## Conversation History\n${formatted}`;
}

const agent = new Agent<HonchoContext>({
  name: 'HonchoMemoryAgent',
  instructions: honchoInstructions,
  tools: [queryMemory],
  model: 'gpt-4.1-mini',
});

async function chat(userId: string, message: string, sessionId: string): Promise<string> {
  const ctx = createContext(userId, sessionId);
  await saveMemory(userId, message, 'user', sessionId);
  const result = await run(agent, message, { context: ctx });
  const response = String(result.finalOutput);
  await saveMemory(userId, response, 'assistant', sessionId);
  return response;
}

const response = await chat('alice', 'I love hiking in the mountains', 'session-1');
console.log(response);
```

### Run the demo

```bash
cd typescript
bun run main.ts
```

### Type-check

```bash
cd typescript
bun run tsc --noEmit
```

---

## How It Works

### 1. Dynamic Instructions

The agent uses a callable `instructions` function instead of a static string. Before every LLM call, the SDK invokes this function with the current run context. The function calls `getContext()` to fetch recent messages from Honcho and injects them into the system prompt:

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tools

The `queryMemory` tool is exposed to the LLM. When the user asks "What do you remember about me?", the agent calls this tool to query Honcho's Dialectic API — a semantic memory layer that synthesizes observations about the user into a natural language answer.

### 3. Auto-Save

The `chat()` helper wraps `Runner.run()` (Python) / `run()` (TypeScript) to save the user message before the run and the assistant response after. This keeps Honcho in sync with every conversation turn.

## Concept Mapping

| OpenAI Agents SDK | Honcho |
|---|---|
| `context.userId` | Peer (human) |
| `context.assistantId` | Peer (agent) |
| `context.sessionId` | Session |
| Agent input | Message |

## License

AGPL-3.0-or-later
