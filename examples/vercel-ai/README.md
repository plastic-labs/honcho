# Honcho Memory Integration for Vercel AI SDK

Give your [Vercel AI SDK](https://sdk.vercel.ai) agents persistent memory using [Honcho](https://honcho.dev).

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's system prompt on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and formatted as a dynamic `system` string passed to `generateText()`.
- **Multi-Step Tool Calls**: `maxSteps: 5` allows the agent to call `query_memory` and continue responding in a single turn.

## Structure

```
vercel-ai/
├── README.md
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

Create a `.env` file in the `typescript/` directory:

```env
HONCHO_API_KEY=your-honcho-api-key
HONCHO_WORKSPACE_ID=default
OPENAI_API_KEY=your-openai-api-key
```

Get your Honcho API key at [honcho.dev](https://honcho.dev).

## Installation

```bash
cd typescript
bun install
# or: npm install
```

## Quick Start

```typescript
import { chat } from './main.js';

const response = await chat('alice', 'I love hiking in the mountains', 'session-1');
console.log(response);
```

## Run the Demo

```bash
cd typescript
bun run main.ts
```

## How It Works

### 1. Dynamic System String

Before every `generateText()` call, `chat()` fetches recent messages from Honcho and injects them into the `system` parameter:

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tool Factory

`makeQueryMemoryTool(ctx)` returns a Vercel AI `tool()` with a Zod schema, closing over the user context. When the model calls `query_memory`, the execute function queries Honcho's Dialectic API.

### 3. Multi-Step Execution

`maxSteps: 5` allows the model to call `query_memory` and then generate a natural language response in a single `chat()` call — no extra orchestration needed.

### 4. Auto-Save

The `chat()` function saves the user message before `generateText()` runs and the assistant response after, keeping Honcho in sync with every turn.

## Concept Mapping

| Vercel AI SDK | Honcho |
|---|---|
| `userId` | Peer (human) |
| `"assistant"` | Peer (agent) |
| `sessionId` | Session |
| `system` string | Context injection |

## License

AGPL-3.0-or-later
