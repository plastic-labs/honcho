# Honcho Memory Integration for Mastra

Give your [Mastra](https://mastra.ai) agents persistent memory using [Honcho](https://honcho.dev).

## Features

- **Persistent Memory**: Every conversation turn is saved to Honcho and automatically injected into the agent's instructions on the next turn.
- **Natural Language Recall**: The agent can query Honcho's Dialectic API to answer questions like "What are my hobbies?" or "What did we talk about last time?"
- **Context Injection**: Conversation history is retrieved from Honcho and formatted into the `instructions` string passed to `new Agent()` on every turn.
- **Zero Boilerplate**: A single `chat()` function handles context injection, agent creation, and memory persistence automatically.

## Structure

```
mastra/
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

### 1. Dynamic Instructions

`buildInstructions(ctx)` fetches recent messages from Honcho and passes them as the `instructions` string to `new Agent()` before every `generate()` call:

```
You are a helpful assistant with persistent memory powered by Honcho.

## Conversation History
User: I love hiking
Assistant: That sounds wonderful! Do you have a favorite trail?
```

### 2. Memory Tool Factory

`makeQueryMemoryTool(ctx)` returns a Mastra `createTool()` instance, closing over the user context. When the model calls `query_memory`, the execute function queries Honcho's Dialectic API for long-term user facts.

### 3. Auto-Save

The `chat()` function saves the user message before `agent.generate()` runs and the assistant response after, keeping Honcho in sync with every turn.

## Concept Mapping

| Mastra | Honcho |
|---|---|
| `userId` | Peer (human) |
| `"assistant"` | Peer (agent) |
| `sessionId` | Session |
| `instructions` string | Context injection |

## License

AGPL-3.0-or-later
