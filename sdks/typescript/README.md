# Honcho TypeScript SDK

A high-level, ergonomic TypeScript SDK for the Honcho conversational memory platform. This library wraps [honcho-node-core](../honcho-node-core) to provide a user-friendly, Pythonic API for managing peers, sessions, and conversational context.

## Installation

```
npm install @honcho-ai/sdk
```

## Usage

```ts
import { Honcho } from "@honcho-ai/sdk";

const honcho = new Honcho({
  apiKey: process.env.HONCHO_API_KEY,
  baseUrl: "http://localhost:8000",
  workspaceId: "test",
});

const assistant = honcho.peer("bob");
const alice = honcho.peer("alice");

await honcho.getPeers();

const session = honcho.session("session_1");
await session.addPeers([alice, assistant]);

await session.addMessages([
  assistant.message("What did you have for breakfast today, alice?"),
  alice.message("I had oatmeal."),
]);

const response = await alice.chat("what did alice have for breakfast today?");
console.log(response);
```

See `examples/` for more.

## Development

### Type checking

```bash
bun run typecheck
```

### Testing

Tests for the SDK live in `tests/` at the monorepo root and are run via pytest, which orchestrates a test server. Do not run `bun test` directly.

```bash
# From the monorepo root
uv run pytest tests/ -k typescript
```
