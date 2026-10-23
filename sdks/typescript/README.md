# Honcho TypeScript SDK

A high-level, ergonomic TypeScript SDK for the Honcho conversational memory platform. Provides a user-friendly API for managing peers, sessions, and conversational context.

## Installation

```
npm install @honcho-ai/sdk
```

## Usage

```ts
import { Honcho } from "@honcho-ai/sdk";

const honcho = new Honcho({
  apiKey: process.env.HONCHO_API_KEY,
  baseURL: "http://localhost:8000",
  workspaceId: "test",
});

const assistant = await honcho.peer("bob");
const alice = await honcho.peer("alice");

await honcho.peers();

const session = await honcho.session("session_1");
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

The tests in `__tests__/` run against a real Honcho server. `bun run test` starts a throwaway one (a fresh database on the Postgres named by `DB_CONNECTION_URI`, the API, and the mock LLM provider), runs `bun test` against it, and tears it down. Bare `bun test` has no server to talk to.

```bash
bun run test              # whole suite
bun run test peer.test    # bun test args pass through
```
