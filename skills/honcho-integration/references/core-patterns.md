# Core Integration Patterns

The base SDK boilerplate for any Honcho integration: choosing sync vs async, initializing the client, creating peers, configuring sessions, and adding messages. Read this once you've chosen your entities and session structure (Phases 1–2). For the agent-facing recall patterns (tool call, pre-fetch, `context()`), see `agent-patterns.md`.

## Sync vs Async

**TypeScript** — The SDK is async by default. All methods return promises. No separate sync API.

**Python** — The SDK provides both sync and async interfaces:

- **Sync** (default): `from honcho import Honcho` — use in sync frameworks (Flask, Django, CLI scripts)
- **Async**: `from honcho import Honcho` with `.aio` namespace — use in async frameworks (FastAPI, Starlette, async workers)

```python
# Sync usage (Flask, Django, scripts)
from honcho import Honcho
honcho = Honcho(workspace_id="my-app", api_key=os.environ["HONCHO_API_KEY"])
peer = honcho.peer("user-123")
response = peer.chat("What does this user prefer?")

# Async usage (FastAPI, Starlette)
from honcho import Honcho
honcho = Honcho(workspace_id="my-app", api_key=os.environ["HONCHO_API_KEY"])
peer = await honcho.aio.peer("user-123")
response = await peer.aio.chat("What does this user prefer?")
```

Match the client to the framework — check whether the codebase uses `async def` handlers or sync `def` handlers and choose accordingly. The examples below show sync Python; swap to `.aio` equivalents for async codebases.

## 1. Initialize with a Single Workspace

Use ONE workspace for your entire application. The workspace name should reflect your app/product.

**Python:**

```python
from honcho import Honcho
import os

# Sync client (Flask, Django, scripts)
honcho = Honcho(
    workspace_id="your-app-name",
    api_key=os.environ["HONCHO_API_KEY"],
    environment="production"
)

# Async client (FastAPI, Starlette) — use honcho.aio for all operations
# honcho.aio.peer(), honcho.aio.session(), etc.
```

**TypeScript:**

```typescript
import { Honcho } from '@honcho-ai/sdk';

// All methods are async by default
const honcho = new Honcho({
    workspaceId: "your-app-name",
    apiKey: process.env.HONCHO_API_KEY,
    environment: "production"
});
```

## 2. Create Peers for ALL Entities

Create peers for **every entity** in your business logic - users AND AI assistants.

**Python:**

```python
from honcho.api_types import PeerConfig

# Human users
user = honcho.peer("user-123")

# AI assistants - set observe_me=False so Honcho doesn't model the AI
assistant = honcho.peer("assistant", configuration=PeerConfig(observe_me=False))
support_bot = honcho.peer("support-bot", configuration=PeerConfig(observe_me=False))
```

**TypeScript:**

```typescript
// Human users
const user = await honcho.peer("user-123");

// AI assistants - set observeMe=false so Honcho doesn't model the AI
const assistant = await honcho.peer("assistant", { configuration: { observeMe: false } });
const supportBot = await honcho.peer("support-bot", { configuration: { observeMe: false } });
```

## 3. Multi-Peer Sessions

Sessions can have multiple participants. Configure observation settings per-peer.

**Python:**

```python
from honcho.api_types import SessionPeerConfig

session = honcho.session("conversation-123")

# User is observed (Honcho builds a model of them)
user_config = SessionPeerConfig(observe_me=True, observe_others=True)

# AI is NOT observed (no model built of the AI)
ai_config = SessionPeerConfig(observe_me=False, observe_others=True)

session.add_peers([
    (user, user_config),
    (assistant, ai_config)
])
```

**TypeScript:**

```typescript
const session = await honcho.session("conversation-123");

await session.addPeers([
    [user, { observeMe: true, observeOthers: true }],
    [assistant, { observeMe: false, observeOthers: true }]
]);
```

## 4. Add Messages to Sessions

**Python:**

```python
session.add_messages([
    user.message("I'm having trouble with my account"),
    assistant.message("I'd be happy to help. What seems to be the issue?"),
    user.message("I can't reset my password")
])
```

**TypeScript:**

```typescript
await session.addMessages([
    user.message("I'm having trouble with my account"),
    assistant.message("I'd be happy to help. What seems to be the issue?"),
    user.message("I can't reset my password")
]);
```
