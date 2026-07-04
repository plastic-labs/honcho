---
name: migrate-honcho-ts
description: Migrates Honcho TypeScript SDK code from v1.6.0 to v2.1.1. Use when upgrading @honcho-ai/sdk, fixing breaking changes after upgrade, or when errors mention removed APIs like .core, getConfig, observations, or snake_case properties.
---

# Honcho TypeScript SDK Migration (v1.6.0 → v2.1.1)

## Overview

This skill migrates code from `@honcho-ai/sdk` v1.6.0 to v2.1.1 (required for Honcho 3.0.0+).

**Key breaking changes:**

- `@honcho-ai/core` dependency removed
- "Observation" → "Conclusion" terminology
- "Deriver" → "Queue" terminology
- `getConfig`/`setConfig` → `getConfiguration`/`setConfiguration`
- `snake_case` → `camelCase` throughout
- Streaming via `chatStream()` instead of `chat({ stream: true })`
- `Representation` class removed (returns string now)

## Quick Migration

### 1. Update dependencies

Remove `@honcho-ai/core` from package.json. The SDK now has its own HTTP client.

### 2. Replace `.core` with `.http`

```typescript
// Before
const workspace = await client.core.workspaces.getOrCreate({ id: 'my-workspace' })

// After
const response = await client.http.post('/v3/workspaces', { body: { id: 'my-workspace' } })
```

### 3. Rename configuration methods

```typescript
// Before
await honcho.getConfig()
await honcho.setConfig({ key: 'value' })
await peer.getConfig()
await session.getConfig()

// After
await honcho.getConfiguration()
await honcho.setConfiguration({ reasoning: { enabled: true } })
await peer.getConfiguration()
await session.getConfiguration()
```

### 4. Rename listing methods

```typescript
// Before
const peers = await honcho.getPeers()
const sessions = await honcho.getSessions()
const workspaces = await honcho.getWorkspaces()  // string[]

// After
const peers = await honcho.peers()
const sessions = await honcho.sessions()
const workspaces = await honcho.workspaces()  // Page<string>
```

### 5. Update streaming

```typescript
// Before
const stream = await peer.chat('Hello', { stream: true })

// After
const stream = await peer.chatStream('Hello')
```

### 6. Update observations → conclusions

```typescript
// Before
peer.observations
peer.observationsOf('bob')
maxObservations: 50
includeMostDerived: true

// After
peer.conclusions
peer.conclusionsOf('bob')
maxConclusions: 50
includeMostFrequent: true
```

### 7. Update queue status methods

```typescript
// Before
await honcho.getDeriverStatus({ observer: peer })
await honcho.pollDeriverStatus({ timeoutMs: 60000 })  // REMOVE - see note below

// After
await honcho.queueStatus({ observer: peer })
// pollDeriverStatus() has no replacement - see note below
```

**Important:** `pollDeriverStatus()` and its polling pattern have been removed entirely. Do not rely on the queue ever being empty. The queue is a continuous processing system—new messages may arrive at any time, and waiting for "completion" is not a valid pattern. If your code previously polled for queue completion, redesign it to work without that assumption.

### 8. Convert snake_case to camelCase

```typescript
// Before
message.peer_id
message.session_id
message.created_at
message.token_count
{ observe_me: true, observe_others: false }
{ created_at: '2024-01-01' }

// After
message.peerId
message.sessionId
message.createdAt
message.tokenCount
{ observeMe: true, observeOthers: false }
{ createdAt: '2024-01-01' }
```

### 9. Update representation calls

```typescript
// Before
const rep = await peer.workingRep(session, target, options)
console.log(rep.explicit)  // ExplicitObservation[]
console.log(rep.deductive) // DeductiveObservation[]

// After
const rep = await peer.representation({ session, target, ...options })
console.log(rep)  // string
```

### 10. Move updateMessage to session

```typescript
// Before
await honcho.updateMessage(message, metadata, session)

// After
await session.updateMessage(message, metadata)
```

### 11. Update card() to getCard() (v2.0.1+)

```typescript
// Before
const card = await peer.card(target)

// After (v2.0.1+)
const card = await peer.getCard(target)  // Returns string[] | null

// peer.card() still works but is deprecated — use getCard()

// New: setPeerCard / setCard
await peer.setCard(['Prefers dark mode', 'Located in US'])
```

### 12. Strict input validation (v2.0.2+)

Client constructor and all input schemas now reject unknown options via `.strict()` Zod validation.

```typescript
// Before (v2.0.1 and earlier) — silently ignored
const honcho = new Honcho({ baseUrl: 'http://...' })  // typo: baseUrl vs baseURL — silently fell back to default

// After (v2.0.2+) — throws ZodError
const honcho = new Honcho({ baseUrl: 'http://...' })  // ZodError! Use baseURL
```

### 13. peer() and session() always make API calls (v2.1.0+)

**Breaking**: `peer()` and `session()` now always make a get-or-create API call. Previously, calling without metadata/configuration returned a lazy object with no API call.

```typescript
// Before (v2.0.x) — no API call without options
const session = honcho.session('my-session')  // Lazy, no network request

// After (v2.1.0+) — always hits the API
const session = await honcho.session('my-session')  // Makes POST to /sessions (get-or-create)
```

### 14. New properties and methods (v2.1.0+)

```typescript
// createdAt on Peer and Session
const peer = await honcho.peer('user-123')
console.log(peer.createdAt)  // string | undefined

const session = await honcho.session('sess-1')
console.log(session.createdAt)  // string | undefined

// isActive on Session
console.log(session.isActive)  // boolean | undefined

// getMessage() on Session
const msg = await session.getMessage('msg-id')
```

### 15. Pagination parameters on list methods (v2.1.0+)

All list methods now accept `page`, `size`, and `reverse` parameters:

```typescript
// Before (v2.0.x) — only filters
const peers = await honcho.peers({ metadata: { role: 'admin' } })

// After (v2.1.0+) — pagination controls via options object
const peers = await honcho.peers({
  filters: { metadata: { role: 'admin' } },
  page: 2,
  size: 25,
  reverse: true
})

// Legacy raw-filter form still works:
const peers = await honcho.peers({ metadata: { role: 'admin' } })

// Works on: honcho.peers(), honcho.sessions(), honcho.workspaces(),
// peer.sessions(), session.messages(), scope.list()
```

### 16. searchQuery moved in context() (v2.1.0+)

**Breaking**: `searchQuery` removed from top-level `context()` options. Use `representationOptions.searchQuery` instead.

```typescript
// Before (v2.0.x)
await session.context({ searchQuery: '...' })

// After (v2.1.0+)
await session.context({ representationOptions: { searchQuery: '...' } })
```

### 17. Broader fetch retry logic (v2.1.1+)

The SDK now retries on all `TypeError` network failures (connection resets, DNS errors, etc.) instead of only those with `'fetch'` in the message. No code changes needed — this is transparent.

## Quick Reference Table

| v1.6.0 | v2.0.0 |
|--------|--------|
| `client.core` | `client.http` |
| `getConfig()` | `getConfiguration()` |
| `setConfig()` | `setConfiguration()` |
| `getPeers()` | `peers()` |
| `getSessions()` | `sessions()` |
| `getWorkspaces()` | `workspaces()` |
| `getDeriverStatus()` | `queueStatus()` |
| `pollDeriverStatus()` | *Removed - do not poll* |
| `peer.chat(q, { stream: true })` | `peer.chatStream(q)` |
| `peer.workingRep()` | `peer.representation()` |
| `peer.getContext()` | `peer.context()` |
| `peer.observations` | `peer.conclusions` |
| `peer.observationsOf()` | `peer.conclusionsOf()` |
| `session.getPeers()` | `session.peers()` |
| `session.getMessages()` | `session.messages()` |
| `session.getSummaries()` | `session.summaries()` |
| `session.getContext()` | `session.context()` |
| `session.workingRep()` | `session.representation()` |
| `session.peerConfig()` | `session.getPeerConfiguration()` |
| `session.setPeerConfig()` | `session.setPeerConfiguration()` |
| `{ timeoutMs: 60000 }` | `{ timeout: 60000 }` |
| `{ maxObservations: 50 }` | `{ maxConclusions: 50 }` |
| `{ includeMostDerived }` | `{ includeMostFrequent }` |
| `{ lastUserMessage }` | `{ searchQuery }` |
| `{ config: ... }` | `{ configuration: ... }` |
| `message.peer_id` | `message.peerId` |
| `message.created_at` | `message.createdAt` |
| `peer.card()` | `peer.getCard()` *(card() deprecated)* |
| *(new)* | `peer.setCard(string[])` |
| `Observation` | `Conclusion` |
| `ObservationScope` | `ConclusionScope` |
| *(new v2.1.0)* | `peer.createdAt` / `session.createdAt` |
| *(new v2.1.0)* | `session.isActive` |
| *(new v2.1.0)* | `session.getMessage(id)` |
| *(new v2.1.0)* | `page`, `size`, `reverse` on list methods |
| `context({ searchQuery })` | `context({ representationOptions: { searchQuery } })` |

## Detailed Reference

For comprehensive details on each change, see:

- [DETAILED-CHANGES.md](DETAILED-CHANGES.md) - Full API change documentation
- [MIGRATION-CHECKLIST.md](MIGRATION-CHECKLIST.md) - Step-by-step checklist

## New Error Types

```typescript
import {
  HonchoError,
  AuthenticationError,
  BadRequestError,
  NotFoundError,
  PermissionDeniedError,
  RateLimitError,
  ConflictError,
  UnprocessableEntityError,
  ServerError,
  ConnectionError,
  TimeoutError
} from '@honcho-ai/sdk'
```

## New Configuration Types

Configurations are now strongly typed:

```typescript
await honcho.setConfiguration({
  reasoning: {
    enabled: true,
    customInstructions: 'Be concise'
  },
  peerCard: { use: true, create: true },
  summary: {
    enabled: true,
    messagesPerShortSummary: 20,
    messagesPerLongSummary: 60
  },
  dream: { enabled: true }
})
```
