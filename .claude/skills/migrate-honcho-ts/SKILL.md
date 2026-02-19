---
name: migrate-honcho-ts
description: Migrates Honcho TypeScript SDK code from v1.6.0 to v2.0.0. Use when upgrading @honcho-ai/sdk, fixing breaking changes after upgrade, or when errors mention removed APIs like .core, getConfig, observations, or snake_case properties.
---

# Honcho TypeScript SDK Migration (v1.6.0 → v2.0.0)

## Overview

This skill migrates code from `@honcho-ai/sdk` v1.6.0 to v2.0.0 (required for Honcho 3.0.0+).

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
| `{ timeoutMs: 60000 }` | `{ timeout: 60 }` |
| `{ maxObservations: 50 }` | `{ maxConclusions: 50 }` |
| `{ includeMostDerived }` | `{ includeMostFrequent }` |
| `{ lastUserMessage }` | `{ searchQuery }` |
| `{ config: ... }` | `{ configuration: ... }` |
| `message.peer_id` | `message.peerId` |
| `message.created_at` | `message.createdAt` |
| `Observation` | `Conclusion` |
| `ObservationScope` | `ConclusionScope` |

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
