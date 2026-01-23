# Detailed API Changes

## Client Changes

### `.core` Property Removed

The `.core` property (which exposed the raw `@honcho-ai/core` client) has been removed. Use `.http` for advanced HTTP access.

```typescript
// Before
const workspace = await client.core.workspaces.getOrCreate({ id: 'my-workspace' })

// After - SDK handles workspace creation automatically
// For advanced usage:
const response = await client.http.post('/v3/workspaces', { body: { id: 'my-workspace' } })
```

### Listing Methods Return Type Changes

- `workspaces()` now returns `Page<string>` instead of `string[]`
- `session.peers()` now returns `Peer[]` instead of `Page<Peer>`

```typescript
const workspacePage = await honcho.workspaces()
for (const id of workspacePage.items) {
  console.log(id)
}
```

### `updateMessage()` Moved to Session

```typescript
// Before
await honcho.updateMessage(message, { key: 'value' }, session)

// After
await session.updateMessage(message, { key: 'value' })
```

### `config` Option Renamed to `configuration`

```typescript
// Before
const peer = await honcho.peer('user-id', { config: { observe_me: true } })
const session = await honcho.session('session-id', { config: { ... } })

// After
const peer = await honcho.peer('user-id', { configuration: { observeMe: true } })
const session = await honcho.session('session-id', { configuration: { reasoning: { enabled: true } } })
```

---

## Peer Changes

### Streaming API

The `stream` option on `chat()` has been removed. Use `chatStream()` instead.

```typescript
// Before
const stream = await peer.chat('Hello', { stream: true })
for await (const chunk of stream) {
  process.stdout.write(chunk)
}

// After
const stream = await peer.chatStream('Hello')
for await (const chunk of stream) {
  process.stdout.write(chunk)
}
```

Non-streaming `chat()` now only returns `string | null`:

```typescript
const response = await peer.chat('Hello')  // Returns string | null
```

### New `reasoningLevel` Option

```typescript
const response = await peer.chat('Complex question', {
  reasoningLevel: 'high'  // 'minimal' | 'low' | 'medium' | 'high' | 'max'
})
```

### `workingRep()` Renamed to `representation()`

```typescript
// Before
const rep = await peer.workingRep(session, target, options)
console.log(rep.toString())
console.log(rep.explicit)
console.log(rep.deductive)

// After
const rep = await peer.representation({
  session,
  target,
  searchQuery: options?.searchQuery,
  maxConclusions: options?.maxObservations,
  includeMostFrequent: options?.includeMostDerived,
})
console.log(rep)  // Returns string directly
```

### `getContext()` Renamed to `context()`

Options are now passed as a single object:

```typescript
// Before
const ctx = await peer.getContext(target, options)

// After
const ctx = await peer.context({ target, ...options })
```

### `card()` Return Type Changed

```typescript
// Before
const card = await peer.card(target)  // Returns string

// After
const card = await peer.card(target)  // Returns string[] | null
```

### `message()` Options Changed

```typescript
// Before
const msg = peer.message('Hello', {
  metadata: { key: 'value' },
  configuration: { deriver: { enabled: true } },
  created_at: '2024-01-01T00:00:00Z'
})
// Returns ValidatedMessageCreate with peer_id, created_at

// After
const msg = peer.message('Hello', {
  metadata: { key: 'value' },
  configuration: { reasoning: { enabled: true } },
  createdAt: '2024-01-01T00:00:00Z'
})
// Returns MessageInput with peerId, createdAt
```

### `PeerContext.representation` Type Changed

```typescript
// Before
const ctx = await peer.getContext()
if (ctx.representation) {
  console.log(ctx.representation.explicit)  // Representation object
  console.log(ctx.representation.deductive)
}

// After
const ctx = await peer.context()
if (ctx.representation) {
  console.log(ctx.representation)  // Now a string
}
```

---

## Session Changes

### `getPeers()` Return Type Changed

```typescript
// Before
const peers = await session.getPeers()  // Returns Page<Peer>

// After
const peers = await session.peers()  // Returns Peer[]
```

### `getContext()` Renamed to `context()`

```typescript
// Before
const ctx = await session.getContext({
  summary: true,
  peerTarget: user,
  peerPerspective: assistant,
  lastUserMessage: "What are my preferences?",
  representationOptions: {
    maxObservations: 50,
    includeMostDerived: true
  }
})

// After
const ctx = await session.context({
  summary: true,
  peerTarget: user,
  peerPerspective: assistant,
  searchQuery: "What are my preferences?",
  representationOptions: {
    maxConclusions: 50,
    includeMostFrequent: true
  }
})
```

### `SessionPeerConfig` Uses camelCase and Methods Renamed

```typescript
// Before
await session.setPeerConfig(peer, {
  observe_me: true,
  observe_others: false
})
const config = await session.peerConfig(peer)

// After
await session.setPeerConfiguration(peer, {
  observeMe: true,
  observeOthers: false
})
const config = await session.getPeerConfiguration(peer)
```

---

## Message Changes

### Message Properties Use camelCase

```typescript
// Before (from @honcho-ai/core)
message.peer_id
message.session_id
message.workspace_id
message.created_at
message.token_count

// After
message.peerId
message.sessionId
message.workspaceId
message.createdAt
message.tokenCount
```

### MessageInput Type

```typescript
// Before
interface ValidatedMessageCreate {
  peer_id: string
  content: string
  metadata?: Record<string, unknown>
  configuration?: Record<string, unknown>
  created_at?: string
}

// After
interface MessageInput {
  peerId: string
  content: string
  metadata?: Record<string, unknown>
  configuration?: MessageConfiguration
  createdAt?: string
}
```

---

## Streaming Changes

### `DialecticStreamDelta` Removed

```typescript
// Before
import { DialecticStreamDelta, DialecticStreamChunk } from '@honcho-ai/sdk'

// After
import { DialecticStreamChunk, DialecticStreamResponse } from '@honcho-ai/sdk'
```

---

## Configuration Changes

### Workspace Configuration

Configurations are now strongly typed objects instead of `Record<string, unknown>`.

```typescript
// Before
await honcho.setConfig({
  deriver: { enabled: true },
  some_custom_key: 'value'
})

// After
await honcho.setConfiguration({
  reasoning: {
    enabled: true,
    customInstructions: 'Be concise'
  },
  peerCard: {
    use: true,
    create: true
  },
  summary: {
    enabled: true,
    messagesPerShortSummary: 20,
    messagesPerLongSummary: 60
  },
  dream: {
    enabled: true
  }
})
```

### Peer Configuration

```typescript
// Before
await peer.setConfig({ observe_me: false })

// After
await peer.setConfiguration({ observeMe: false })
```

### Message Configuration

```typescript
// Before
peer.message('Hello', {
  configuration: {
    deriver: { enabled: true }
  }
})

// After
peer.message('Hello', {
  configuration: {
    reasoning: {
      enabled: true,
      customInstructions: 'Focus on emotions'
    }
  }
})
```

---

## Type Changes

### Removed Exports

- `Observation` (use `Conclusion`)
- `ObservationScope` (use `ConclusionScope`)
- `ObservationData`, `ObservationCreateParam`, `ObservationQueryParams`
- `Representation`, `RepresentationData`, `RepresentationOptions` (class removed)
- `ExplicitObservation`, `DeductiveObservation`
- `DialecticStreamDelta`
- `DeriverStatusOptions` (use `QueueStatusOptions`)
- `MessageCreate` (use `MessageInput`)
- `WorkingRepParams`

### New Exports

```typescript
import {
  // Domain classes
  Conclusion,
  ConclusionScope,
  ConclusionCreateParams,

  // Error types
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
  TimeoutError,

  // Message types
  Message,
  MessageInput,

  // Configuration types
  WorkspaceConfig,
  SessionConfig,
  PeerConfig,
  SessionPeerConfig,
  MessageConfiguration,
  ReasoningConfig,
  PeerCardConfig,
  SummaryConfig,
  DreamConfig,

  // API response types
  QueueStatus,
  QueueStatusOptions,
  RepresentationOptions,
  ConclusionQueryParams,
  ConclusionResponse,
} from '@honcho-ai/sdk'
```

### SummaryData Type Changed

```typescript
// Before
interface SummaryData {
  content: string
  message_id: string
  summary_type: string
  created_at: string
  token_count: number
}

// After
interface SummaryData {
  content: string
  messageId: string
  summaryType: string
  createdAt: string
  tokenCount: number
}
```
