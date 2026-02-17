# Migration Checklist

Use this checklist to track migration progress. Copy into your working notes and check off items as completed.

## Dependencies

- [ ] Remove `@honcho-ai/core` from dependencies
- [ ] Update `@honcho-ai/sdk` to v2.0.0

## Client-Level Changes

- [ ] Replace all `.core` usages with `.http` or remove
- [ ] Rename `getConfig()` → `getConfiguration()`
- [ ] Rename `setConfig()` → `setConfiguration()`
- [ ] Rename `getPeers()` → `peers()`
- [ ] Rename `getSessions()` → `sessions()`
- [ ] Rename `getWorkspaces()` → `workspaces()` (returns `Page<string>` now)
- [ ] Rename `getDeriverStatus()` → `queueStatus()`
- [ ] Remove `pollDeriverStatus()` calls entirely (no replacement—do not rely on queue being empty)
- [ ] Move `updateMessage()` calls from client to session

## Peer-Level Changes

- [ ] Replace `peer.chat(q, { stream: true })` with `peer.chatStream(q)`
- [ ] Rename `getSessions()` → `sessions()`
- [ ] Rename `getConfig()` → `getConfiguration()`
- [ ] Rename `setConfig()` → `setConfiguration()`
- [ ] Rename `peerConfig()` → `getPeerConfiguration()`
- [ ] Rename `setPeerConfig()` → `setPeerConfiguration()`
- [ ] Rename `workingRep()` → `representation()` (returns string now)
- [ ] Rename `getContext()` → `context()`
- [ ] Replace `observations` → `conclusions`
- [ ] Replace `observationsOf()` → `conclusionsOf()`
- [ ] Handle `card()` returning `string[] | null` instead of `string`

## Session-Level Changes

- [ ] Rename `getPeers()` → `peers()` (returns `Peer[]` now, not `Page<Peer>`)
- [ ] Rename `getMessages()` → `messages()`
- [ ] Rename `getConfig()` → `getConfiguration()`
- [ ] Rename `setConfig()` → `setConfiguration()`
- [ ] Rename `getContext()` → `context()`
- [ ] Rename `getSummaries()` → `summaries()`
- [ ] Rename `getDeriverStatus()` → `queueStatus()`
- [ ] Remove `pollDeriverStatus()` calls entirely (no replacement—do not rely on queue being empty)
- [ ] Rename `workingRep()` → `representation()` (returns string now)

## Terminology Changes

- [ ] Rename `maxObservations` → `maxConclusions`
- [ ] Rename `includeMostDerived` → `includeMostFrequent`
- [ ] Rename `lastUserMessage` → `searchQuery`
- [ ] Rename `Observation` type → `Conclusion`
- [ ] Rename `ObservationScope` type → `ConclusionScope`

## snake_case → camelCase

- [ ] Update all `{ config: ... }` to `{ configuration: ... }`
- [ ] Update `observe_me` → `observeMe`
- [ ] Update `observe_others` → `observeOthers`
- [ ] Update `created_at` → `createdAt`
- [ ] Update message property access:
  - [ ] `peer_id` → `peerId`
  - [ ] `session_id` → `sessionId`
  - [ ] `workspace_id` → `workspaceId`
  - [ ] `created_at` → `createdAt`
  - [ ] `token_count` → `tokenCount`
- [ ] Update summary property access:
  - [ ] `message_id` → `messageId`
  - [ ] `summary_type` → `summaryType`

## Configuration Objects

- [ ] Update workspace configuration to typed structure
- [ ] Update session configuration to typed structure
- [ ] Update peer configuration to typed structure
- [ ] Replace `deriver` config with `reasoning` config

## Error Handling

- [ ] Update error handling to use new error types if needed

## Type Imports

- [ ] Remove imports of deleted types:
  - `Observation`, `ObservationScope`, `ObservationData`
  - `Representation`, `RepresentationData`
  - `ExplicitObservation`, `DeductiveObservation`
  - `DialecticStreamDelta`
  - `DeriverStatusOptions`
  - `MessageCreate`, `ValidatedMessageCreate`
  - `WorkingRepParams`
- [ ] Add imports of new types as needed:
  - `Conclusion`, `ConclusionScope`
  - `MessageInput`
  - `QueueStatusOptions`
  - Error types

## Representation Handling

- [ ] Remove usage of `Representation` class methods (`.explicit`, `.deductive`, `.isEmpty()`, `.diff()`)
- [ ] Handle representation as plain string

## Final Verification

- [ ] Run TypeScript compiler with no errors
- [ ] Run tests
- [ ] Verify streaming functionality works
- [ ] Verify configuration changes take effect
