# Migration Checklist

Use this checklist to track migration progress. Copy into your working notes and check off items as completed.

## Dependencies

- [ ] Remove `@honcho-ai/core` from dependencies
- [ ] Update `@honcho-ai/sdk` to v2.1.1

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

## Card Method Updates (v2.0.1)

- [ ] Replace `peer.card()` with `peer.getCard()` (card() is deprecated)
- [ ] Use `peer.setCard(string[])` if setting peer cards

## Strict Validation (v2.0.2)

- [ ] Verify no constructor options or input schemas pass unknown/misspelled fields (now throws `ZodError`)
- [ ] Check for `baseUrl` vs `baseURL` typo in Honcho constructor

## peer() / session() API Call Change (v2.1.0)

- [ ] Update code that relied on lazy `peer()` / `session()` — they now always make API calls
- [ ] Ensure all `peer()` and `session()` calls are `await`ed

## New Properties (v2.1.0)

- [ ] Use `peer.createdAt` / `session.createdAt` where creation time is needed
- [ ] Use `session.isActive` where session active status is needed

## New Methods (v2.1.0)

- [ ] Use `session.getMessage(messageId)` to fetch single messages by ID

## Pagination Parameters (v2.1.0)

- [ ] Add `page`, `size`, `reverse` parameters to list calls where needed:
  - [ ] `honcho.peers()`
  - [ ] `honcho.sessions()`
  - [ ] `honcho.workspaces()`
  - [ ] `peer.sessions()`
  - [ ] `session.messages()`
  - [ ] `scope.list()`

## searchQuery Location Change (v2.1.0)

- [ ] Move `searchQuery` from top-level `context()` options to `representationOptions.searchQuery`

## Final Verification

- [ ] Run TypeScript compiler with no errors
- [ ] Run tests
- [ ] Verify streaming functionality works
- [ ] Verify configuration changes take effect
