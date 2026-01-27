# Migration Checklist

Use this checklist to track migration progress. Copy into your working notes and check off items as completed.

## Dependencies

- [ ] Update `honcho` package to v2.0.0
- [ ] Remove any `honcho-core` imports

## Async Architecture Changes

- [ ] Remove `AsyncHoncho` imports → use `Honcho` with `.aio` accessor
- [ ] Remove `AsyncPeer` imports → use `Peer` with `.aio` accessor
- [ ] Remove `AsyncSession` imports → use `Session` with `.aio` accessor
- [ ] Update all async client usage to use `.aio` accessor pattern
- [ ] Update type hints: `AsyncPeer` → `Peer`, `AsyncSession` → `Session`

## Terminology: Observations → Conclusions

- [ ] Replace `Observation` import with `Conclusion`
- [ ] Replace `ObservationScope` import with `ConclusionScope`
- [ ] Replace `AsyncObservationScope` import with `ConclusionScopeAio`
- [ ] Replace `.observations` property with `.conclusions`
- [ ] Replace `.observations_of()` method with `.conclusions_of()`
- [ ] Replace `.get_representation()` with `.representation()`

## Representation Changes

- [ ] Remove `Representation` import (now returns `str`)
- [ ] Remove `ExplicitObservation` import
- [ ] Remove `DeductiveObservation` import
- [ ] Replace `working_rep()` with `representation()`
- [ ] Update type hints from `Representation` to `str`
- [ ] Remove `.explicit` property access
- [ ] Remove `.deductive` property access
- [ ] Replace `.is_empty()` checks with `not rep`
- [ ] Remove `.merge_representation()` calls
- [ ] Remove `.diff_representation()` calls
- [ ] Remove `.str_no_timestamps()` calls
- [ ] Remove `.format_as_markdown()` calls

## Configuration Changes

- [ ] Replace all `config=` parameters with `configuration=`
- [ ] Replace `.get_config()` with `.get_configuration()`
- [ ] Replace `.set_config()` with `.set_configuration()`
- [ ] Rename `.get_peer_config()` → `.get_peer_configuration()`
- [ ] Rename `.set_peer_config()` → `.set_peer_configuration()`
- [ ] Import typed config classes from `honcho.api_types` if needed:
  - [ ] `PeerConfig`
  - [ ] `SessionConfiguration`
  - [ ] `WorkspaceConfiguration`

## Method Renames

### Peer Methods

- [ ] `peer.working_rep()` → `peer.representation()`
- [ ] `peer.get_context()` → `peer.context()`
- [ ] `peer.get_sessions()` → `peer.sessions()`
- [ ] `peer.chat(stream=True)` → `peer.chat_stream()`

### Session Methods

- [ ] `session.get_context()` → `session.context()`
- [ ] `session.get_summaries()` → `session.summaries()`
- [ ] `session.get_messages()` → `session.messages()`
- [ ] `session.get_peers()` → `session.peers()`
- [ ] `session.get_peer_config()` → `session.get_peer_configuration()`
- [ ] `session.set_peer_config()` → `session.set_peer_configuration()`
- [ ] `session.working_rep()` → `session.representation()`
- [ ] `session.get_deriver_status()` → `session.queue_status()`
- [ ] Remove `session.poll_deriver_status()` calls

### Client Methods

- [ ] `client.get_peers()` → `client.peers()`
- [ ] `client.get_sessions()` → `client.sessions()`
- [ ] `client.get_workspaces()` → `client.workspaces()`
- [ ] `client.get_deriver_status()` → `client.queue_status()`
- [ ] Remove `client.poll_deriver_status()` calls
- [ ] Move `client.update_message()` → `session.update_message()`

## Parameter Renames

- [ ] `include_most_derived=` → `include_most_frequent=`
- [ ] `max_observations=` → `max_conclusions=`
- [ ] `last_user_message=` → `search_query=`

## Return Type Changes

- [ ] Handle `card()` returning `list[str] | None` instead of `str`
- [ ] Handle `.list()` on conclusions returning `SyncPage` instead of `list`

## Removed Features

- [ ] Remove any usage of `client.core` property
- [ ] Remove usage of `"demo"` environment (only `"local"` and `"production"` remain)
- [ ] Implement custom polling if you were using `poll_deriver_status()`

## Type Import Updates

- [ ] Replace `PeerContext` import with `PeerContextResponse` from `honcho.api_types`
- [ ] Replace `DeriverStatus` import with `QueueStatusResponse` from `honcho.api_types`
- [ ] Replace `MessageCreateParam` with `MessageCreateParams` (plural)
- [ ] Move `SessionPeerConfig` import from `honcho.session` to `honcho.api_types`

## Exception Handling (Optional)

- [ ] Update exception handling to use new exception types if needed:
  - `HonchoError`, `APIError`, `BadRequestError`, `AuthenticationError`
  - `PermissionDeniedError`, `NotFoundError`, `ConflictError`
  - `UnprocessableEntityError`, `RateLimitError`, `ServerError`
  - `TimeoutError`, `ConnectionError`

## Final Verification

- [ ] Run type checker (mypy/pyright) with no errors
- [ ] Run tests
- [ ] Verify async operations work with `.aio` accessor
- [ ] Verify streaming functionality works with `chat_stream()`
- [ ] Verify configuration changes take effect
