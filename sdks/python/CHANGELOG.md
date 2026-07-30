# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.2.0] - 2026-07-02

### Added

- `ConclusionLevel` type (`explicit`, `deductive`, `inductive`, `contradiction`) and a `level` field on `Conclusion`, exposing the reasoning level the server already tracked but previously stripped from responses.
- `filters` parameter on `ConclusionScope.list()` and `ConclusionScope.query()` (sync and async), passed through to the same dynamic server-side filter logic as `peers()`/`sessions()`/`messages()`. Filter explicit-only conclusions with `filters={"level": "explicit"}`, or by any other supported field/operator. Requires a Honcho server with the matching API support (Honcho v3.0.11+).

### Fixed

- Scope-managed filter keys (`observer`, `observed`, `session`) are now rejected with a clear `ValueError` if passed in `filters`, instead of silently overriding the scope and returning conclusions from a different peer pair. Use `peer.conclusions` / `conclusions_of(target)` and the `session=` parameter instead. `session_id` remains a valid filter on `query()`.

## [2.1.2] - 2026-05-21

### Added

- `page`, `size`, and `reverse` pagination parameters on `Honcho.workspaces()` and `HonchoAio.workspaces()`, closing the gap from 2.1.0 which added these to `peers()`, `sessions()`, `messages()`, and `conclusions.list()` but not to `workspaces()`. Honoring `reverse` on the workspace/peer/session list routes also requires a Honcho server with the matching API fix; older servers silently ignore the parameter.
- `peers` parameter on `Honcho.session()` and `HonchoAio.session()` — attach peers to a session at creation time instead of needing a follow-up `session.add_peers()` call. Accepts the same shapes as `Session.add_peers` (peer ID string, `Peer` object, list of either, or tuples with `SessionPeerConfig`).

### Changed

- `WorkspaceCreateParams`, `PeerCreateParams`, and `SessionCreateParams` now accept IDs up to 512 characters (was 100), matching the server-side schema change in Honcho v3.0.7.

## [2.1.1] - 2026-04-01

### Fixed

- Broadened HTTP retry logic to cover `httpx.NetworkError` and `httpx.RemoteProtocolError` in addition to `httpx.TimeoutException` and `httpx.ConnectError`, improving resilience against transient network failures

## [2.1.0] - 2026-03-25

### Added

- `created_at` property on `Peer` and `Session` objects
- `is_active` property on `Session` objects
- `get_message(message_id)` method on `Session` (sync and async) to fetch a single message by ID
- `page`, `size`, and `reverse` pagination parameters on all list methods: `peers()`, `sessions()`, `messages()`, and `conclusions.list()`

### Changed

- **Breaking**: `peer()` and `session()` now always make a get-or-create API call. Previously, calling without metadata/configuration returned a lazy object with no API call. All Peer/Session objects now have `created_at` populated immediately.
- Response configuration models (`WorkspaceConfigurationResponse`, `SessionConfigurationResponse`) now tolerate unknown fields from newer servers for forward compatibility

### Fixed

- Sync and async `Session.get_metadata()`, `get_configuration()`, and `refresh()` now refresh cached `created_at` and `is_active` values along with metadata and configuration.
- `honcho.__version__` now derives from package metadata, with a `pyproject.toml` fallback in source checkouts, so it stays aligned with SDK releases.

## [2.0.2] - 2026-03-10

### Changed

- All input models now reject unknown fields via `extra="forbid"` Pydantic validation. Previously, misspelled or extraneous fields were silently ignored. Now a `ValidationError` is raised with the unrecognized field name.

## [2.0.1] - 2026-02-09

### Added

- `set_peer_card` method

### Changed

- `card` is now `get_card` with `card` kept for backwards compatibility and marked as deprecated

## [2.0.0] - 2026-01-13

### Added

- `ConclusionScope` object for CRUD operations on conclusions (renamed from observations)
- Representation configuration support

### Changed

- Observations renamed to Conclusions across the SDK
- Major SDK refactoring and cleanup
- Simplified method signatures throughout
- Representation endpoints now return `string` instead of old Representation object

### Removed

- Standalone types module (now uses honcho-core types)
- Representation object

## [1.6.0] - 2025-12-03

### Added

- metadata and configuration fields to Workspace, Peer, Session, and Message objects
- Session Clone methods
- Peer level get_context method
- `ObservationScope` object to perform CRUD operations on observations
- Representation object for WorkingRepresentations

### Changed

- methods that take IDs, can all optionally take an object of the same type

## [1.5.0] - 2025-10-09

### Added

- Delete workspace method

### Changed

- message_id of `Summary` model is a string nanoid
- Get Context can return Peer Card & Peer Representation

## [1.4.1] — 2025-10-01

### Added

- Get Peer Card method
- Update Message metadata method
- Session level deriver status methods
- Delete session message

### Fixed

- Dialectic Stream returns Iterators
- Type warnings

### Changed

- Pagination class to match core implementation

## [1.4.0] — 2025-08-12

### Added

- getSummaries API returning structured summaries
- Webhook support

### Changed

- Messages can take an optional `created_at` value, defaulting to the current
  time (UTC ISO 8601)

## [1.3.0] - 2025-08-04

### Added

- Added get_peer_config to sessions module

### Changed

- Summaries are now included in `toOpenAI` and `toAnthropic` functions
- `SessionContext.__len__` now counts the summary in its total
- `filter` keyword changed to `filters`

### Fixed

- Added missing metadata inputs in many places
- Better documentation all over

## [1.2.2] - 2025-07-21

### Added

- Filter parameter to various endpoints

## [1.2.1] - 2025-07-17

### Fixed

- honcho utils import path

## [1.2.0] - 2025-07-16

### Added

- Get/poll deriver queue status endpoints added to workspace
- Added endpoint to upload files as messages

### Removed

- Removed peer messages in accordance with Honcho 2.1.0

### Changed

- Updated chat endpoint to use singular `query` in accordance with Honcho 2.1.0

## [1.1.0] - 2025-07-08

### Fixed

- Properly handle AsyncClient
