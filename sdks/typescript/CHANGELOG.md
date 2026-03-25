# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.1.0] - 2026-03-25

### Added

- `createdAt` property on `Peer` and `Session` wrapper objects
- `isActive` property on `Session` wrapper objects
- `getMessage(messageId)` method on `Session` to fetch a single message by ID
- `Peer.representation()`, `Session.representation()`, and `Session.context()` now accept `Message` objects for `searchQuery` (extracts `.content` automatically)
- `page`, `size`, and `reverse` pagination controls on all list methods: `peers()`, `sessions()`, `messages()`, `workspaces()`, and `conclusions.list()`

### Changed

- **Breaking**: `searchQuery` removed from top-level `context()` options. Use `representationOptions.searchQuery` instead — this eliminates the duplicate parameter and is consistent with the API structure.
- **Breaking**: List methods (`peers()`, `sessions()`, `messages()`, `workspaces()`) now take an options object instead of a raw filter: `peers({ filters, page, size, reverse })` instead of `peers(filters)`.
- `RepresentationOptionsSchema` now accepts `string | MessageResponse` for `searchQuery`
- **Breaking**: `peer()` and `session()` now always make a get-or-create API call. Previously, calling without metadata/configuration returned a lazy object with no API call. All Peer/Session objects now have `createdAt` populated immediately.
- Response configuration models (`WorkspaceConfigurationResponse`, `SessionConfigurationResponse`) now tolerate unknown fields from newer servers for forward compatibility
- Reusable `PeerIdObjectSchema` and `SessionIdObjectSchema` helpers for union validation
- Moved `@types/node` from `dependencies` to `devDependencies`

## [2.0.2] - 2026-03-10

### Changed

- **Breaking**: Client constructor now rejects unknown options via `.strict()` Zod validation. Previously, misspelled options (e.g., `baseUrl` instead of `baseURL`) were silently ignored, causing the SDK to fall back to defaults. Now a `ZodError` is thrown with the unrecognized key name.

## [2.0.1] - 2026-02-09

### Added

- `setPeerCard` method

### Changed

- `card` is now `getCard` with `card` kept for backwards compatibility and marked as deprecated

## [2.0.0] - 2026-01-13

### Added

- `ConclusionScope` object for CRUD operations on conclusions (renamed from observations)
- Representation configuration support

### Changed

- Observations renamed to Conclusions across the SDK
- Major SDK refactoring and cleanup
- Simplified method signatures throughout
- Representation endpoints now return `string` instead of old Representation object

### Fixed

- Pagination `this` binding issue

### Removed

- Representation object
- Stainless "core" SDK -- this SDK is now standalone

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

## [1.4.0] - 2025-08-12

### Added

- getSummaries API returning structured summaries
- Webhook support

### Changed

- Messages can take an optional `created_at` value, defaulting to the current
  time (UTC ISO 8601)

## [1.3.0] - 2025-08-04

### Added

- Zod validation
- Added getPeerConfig to Session object

### Changed

- Moved parameters out of random `opts` dictionaries in many places
- Peer and Session objects now use inner client like python SDK

### Fixed

- Enabled missing `metadata` options in many places
- Proper default behavior for SessionPeerConfig

## [1.2.1] - 2025-07-21

### Fixed

- Order of parameters in `getSessions` endpoint

### Added

- Linting via Biome
- Adding filter parameter to various endpoints

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

- Create default workspace on Honcho client instantiation
- Simplified Honcho client import path
