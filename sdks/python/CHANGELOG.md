# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.7.0] - 2026-01-13

### Added

- `ConclusionScope` object for CRUD operations on conclusions (renamed from observations)
- Representation configuration support

### Changed

- Observations renamed to Conclusions across the SDK
- Major SDK refactoring and cleanup
- Simplified method signatures throughout

### Removed

- Standalone types module (now uses honcho-core types)

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
