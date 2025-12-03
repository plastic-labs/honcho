# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.6.0] - 2025-12-03

### Added

- metadata and configuration fields to Workspace, Peer, Session, and Message objects
- Session Clone methods
- Peer level get_context method
- `ObservationScope` object to query and delete observations
- Representation object for WorkingRepresentations

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
