# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.3.0] - 2025-08-04

### Added

- Zod validation
- Added getPeerConfig to Session object

### Changed

- Moved parameters out of random `opts` dictionaries in many places
- Peer and Session objects now use inner client like python SDK
- `SessionContext.length` now counts the summary in its total as +1 messages if it exists

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
