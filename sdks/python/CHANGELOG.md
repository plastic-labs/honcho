# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.2.3] - 2025-08-04

### Fixed

- Added missing metadata inputs in many places
- Better documentation all over

### Added

- getPeerConfig to sessions

### Changed

- Summaries now included in toOpenAI and toAnthropic functions for context
- Length calculation for context object includes summary


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
