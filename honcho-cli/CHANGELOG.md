# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.1.2] - 2026-07-20

### Added

- Device-code OAuth login for managed Honcho servers. `honcho init` now offers browser-based login (RFC 8628 device authorization grant) when the host advertises the device grant in its OAuth authorization-server metadata; tokens are persisted to `~/.honcho/config.json` and auto-refreshed (#891)

## [0.1.1] - 2026-06-15

### Fixed

- Declare `click` as an explicit dependency. The CLI imported `click` directly but relied on it being pulled in transitively, so installs without it on the path could fail at runtime (#787)

## [0.1.0] - 2026-04-20

### Added

- Initial release of `honcho-cli` — a terminal for inspecting and managing a Honcho deployment (#424)
- `workspace`, `peer`, `session`, `message`, `conclusion`, and `config` command groups for managing resources against any Honcho server
- `init` onboarding flow that prompts for and persists connection settings, with flag/env-var pre-seeding for non-interactive use
- Per-command flags, environment variables, and a config file for pointing the CLI at different servers (local, self-hosted, or hosted)
- Rich terminal output and an agent-usage mode for scripting against the CLI
- Documentation and an agent skill for the CLI (#589)
