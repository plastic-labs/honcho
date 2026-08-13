# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased]

### Added

- `honcho session view` — session transcript table (`--last N`, `--page N --size M`, `--all`, `--reverse`, `--ids`, peer filter via `-p`). Content is shown verbatim, timestamps are normalized to UTC, and the command is read-only: unlike the other session commands it never get-or-creates the session

### Fixed

- `honcho message list --last N` no longer stops at the first page of 50 — it walks pages to fill the requested window

## [0.1.2] - 2026-07-20

### Added

- Device-code OAuth login for managed Honcho servers. `honcho init` now offers browser-based login (RFC 8628 device authorization grant) when the host advertises the device grant in its OAuth authorization-server metadata; tokens are persisted to `~/.honcho/config.json` and auto-refreshed (#891)
- `HONCHO_CONFIG_DIR` environment variable for pointing the CLI at an alternate config directory (#891)

### Changed

- An OAuth grant now records the host it was minted against and is ignored — neither used nor refreshed — when `base_url` points elsewhere, so a staging grant is never sent to production. A live OAuth token takes precedence over a stored `apiKey`, and a dead grant degrades to the saved key with a warning instead of aborting. Device login no longer deletes the shared `apiKey`, which sibling tools read from the same config file (#891)

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
