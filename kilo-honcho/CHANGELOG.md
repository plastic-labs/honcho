# Changelog

All notable changes to this package are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the package uses [Semantic Versioning](https://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and other host plugins.

## [Unreleased]

## [0.1.0] - Unreleased

### Added

- First release of the Kilo Code plugin, forked from `@honcho-ai/opencode-honcho` 0.2.1. Kilo's plugin API matches OpenCode 1.x, so the hooks, tools, TUI commands and transcript import carry over.
- Kilo defaults: the `hosts.kilo` config block, workspace and agent peer `kilo`, telemetry headers `X-Honcho-Host: kilo` and `X-Honcho-Plugin: kilo-honcho`, and the `KILO_HONCHO_CONFIG_PATH` override.
- The `honcho-memory` skill installs to `~/.config/kilo/skills`, or `$KILO_CONFIG_DIR/skills`, where Kilo looks for skills.
- `.kilo` and `.kilocode` count as project-root markers alongside `.git`.

### Removed

- The OpenCode 2.x adapter. Kilo uses the 1.x plugin API.
