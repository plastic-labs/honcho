# Changelog

All notable changes to `honcho-mcp` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and host plugins.

## [Unreleased]

## [0.1.0] - 2026-09-08

### Added

- Scopes: `list_scopes`, `get_scope_sessions`, and `scope`/`sessions` on `chat` (Honcho v3.1.0+)
- Pagination on `list_sessions` and `get_session_messages`
- `workspace_chat` for reasoned questions across all peers (Honcho v3.1.0+)
