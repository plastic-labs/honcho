# Changelog

All notable changes to `honcho-mcp` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and host plugins.

## [Unreleased]

### Added

- Scope membership tools: `create_scope`, `add_sessions_to_scope`,
  `remove_session_from_scope`, and `get_scope_status` (Honcho v3.1.0+), so
  clients can provision recall boundaries instead of only reading them.
- Scope read options on the remaining read tools, matching the SDKs
  (Honcho v3.1.0+): `scopes` on `create_session`, `scope`/`sessions` on
  `get_representation`, `scope` on `search`, and `peer_target` /
  `peer_perspective` / `scope` / `limit_to_session` on `get_session_context`
  (which now returns `peer_representation` and `peer_card` when a target is
  given).

## [0.1.0] - 2026-09-09

### Added

- Scopes: `list_scopes`, `get_scope_sessions`, and `scope`/`sessions` on `chat` (Honcho v3.1.0+)
- Pagination on `list_sessions` and `get_session_messages`
- `workspace_chat` for reasoned questions across all peers (Honcho v3.1.0+)
- Every Honcho request carries `X-Honcho-Host: honcho-mcp/<version>`.
  `X-Honcho-Plugin` is the caller's `User-Agent`, verbatim (absent on stdio
  and when the caller sends none). `X-Honcho-Agent-Model` is never sent.
