# Changelog

All notable changes to `honcho-mcp` will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

This package versions independently of the Honcho API, `@honcho-ai/sdk`, and host plugins.

## [Unreleased]

## [3.0.1] - 2026-09-22

### Added

- `list_peers` takes `page`, `size` (max 100) and `reverse`, so a client can
  walk past the first page of peers.
- Every tool carries MCP annotations: a title, and `readOnlyHint` or
  `destructiveHint`, so hosts can label tools and tell reads from writes.

### Changed

- The server instructions describe two modes. Recall mode only reads
  (`chat`, `workspace_chat`, `search`, and the other read tools) and is the
  default. Memory-store mode sets up sessions and records messages, and agents
  are told to use it only when the user asks for the conversation to be
  recorded. The per-tool listings are gone, since tool descriptions already
  carry them.

### Fixed

- `chat` and `workspace_chat` no longer fail when the answer takes more than
  10 seconds on the HTTP host (Bun's idle timeout dropped the connection) or
  more than 60 seconds on any host (the SDK timed out and retried the chat).
  Honcho API requests now time out after 5 minutes, configurable with
  `HONCHO_TIMEOUT_MS`.

## [0.2.0] - 2026-09-17

### Added

- Conclusion attribution (Honcho v3.2.0+). `list_conclusions` and
  `query_conclusions` now return `level`, `source_ids` and `times_derived`
  alongside the content the server was already sending, so a client can tell an
  extracted fact from a dreamed one and see what a derived conclusion rests on.
- `get_conclusions` fetches conclusions by ID from anywhere in the workspace,
  with no observer/observed pair required, and reports which of the requested
  IDs no longer exist. Feed it a conclusion's `source_ids` to walk a reasoning
  chain down to the explicit facts underneath it.
- `get_derived_conclusions` walks the same edge upward, listing what was built
  on top of a given conclusion — worth checking before deleting or correcting
  one.
- `include_evidence` on `chat` and `workspace_chat` (Honcho v3.2.0+). Opting in
  returns the answer together with the conclusions and messages the agent read
  and the tools it called, for auditing what an answer was built from. Evidence
  is collated from what the agent accessed rather than reported by the model, so
  it over-reports and costs no extra model tokens. Without the flag both tools
  return the bare answer exactly as before.
- Paging and filtering on `list_conclusions`: `page`, `size`, `reverse`,
  `session_id`, and a `filters` passthrough. Previously it always returned the
  first page of 50 with no way to narrow it.

### Changed

- Requires `@honcho-ai/sdk` 2.5.0 (was 2.4.0), which carries the attribution
  fields and the evidence-bearing chat response.

## [0.1.1] - 2026-09-11

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
