# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2025-05-13

### Breaking Changes

- **R-03**: `Session::context_with_options` now takes `&SessionContextOptions` instead of `(bool, bool)`. Use the builder pattern for options.
- **R-07**: `Page::next_page` now returns `Result<Option<Page<T>>>` instead of `Option<Page<T>>`. HTTP errors propagate as `Err` instead of being silently swallowed as `None`.
- **R-08**: `collect_all_pages` now returns `Result<Vec<T>>` instead of `Vec<T>`. Pagination errors are no longer silently dropped.
- **R-20**: Session message methods (`add_messages`, `get_message`, `update_message`, `search`, `search_with_options`) now return `Message` (accessor methods) instead of `MessageResponse` (direct field access). Blocking equivalents also updated.
- **R-22**: All `bon::Builder` structs use `finish_fn = build`. No migration needed if already calling `.build()`.
- **R-27**: `DialecticStream::final_response` returns `FinalResponse` struct (`.content` field) instead of `&str`.

### Added

- `Conclusion` wrapper with accessor methods and custom `Debug`/`Display`
- `ConclusionScope` for self-scoped and cross-peer conclusion access
- `ConclusionScope::create`, `create_batch`, `list`, `get`, `delete`, `query` (semantic search), `representation`
- `Peer::conclusions()` and `Peer::conclusions_of(target)` for scoped access
- `Peer::representation_builder()` with fine-grained parameters (search_query, search_top_k, search_max_distance, include_most_frequent, max_conclusions)
- `Peer::chat_with_options` for full dialectic options (session, target, reasoning level)
- `Peer::chat_stream` builder with `.target()`, `.session()`, `.reasoning_level()` chainable methods
- `Peer::context_with_target` for scoped context retrieval
- `Peer::get_card`, `get_card_with_target`, `set_card`, `set_card_with_target` for peer card CRUD
- `Peer::sessions` and `sessions_with_options` for paginated session listing
- `Peer::search` and `search_with_options` for message search
- `Peer::update` for patch-style metadata updates
- `Session::peers()` returning `Vec<Peer>` wrappers
- `Session::add_peer`, `add_peers`, `set_peers`, `remove_peers` with `PeerSpec` enum (bare ID or with config)
- `Session::get_peer_configuration`, `set_peer_configuration` for per-peer session config
- `Session::upload_file` and `upload_file_streamed` with builder pattern (`.peer()`, `.metadata()`, `.configuration()`, `.created_at()`)
- `Session::clone_session`, `clone_session_with_message`
- `Session::representation(peer_id)` scoped to session
- `Session::queue_status()` for processing status
- `Session::summaries()` for short/long summary retrieval
- `Session::messages()` paginated message listing
- `Session::delete()` for session removal
- `Honcho::force_ensure()` for explicit workspace creation
- `Honcho::schedule_dream(observer)` for memory consolidation
- `Honcho::search` for workspace-wide message search
- `Honcho::queue_status` for workspace processing status
- `Honcho::peers`, `peers_with_filters`, `sessions`, `sessions_with_filters`, `workspaces` for paginated listing
- `Honcho::get_configuration`, `set_configuration` for typed workspace config
- `Honcho::get_configuration_raw`, `set_configuration_raw` for raw JSON config access
- `Honcho::delete_workspace` for workspace removal
- `Message` wrapper type with `id()`, `content()`, `peer_id()`, `session_id()`, `metadata()`, `created_at()`, `token_count()`, `workspace_id()` accessors and `Display` impl
- `FileSource` enum for file uploads (`bytes`, `path`, `stream`)
- `DialecticStream` adapter that accumulates SSE content and provides `final_response()` / `is_complete()`
- `blocking` feature: sync wrappers over the full async API with runtime guard
- `tracing` feature: `#[tracing::instrument]` on all public async methods
- `SessionContext::to_openai` and `to_anthropic` for provider-compatible message format conversion
- `Page::into_stream` for auto-fetching paginated stream
- `Page::map` for item transform chaining
- Options types: `SessionListOptions`, `DialecticOptions`, `MessageSearchOptions` with builder pattern

### Changed

- MSRV raised to 1.88
- All `bon::Builder` structs use `finish_fn = build` for consistency
- Error type provides `code()` method returning machine-readable string identifiers

### Fixed

- Duplicate `Page::next_page` section in MIGRATION.md removed
- SSE stream handles UTF-8 splits at chunk boundaries
- Cancel-safety for SSE stream drops

### Known Limitations

- No webhooks or API keys endpoints
- No automatic SSE reconnection
- MSRV 1.88

## [0.1.0] - 2025-05-13

### Added

- Error types with HTTP status mapping (5xx → Server, 429 → RateLimit, etc.) and Retry-After parsing
- 55+ request/response type schemas with OpenAPI validation and serde roundtrip tests
- HTTP client with automatic retries, exponential backoff, and configurable max retries
- Paginated collection streaming (Page → Stream → Iterator)
- Honcho client: workspace auto-creation, metadata/configuration CRUD
- Peer: chat (dialectic), streaming chat, representation, context, card, conclusions
- Session: peer management, messages (batch up to 100), file upload (multipart streaming), clone, summaries
- SSE streaming with cancel-safety and UTF-8 split handling
- Conclusion & ConclusionScope: create, list, query (semantic search), representation, delete
- Blocking facade (feature-gated): sync wrappers over async API with runtime guard
- Compile-time assertions: Send + Sync + 'static bounds on all public types
- Session context parity with Python SDK (to_openai / to_anthropic)
- CI workflow: fmt, clippy, test, doc, MSRV verify
- Integration tests with graceful skip when no server available
