# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

### Known Limitations

- No webhooks or API keys endpoints
- No automatic SSE reconnection
- MSRV 1.80
