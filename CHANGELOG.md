# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [3.0.12] - 2026-08-10

### Added

- Session allowlist on the Dialectic and representation via a constrained `filters` body on `POST /peers/{peer_id}/chat` and `/representation`, supporting only the `session_id` key (a session id, a bare list, or `{"in": [...]}`). Unsupported keys and shapes are rejected with 422 rather than silently ignored, it composes with `session_id` (which must be included in the allowlist when both are given), and it is capped at 1,000 sessions per request. Enforcement is uniform and fail-closed at every recall chokepoint: scoped conclusion recall is restricted to `level == "explicit"` (dream-derived conclusions carry a single `session_name` but are synthesized across all sessions, so that stamp can't be scoped on), `get_reasoning_chain` is unavailable under an allowlist, and an empty allowlist short-circuits to empty results everywhere. Workspace keys pass the allowlist as-given; peer-scoped JWTs must be an active member of every allowlisted session (401 otherwise) (#882)
- Bare-list membership sugar in the filter DSL: `{"session_id": ["s1", "s2"]}` is now shorthand for `{"session_id": {"in": [...]}}` on regular columns generically. JSONB metadata columns are excluded and keep containment semantics. Strictly additive, since a bare list on a regular column previously compiled to a type-mismatched equality that matched nothing (#881)
- Optional structured outputs on the Dialectic: `response_format` (a JSON Schema with root type `object`) on peer chat makes `content` a JSON string conforming to that schema. Only a conservative subset of JSON Schema is supported, with DoS guards and non-recursive `$ref` support (#896)
- Combined tool calling and structured output in the LLM transport layer, with per-backend request shaping: OpenAI routes tool-carrying structured requests through `create()` with an explicit `json_schema` response format (`parse()` 500s on non-strict function tools), Anthropic skips the `{` JSON prefill when tools are present so `tool_use` blocks stay reachable, and Gemini injects a schema instruction into the final turn instead of using native `response_schema` (rejected alongside function calling before Gemini 3). All backends skip structured-output parsing on tool-call turns, which carry no consumable content (#907)
- `card_refresh` dream type: a lightweight dream that runs only the peer-card update, for event-driven refreshes such as membership changes and cold starts. Handled by a new `CardRefreshSpecialist` restricted to `get_recent_observations`, `search_memory`, and `update_peer_card` (no observation-mutating tools) with a tool-iteration cap of `min(6, DREAM.MAX_TOOL_ITERATIONS)`. `POST /v3/workspaces/{workspace_id}/schedule_dream` accepts `dream_type=card_refresh` plus a `rebuild` flag, which omits the existing card from the prompt so the specialist rebuilds it solely from observations present in the collection. Card refreshes never advance the omni dream guard pair (`last_dream_at` / `last_dream_document_count`) (#883)
- Full-fidelity LLM trace stream, with Langfuse as one projection over it: each call is captured once (`CapturedLLMCall`) and fanned out to a CloudEvents trace stream (`llm.call.traced` / `trace.content`) and a Langfuse exporter, both reconstructing trace → run → step → generation from the same source of truth. Adds `TELEMETRY_TRACE_PAYLOADS_ENABLED` (default `false`), `TELEMETRY_TRACE_MAX_BYTES` (default 262144, per-message cap with oversized content clipped), `TELEMETRY_TRACE_PURPOSES` (JSON list of `CallPurpose` values; empty means all), and `LANGFUSE_EXPORTER_MODE` (`exporter` by default; `inline` is kept for one release for side-by-side validation). Embedding calls are traced, dreamer branches nest under one dream trace, tool calls become spans under their step, and high-volume events are sampled deterministically. `TRACE_ENDPOINT` is dropped (#845)
- Redis Cluster support via `CACHE_CLUSTER` (for example GCP Memorystore for Redis Cluster), alongside a new `CACHE_LOCK_WAIT_CHECK_INTERVAL_SECONDS` (#905)
- `EMBEDDING_MODEL_CONFIG__MAX_BATCH_SIZE` caps texts per embedding request for OpenAI-compatible providers with smaller limits than OpenAI's, such as DashScope `text-embedding-v4` (10) and Alibaba Bailian `qwen3.7-text-embedding` (20). When unset, native provider defaults are preserved (OpenAI 2048, Gemini 100) (#983)
- Per-request provider timeouts via `provider_params.timeout` on any model config, validated at config load so a bad value fails at startup with the exact config path instead of surfacing per-request as a retried 500. Good values normalize to float seconds; Gemini's is converted to milliseconds (#832)
- `RepresentationCompletedEvent` now reports deduplication counts: `exact_dup_in_batch_count`, `exact_dup_existing_count`, `semantic_dup_rejected_count`, and `semantic_dup_replaced_count` (#910)
- OAuth discovery for MCP clients: the MCP worker serves `/.well-known/oauth-protected-resource` (RFC 9728) without auth so clients can discover the authorization server, and a 401 now carries `WWW-Authenticate: Bearer resource_metadata="..."` (exposed cross-origin) to start the flow (#923)
- Prometheus metrics for the immediate-embed fast path: tasks shed because `EMBEDDING_MAX_PENDING_EMBED_TASKS` was reached, and the current in-flight task count (#892)
- Docs: a detailed system architecture diagram, a Codex integration guide (#879), a structured-outputs page (#896), a section on filtering conclusions by reasoning level (#851), a health-check endpoint reference, and SDK updates (#867)

### Changed

- **Breaking config change:** `DERIVER_REPRESENTATION_BATCH_MAX_TOKENS` is split into two settings that were previously conflated — `DERIVER_REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS` (default 512), the producer-side minimum a work unit accumulates before the deriver claims it, where `0` disables the gate; and `DERIVER_REPRESENTATION_BATCH_TARGET_INPUT_TOKENS` (default 1024), the consumer-side maximum context-window tokens per deriver LLM call. Deployments setting the old name must migrate (#889)
- The immediate-embed fast path now applies backpressure: `EMBEDDING_MAX_PENDING_EMBED_TASKS` (default 50) caps in-flight embed tasks, and once saturated, message creation skips the fast path entirely and the reconciler embeds on its next cycle. `0` disables the fast path (#892)
- Explicit-level documents are now kept session-pure, so memory can be built by copying explicit documents between collections. Enforcement refuses rather than rewrites: `create_documents` rejects explicit documents with a null `session_name`, exact dedup keys on (content, level, session-for-explicit), semantic dedup scopes candidate search to the same level and — for explicit documents — the same session, and the generic `create_observations` tool rejects `level='explicit'` outside message-ingestion (deriver) context. Derived levels keep cross-session consolidation (#883)
- Sentry's `before_send` filter is centralized as `default_before_send` in `src/telemetry/sentry.py` instead of living only in the API's `main.py`, so the deriver gets the same non-actionable-exception filtering. All Sentry events also carry a `namespace` tag for correlation (#934, #870)
- The minimal deriver's extraction examples no longer teach inferences its own output schema forbids. The `EXAMPLES` block demonstrated deriving a specific birthday from "I just had my 25th birthday last Saturday", deriving residence from a single visit ("I took my dog for a walk in NYC" → "alice lives in NYC"), and a "+ general knowledge" deductive output the deriver has no channel for. The replacements stay inside the schema's contract and teach the boundary: the dog/NYC message is kept and shown extracting correctly, and a separate example shows "lives in NYC" is valid when actually stated (#985)
- Dreamer specialists are instructed not to output summaries (#894)
- `session_name` is deprecated for scoping in favor of the session allowlist. It is not removed and not aliased: it also pins the query to one session, bypasses observer scoping, and drives session-history injection into the dialectic prompt, so it has no drop-in replacement (#882)
- The MCP worker no longer requires the `X-Honcho-User-Name` or `X-Honcho-Assistant-Name` headers (#923)

### Fixed

- Session scoping was applied to only one of the three working-representation query paths: `session_name` reached the recent-documents query, but the semantic and most-derived paths ignored it, so `limit_to_session` leaked cross-session conclusions into perspectives. The allowlist is now threaded uniformly through all three paths and pushed down to pgvector and external vector stores (#881)
- Empty membership lists failed open in the vector-store filter builders, silently widening scope: LanceDB dropped empty `IN` clauses and Turbopuffer emitted a bare `In []` with undocumented semantics. Both now emit an explicit always-false predicate, and `_build_filter_conditions` checks `is not None` rather than truthiness so an empty list is no longer treated like `None` (#881, #882)
- Session-scoped CRUD helpers ignored the session allowlist entirely, so a caller could read a session the allowlist forbids. The API routes guarded this with a 422, but the dialectic tools call these CRUD functions directly and bypassed it. `_semantic_search_messages` (covering `search_messages` and `search_messages_temporal`), `grep_messages`, `get_messages_by_date_range`, `get_recent_history`, and `get_observation_context` now return `[]` when `session_name` is set and outside the allowlist (#882)
- The cache client logged the full Redis URL — including the password — at INFO and WARNING on every connection attempt and failure, exposing the live credential in container logs and downstream aggregation. Credentials are now redacted across userinfo, the `?password=` (redis-py) and `?secret=` (cashews) query params, scheme-less URLs whose password is invisible to `.port`/`.password` parsing, and malformed URLs, whose fallback previously echoed the raw input verbatim (#869)
- A `top_k` of `0` reached the vector store, where Turbopuffer rejects it with a 400 (`top_k must be between 1 and 10000`). A non-positive `top_k` now returns `[]` before the embedding call, and the semantic budget floors at 1 so an explicitly requested search isn't silently allocated zero (#970)
- Gemini clients had no HTTP timeout, so a stalled socket wedged the deriver worker's uvloop event loop, which the in-process reconciler shares. A 10-minute timeout is now set on both the Gemini LLM client and the Gemini embedding client (#903)
- Dreamer conclusions were dated to ingestion time rather than their latest source observation, and their timestamps are now normalized (#890)
- Langfuse I/O annotation was gated on `LANGFUSE_PUBLIC_KEY` instead of `langfuse_inline_enabled`, so in the default `exporter` mode it called `update_current_generation()` with no active span — logging "No active span in current context" roughly 14 times per dialectic run and building throwaway `model_dump` payloads on every LLM call. Separately, `AgentToolSummaryCreatedEvent` hardcoded `run_id="deriver"` / `iteration=0`, polluting `run_id` grouping in the CloudEvents stream with a phantom run; both fields are now optional and the resource id is keyed on `message_id:summary_type` (schema_version 2 → 3) (#845)
- Assistant tool calls were dropped from the captured trace stream for OpenAI and Gemini: `build_captured_messages` read only `{role, content, tool_call_id}`, but those providers keep tool calls outside `content`, so replayed tool-call turns landed as empty content and Gemini lost its text and tool results entirely. Tool calls are now normalized per provider into a unified `tool_calls` field and folded into the content hash. Gemini's `thought_signature` is bytes, so `model_dump(mode="json")` raised `UnicodeDecodeError` inside `emit_trace`, silently dropping whole tool-calling iterations from the trace stream (billing and Langfuse were unaffected); it is now base64-encoded on the telemetry path while replay keeps the raw bytes (#845)
- `EmbeddingClient.encoding` forced full client construction, raising "OpenAI API key is required" even though tiktoken needs no credentials. The document dedup tie-break only needs `.encoding` for token counting, so any test hitting that path failed in environments without embedding keys — notably CI for pull requests from forks. The encoding is now resolved from the configured model directly, falling back to `cl100k_base`, and the underlying client's encoding is reused only when it has already been constructed (#955)
- The Docker build failed under Podman because the uv build inputs weren't copied (#878)
- LanceDB was installed on macOS Intel, where it doesn't work. A PEP 508 marker excludes `darwin/x86_64` and the LanceDB vector-store import is wrapped so a misconfiguration surfaces as a clear config error (#496)
- Prompt checks requiring the literal token "json" for `json_object` mode are now satisfied in lowercase (#887)
- Reverted an unintended `RepresentationCompletedEvent` schema-version increment
- Documented preinstalling pgvector as a privileged role for deployments where the `DB_CONNECTION_URI` role deliberately cannot create extensions (managed Postgres, Kubernetes operators, NixOS). `CREATE EXTENSION IF NOT EXISTS vector` does not help there, because Postgres checks the privilege before checking whether the extension exists. Docker Compose is unaffected, since the bundled stack connects as the `postgres` superuser (#984)

## [3.0.11] - 2026-06-24

### Added

- `api_request_duration_seconds` Prometheus histogram tracking per-route request latency, labeled by method and endpoint (#837)
- LLM `provider_params` passthroughs (`extra_body` / `extra_headers` / `extra_query`) are now forwarded to the underlying provider transport across all backends, with shape validation that rejects non-mapping values (#821)
- `structured_output_mode` model-config option to use `json_object` mode for OpenAI-compatible providers that lack native Structured Outputs support (used by the deriver) (#820)
- OpenRouter app-attribution headers (`HTTP-Referer` / `X-Openrouter-Title`) are now sent on OpenAI-compatible clients when the configured base URL is OpenRouter, so requests are attributed to "Honcho" in OpenRouter's dashboard (#805)
- Langfuse traces are now tagged with user and session IDs for easier trace filtering (#814)
- `DERIVER_REPRESENTATION_BATCH_MAX_AGE_SECONDS` (default 1800s) lets sub-threshold representation work units flush once their oldest unprocessed queue item ages out. Set it to `0` to keep the legacy behavior where sub-threshold tails wait indefinitely unless `DERIVER_FLUSH_ENABLED=true` (#826)
- Conclusion responses now include a `level` field (`explicit`, `deductive`, `inductive`, `contradiction`); list/query endpoints support filtering by `level` via `filters`, with reserved filter keys protected from being overridden by user-supplied filters (#851)

### Changed

- Peer-scoped JWTs now get read-only access to the sessions their peer is an active member of (session context, summaries, peers, their own per-session config, search, and message reads). Session-scoped JWTs remain confined to their session and cannot reach peer routes (#679)
- Compacted Honcho's log output, with guarded ms/s metric formatting that falls back to a plain string for non-numeric values (#836)
- Sentry now drops noisy infra/scrape transactions: the reconciler opens a transaction only once a batch has rows (idle cycles emit none), and a `traces_sampler` returns `0.0` for `/metrics`, `/health`, `/openapi.json`, `/docs`, `/redoc`, and the deriver metrics server. `SENTRY.TRACES_SAMPLE_RATE` still governs real traffic (#834)

### Fixed

- Peer- and session-scoped JWTs were effectively workspace-scoped: authorization walked the route's declared scope and fell through to a workspace match, so a `{w, p: alice}` token could act on any peer in the workspace. JWTs are now authorized by their narrowest claim and never widen to workspace access (#679)
- The keys API now rejects creating a peer- or session-scoped key without a workspace. Such keys were minted successfully but failed verification on every request (#679)
- Agent-supplied observation IDs carrying the display-format `id:` prefix are now normalized (prefix and trailing whitespace stripped) before `source_ids` are stored and on `get_reasoning_chain` lookups, fixing corrupted provenance links and broken reasoning-chain traversal (#795)
- Fixed a `create_tree` keyword-argument mismatch in the Dreamer's surprisal tree construction (#749)
- Providers that omit output-token counts (observed with Gemini on tool-loop completions) returned `output_tokens=None`, which raised a Pydantic validation error that aborted the call and crashed the Dreamer's induction phase before inductive conclusions were persisted. `None` is now coerced to `0` so token accounting degrades gracefully (#809)
- Document creation now performs exact (case-insensitive, whitespace-trimmed) content deduplication before the existing semantic dedup step: exact duplicates within a batch collapse to a single insert, and an exact match against a live document reinforces it (atomic `times_derived` increment) instead of creating a new row (#861)
- The OpenAI backend passed `tool_choice` through raw while the Anthropic and Gemini backends translate Honcho's canonical vocabulary to their native form, so on a mixed-provider fallback chain (for example Gemini primary → OpenAI backup) a canonical `"any"` reached OpenAI unchanged and was rejected as an invalid param. The OpenAI backend now converts it, mirroring the others: `any`/`required` → `required`, `auto`/`none` pass through, and a tool-name string or `{"name": ...}` dict becomes a function selection (#850)
- Langfuse `@observe` auto-capture serialized every argument of `honcho_llm_call_inner` into the generation span input, including `client_override` (a live `AsyncOpenAI`/`genai` client) and `selected_config` (which carries `api_key`). Auto-capture deep-copied the client into a half-constructed object whose teardown raised (`AsyncHttpxClientWrapper ... no attribute '_state'` on OpenAI, flooding stderr; `BaseApiClient ... no attribute '_http_options'` on Gemini), and it leaked `ModelConfig.api_key` into traces. Capture is now an explicit allowlist: `capture_input`/`capture_output` are disabled and curated, serializable input and output are stamped instead, with tuning knobs surfaced as `model_parameters` via a secret-bearing denylist and per-call token usage mirrored as `usage_details` (#849)

## [3.0.10] - 2026-06-15

### Added

- Messages are now embedded via a background task rather than blocking API request
- Read-only DB session mode (`get_read_db` / `tracked_db(..., read_only=True)`) so reads don't hold a transaction open across the work
- `CORS_ORIGINS` env var to configure CORS allowed origins without editing source; defaults match the prior hardcoded list, so self-hosted deployments behind custom domains can whitelist their frontend (#697)
- `scripts/generate_jwt.py` — utility for minting scoped or admin Honcho JWTs (`--admin`, `--workspace`/`--peer`/`--session`, `--expires` with human-friendly durations, `--print-only`) without calling the keys API (#757)
- `STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS` (default 60s) — minimum jittered spacing between deriver stale-work-unit cleanup runs, so cleanup no longer runs on every seconds-scale poll (`0.0` keeps the legacy every-poll behavior) (#773)

### Changed

- Optimized the deriver and dreamer prompt cache prefixes to improve prompt-cache hit rates (#806)

### Fixed

- `times_derived` is now properly reinforced when a duplicate conclusion is detected. It had been pinned at 1 for nearly every conclusion (the reject-new branch dropped the increment and the new-wins branch reset the count to 1), so `ORDER BY times_derived DESC` fell back to arbitrary heap order and froze stale conclusions to the front of injected context. Reinforcement is now an atomic increment and both most-derived queries gained a `created_at DESC` recency tiebreaker (#768)
- Webhook creation now correctly rejects private/internal IP addresses (#793)

## [3.0.9] - 2026-06-02

### Changed

- Connection acquisition is now a single attempt with no server-side retry, on a vanilla `AsyncSession`. A new `DB_CONNECT_TIMEOUT_SECONDS` (default 2s) bounds the attempt so a saturated or unreachable pooler fails fast instead of holding a client connection open to re-knock. A saturated DB now surfaces to the caller — the API returns an error and the deriver backs off and retries on a later poll — which lets the pooler drain rather than amplifying saturation.

### Added

- Deriver poll jitter so instances that start together don't poll in lockstep: `DERIVER_POLLING_STARTUP_JITTER_SECONDS` (random delay before the first poll, default 30s) and `DERIVER_POLLING_JITTER_RATIO` (±fraction applied to every poll sleep, default 0.5). Both disable at `0.0`; the underlying backoff schedule is unchanged.

### Removed

- Reverted the connection-checkout retry and `HonchoAsyncSession` custom session introduced in 3.0.8. Removed the `DB_CONNECTION_RETRY_ENABLED` / `DB_CONNECTION_RETRY_MAX_DELAY_SECONDS` / `DB_CONNECTION_RETRY_BACKOFF_INITIAL_SECONDS` / `DB_CONNECTION_RETRY_BACKOFF_MAX_SECONDS` settings, the `db_connection_acquisitions{outcome=...}` Prometheus counter, and the `db.pool.acquire` Sentry span. Alerting built on `db_connection_acquisitions` should migrate to `db_pool_connections` / `db_queries_in_flight`.

## [3.0.8] - 2026-06-01

### Added

- Connection-checkout retry with bounded exponential backoff (tenacity) on `get_db`/`tracked_db`: transient transaction-pooler (Supavisor) rejections — SQLAlchemy `TimeoutError` and `OperationalError` — now retry with backoff instead of surfacing as 500s under client-connection saturation. Gated by
  `DB_CONNECTION_RETRY_ENABLED` with configurable delay/backoff knobs; ~10s default budget (#758)
- `HonchoAsyncSession` — a lazy `AsyncSession` that checks out its pooled connection (with retry) on the first DB-touching call rather than at construction. Request handlers doing non-DB work (embedding, file, LLM) before their first query no longer pin a pooler connection across it. Only the checkout is retried;
  the statement still runs exactly once, so writes are never duplicated (#758)
- Adaptive deriver queue polling: the poll interval backs off when the queue is idle or erroring (base → max, doubling each cycle) and snaps back to base the moment work is claimed, cutting steady-state query load against the DB. Gated by `DERIVER_POLLING_BACKOFF_ENABLED` with configurable max/multiplier (#758)
- New Prometheus `db_pool_connections` gauge (checked_out / checked_in / size / overflow), labeled `api`|`deriver`, registered in both the API lifespan and the deriver metrics server (#758)
- New Prometheus `db_connection_acquisitions{outcome=ok|retried|exhausted}` counter — the alertable early-warning signal that connection checkouts are retrying through pooler rejection, before requests start failing (#758)
- New Prometheus `db_queries_in_flight` gauge — statements actually executing on the wire (via SQLAlchemy cursor-execute events). Paired with `checked_out`, the gap reveals connections held but parked (the "idle in transaction during an external call" antipattern). Gated on `METRICS.ENABLED` for zero overhead when
  off (#758)
- Explicit `SqlalchemyIntegration` in both the API and deriver Sentry inits; connection acquisition wrapped in a `db.pool.acquire` span with live pool stats captured on retry exhaustion (#758)

### Changed

- Default `POOL_TIMEOUT` lowered to 5s, with validation that it stays under the connection-retry budget when a pooled (non-null) `POOL_CLASS` is configured; `config.toml.example` and the v2/v3 configuration docs updated to match (#758)
- `HonchoAsyncSession` wraps every DB-touching session method (execute / scalar / scalars / flush / merge / refresh / commit / get / get_one / stream / stream_scalars / delete) so the lazy-checkout-with-retry guarantee has no holes; the acquired flag resets on `close()`/`reset()` so a reused session re-acquires on
  next use (#758)

### Fixed

- Roll the session back on a retryable checkout failure before retrying — a failed autobegin could otherwise leave it pending-rollback, making the next connection attempt raise instead of cleanly re-checking-out (#758)
- Guard `DBPoolCollector.collect()` so a pool-read/import hiccup can't raise and abort the entire `/metrics` scrape (Prometheus drops all metrics if any collector raises) (#758)
- Clamp the pool overflow gauge to ≥ 0 (it could report negative before the pool fills) (#758)
- Removed a double-sleep in the deriver idle poll so the backoff cap is a true cap rather than 2× (#758)

## [3.0.7] - 2026-05-21

### Added

- New `src/llm/` module as the single owner of provider runtime: clients, backends, history adapters, tool loop, request builder, credentials, and caching policy (#459)
- `AttemptPlan` dataclass captures per-retry provider selection (client, model, reasoning_effort, thinking_budget_tokens, selected_config) and pins it across stream-final retries so streaming doesn't bounce back to primary after the tool loop has settled on fallback (#459)
- Gemini JSON-schema sanitizer for `function_declarations` — strips keywords Gemini's validator rejects (`additionalProperties`, `allOf`, etc.) while preserving semantics for all other backends (#459)
- Dreamer specialists derive `effective_max_tokens` from `model_config.max_output_tokens` with a per-specialist default fallback (#459)
- New cloudevent `LLMCallCompletedEvent` (`llm.call.completed`) fires once per provider hit with full cost-attribution context: transport/provider_label, model, token counts with cache breakdown, finish_reason, outcome, `is_final_attempt`, retry/fallback state, duration, tool-call shape, streaming flag, and agent correlation (`run_id` + iteration). Includes a `CallPurpose` closed enum (`deriver.representation`, `dialectic.answer`, `dream.deduction|induction`, `summary.short|long`) (#637)
- `RepresentationCompletedEvent` now carries `total_input_tokens` for full-trace cost attribution (#637)
- Per-emitter `honcho_version` injection on all CloudEvents plus emitter health metrics (#637)
- `TelemetrySettings.HIGH_VOLUME_SAMPLE_RATE` (default 1.0) — deterministic per-`run_id` sampler so an entire agent trace is kept or dropped together; aggregate envelopes bypass the sampler (#637)
- Deriver custom instructions: per-workspace/peer guidance threaded into the deriver prompt with a `MAX_CUSTOM_INSTRUCTIONS_TOKENS` budget (default 2000); deriver `MAX_INPUT_TOKENS` raised 23000 → 25000 to make room (#609)
- Configurable embedding dimensions: `EMBEDDING_MODEL_CONFIG__DIMENSIONS_MODE` (`auto`/`always`/`never`) controls whether the OpenAI `dimensions=` parameter is forwarded; `auto` (default) sends it when the operator explicitly set `EMBEDDING_VECTOR_DIMENSIONS` and the model is not on the known-rejecting allowlist (#678)
- New `honcho-cli` package — Python CLI for inspecting and managing peers, sessions, and configuration against a Honcho deployment (#424)
- `HONCHO_API_URL` env var support in the MCP Worker, enabling self-hosted Honcho deployments to point the Worker at their own instance instead of `https://api.honcho.dev` (#575)
- API ID `max_length` increased from 100 to 512 across `WorkspaceCreate`, `PeerCreate`, and `SessionCreate` to align the API contract with the underlying DB schema (#684)
- Regression tests covering fallback-config thinking-param reach, provider_params → extra_params boundary, OpenAI reasoning-model parameter routing, Gemini blocked finish_reason handling, and fail-fast `max_tool_iterations` validation (#459)

### Changed

- All LLM orchestration moved out of `src/utils/clients.py` into `src/llm/` with modules split by responsibility (api, executor, tool_loop, runtime, registry, conversation, request_builder, credentials, caching, backends, history_adapters) (#459)
- Default `ModelConfig` factories (deriver, summary, dreamer specialists, dialectic levels) normalized to `openai/gpt-5.4-mini` with no extra parameters set by default; operators add transport/thinking overrides explicitly (#459)
- OpenAI reasoning-model routing widened via `_uses_max_completion_tokens` heuristic covering `gpt-5.x` and `o1/o3/o4` — these models receive `max_completion_tokens` instead of `max_tokens` (#459)
- Override client factories switched from unbounded `@cache` to `@lru_cache(maxsize=128)` for predictable memory growth on long-running processes (#459)
- `get_backend` now delegates to `client_for_model_config`, so the live-test path and production path share one missing-API-key validation (#459)
- Blocked Gemini responses (`SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `BLOCKLIST`) raise `LLMError` in the streaming path too (previously only the non-streaming path), ensuring retry/fallback logic fires uniformly (#459)
- Transport-change env overrides now strip transport-specific thinking params (thinking_budget_tokens vs. reasoning_effort) during config merge, including at the dialectic-level merge, so switching from Anthropic → OpenAI doesn't leave orphaned Anthropic-only params that the OpenAI backend would reject (#459)
- `max_tool_iterations` out-of-range inputs now raise `ValidationException` instead of being silently clamped (#459)
- Public API schemas (`WorkspaceCreate`, `PeerCreate`, `SessionCreate`) and SDK validation (`api_types.py`, `validation.ts`) accept IDs up to 512 chars (was 100) (#684)
- Peer card prompts reframed as stable identity markers (replaces the prior "biographical/profile facts" language). Induction specialist is now opted out of peer card writes (`can_update_peer_card = False`) so only deduction touches the card (#686)
- Vector store queries no longer fetch embedding vectors — only document metadata is returned, reducing payload size and DB load (pgvector, lancedb, turbopuffer) (#682)
- Langfuse trace metadata now includes `namespace`, `model`, and `provider` so traces can be filtered by deployment slice (#565)
- Deriver: model-aware tokenizer (replaces the previously hardcoded encoding) and explicit guard on empty message content (#647)
- Dialectic level defaults now merge correctly with per-level overrides in `src/config` (#656)
- Default dialectic tool choice switched from forced/required to `auto` (#630)
- Vector sync given a substantial retry budget to tolerate transient embedding provider outages (#604)
- `AgentToolConclusionsDeletedEvent` payload now carries `levels` for parity with the rest of the conclusion event surface (#612)
- Turbopuffer vector store: `InternalServerError` caught and surfaced as a warning rather than a hard failure; unused `upsert_with_retry` and `VectorUpsertResult` removed; explicit silent and explicit-error paths for vector DB server errors (#561)
- Troubleshooting docs updated to reflect nested-env-var form for per-component thinking-budget overrides (#459)
- README refresh (#681)
- CLAUDE.md refreshed against the current `src/` layout (#680)

### Fixed

- Fallback `ModelConfig` temperature and `thinking_budget_tokens` reach the backend on the final retry — previously the primary's values were pre-populated into caller kwargs early and clobbered fallback values via `effective_config_for_call(update=...)` (#459)
- Stream-final retries pin to the `AttemptPlan` that succeeded rather than re-running provider selection through the outer `current_attempt` ContextVar (which could roll streaming back to primary after the tool loop had already switched to fallback) (#459)
- OpenAI structured-output calls continue to use `chat.completions.parse()` with strict schema enforcement, while tool-calling paths use `chat.completions.create()` without `strict:True` for broader proxy compatibility (OpenRouter, vLLM, Ollama) (#459)
- Gemini `cached_content` reuse keys now include `system_instruction` and `tool_config` so cache hits don't cross configurations that differ only in those fields (#459)
- Removed strict parameter validation for thinking params on Anthropic and OpenAI transports — was rejecting valid per-transport configs (#686)
- `reverse` query parameter is now honored on the v3 workspace list (`POST /v3/workspaces/list`), peer list (`POST /v3/workspaces/{workspace_id}/peers/list`), workspace-scoped session list (`POST /v3/workspaces/{workspace_id}/sessions/list`), and peer-scoped session list (`POST /v3/workspaces/{workspace_id}/peers/{peer_id}/sessions`). Honcho SDKs at 2.1.0+ were already sending `reverse=true` for these routes but the server silently ignored it. Ties on `created_at` now fall back to the internal nanoid `id` so ordering remains stable across pages (#685)
- LLM client factories now receive `base_url` from `LLMSettings` for default providers — previously the override path honored `base_url` but the default path didn't, so operators pointing at OpenAI-compatible proxies via `LLM__OPENAI_BASE_URL` were ignored (#643, fixes #641)
- Internal N+1 query in dialectic agent tool execution — collapsed per-iteration DB lookups into a single fetch (#652)
- Dreamer threshold and time-guard semantics: `check_and_schedule_dream` count filter now includes only `documents.level == 'explicit'` (dreamer-created levels are output, not input, and were inflating the threshold and creating a feedback loop); `last_dream_at` write relocated from `enqueue_dream` into `process_dream` so duplicate enqueues or failed runs no longer reset the 8-hour time guard (#573)
- Deriver: blank observations are filtered out before embedding (previously triggered noisy embedding calls and persisted empty rows); blank-observation filtering unified across tool paths (#615)
- Surprisal module: filter for level observations changed from `{"level": levels}` to `{"level": {"in": levels}}` — `apply_filter()` requires operator syntax, so the prior call silently returned 0 results and made the entire Surprisal phase of the Dream cycle a no-op (#581, fixes #559)
- Removed hardcoded `stop_sequences` override from Deriver `ModelConfig` (was clobbering operator-configured stop sequences) (#587)
- Removed stale `stop_sequences` from tests (#607)
- Embedding client: `embed()` now wraps single-string input in an array, restoring compatibility with OpenAI-compatible third-party providers that reject scalar input (#586)
- Docker Compose: deriver service startup gated on the API service healthcheck (prevents races where the deriver starts before the API has run migrations) (#689)
- Docker image: `HEALTHCHECK` directive removed from the shared base image — it probed an HTTP endpoint only the API serves, permanently marking deriver containers as unhealthy. Service-level health checks now belong in each service's own configuration (k8s readiness/liveness probes on the API Deployment only) (#530)
- `tests/unified`: `--test-dir`/`--test-file` arguments now use an argparse mutually-exclusive group instead of manual validation (#650)
- CrewAI example updated for the latest CrewAI protocol (#631)

### Removed

- `src/utils/clients.py` deleted; its responsibilities are split across `src/llm/registry.py`, `src/llm/credentials.py`, and the backend-specific modules (#459)
- `HEALTHCHECK` directive removed from the shared Docker image (#530)

## [3.0.6] - 2026-04-10

### Changed

- Tightened transaction scopes across search, agent tools, queue manager, and webhook delivery to minimize DB connection hold time during external operations (#525)
- Search operations refactored to two-phase pattern — external work (embeddings, LLM calls) completes before opening a transaction (#525)
- Agent tool executor performs external operations before acquiring DB sessions (#525)
- Queue manager transaction scope reduced to only the critical section (#525)
- Webhook delivery no longer holds a DB session parameter (#525)

### Fixed

- Session leakage in non-session-scoped dialectic chat calls (#526)

### Added

- Health check endpoint (`/health`) for container orchestration and load balancer probes (#510)

## [3.0.5] - 2026-04-03

### Fixed

- explicit rollback on all transactions to force connection closed

## [3.0.4] - 2026-04-02

### Added

- JSONB metadata validation enforces 100 key limit and max depth of 5 (#419)

### Changed

- Schemas refactored from single `schemas.py` into `schemas/api.py`, `schemas/configuration.py`, and `schemas/internal.py` with backwards-compatible re-exports (#419)

### Fixed

- Missing `deleted_at` filter on `RepresentationManager._query_documents_recent()` and `._query_documents_most_derived()` allowed soft-deleted documents to leak into the deriver's working representation (#456)
- `CleanupStaleItemsCompletedEvent` emitted spuriously when no queue item was actually deleted (#454)
- Empty JSON file uploads caused unhandled errors; now returns normalized error responses (#434)
- Memory leak: `_observation_locks` switched to `WeakValueDictionary` to prevent unbounded growth (#419)
- SQL injection in `dependencies.py`: parameterized `set_config` calls to prevent injection via request context (#419)
- NUL byte crashes: string inputs (message content, queries, peer cards) now stripped at schema level (#419)
- Filter recursion depth capped at 5 to prevent stack overflow (#419)
- Dedup-skipped observations now correctly reflected in created counts (#477)
- External vector store support for message search — routes queries through configured external vector store with oversampling and
  deduplication to handle chunked embeddings (#479)
- Dialectic agent no longer holds a DB connection during LLM calls — embeddings are pre-computed before tool execution, DB sessions isolated in `extract_preferences`, `query_documents` no longer accepts a DB session parameter (#477)

## [3.0.3] - 2026-02-25

### Added

- Consolidated session context into a single DB session with 40/60 token budget allocation between summary and messages
- Observation validation via `ObservationInput` Pydantic schema with partial-success support and batch embedding with per-observation fallback
- Peer card hard cap of 40 facts with case-insensitive deduplication and whitespace normalization
- Safe integer coercion (`_safe_int`) for all LLM tool inputs to handle non-integer values like `"Infinity"`
- Embedding pre-computation and reuse across multiple search calls in dialectic and representation flows
- Peer existence validation in dialectic chat endpoints — raises ResourceNotFoundException instead of silently failing
- Logging filter to suppress noisy `GET /metrics` access logs
- Oolong long-context aggregation benchmark (synth and real variants, 1K–4M token context windows)
- MolecularBench fact quality evaluation (ambiguity, decontextuality, minimality scoring)
- CoverageBench information recall evaluation (gold fact extraction, coverage matching, QA verification)
- LoCoMo summary-as-context baseline evaluation
- Webhook delivery tests, dependency lifecycle tests, queue cleanup tests, summarizer fallback tests
- Parallel test execution via pytest-xdist with worker-specific databases
- `test_reasoning_levels.py` script for LOCOM dataset testing across reasoning levels

### Changed

- Workspace deletion is now async — returns 202 Accepted, validates no active sessions (409 Conflict), cascade-deletes in background
- Redis caching layer now stores plain-dict instead of ORM objects, with v2-prefixed keys, storage, resilient `safe_cache_set`/`safe_cache_delete` helpers, and deferred post-commit cache invalidation
- All `get_or_create_*` CRUD operations now use savepoints (`db.begin_nested()`) instead of commit/rollback for race condition prevention
- Reconciler vector sync uses direct ORM mutation instead of batch parameterized UPDATE statements
- Summarizer enforces hard word limit in prompt and creates fallback text for empty summaries with `summary_tokens = 0`
- Blocked Gemini responses (SAFETY, RECITATION, PROHIBITED_CONTENT, BLOCKLIST) now raise `LLMError` to trigger retry/backup-provider logic
- Gemini client explicitly sets `max_output_tokens` from `max_tokens` parameter
- All deriver and metrics collector logging replaced with structured `logging.getLogger(__name__)` calls
- Dreamer specialist prompts updated to enforce durable-facts-only peer cards with max 40 entries and deduplication
- `GetOrCreateResult` changed from `NamedTuple` to `dataclass` with `async post_commit()` method
- FastAPI upgraded from 0.111.0 to 0.131.0; added pyarrow dependency
- Queue status filtering to only show user-facing tasks (representation, summary, dream); excludes internal infrastructure tasks

### Fixed

- JWT timestamp bug — `JWTParams.t` was evaluated once at class definition time instead of per-instance
- Session cache invalidation on deletion was missing
- `get_peer_card()` now properly propagates `ResourceNotFoundException` instead of swallowing it
- `set_peer_card()` ensures peer exists via `get_or_create_peers()` before updating
- Backup provider failover with proper tool input type safety
- Removed `setup_admin_jwt()` from server startup
- Sentry coroutine detection switched from `asyncio.iscoroutinefunction` to `inspect.iscoroutinefunction`

### Removed

- `explicit.py` and `obex.py` benchmarks replaced by coverage.py and molecular.py
- Claude Code review automation workflow (`.github/workflows/claude.yml`)
- Coverage reporting from default pytest configuration

## [3.0.2] - 2026-01-27

### Added

- Documentation for reasoning_level and Claude Code plugin

### Changed

- Gave dreaming sub-agents better prompting around peer card creation, tweaked overall prompts

### Fixed

- Added message-search fallback for memory search tool, necessary in fresh sessions
- Made FLUSH_ENABLED a config value
- Removed N+1 query in search_messages

## [3.0.1] - 2026-01-27

### Fixed

- Token counting in Explicit Agent Loop
- Backwards compatibility of queue items

## [3.0.0] - 2026-01-19

### Added

- Agentic Dreamer for intelligent memory consolidation using LLM agents
- Agentic Dialectic for query answering using LLM agents with tool use
- Reasoning levels configuration for dialectic (`minimal`, `low`, `medium`, `high`, `max`)
- Prometheus token tracking for deriver and dialectic operations
- n8n integration
- Cloud Events for auditable telemetry
- External Vector Store support for turbopuffer and lancedb with reconciliation flow

### Changed

- API route renaming for consistency
- Dreamer and dialectic now respect peer card configuration settings
- Observations renamed to Conclusions across API and SDKs
- Deriver to buffer representation tasks to normalize workloads
- Local Representation tasks to create singular QueueItems
- getContext endpoint to use `search_query` rather than force `last_user_message`

### Fixed

- Dream scheduling bugs
- Summary creation when start_message_id > end_message_id
- Cashews upgrade to prevent NoScriptError
- Memory leak in `accumulate_metric` call

### Removed

- Peer card configuration from message configuration; peer cards no longer created/updated in deriver process

## [2.5.1] - 2025-12-15

### Fixed

- Backwards compatibility for `message_ids` field in documents to handle legacy tuple format

## [2.5.0] - 2025-12-03

### Added

- Message level configurations
- CRUD operations for observations
- Comprehensive test cases for harness
- Peer level get_context
- Set Peer Card Method
- Manual dreaming trigger endpoint

### Changed

- Configurations to support more flags for fine-grained control of the deriver, peer cards, summaries, etc.
- Working Representations to support more fine-grained parameters

### Fixed

- File uploads to match `MessageCreate` structure
- Cache invalidation strategy

## [2.4.3] - 2025-11-20

### Added

- Redis caching to improve DB IO
- Backup LLM provider to avoid failures when a provider is down

### Changed

- QueueItems to use standardized columns
- Improved Deduplication logic for Representation Tasks
- More finegrained metrics for representation, summary, and peer card tasks
- DB constraint to follow standard naming conventions

## [2.4.2] - 2025-11-03

### Fixed

- Langfuse tracing to have readable waterfalls
- Alembic Migrations to match models.py
- message_in_seq correctly included in webhook payload

### Changed

- Alembic to always use a session pooler
- Statement timeout during alembic operations to 5 min

## [2.4.1] - 2025-10-24

### Added

- Alembic migration validation test suite

### Fixed

- Alembic migrations to batch changes
- Batch message creation sequence number

### Changed

- Logging infrastructure to remove noisy messages
- Sentry integration is centralized

## [2.4.0] - 2025-10-09

### Added

- Unified `Representation` class
- vllm client support
- Periodic queue cleanup logic
- WIP Dreaming Feature
- LongMemEval to Test Bench
- Prometheus Client for better Metrics
- Performance metrics instrumentation
- Error reporting to deriver
- Workspace Delete Method
- Multi-db option in test harness

### Changed

- Working Representations are Queried on the fly rather than cached in metadata
- EmbeddingStore to RepresentationFactory
- Summary Response Model to use public_id of message for cutoff
- Semantic across codebase to reference resources based on `observer` and `observed`
- Prompts for Deriver & Dialectic to reference peer_id and add examples
- `Get Context` route returns peer card and representation in addition to messages and summaries
- Refactoring logger.info calls to logger.debug where applicable

### Fixed

- Gemini client to use async methods

## [2.3.3] — 2025-10-01

### Changed

- Deriver Rollup Queue processes interleaved messages for more context

### Fixed

- Dialectic Streaming to follow SSE conventions
- Sentry tracing in the deriver

## [2.3.2] — 2025-09-25

### Added

- Get peer cards endpoint (`GET /v2/peers/{peer_id}/card`) for retrieving targeted peer context information

### Changed

- Replaced Mirascope dependency with small client implementation for better control
- Optimized deriver performance by using joins on messages table instead of storing token count in queue payload
- Database scope optimization for various operations
- Batch representation task processing for ~10x speed improvement in practice

### Fixed

- Separated clean and claim work units in queue manager to prevent race conditions
- Skip locked ActiveQueueSession rows on delete operations
- Langfuse SDK integration updates for compatibility
- Added configurable maximum message size to prevent token overflow in deriver
- Various minor bugfixes

## [2.3.1] - 2025-09-18

### Fixed

- Added max message count to deriver in order to not overflow token limits

## [2.3.0] — 2025-08-14

### Added

- `getSummaries` endpoint to get all available summaries for a session directly
- Peer Card feature to improve context for deriver and dialectic

### Changed

- Session Peer limit to be based on observers instead, renamed config value to
  `SESSION_OBSERVERS_LIMIT`
- `Messages` can take a custom timestamp for the `created_at` field, defaulting
  to the current time
- `get_context` endpoint returns detailed `Summary` object rather than just
  summary content
- Working representations use a FIFO queue structure to maintain facts rather
  than a full rewrite
- Optimized deriver enqueue by prefetching message sequence numbers (eliminates N+1 queries)

### Fixed

- Deriver uses `get_context` internally to prevent context window limit errors
- Embedding store will truncate context when querying documents to prevent embedding
  token limit errors
- Queue manager to schedule work based on available works rather than total
  number of workers
- Queue manager to use atomic db transactions rather than long lived transaction
  for the worker lifecycle
- Timestamp formats unified to ISO 8601 across the codebase
- Internal get_context method's cutoff value is exclusive now

## [2.2.0] — 2025-08-07

### Added

- Arbitrary filters now available on all search endpoints
- Search combines full-text and semantic using reciprocal rank fusion
- Webhook support (currently only supports queue_empty and test events, more to come)
- Small test harness and custom test format for evaluating Honcho output quality
- Added MCP server and documentation for it

### Changed

- Search has 10 results by default, max 100 results
- Queue structure generalized to handle more event types
- Summarizer now exhaustive by default and tuned for performance

### Fixed

- Resolve race condition for peers that leave a session while sending messages
- Added explicit rollback to solve integrity error in queue
- Re-introduced Sentry tracing to deriver
- Better integrity logic in get_or_create API methods

## [2.1.2] — 2025-07-30

### Fixed

- Summarizer module to ignore empty summaries and pass appropriate one to get_context
- Structured Outputs calls with OpenAI provider to pass strict=True to Pydantic Schema

## [2.1.1] — 2025-07-23

### Added

- Test harness for custom Honcho evaluations
- Better support for session and peer aware dialectic queries
- Langfuse settings
- Added recent history to dialectic prompt, dynamic based on new context window size setting

### Fixed

- Summary queue logic
- Formatting of logs
- Filtering by session
- Peer targeting in queries

### Changed

- Made query expansion in dialectic off by default
- Overhauled logging
- Refactor summarization for performance and code clarity
- Refactor queue payloads for clarity

## [2.1.0] — 2025-07-17

### Added

- File uploads
- Brand new "ROTE" deriver system
- Updated dialectic system
- Local working representations
- Better logging for deriver/dialectic
- Endpoint for deriver queue status

### Fixed

- Document insertion
- Session-scoped and peer-targeted dialectic queries work now

### Removed

- Peer-level messages

### Changed

- Dialectic chat endpoint takes a single query
- Rearranged configuration values (LLM, Deriver, Dialectic, History->Summary)

## [2.0.5] - 2025-07-11

### Fixed

- Groq API client to use the Async library

## [2.0.4] - 2025-07-02

### Fixed

- Migration/provision scripts did not have correct database connection arguments, causing timeouts

## [2.0.3] - 2025-07-01

### Fixed

- Bug that causes runtime error when Sentry flags are enabled

## [2.0.2] - 2025-06-27

### Fixed

- Database initialization was misconfigured and led to provision_db script failing: switch to consistent working configuration with transaction pooler

## [2.0.1] - 2025-06-26

### Added

- Ergonomic SDKs for Python and TypeScript (uses Stainless underneath)
- Deriver Queue Status endpoint
- Complex arbitrary filters on workspace/session/peer/message
- Message embedding table for full semantic search

### Changed

- Overhauled documentation
- BasedPyright typing for entire project
- Resource filtering expanded to include logical operators

### Fixed

- Various bugs
- Use new config arrangement everywhere
- Remove hardcoded responses

## [2.0.0] - 2025-06-24

### Added

- Ability to get a peer's working representation
- Metadata to all data primitives (Workspaces, Peers, Sessions, Messages)
- Internal metadata to store Honcho's state no longer exposed in API
- Batch message operations and enhanced message querying with token and message count limits
- Search and summary functionalities scoped by workspace, peer, and session
- Session context retrieval with summaries and token allocation
- HNSW Index for Documents Table
- Centralized Configuration via Environment Variables or `config.toml` file

### Changed

- API route is now /v2/
- New architecture centered around the concept of a "peer" replaces the former
  "app"/"user"/"session" paradigm
- Workspaces replace "apps" as top-level namespace
- Peers replace "users"
- Sessions no longer nested beneath peers and no longer limited to a single
  user-assistant model. A session exists independently of any one peer and
  peers can be added to and removed from sessions.
- Dialectic API is now part of the Peer, not the Session
- Dialectic API now allows queries to be scoped to a session or "targeted"
  to a fellow peer
- Database schema migrated to adopt workspace/peer/session naming and structure
- Authentication and JWT scopes updated to workspace/peer/session hierarchy
- Queue processing now works on 'work units' instead of sessions
- Message token counting updated with tiktoken integration and fallback heuristic
- Queue and message processing updated to handle sender/target and task types for multi-peer scenarios

### Fixed

- Improved error handling and validation for batch message operations and metadata
- Database Sessions to be more atomic to reduce idle in transaction time

### Removed

- Metamessages removed in favor of metadata
- Collections and Documents no longer exposed in the API, solely internal
- Obsolete tests for apps, users, collections, documents, and metamessages

## [1.1.0] - 2025-05-15

### Added

- Normalize resources to remove joins and increase query performance
- Query tracing for debugging

### Changed

- `/list` endpoints to not require a request body
- `metamessage_type` to `label` with backwards compatibility
- Database Provisioning to rely on alembic
- Database Session Manager to explicitly rollback transactions before closing
  the connection

### Fixed

- Alembic Migrations to include initial database migrations
- Sentry Middleware to not report Honcho Exceptions

## [1.0.0] - 2025-04-10

### Added

- JWT based API authentication
- Configurable logging
- Consolidated LLM Inference via `ModelClient` class
- Dynamic logging configurable via environment variables

### Changed

- Deriver & Dialectic API to use Hybrid Memory Architecture
- Metamessages are not strictly tied to a message
- Database provisioning is a separate script instead of happening on startup
- Consolidated `session/chat` and `session/chat/stream` endpoints

## [0.0.16] - 2025-03-05

### Added

- Detailed custom exceptions for better error handling
- CLAUDE.md for claude code

### Changed

- Deriver to use a new cognitive architecture that only updates on user messages
  and updates user representation to apply more confidence scores to its known
  facts
- Dialectic API token cutoff from 150 tokens to 300
- Dialectic API uses Claude 3.7 Sonnet
- SQLAlchemy echo changed to false by default, can be enabled with SQL_DEBUG
  environment flag

### Fixed

- Self-hosting documentation and README to mention `uv` instead of `poetry`

## [0.0.15] - 2025-01-06

### Added

- Alembic for handling database migrations
- Additional indexes for reading Messages and Metamessages
- Langfuse for prompt tracing

### Changed

- API validation using Pydantic

### Fixed

- Dialectic Streaming Endpoint properly sends text in `StreamingResponse`
- Deriver Queue handles graceful shutdown

## [0.0.14] — 2024-11-14

### Changed

- Query Documents endpoint is a POST request for better DX
- `String` columns are now `TEXT` columns to match postgres best practices
- Docstrings to have better stainless generations

### Fixed

- Dialectic API to use most recent user representation
- Prepared Statements Transient Error with `psycopg`
- Queue parallel worker scheduling

## [0.0.13] — 2024-11-07

### Added

- Ability to clone session for a user to achieve more [loom-like](https://github.com/socketteer/loom/) behavior

## [0.0.12] — 2024-10-21

### Added

- GitHub Actions Testing
- Ability to disable derivations on a session using the `deriver_disabled` flag
  in a session's metadata
- `/v1/` prefix to all routes
- Environment variable to control deriver workers

### Changed

- public_ids to use [NanoID](https://github.com/ai/nanoid) and internal ID to
  use `BigInt`
- Dialectic Endpoint can take a list of queries
- Using `uv` for project management
- User Representations stored in a metamessage rather than using reserved
  collection
- Base model for Dialectic API and Deriver is now Claude 3.5 Sonnet
- Paginated GET requests now POST requests for better developer UX

### Removed

- Mirascope Dependency
- Slowapi Dependency
- Opentelemetry Dependencies and Setup

## [0.0.11] — 2024-08-01

### Added

- `session_id` column to `QueueItem` Table
- `ActiveQueueSession` Table to track, which sessions are being actively
  processed
- Queue can process multiple sessions at once

### Changed

- Sessions do not require a `location_id`
- Detailed printing using `rich`

## [0.0.10] — 2024-07-23

### Added

- Test cases for Storage API
- Sentry tracing and profiling
- Additional Error handling

### Changed

- Document API uses same embedding endpoint as deriver
- CRUD operations use one less database call by removing extra refresh
- Use database for timestampz rather than API
- Pydantic schemas to use modern syntax

### Fixed

- Deriver queue resolution

## [0.0.9] — 2024-05-16

### Added

- Deriver to docker compose
- Postgres based Queue for background jobs

### Changed

- Deriver to use a queue instead of supabase realtime
- Using mirascope instead of langchain

### Removed

- Legacy SDKs in preference for stainless SDKs

## [0.0.8] — 2024-05-09

### Added

- Documentation to OpenAPI
- Bearer token auth to OpenAPI routes
- Get by ID routes for users and collections
- [NodeJS](https://github.com/plastic-labs/honcho-node) SDK support

### Changed

- Authentication Middleware now implemented using built-in FastAPI Security
  module
- Get by name routes for users and collections now include "name" in slug
- Python SDK moved to separate [repository](https://github.com/plastic-labs/honcho-python)

### Fixed

- Error reporting for methods with integrity errors due to unique key
  constraints

## [0.0.7] — 2024-04-01

### Added

- Authentication Middleware Interface

## [0.0.6] — 2024-03-21

### Added

- Full docker-compose for API and Database

### Fixed

- API Response schema removed unnecessary fields
- OTEL logging to properly work with async database engine
- `fly.toml` default settings for deriver set `auto_stop=false`

### Changed

- Refactored API server into multiple route files

## [0.0.5] — 2024-03-14

### Added

- Metadata to all data primitives (Users, Sessions, Messages, etc.)
- Ability to filter paginated GET requests by JSON filter based on metadata
- Optional Sentry error monitoring
- Optional Opentelemetry logging
- Dialectic API to interact with honcho agent and get insights about users
- Automatic Fact Derivation Script for automatically generating simple memory

### Changed

- API Server now uses async methods to make use of benefits of FastAPI

## [0.0.4] — 2024-02-22

### Added

- apps table with a relationship to the users table
- users table with a relationship to the collections and sessions tables
- Reverse Pagination support to get recent messages, sessions, etc. more easily
- Linting Rules

### Changed

- Get sessions method returns all sessions including inactive
- using timestampz instead of timestamp

## [0.0.3] — 2024-02-15

### Added

- Collections table to reference a collection of embedding documents
- Documents table to hold vector embeddings for RAG workflows
- Local scripts for running a postgres database with pgvector installed
- OpenAI Dependency for embedding models
- PGvector dependency for vector db support

### Changed

- session_data is now metadata
- session_data is a JSON field used python `dict` for compatibility

## [0.0.2] — 2024-02-01

### Added

- Pagination for requests via `fastapi_pagination`
- Metamessages
- `get_message` routes
- `created_at` field added to each Table
- Message size limits

### Changed

- IDs are now UUIDs
- default rate limit now 100 requests per minute

### Removed

- Removed messages from session response model

## [0.0.1] — 2024-02-01

### Added

- Rate limiting of 10 requests for minute
- Application level scoping
