# Local Codex Responses relay

`src.codex_relay` is a narrow, local-only OpenAI Chat Completions compatibility
process for Honcho development. It translates the supported Chat request subset
to a streaming Codex Responses request and translates the response back.

## Security and credential ownership

The relay does not refresh or write Hermes credentials. Hermes remains the sole
owner of OAuth rotation, locks, profile selection, pool selection, and persistence.
The optional `--auth-path` is a read-only JSON source for a single canonical
`providers.openai-codex.tokens.access_token` record; it is not a Hermes credential
adapter and must not be used as a shared refresh store. For rotation, the embedding
process must inject a token provider (`TokenProvider(force_refresh)`) that delegates
to the active Hermes profile. No credential value is logged by the relay.

The default listener is loopback-only and may be used without an inbound key:

```bash
uv run python -m src.codex_relay --host 127.0.0.1 --port 8787 \
  --auth-path "$HERMES_HOME/profiles/$HERMES_PROFILE/auth.json"
```

Every non-loopback bind requires a separately configured relay key. The key is
accepted as an Authorization Bearer relay key and is compared in constant time.
Configure it through the environment or a secret manager; do not put it in
source control or logs:

```bash
CODEX_RELAY_INBOUND_KEY='use-a-secret-manager-value' \
  uv run python -m src.codex_relay --host 192.0.2.10 --port 8787
```

Do not use `0.0.0.0` as a claim of privacy. If wildcard binding is unavoidable,
require the inbound key, restrict the published port with a firewall/container
network policy, and treat every reachable peer as untrusted. The upstream Codex
transport is HTTPS by default at
`https://chatgpt.com/backend-api/codex/responses`.

## Honcho configuration

Point the current upstream `OpenAIBackend`/`AsyncOpenAI` client at
`http://127.0.0.1:8787/v1` (or the explicitly trusted relay address), use a model
name accepted by the upstream configuration, and provide Honcho's required
OpenAI API-key field with the same relay key. The relay key is inbound relay
authentication; it is not an upstream Codex OAuth token.

Supported Chat controls are: messages, model, stream, tools, tool_choice
(`none`, `auto`, `required`, or a named function), parallel_tool_calls,
reasoning_effort, response_format (`json_object`/`json_schema`), and
max_tokens/max_completion_tokens (translated to `max_output_tokens`). The current
Honcho OpenAI streaming backend's `stream_options: {"include_usage": true}` is
accepted; the relay emits an OpenAI-compatible final usage chunk before `[DONE]`.
Conflicting output-limit aliases and invalid values are rejected. Controls that
cannot be represented safely by Codex Responses (temperature, top_p, stop,
penalties, seed, logprobs, n, user, verbosity, reasoning, and SDK passthrough
fields such as extra_body) are rejected with a clear 400 rather than silently
dropped. Other unknown top-level fields and malformed supported fields are also
rejected with a clear 400.

Responses streams must end in `response.completed` or `response.incomplete`.
The relay stops reading immediately after the first valid terminal event and
closes the upstream response. Provider HTTP error bodies are sanitized to a
stable OpenAI-compatible error while preserving the upstream status code;
credential, malformed data, malformed usage, EOF/truncation, and transport
failures are sanitized as well. An already-started downstream stream emits one
error event followed by one `[DONE]`. Incomplete content-filter results map to
`content_filter`; other incomplete results map to `length`.

The credential document must contain a mapping at
`providers.openai-codex.tokens.access_token` with a non-empty string token.
Missing, corrupt, or malformed documents and non-ASCII/empty transport tokens
return a sanitized 503 credential error. Malformed or opaque JWT claims are
never decoded into headers or reflected in errors; if the upstream rejects such
a token, its HTTP error is sanitized.

## Reproducible checks

From a clean checkout with the locked development dependencies available:

```bash
uv sync --frozen
uv run pytest tests/llm/test_codex_relay.py -q -n 0
uv run ruff check src/codex_relay.py tests/llm/test_codex_relay.py
uv run basedpyright src/codex_relay.py tests/llm/test_codex_relay.py
```

These tests use synthetic tokens and `httpx.MockTransport`/in-process ASGI only;
they do not read live credentials or call a live provider. Before review, also
run `git diff --check` and the repository's approved secret scanner. The targeted
type command may report pre-existing warnings in a dependency-complete checkout;
record its actual exit code rather than treating warnings as success.
