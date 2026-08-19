# Live LLM Tests

These tests call real provider APIs and are disabled by default.

Run them with:

```bash
uv run pytest tests/live_llm -n 0 --live-llm --no-header -q
```

Required API key env vars:

- `LLM_ANTHROPIC_API_KEY`
- `LLM_OPENAI_API_KEY`
- `LLM_GEMINI_API_KEY`

Model-family env vars:

- `LIVE_LLM_ANTHROPIC_45_PLUS_MODELS`
- `LIVE_LLM_OPENAI_GPT4_MODELS`
- `LIVE_LLM_OPENAI_GPT5_MODELS`
- `LIVE_LLM_OPENAI_OPENROUTER_NON_REASONING_MODELS` (OpenAI-transport → OpenRouter-served non-reasoning models)
- `LIVE_LLM_GEMINI_25_MODELS`
- `LIVE_LLM_GEMINI_30_MODELS`
- `LIVE_LLM_GEMINI_31_MODELS`

Embedding-model env vars:

- `LIVE_EMBEDDING_GEMINI_MODELS` (default: `gemini-embedding-001,gemini-embedding-2`; add `gemini-embedding-2-preview` to cover the preview twin)
- `LIVE_EMBEDDING_OPENAI_MODELS` (default: `text-embedding-3-small`)
- `LIVE_EMBEDDING_OPENAI_COMPATIBLE_MODELS` (no default → skipped) — OpenAI transport pointed at a third-party OpenAI-compatible provider. Also reads `OPENROUTER_API_KEY`, `LIVE_EMBEDDING_OPENAI_COMPATIBLE_BASE_URL` (default `https://openrouter.ai/api/v1`), `LIVE_EMBEDDING_OPENAI_COMPATIBLE_DIMENSIONS` (default `3072`) and `LIVE_EMBEDDING_OPENAI_COMPATIBLE_SEND_DIMENSIONS` (default on; set to `0` for a provider that rejects OpenAI's `dimensions` param)

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
export LIVE_EMBEDDING_OPENAI_COMPATIBLE_MODELS="google/gemini-embedding-001"
```

Each model env var accepts a comma-separated list of bare model ids or provider-qualified ids.

Examples:

```bash
export LIVE_LLM_ANTHROPIC_45_PLUS_MODELS="claude-sonnet-4-5,claude-sonnet-4-6"
export LIVE_LLM_OPENAI_GPT4_MODELS="gpt-4.1"
export LIVE_LLM_OPENAI_GPT5_MODELS="gpt-5,gpt-5.4,gpt-5.4-mini"
export LIVE_LLM_OPENAI_OPENROUTER_NON_REASONING_MODELS="inception/mercury-2"
export LIVE_LLM_GEMINI_25_MODELS="gemini-2.5-flash,gemini-2.5-pro"
export LIVE_LLM_GEMINI_30_MODELS="gemini-3-flash-preview"
export LIVE_LLM_GEMINI_31_MODELS="gemini-3.1-pro-preview"
```

OpenRouter-routed models require additional env for the proxy endpoint:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
# Per-feature config example:
#   DERIVER_MODEL_CONFIG__TRANSPORT=openai
#   DERIVER_MODEL_CONFIG__MODEL=inception/mercury-2
#   DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL=https://openrouter.ai/api/v1
#   DERIVER_MODEL_CONFIG__OVERRIDES__API_KEY_ENV=OPENROUTER_API_KEY
```

Coverage by provider:

- Anthropic: structured output path, prompt caching metrics, thinking blocks, multi-turn tool replay
- OpenAI GPT-4 class: structured outputs, prompt caching
- OpenAI GPT-5 class (incl. gpt-5.x point-releases): structured outputs, prompt caching, `reasoning_effort`, `max_completion_tokens` routing
- OpenAI transport → OpenRouter non-reasoning models (e.g. `inception/mercury-2`): non-chat / diffusion architectures must stay on `max_tokens`, no `reasoning_effort`, tool-calling parameter-schema compatibility is the canary for exotic OR-served providers
- Gemini 2.5/3.0 classes: structured outputs, cached-content reuse, thought signatures, multi-turn tool replay
- Gemini 3.1 class: thinking and tool replay coverage by default; structured-output/caching coverage should only be added once Google documents support for that path
- Embeddings (`test_live_embeddings.py`): single embed, batched embed, batch-vs-single alignment, and chunk-to-id mapping for every configured embedding model. `gemini-embedding-2*` is the reason this exists — those models collapse a list of bare strings into one document (#745), and only a live call catches it. Also covers first-class `EmbeddingModelConfig.timeout` plumbing (one representative model per transport): configured timeout lands on the SDK client, and a near-zero timeout aborts before the provider answers
- OpenAI-compatible embedding providers (e.g. OpenRouter's `google/gemini-embedding-001`): the #932 surface. Those providers reject a base64 embedding request outright (HTTP 400) or answer HTTP 200 with empty data, so the whole matrix fails without `encoding_format="float"`. Real OpenAI accepts base64 happily, so only a third-party provider catches it. Note that OpenRouter load-balances across upstreams, so the base64 failure is per-attempt rather than guaranteed: a retry can land on an endpoint that accepts it. `test_live_openai_float_encoding_matches_base64` covers the other side, that the float switch must not move vectors on real OpenAI
