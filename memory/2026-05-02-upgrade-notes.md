# Honcho Upgrade Notes (2026-05-02)

## Critical: LLM Module Restructuring

The upstream `plastic-labs/honcho` repo completely restructured the LLM client code:
- `src/utils/clients.py` → **DELETED** (was the "custom" provider path for Ollama)
- New: `src/llm/` module with registry, backends (openai, anthropic, gemini), structured output
- `ModelTransport = Literal["anthropic", "openai", "gemini"]` — no "custom" transport
- OpenAI-compatible endpoints (Ollama, OpenRouter, etc.) use `transport=openai` with `base_url` override

## New Config Format

Old (DEPRECATED, silently ignored):
```
DERIVER_PROVIDER=custom
DERIVER_MODEL=qwen3-coder:480b
LLM_OPENAI_COMPATIBLE_BASE_URL=https://ollama.com/v1
LLM_OPENAI_COMPATIBLE_API_KEY=...
```

New (REQUIRED):
```
DERIVER_MODEL_CONFIG__TRANSPORT=openai  # or gemini, anthropic
DERIVER_MODEL_CONFIG__MODEL=qwen3-coder:480b
DERIVER_MODEL_CONFIG__OVERRIDES__BASE_URL=https://ollama.com/v1
DERIVER_MODEL_CONFIG__OVERRIDES__API_KEY=...
```

## Current Issue: Both LLM Providers Exhausted

- **Ollama.com/v1**: Returns 401 "unauthorized" for chat completions (was working ~06:00 UTC)
- **Gemini free tier**: Quota exhausted (429 RESOURCE_EXHAUSTED) for both embeddings AND generation
- Gemini quota resets at midnight Pacific (07:00 UTC)
- Deriver is configured to use Gemini as fallback; will resume when quota resets

## Embedding Resilience Patches Applied

Patched 4 files on `origin/main + 1 commit`:
1. `src/crud/representation.py`: Catch embedding errors, save docs with `embedding=None`
2. `src/schemas/internal.py`: Make `embedding` field `Optional` (list[float] | None)
3. `src/crud/document.py`: Skip dedup when `embedding is None`
4. `src/embedding_client.py`: Fail-fast on 429 RESOURCE_EXHAUSTED (Gemini daily quota)

## Key Upstream Fixes Now In Place

- `fix(deriver): ignore blank observations before embedding (#615)`
- `fix(dreamer): threshold and time-guard semantics (#573)`
- `fix(config): use auto tool choice for dialectic defaults (#630)`
- `fix: give vector sync a substantial retry budget (#604)`
- Groq json_schema response_format fix (upstream refactored to src/llm/backends/)
- OpenAI dimensions parameter (18 call sites with `dimensions` param)