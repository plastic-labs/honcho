## 1. Core Fallback Logic Changes

- [x] 1.1 Modify `select_model_config_for_attempt()` in `src/llm/runtime.py` to support `force_fallback` parameter — when True, swap to fallback config immediately instead of waiting for final retry
- [x] 1.2 Modify `plan_attempt()` in `src/llm/runtime.py` to accept and propagate `force_fallback` flag
- [x] 1.3 Modify `honcho_llm_call()` in `src/llm/api.py` to detect first failure (429/5xx/timeout) and set `force_fallback=True` for subsequent attempts
- [x] 1.4 Add WARNING-level log in `plan_attempt()` when fallback is activated, including agent context (provider, model, failure reason)
- [x] 1.5 Add `is_fallback` signal to Langfuse generation metadata via `update_current_langfuse_observation()`

## 2. Configuration Changes

- [x] 2.1 Add `FALLBACK__TRANSPORT` and `FALLBACK__MODEL` env var support to `ConfiguredModelSettings` in `src/config.py` (via existing `fallback: FallbackModelSettings | None` field — no new fields needed, just env var mapping)
- [x] 2.2 Add env var documentation to `.env.example` for all 4 agents
- [x] 2.3 Add fallback config to `docker-compose/honcho/.env` with example values (commented out)

## 3. Agent Integration

- [x] 3.1 Verify Deriver (`src/deriver/deriver.py`) passes model_config correctly — no code change needed, just verify fallback flows through
- [x] 3.2 Verify Dialectic (`src/dialectic/core.py`) passes model_config correctly — no code change needed, just verify fallback flows through
- [x] 3.3 Verify Dreamer (`src/dreamer/specialists.py`) passes model_config correctly — no code change needed, just verify fallback flows through
- [x] 3.4 Verify Summary (`src/utils/summarizer.py`) passes model_config correctly — no code change needed, just verify fallback flows through

## 4. Unit Tests

- [x] 4.1 Write test for `select_model_config_for_attempt()` with `force_fallback=True` — verifies fallback config is returned on first attempt
- [x] 4.2 Write test for `select_model_config_for_attempt()` with `force_fallback=False` — verifies existing behavior (primary on non-final attempts)
- [x] 4.3 Write test for `plan_attempt()` with `force_fallback=True` — verifies AttemptPlan uses fallback provider/model
- [x] 4.4 Write test for `_is_retryable_error()` — verifies 429/5xx/Timeout/Connection classified as retryable
- [x] 4.5 Write test for non-retryable errors — verifies 400/200/ValueError NOT retryable
- [x] 4.6 Write test for cross-provider fallback — primary=openai→fallback=anthropic, primary=nous→fallback=openai

## 5. Integration Test

- [x] 5.1 Write integration test: configure primary with invalid URL, fallback with valid URL, verify request succeeds via fallback
- [x] 5.2 Write integration test: configure primary to return 429, verify fallback triggers within 1 retry (not 3)

## 6. Documentation

- [x] 6.1 Update `CLAUDE.md` with fallback configuration instructions
- [x] 6.2 Add fallback section to deployment docs (if exists) — no separate deployment docs exist; fallback documented in CLAUDE.md and .env.template
