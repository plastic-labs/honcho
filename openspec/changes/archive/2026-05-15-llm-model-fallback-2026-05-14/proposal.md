## Why

Honcho's LLM calls fail entirely when the primary AI model is unavailable (rate limit, timeout, API error). The codebase already has a basic single-model fallback mechanism (`ModelConfig.fallback`), but it is never configured in production — the `.env` has no `FALLBACK_MODEL_CONFIG__*` variables. This means any transient provider outage causes complete service disruption for memory formation (Deriver), conversational reasoning (Dialectic), knowledge consolidation (Dreamer), and session summaries.

## What Changes

- Add `FALLBACK_MODEL_CONFIG__TRANSPORT` and `FALLBACK_MODEL_CONFIG__MODEL` to `.env` / settings as the global fallback model
- Configure all agents (Deriver, Dialectic, Dreamer, Summary) to use the configured fallback when their primary model fails
- Fallback triggers on the **first** failure (not only on the final retry attempt), reducing latency during provider issues
- Log fallback events for observability (which agent, which provider, which model, failure reason)
- Fail only when both primary and fallback models are exhausted

## Capabilities

### New Capabilities
- `llm-model-fallback`: Global fallback model configuration and per-agent fallback chain with observability

### Modified Capabilities
None. This is a new capability — no existing spec-level behavior changes.

## Impact

- **Affected code**: `src/config.py` (settings + `ConfiguredModelSettings`), `src/llm/runtime.py` (`select_model_config_for_attempt`, `plan_attempt`), `src/llm/api.py` (`honcho_llm_call`), `src/deriver/deriver.py`, `src/dialectic/chat.py`, `src/dreamer/`
- **Affected config**: `.env` (new `FALLBACK_MODEL_CONFIG__TRANSPORT`, `FALLBACK_MODEL_CONFIG__MODEL` variables)
- **Affected deployment**: `docker-compose/honcho/.env` (same new variables)
- **Dependencies**: None new — uses existing `FallbackModelSettings` / `ResolvedFallbackConfig` models already in codebase
- **Breaking changes**: None. Fallback is opt-in via config; existing behavior unchanged when not configured.
