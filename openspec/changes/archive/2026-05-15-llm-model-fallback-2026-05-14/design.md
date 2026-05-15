## Context

Honcho's LLM layer (`src/llm/`) already has a complete fallback mechanism:
- `ModelConfig.fallback` field (runtime-resolved `ResolvedFallbackConfig`)
- `FallbackModelSettings` in `ConfiguredModelSettings` (operator-facing config)
- `select_model_config_for_attempt()` — swaps to fallback on final retry
- `plan_attempt()` — builds `AttemptPlan` with fallback provider/model
- `honcho_llm_call()` — tenacity retry + fallback on last attempt

**Gap:** The fallback mechanism exists in code but is NEVER configured in production. None of the 4 agents (Deriver, Dialectic, Dreamer, Summary) have `fallback` set in their `MODEL_CONFIG`. Additionally, the current code only triggers fallback on the **final** retry attempt, not on first failure.

**Constraints:**
- Must be backward compatible — no fallback config = existing behavior unchanged
- Must support cross-provider failover (different transport for primary vs fallback)
- Must not break existing `honcho_llm_call()` signature or behavior
- Must work with existing tenacity retry mechanism

## Goals / Non-Goals

**Goals:**
1. Configure fallback models for all 4 agents via environment variables
2. Change fallback trigger from "final retry" to "first failure" for faster failover
3. Add observability (WARNING-level log) for fallback events
4. Support per-agent independent fallback configuration
5. Support cross-provider failover with automatic parameter mapping

**Non-Goals:**
- Adding a third model to the fallback chain (only primary + 1 fallback)
- Dynamic fallback selection based on error type (all errors trigger same fallback)
- Fallback for embedding client (only LLM calls)
- UI/dashboard for fallback monitoring (logging only)

## Decisions

### Decision 1: Fallback trigger on first failure (not final retry)

**Current behavior:** `select_model_config_for_attempt()` only swaps to fallback when `attempt == retry_attempts` (the last attempt).

**New behavior:** Swap to fallback on the **first** failure, then use fallback for remaining retries.

**Rationale:** During a provider outage, waiting for all primary retries to exhaust before switching adds unnecessary latency. A single 429/5xx/timeout is a strong signal that the provider is having issues.

**Alternative considered:** Keep final-retry fallback only. Rejected because it adds 2-3 extra failed attempts (with exponential backoff) before switching, which can add 10-30 seconds of latency per LLM call.

**Implementation:** Modify `select_model_config_for_attempt()` to accept a `force_fallback` flag. When a failure is detected, set the flag and swap to fallback for the next attempt.

### Decision 2: Per-agent fallback via environment variables

**Approach:** Add `FALLBACK__TRANSPORT` and `FALLBACK__MODEL` to each agent's model config env vars:
- `DERIVER_MODEL_CONFIG__FALLBACK__TRANSPORT` / `DERIVER_MODEL_CONFIG__FALLBACK__MODEL`
- `DIALECTIC_MODEL_CONFIG__FALLBACK__TRANSPORT` / `DIALECTIC_MODEL_CONFIG__FALLBACK__MODEL`
- `DREAM_MODEL_CONFIG__FALLBACK__TRANSPORT` / `DREAM_MODEL_CONFIG__FALLBACK__MODEL`
- `SUMMARY_MODEL_CONFIG__FALLBACK__TRANSPORT` / `SUMMARY_MODEL_CONFIG__FALLBACK__MODEL`

**Rationale:** Consistent with existing env var naming convention (`<AGENT>_MODEL_CONFIG__<FIELD>`). Per-agent config allows operators to set different fallbacks for different agents (e.g., Deriver uses cheaper fallback, Dialectic uses same-tier fallback).

**Alternative considered:** Single global `FALLBACK_MODEL_CONFIG__*` for all agents. Rejected because different agents may need different fallback strategies.

### Decision 3: Cross-provider parameter mapping

**Approach:** When the fallback uses a different transport, the system SHALL automatically map transport-specific parameters:
- `thinking_budget_tokens` → only for Anthropic (ignored for OpenAI/Gemini)
- `thinking_effort` / `reasoning_effort` → mapped to provider-appropriate values

**Rationale:** The existing `plan_attempt()` already handles this via `call_thinking_budget_tokens if is_primary else selected.thinking_budget_tokens`. The fallback config carries its own thinking params.

### Decision 4: Observability via WARNING-level logs

**Approach:** Add `logger.warning()` in `select_model_config_for_attempt()` when fallback is activated, including agent context from the caller.

**Rationale:** WARNING level is appropriate — fallback is not an error (the system recovers), but operators should know it happened. Structured log messages enable log aggregation tools to alert on fallback frequency.

**Alternative considered:** Custom telemetry event. Rejected as over-engineering for v1. Logs are sufficient; can add telemetry events later if needed.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Fallback model also fails | System exhausts fallback retries then returns error. No infinite loop. |
| Cross-provider param mismatch | Fallback config carries its own params. `plan_attempt()` already handles this. |
| Env var misvalidation | Pydantic validation on `FallbackModelSettings` catches invalid transport/model at startup. |
| Increased cost from fallback | Fallback is only used when primary fails. Normal traffic uses primary only. |
| Stale fallback config | If fallback model is decommissioned, operators will see errors in logs and can update config. |

## Migration Plan

1. **Deploy code changes** — modify `select_model_config_for_attempt()` and `honcho_llm_call()` to support first-failure fallback
2. **Add env vars** — add `FALLBACK__TRANSPORT` and `FALLBACK__MODEL` to `.env` for each agent
3. **Restart gateway** — pick up new config
4. **Verify** — check logs for fallback events during normal operation (should be none)
5. **Test** — temporarily point primary to invalid URL, verify fallback triggers

**Rollback:** Remove fallback env vars from `.env` and restart. Code is backward compatible — without config, behavior is identical to before.

## Open Questions

1. Should we add a metric counter for fallback events (e.g., `honcho_llm_fallback_total{agent, from, to}`)?
   - **Recommendation:** Yes, but as a follow-up. Logs are sufficient for v1.
2. Should the fallback chain support more than 2 models?
   - **Recommendation:** No. Primary + 1 fallback covers 99% of use cases. Can extend later.
3. Should we add a circuit breaker pattern (e.g., don't retry primary at all after N consecutive failures)?
   - **Recommendation:** Out of scope for this change. Consider as separate improvement.
