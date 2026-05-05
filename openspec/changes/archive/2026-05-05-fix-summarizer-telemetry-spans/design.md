## Context

Langfuse aggregates model identification and token usage at the `GENERATION` observation level. When an observation is created as a `SPAN`, it acts purely as a grouping container and its table view lacks these columns. 

In `src/utils/summarizer.py`, the `create_short_summary` and `create_long_summary` functions are wrapped in `@conditional_observe` decorators, emitting `SPAN` observations. However, they internally call `honcho_llm_call`, which emits a nested `GENERATION` observation. Langfuse captures the model/tokens on the inner Generation, leaving the outer Span telemetry empty in the UI. 

Other modules (like the Dialectic Agent) avoid this by not using an outer decorator, and instead passing `track_name` directly to `honcho_llm_call()`, creating a single top-level `GENERATION` named exactly what it needs to be.

## Goals / Non-Goals

**Goals:**
- Consolidate the Langfuse telemetry for summarizer operations into single `GENERATION` observations to expose model and token data at the root level of the trace.
- Align `src/utils/summarizer.py` with the `track_name` pattern established elsewhere in the codebase.

**Non-Goals:**
- Modifying the behavior, logic, or model choices of the summarizer algorithms themselves.
- Altering the implementation of `honcho_llm_call` or its core telemetry structure.

## Decisions

**1. Remove outer `@conditional_observe` from summarizers**
Instead of having a `SPAN` encompassing a `GENERATION`, we will completely remove the outer `SPAN`.
*Rationale:* Nested observations where the inner contains all the actionable metadata lead to confusing UI experiences. A single pure generation is exactly what these functions represent.

**2. Pass `track_name` to `honcho_llm_call`**
We will add `track_name="Create Short Summary"` and `track_name="Create Long Summary"` to their respective `honcho_llm_call` arguments.
*Rationale:* `honcho_llm_call` is already designed to consume a `track_name` and correctly name the `GENERATION` trace. This perfectly satisfies the observability requirement without redundant decorators.

## Risks / Trade-offs

- **Risk:** Any manual metadata that the outer `@conditional_observe` might have been capturing could be lost.
- **Mitigation:** The summarizer functions do not pass any special manual context kwargs or tags to the decorator; they simply name the span. The inner generation captures all input/output payload data reliably.
