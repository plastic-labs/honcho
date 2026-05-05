# Exploration: Nested Langfuse Observations & Missing Telemetry Columns

**Date**: 2026-05-05
**Topic**: Root cause analysis for `Create Short Summary` (and similar traces) missing model and token attribution in the Langfuse UI.

## The Problem
The user noticed that the trace observation named **"Create Short Summary"** appears in Langfuse without a defined model or token usage, despite the underlying LLM call functioning correctly. They hypothesized that this might be a widespread issue for other traces.

## System Analysis

### How the Summarizer works
In `src/utils/summarizer.py`, the functions are structured like this:
```python
@conditional_observe(name="Create Short Summary")
async def create_short_summary(...) -> HonchoLLMCallResponse[str]:
    # ...
    return await honcho_llm_call(...)
```

### The "Nested Observation" Conflict
1. The `@conditional_observe` decorator on the outer function creates a **SPAN** observation (since it does not specify `as_type="generation"`).
2. Inside that function, `honcho_llm_call` executes. `honcho_llm_call` is wrapped with `@conditional_observe(name="LLM Call", as_type="generation")`.
3. Consequently, Langfuse records a nested hierarchy:
   - **SPAN**: "Create Short Summary" *(No model/tokens)*
     - **GENERATION**: "LLM Call" *(Contains model/tokens)*

Because Langfuse UI (specifically the trace and generations tables) only aggregates model and token statistics at the **GENERATION** level, the top-level "Create Short Summary" span appears empty.

### How other modules (e.g. Dialectic Agent) avoid this
In `src/dialectic/core.py`, the `Dialectic Agent` does **not** use an outer `@conditional_observe` span. Instead, it utilizes the `track_name` argument directly:
```python
return await honcho_llm_call(
    # ...
    track_name="Dialectic Agent",
)
```
This causes the inner `honcho_llm_call` GENERATION observation to dynamically rename itself to "Dialectic Agent", cleanly consolidating the trace into a single generation with full token/model attribution.

## Scope of the Issue
A codebase-wide search reveals that ONLY the `create_short_summary` and `create_long_summary` functions suffer from this nested `@conditional_observe` pattern. 

## Recommended Path Forward (Actionable Fix)
We can fix this permanently and elegantly by aligning the summarizer module with the `Dialectic Agent` pattern:
1. **Remove** `@conditional_observe(name="Create Short Summary")` and `name="Create Long Summary"` from `src/utils/summarizer.py`.
2. **Inject** `track_name="Create Short Summary"` and `track_name="Create Long Summary"` into their respective `honcho_llm_call(...)` invocations.

This will collapse the nested traces into single, pure GENERATION observations that fully populate the Langfuse UI.
