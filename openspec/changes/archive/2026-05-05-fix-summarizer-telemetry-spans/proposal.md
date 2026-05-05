## Why

Langfuse UI only aggregates and displays `model` and `tokens` explicitly at the `GENERATION` observation level. Currently, the `create_short_summary` and `create_long_summary` functions in `src/utils/summarizer.py` are wrapped with `@conditional_observe`, which creates a top-level `SPAN` observation. Inside them, `honcho_llm_call` creates a nested `GENERATION` observation. Because the root observation is a `SPAN`, the Langfuse dashboard does not display model and token usage for the summarizer traces at a glance, obscuring critical telemetry for these operations.

## What Changes

- Remove the `@conditional_observe(name="Create Short Summary")` and `@conditional_observe(name="Create Long Summary")` decorators from the summarizer functions.
- Inject the names explicitly via the `track_name="Create Short Summary"` and `track_name="Create Long Summary"` parameters in the inner `honcho_llm_call` invocations.
- This collapses the traces into single, pure `GENERATION` observations that properly bubble up their telemetry in the Langfuse UI, mirroring the successful pattern used in `Dialectic Agent`.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `observability-langfuse`: Require that all LLM interactions, including summarizers, cleanly propagate top-level generation traces without opaque span wrappers.

## Impact

- `src/utils/summarizer.py`: The `@conditional_observe` decorators will be removed and replaced with explicit `track_name` kwargs in the LLM calls.
- **Langfuse Telemetry**: Summarizer operations will appear as top-level `GENERATION`s instead of nested inside `SPAN`s, granting full visibility to token and model metadata.
