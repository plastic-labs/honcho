# observability-langfuse

## Purpose
TBD: Core capabilities for integrating Langfuse LLM observability and tracing telemetry within Honcho.

## Requirements

### Requirement: LLM calls are traced as Langfuse generations
When Langfuse tracing is enabled, each `honcho_llm_call` invocation MUST be recorded as a Langfuse observation with generation semantics.

#### Scenario: Generation observation type is used
- **WHEN** `LANGFUSE_PUBLIC_KEY` is configured and application code executes `honcho_llm_call`
- **THEN** the active Langfuse observation for that call is created with `as_type="generation"`

### Requirement: Generation model is attributed from attempt planning
The implementation MUST set the generation `model` field from the resolved model for the active attempt.

#### Scenario: Model field follows selected attempt model
- **WHEN** `plan_attempt` resolves provider/model (including final-attempt fallback)
- **THEN** `update_current_langfuse_observation` updates the active generation via `update_current_generation(model=<resolved-model>)`

### Requirement: Provider and namespace remain in metadata
Langfuse generation updates MUST include operational context metadata for provider and namespace.

#### Scenario: Metadata dimensions are preserved
- **WHEN** generation data is updated for an LLM call
- **THEN** metadata includes `provider` and `namespace` values used by Honcho routing context
