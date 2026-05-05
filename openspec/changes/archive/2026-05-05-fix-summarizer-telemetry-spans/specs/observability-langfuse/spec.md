## MODIFIED Requirements

### Requirement: Langfuse Observability Tracing

#### Scenario: Summarization Tracing
- **WHEN** a background task or explicit request triggers `create_short_summary` or `create_long_summary`
- **THEN** the system MUST trace it as a top-level `GENERATION` observation without nested `SPAN` wrappers to ensure accurate model and token attribution in the Langfuse UI
