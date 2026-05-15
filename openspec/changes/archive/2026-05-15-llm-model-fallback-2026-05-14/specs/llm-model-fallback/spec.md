## ADDED Requirements

### Requirement: System SHALL support configurable LLM model fallback per agent
The system SHALL allow operators to configure a fallback model for each agent (Deriver, Dialectic, Dreamer, Summary) independently. When the primary model fails, the system SHALL automatically switch to the configured fallback model.

#### Scenario: Fallback configured for Deriver
- **WHEN** `DERIVER_MODEL_CONFIG__FALLBACK__TRANSPORT` and `DERIVER_MODEL_CONFIG__FALLBACK__MODEL` are set in environment
- **THEN** the Deriver agent SHALL use the configured fallback model when its primary model fails

#### Scenario: Fallback configured for Dialectic
- **WHEN** `DIALECTIC_MODEL_CONFIG__FALLBACK__TRANSPORT` and `DIALECTIC_MODEL_CONFIG__FALLBACK__MODEL` are set in environment
- **THEN** the Dialectic agent SHALL use the configured fallback model when its primary model fails

#### Scenario: Fallback configured for Dreamer
- **WHEN** `DREAM_MODEL_CONFIG__FALLBACK__TRANSPORT` and `DREAM_MODEL_CONFIG__FALLBACK__MODEL` are set in environment
- **THEN** the Dreamer agent SHALL use the configured fallback model when its primary model fails

#### Scenario: Fallback configured for Summary
- **WHEN** `SUMMARY_MODEL_CONFIG__FALLBACK__TRANSPORT` and `SUMMARY_MODEL_CONFIG__FALLBACK__MODEL` are set in environment
- **THEN** the Summary generator SHALL use the configured fallback model when its primary model fails

### Requirement: Fallback SHALL trigger on first model failure
The system SHALL switch to the fallback model on the **first** detected failure (rate limit, timeout, API error), not only on the final retry attempt. This reduces latency during provider outages.

#### Scenario: Primary model returns rate limit error
- **WHEN** the primary model returns a 429 (Too Many Requests) error on the first attempt
- **THEN** the system SHALL immediately switch to the fallback model for the next attempt without exhausting all primary retries

#### Scenario: Primary model times out
- **WHEN** the primary model exceeds the configured timeout on the first attempt
- **THEN** the system SHALL immediately switch to the fallback model for the next attempt

#### Scenario: Primary model returns server error
- **WHEN** the primary model returns a 5xx error on the first attempt
- **THEN** the system SHALL immediately switch to the fallback model for the next attempt

### Requirement: System SHALL log fallback events for observability
The system SHALL emit a log entry at WARNING level every time a fallback event occurs, including: agent name, primary provider/model, fallback provider/model, and failure reason.

#### Scenario: Fallback triggered for Deriver
- **WHEN** the Deriver agent falls back from primary to fallback model
- **THEN** the system SHALL log a WARNING message containing: agent=deriver, primary=<provider>/<model>, fallback=<provider>/<model>, reason=<error>

#### Scenario: Fallback triggered for Dialectic
- **WHEN** the Dialectic agent falls back from primary to fallback model
- **THEN** the system SHALL log a WARNING message containing: agent=dialectic, primary=<provider>/<model>, fallback=<provider>/<model>, reason=<error>

### Requirement: System SHALL fail only when all models in chain are exhausted
The system SHALL only return an error to the caller when both the primary model AND the fallback model have been tried and failed. If the fallback succeeds, the response SHALL be returned normally.

#### Scenario: Primary fails, fallback succeeds
- **WHEN** the primary model fails and the fallback model succeeds
- **THEN** the system SHALL return the fallback model's response as if it were a normal response

#### Scenario: Both primary and fallback fail
- **WHEN** both the primary model and the fallback model fail after exhausting their retry attempts
- **THEN** the system SHALL return an error indicating all models in the chain were exhausted

### Requirement: Fallback SHALL support cross-provider failover
The fallback model MAY use a different provider than the primary model (e.g., primary=openai, fallback=anthropic). The system SHALL handle cross-provider parameter mapping automatically.

#### Scenario: Cross-provider fallback from OpenAI to Anthropic
- **WHEN** primary is `openai/gpt-4o` and fallback is `anthropic/claude-sonnet-4`
- **THEN** the system SHALL automatically map transport parameters (reasoning_effort, thinking_budget_tokens) to provider-appropriate values

#### Scenario: Cross-provider fallback from Nous to OpenAI
- **WHEN** primary is `nous/model` and fallback is `openai/gpt-4o-mini`
- **THEN** the system SHALL automatically map transport parameters to provider-appropriate values

### Requirement: Fallback configuration SHALL be backward compatible
When no fallback is configured, the system SHALL behave exactly as before — using only the primary model with existing retry logic. Fallback is opt-in via configuration.

#### Scenario: No fallback configured
- **WHEN** no `FALLBACK__TRANSPORT` / `FALLBACK__MODEL` environment variables are set
- **THEN** the system SHALL use only the primary model with existing retry behavior (no change from current behavior)

#### Scenario: Partial fallback config (only transport, no model)
- **WHEN** `FALLBACK__TRANSPORT` is set but `FALLBACK__MODEL` is not set
- **THEN** the system SHALL ignore the incomplete fallback config and use only the primary model
