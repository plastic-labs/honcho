# Unified Honcho Test System

This system allows for defining comprehensive, step-based tests for Honcho in a unified JSON format. It supports testing configuration hierarchy, multi-turn interactions, and complex assertions including LLM-as-a-judge.

## Running Tests

```bash
# Run all tests in the test_cases directory
python -m tests.unified.run

# Run a specific test file
python -m tests.unified.run --test-dir tests/unified/test_cases
```

## Test Schema

Tests are defined in JSON files. A test definition consists of a name, optional description, and a list of steps.

### Structure

```json
{
  "name": "my_test",
  "workspace_config": { ... },
  "steps": [
    { "step_type": "..." },
    ...
  ]
}
```

### Actions

1. **Configuration**:
    * `set_workspace_config`: Update workspace settings.
    * `set_session_config`: Update session settings.

2. **Interaction**:
    * `create_session`: Create a new session, optionally with peers and config.
    * `add_message`: Add a single message.
    * `add_messages`: Add multiple messages.

3. **Waiting**:
    * `wait`: Wait for duration or "queue_empty".

4. **Querying & Assertions**:
    * `query`: Perform an action and assert on the result.
        * `target`: "chat", "get_context", "get_peer_card", "get_representation"

5. **Agentic FDE** (Self-adapting Honcho):
    * `set_agent_config`: Set custom rules for deriver and/or dialectic prompts.
        * `deriver_rules`: Custom rules injected into observation extraction.
        * `dialectic_rules`: Custom rules injected into query responses.
    * `submit_feedback`: Submit natural language feedback to configure Honcho.
        * `message`: The feedback message.
        * `include_introspection`: Include latest introspection report (default: true).
        * `assertions`: Optional assertions on the feedback response.
    * `trigger_introspection`: Trigger a meta-cognitive introspection dream.
        * `wait_for_completion`: Wait for introspection to finish (default: true).
        * `timeout`: Timeout in seconds (default: 120).
    * `query_introspection`: Query the latest introspection report.
        * `assertions`: Assertions to run on the report.

### Assertions

* `llm_judge`: Use Claude to evaluate the result against a natural language prompt.
* `contains` / `not_contains`: Substring matching.
* `exact_match`: Strict equality.
* `json_match`: specific key-value checks.

## Example

```json
{
  "name": "demo_config_flow",
  "steps": [
    {
      "step_type": "create_session",
      "session_id": "s1",
      "peer_configs": {
        "user": { "observe_me": true },
        "agent": { "observe_others": true }
      }
    },
    {
      "step_type": "add_message",
      "session_id": "s1",
      "peer_id": "user",
      "content": "My name is Alice."
    },
    {
      "step_type": "wait",
      "target": "queue_empty"
    },
    {
      "step_type": "query",
      "target": "chat",
      "peer_id": "agent",
      "session_id": "s1",
      "input": "Who am I?",
      "assertions": [
        {
          "assertion_type": "contains",
          "text": "Alice"
        }
      ]
    }
  ]
}
```
