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

## Failure-mode reproducer test cases

Where existing benchmarks (LoCoMo, MemoryAgentBench, etc.) test memory at a high level of aggregation and bury specific failure modes inside coarse scores, this family of tests goes the other way: each test case is a small, named reproducer for one entry in Honcho's failure-modes catalog. Fixtures pair a positive control that should pass on current Honcho with a negative reproducer that targets the bug; per a verify-first discipline, polarity starts as `pass_if=true` (invariant) and flips to `pass_if=false` only after a fixture has been observed firing on real code, with the observed behavior documented in the fixture's `description` field. The first such fixture in this directory is `deriver_no_memory_signal.json`. New fixtures should reference the catalog entry they reproduce in their `description` field.

## Run artifacts + manual snapshot replay

Each test run writes to a per-test directory under `tests/unified/artifacts/<test_name>_<utc-timestamp>/` (override with `UNIFIED_TEST_ARTIFACTS_DIR`). Two kinds of files land there:

- **`_run_metadata.json`** is written at test start. It captures the state of the system under test so a later comparison run can be diffed apples-to-apples: the honcho repo's git SHA (and whether the working tree was dirty), the transport + model in use by each LLM-driven agent (deriver / dialectic / summary / dream / peer_card), SHA-256 hashes of the prompt source files, the judge model, and the run timestamp.
- **Artifact JSONs** come from explicit `save_artifact` steps in a fixture. Use this step type to dump the raw result of a `get_representation`, `get_peer_card`, or `get_context` call without coupling the dump to a pass/fail assertion — handy when the binary gate doesn't have the granularity you need offline (e.g., classifying *which* observations leaked, computing signal-to-noise ratios, comparing representations between runs).

Why the metadata matters: LLM-driven tests aren't deterministic in the way unit tests are. A 6/10-leak baseline today, fixed to 3/10 next week, could mean (a) your fix worked, (b) Anthropic shipped a new Haiku, (c) `config.toml` got tuned, or (d) the judge prompt drifted. Capturing metadata lets you tell which.

### Manual replay procedure

We don't currently automate replay (deferred until we've done a fix-eval comparison and know which variables matter to control for). To compare two runs by hand:

1. Read the original run's `_run_metadata.json`. Note the `honcho_git_sha`, the relevant `agents.*.{transport,model}`, and the `prompt_shas` for any file you didn't intentionally change.
2. Check out the repo at that SHA: `git checkout <honcho_git_sha>`.
3. Apply your intentional change on top (e.g., edit the deriver prompt), and only that change.
4. Set env vars to match the original `agents.*` configuration. For the deriver model, that's `DERIVER_MODEL_CONFIG__TRANSPORT=<transport>` and `DERIVER_MODEL_CONFIG__MODEL=<model>`; the equivalent prefixes are `DIALECTIC_`, `SUMMARY_`, `DREAM_`, `PEER_CARD_`.
5. Run the same fixture. Diff the new artifacts against the originals.

When `honcho_git_dirty: true` in metadata, the run included uncommitted changes — replay requires either committing them (so the SHA covers them) or recreating the working-tree state from saved patches.

