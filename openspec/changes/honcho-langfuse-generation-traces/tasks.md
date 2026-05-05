## 0. Spec artifacts (after propose — correct sequencing)

- [x] 0.1 Create delta capability spec at `openspec/changes/honcho-langfuse-generation-traces/specs/observability-langfuse/spec.md` using OpenSpec schema instructions.

## 1. Implementation (retroactive — verify in tree)

- [x] 1.1 Extend **`conditional_observe`** to forward **`as_type`** to Langfuse **`observe()`** (`src/telemetry/logging.py`).
- [x] 1.2 Decorate **`honcho_llm_call`** with **`as_type="generation"`** (`src/llm/api.py`).
- [x] 1.3 Replace **`update_current_span`** with **`update_current_generation`** and set **`model`** + **`metadata`** (`src/llm/runtime.py`).
- [x] 1.4 Update unit test **`test_track_name_updates_langfuse_span_name`** to assert **`update_current_generation`** (`tests/utils/test_clients.py`).

## 2. Verification

- [x] 2.1 Logic-only verification in current local environment (no external Langfuse integration run):

  ```bash
  openspec validate --changes --no-interactive --json
  rg "@conditional_observe\(name=\"LLM Call\", as_type=\"generation\"\)" src/llm/api.py
  rg "update_current_generation\(" src/llm/runtime.py
  rg "update_current_generation\.assert_called_once_with" tests/utils/test_clients.py
  ```

  Evidence captured:
  - `openspec validate` reports `valid: true` for `honcho-langfuse-generation-traces`.
  - Source assertions confirm generation tracing decorator, runtime generation update call, and updated unit-test expectation.

- [ ] 2.2 Optional smoke: trigger one **`honcho_llm_call`** against a deployment with Langfuse configured; confirm in Langfuse UI that the observation shows type **generation** and **model** column populated. Record trace ID or screenshot reference.

## 3. Follow-ups (not blocking this change)

- [ ] 3.1 Consider renaming test **`test_track_name_updates_langfuse_span_name`** → **`test_track_name_updates_langfuse_generation`** for clarity (separate trivial PR acceptable).
