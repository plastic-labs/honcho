## Why

Honcho integrates Langfuse for LLM observability. Observations created by `@observe` defaulted to **span** type while **model** was only placed in **metadata** via `update_current_span`. Langfuse’s tracing UI attributes **model**, usage, and cost primarily to **generation** observations with a top-level **`model`** field.

Related work was implemented before an OpenSpec change existed. This proposal **retroactively formalizes** the requirement and ties verification to evidence so QA and future edits follow the workflow.

## What Changes

- Define capability **`observability-langfuse`**: LLM calls through `honcho_llm_call` MUST emit Langfuse **generation** observations with **`model`** set to the resolved model for each attempt (including fallback retries).
- **`conditional_observe`** MUST forward **`as_type`** to Langfuse `observe()` so call sites can request generation (or other observation types) when needed.
- **`update_current_langfuse_observation`** MUST use **`update_current_generation`** with **`model`** at top level and **`metadata`** for `namespace` and `provider` (not as a substitute for generation `model`).
- Align unit tests with the generation API.

**Implementation status**: The above behaviors are already present in the codebase as of this proposal; remaining work is **verification evidence** and optional documentation polish.

## Capabilities

### New Capabilities

- **`observability-langfuse`** — Langfuse **generation** semantics for `honcho_llm_call`, defined by the delta spec at `specs/observability-langfuse/spec.md`.

### Modified Capabilities

None.

## Impact

- `src/telemetry/logging.py` — `conditional_observe(as_type=...)`.
- `src/llm/api.py` — `@conditional_observe(..., as_type="generation")` on `honcho_llm_call`.
- `src/llm/runtime.py` — `update_current_generation`.
- `tests/utils/test_clients.py` — expectations for `update_current_generation`.

Out of scope for this change: enriching traces with **`usage_details`** / **`model_parameters`** (follow-up change).
