## Context

Honcho uses the Langfuse Python SDK (`langfuse>=3.3.2`). The public LLM entrypoint is **`honcho_llm_call`** in `src/llm/api.py`, wrapped with **`conditional_observe`** so tracing is enabled only when **`LANGFUSE_PUBLIC_KEY`** is set.

Langfuse distinguishes observation types; **`generation`** observations carry **`model`** and related metrics in forms the UI expects. Generic **spans** with `metadata.model` do not populate the same UX.

Exploration traceback: `explorations/2026-05-05-langfuse-generation-observations-openspec-gap.md`.

## Goals / Non-Goals

**Goals:**

- Ensure **`honcho_llm_call`** creates a **generation** observation when Langfuse is configured.
- Set **`model`** on that generation to the effective model ID from **`plan_attempt`** (primary or fallback per retry attempt).
- Preserve **`provider`** and **`namespace`** in **metadata** for filtering and debugging.

**Non-Goals:**

- Emit token usage / cost into Langfuse via **`usage_details`** (deferred).
- Change non-LLM **`@conditional_observe`** call sites unless they explicitly need generation semantics later.

## Decisions

**1. `as_type="generation"` on `honcho_llm_call`**

The decorator wraps the entire `honcho_llm_call`; Langfuse treats this span as a generation-capable observation when **`as_type="generation"`** is passed through **`observe()`**.

**Rationale:** Matches Langfuse Python instrumentation guidance for LLM-shaped functions.

**2. `update_current_generation` instead of `update_current_span`**

After planning each attempt, **`update_current_langfuse_observation`** updates the active observation with **`model`** and **`metadata`** (`namespace`, `provider`).

**Rationale:** **`model`** belongs on the generation observation per SDK semantics; metadata carries ancillary dimensions.

**3. Extend `conditional_observe` with `as_type`**

Other modules can reuse the same decorator with explicit types without importing Langfuse **`observe`** directly everywhere.

## Risks / Trade-offs

- **Risk:** Full test suite may not run on Windows hosts that lack **`fcntl`** in `tests/conftest.py`.
  - **Mitigation:** Run targeted pytest on Linux/macOS/WSL or CI; record command + output in **`tasks.md`** verification.
- **Risk:** Oversized trace I/O from decorator input/output capture.
  - **Mitigation:** Existing Langfuse env toggles (`LANGFUSE_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED`, per-decorator `capture_input`/`capture_output`) remain available for tuning; not changed in this change.

## References

- [Langfuse Python instrumentation](https://langfuse.com/docs/observability/sdk/python/instrumentation)
