# Exploration: Langfuse generation observations & OpenSpec compliance gap

> **Relocation**: Moved from `openspec/workspace/explorations/` to `openspec/changes/honcho-langfuse-generation-traces/explorations/` on 2026-05-05 per `/opsx-propose` operational binding.

**Date**: 2026-05-05  
**Project**: honcho (`O:\workspaces\oss\honcho`)  
**Mode**: `/opsx-explore` — investigation and requirement clarification only (no further application code in this phase).

---

## 1. Trigger

Work was implemented **outside** the OpenSpec lifecycle (Explore → Propose → Apply → Verify → Archive):

- Honcho’s Langfuse integration was adjusted so LLM calls surface **`model`** correctly in the Langfuse UI.
- Collaboration protocol (`openspec_workflow` SKILL) expects **PRE-proposal exploration** under `openspec/workspace/explorations/` and substantive changes to flow through **`/opsx-propose`** before apply.

This document **retroactively captures** the problem statement, grounded findings, and a Hard Exit package for formal proposal.

---

## 2. Ground truth (codebase — verified)

| Area | Finding |
|:-----|:--------|
| LLM entrypoint | `honcho_llm_call` uses `@conditional_observe(name="LLM Call", as_type="generation")` — observation type is **generation**, not default span. |
| Span updates | `update_current_langfuse_observation` calls `get_client().update_current_generation(...)` with top-level **`model`** plus metadata (`namespace`, `provider`). |
| Decorator plumbing | `conditional_observe(..., as_type=...)` forwards `as_type` into Langfuse `observe(**observe_kw)`. |
| Tests | `tests/utils/test_clients.py` expects `update_current_generation` (not `update_current_span`) when `track_name` is set. |

**Symptom addressed**: Langfuse UI commonly treats **`generation`** observations with a **`model`** field as first-class LLM rows; **`span`** + `metadata.model` did not satisfy that UX.

---

## 3. External constraint (Langfuse SDK — conceptual)

Langfuse Python instrumentation documents **`@observe(..., as_type="generation")`** for LLM-shaped work and distinguishes **`update_current_generation`** vs **`update_current_span`** for active-context updates.

Reference (orientation): [Langfuse Python instrumentation](https://langfuse.com/docs/observability/sdk/python/instrumentation).

---

## 4. Requirement statement (what OpenSpec should capture)

1. **Observability**: When `LANGFUSE_PUBLIC_KEY` is set, each **`honcho_llm_call`** MUST emit a Langfuse observation typed as **generation** with **model** set to the resolved model ID for that attempt (primary or fallback retry).
2. **Metadata**: Preserve **provider** and **namespace** (and any future dimensions) as **metadata**, not as a substitute for **`model`** on non-generation types.
3. **Decorators**: `conditional_observe` MUST support forwarding observation type to Langfuse for any call site that requires generation/embedding semantics.
4. **Process**: Future edits to Langfuse wiring MUST go through OpenSpec **unless** classified trivial per workflow scale table.

---

## 5. Process gap analysis

| Gap | Severity | Note |
|:----|:---------|:-----|
| No prior `proposal.md` / delta specs | Process | Violates **OpenSpec before code** for non-trivial behavioral change to tracing. |
| No `tasks.md` checklist | Process | Verification steps (pytest on POSIX, Langfuse UI smoke) not formally tracked. |
| Exploration not filed first | Process | This file corrects placement under `openspec/workspace/explorations/`. |

**Important**: Code may already be merged locally or deployed — OpenSpec artifacts here serve **audit trail + future constraint**, not necessarily “redo the implementation.”

---

## 6. Options for formalization (`/opsx-propose`)

| Option | Description | Pros | Cons |
|:-------|:-------------|:-----|:-----|
| **A — Retro capability spec** | Single capability (e.g. `observability-langfuse`) documenting WHEN/THEN for generation + model | Clear requirement source for QA | Retro active change naming/scoping needed |
| **B — Amend existing spec** | If `openspec/specs/` later gains monitoring/telemetry capability | Fits incremental specs | Requires discovering/creating parent capability |
| **C — Document-only lesson** | Project `docs/lessons` + skip change | Fast | Weak traceability vs spec-driven verify |

**Recommendation**: **Option A** — small dedicated delta: *“LLM paths integrated with Langfuse MUST use generation observations and set `model`.”* Tie **Verify** phase to: unit test mock expectations + optional manual Langfuse trace check.

---

## 7. Follow-up ideas (not decisions)

- **`usage_details` / `model_parameters`**: Improve cost/token columns in Langfuse UI; separate small change or same proposal scope — Commander chooses.
- **Windows pytest**: Full suite blocked on `fcntl` in `tests/conftest.py`; verification mandate may require POSIX CI or shim — flag in tasks.md if relevant.

---

## 8. Architecture sketch (trace context)

```
honcho_llm_call (@observe as_type=generation)
        │
        ▼
plan_attempt → update_current_generation(model, metadata{provider, namespace})
        │
        ▼
honcho_llm_call_inner → provider clients → Langfuse buffers flush async
```

---

## 9. Hard Exit Gate — package for `/opsx-propose`

Use the following as seeds for **`proposal.md`** / **`tasks.md`**:

**Intent (proposal seed)**

- Formalize Honcho’s Langfuse integration rules: generation-type observations for `honcho_llm_call`, **`model`** on generation, **`metadata`** for routing context.
- Record that implementation **already exists** in tree as of exploration date; remaining work is **spec alignment + verification evidence**.

**Scope**

- In: `conditional_observe`, `honcho_llm_call`, `update_current_langfuse_observation`, related tests.
- Out (unless Commander expands): token/cost enrichment (`usage_details`), non-LLM `@conditional_observe` call sites.

**Verification (tasks seed)**

- [ ] Run targeted pytest on POSIX/WSL: `tests/utils/test_clients.py::TestAnthropicCalls::test_track_name_updates_langfuse_span_name` (rename reflects generation API).
- [ ] Manual or scripted smoke: one trace on self-hosted Langfuse shows **generation** + **model** column populated.

**Exit states**

- **Proceed**: Commander runs `/opsx-propose` (or `/opsx-ff` if trivial path accepted) with change name TBD (e.g. `honcho-langfuse-generation-traces`).
- **No action**: Only if Commander decides observability semantics need no written spec — explicitly reject with rationale (not recommended given collaboration rules).

---

## 10. Status

**Exploration**: Closed with **decision to formalize via OpenSpec proposal** (recommended Option A).  
**Next command**: **`/opsx-propose`** (or workspace equivalent) when Commander approplies change name and scope boundaries.
