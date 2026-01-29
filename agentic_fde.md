# Agentic FDE: Self-Adapting Honcho

## Vision

The software market is bifurcating: massive enterprise vs solopreneur vibe-coders. Enterprise requires human FDEs; vibe-coders won't pay for human help but also won't pay for one-size-fits-all SaaS. The solution: make Honcho itself an "Agentic FDE" that adapts to each developer's use case.

Honcho observes usage patterns, engages in meta-cognition about developer goals, and adapts its behavior accordingly. A companion app needs emotional memory extraction and biographical recall. A coding agent needs preference/constraint extraction and should ignore stack traces. An email ingestion pipeline needs RAG, not conversation memory.

The same primitives (workspaces, peers, sessions, messages, documents) can achieve any memory pattern - but the prompts and retrieval strategies must adapt.

## Core Principles

1. **Stable API, stable schema** - Honcho adapts its behavior, not its interface
2. **Developer feedback is highest priority** - Observed patterns can be overridden
3. **Two adaptation questions**:
   - How should I handle the next marginal message? (deriver)
   - How should I handle the next marginal .chat query? (dialectic)
4. **Constraints**: No touching deletion endpoints, workspace isolation, or the core reasoning model

## The 5-Phase Plan

### Phase 1: Instrumentation ✅

**Goal**: Log dialectic interactions so the dreamer can analyze performance.

**Built**:

- `DialecticTrace` model: workspace, session, observer, observed, query, retrieved_doc_ids, tool_calls, response, reasoning_level, duration, tokens, timestamps
- CRUD operations: `create_dialectic_trace()`, `get_dialectic_traces()`, `get_dialectic_trace_stats()`
- Abstention detection via regex patterns
- Integration: traces written at end of `DialecticAgent._log_response_metrics()`

**Files**: `src/models.py`, `src/crud/dialectic_trace.py`, `src/dialectic/core.py`, `tests/test_dialectic_trace.py`

### Phase 2: Prompt Injection Points ✅

**Goal**: Enable workspace-level prompt customization without changing default behavior.

**Built**:

- `WorkspaceAgentConfig` schema with `deriver_rules` and `dialectic_rules` fields
- Storage in `workspace.metadata["_agent_config"]`
- CRUD helpers: `get_workspace_agent_config()`, `set_workspace_agent_config()`
- Deriver prompt injection: `custom_rules` parameter in `minimal_deriver_prompt()`
- Dialectic prompt injection: `custom_rules` parameter in `agent_system_prompt()`
- Config threading through deriver and dialectic paths

**Files**: `src/schemas.py`, `src/crud/workspace.py`, `src/deriver/prompts.py`, `src/deriver/deriver.py`, `src/dialectic/prompts.py`, `src/dialectic/core.py`, `src/dialectic/chat.py`, `tests/test_workspace_agent_config.py`

### Phase 3: Meta-Cognitive Dreamer ✅

**Goal**: Dreamer analyzes logs and generates configuration suggestions.

**Built**:

- `DreamType.INTROSPECTION` enum value
- `IntrospectionSignals`, `IntrospectionSuggestion`, `IntrospectionReport` schemas
- `gather_introspection_context()` - collects dialectic stats, observation counts, peer/session patterns
- `build_introspection_prompt()` - formats signals for LLM analysis
- `run_introspection()` - calls LLM, parses structured suggestions
- `store_introspection_report()` - saves reports as documents in `_system`/`_introspection` collection
- `get_latest_introspection_report()` - retrieves most recent report
- Wired into `DreamType.INTROSPECTION` in orchestrator

**Files**: `src/schemas.py`, `src/dreamer/introspection.py`, `src/dreamer/orchestrator.py`, `tests/test_introspection.py`

### Phase 4: Developer Feedback Channel ✅

**Goal**: Developers can talk to Honcho about Honcho.

**Built**:

- `POST /workspaces/{id}/feedback` endpoint
- `FeedbackRequest`, `ConfigChange`, `FeedbackResponse` schemas
- `process_feedback()` - handles natural language feedback
- `build_feedback_prompt()` - formats context for LLM
- Interview mode: empty config + greeting triggers onboarding questions
- Incremental updates: preserves existing rules when adding new ones
- Introspection context: optionally includes latest report
- Uses `settings.DREAM` for LLM calls (not billed as dialectic)

**Files**: `src/schemas.py`, `src/feedback.py`, `src/routers/workspaces.py`, `src/dreamer/introspection.py`, `tests/test_feedback.py`

### Phase 5: Closed Loop (Future)

**Goal**: Automatic adaptation with developer oversight.

**To build**:

- Dreamer introspection generates draft config changes
- Surfaces to developer via webhook or dashboard
- Developer approves/rejects/modifies
- Approved changes written to config
- Optional: `workspace.meta.auto_adapt = true` for brave workspaces

## Summary Statistics

| Phase | Lines Added | Test Coverage |
|-------|-------------|---------------|
| 1     | ~500        | 376 lines (13 tests) |
| 2     | ~200        | 246 lines (15 tests) |
| 3     | ~500        | 423 lines (10 tests) |
| 4     | ~350        | 514 lines (23 tests) |
| **Total** | **~2,950** | **1,559 lines (61 tests)** |

## Testing Checklist

- [ ] Run full test suite: `uv run pytest tests/`
- [ ] Test Phase 1: Create a dialectic query, verify trace is logged
- [ ] Test Phase 2: Set workspace agent config, verify rules appear in prompts
- [ ] Test Phase 3: Trigger introspection dream, verify report generated
- [ ] Test Phase 4: Submit feedback, verify config updated
- [ ] Test interview flow: New workspace + greeting triggers questions
- [ ] Test incremental updates: Existing rules preserved when adding new ones
