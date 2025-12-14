# CLAUDE.md - Arceus ARC-AGI Solver

This file provides guidance to Claude Code when working with the Arceus codebase.

## Quick Context

**What is Arceus?**
Memory-augmented ARC-AGI puzzle solver using three-layer cognitive architecture, Honcho memory infrastructure, and adaptive learning from experience.

**Critical Context Document**: `docs/ARCEUS_DEVELOPMENT_LOG.md`
⚠️ **READ THIS FIRST** before making changes - contains comprehensive architecture, patterns, and change history.

## Core Architecture

### Three-Layer + Guidance System

1. **Strategy Selection Layer** → Decides deep exploration vs fast solving
2. **Layer 1: Meta-Strategy** → Determines HOW to think (problem type, approach)
3. **Layer 2: Adaptive Memory** → Problem-type-specific queries
4. **Layer 3: Curiosity** → Post-failure alternative exploration
5. **Solution Guidance** → Extracts actionable "HOW TO SOLVE" from memory

### Memory Infrastructure (Honcho)

- **Peers**: `task_analyst`, `solution_generator`, `verifier`
- **Sessions**: Contexts with multiple peers
- **Messages**: Stored experiences processed into peer representations
- **Dialectic**: Natural language queries to memory

### Key Files

| File | Purpose | Lines | Key Methods |
|------|---------|-------|-------------|
| `cognitive_layers.py` | Three-layer architecture + guidance | 714 | `should_use_deep_exploration()`, `meta_strategy_planning()`, `adaptive_memory_reflection()`, `curiosity_driven_reflection()`, `get_solution_guidance_from_memory()` |
| `solver.py` | Main solving pipeline | ~1400 | `solve_task()`, `generate_solution_with_memory()`, `store_solution()` |
| `metrics.py` | Tracking costs and memory | ~250 | `calculate_api_cost()`, `update_peer_facts()` |
| `tui.py` | Rich terminal UI | ~600 | `_make_metrics_panel()`, `_make_memory_viz_panel()` |
| `config.py` | Model pricing and settings | ~200 | `MODEL_PRICING` dict |

## Critical Patterns

### Context Passing
```python
# solving_context flows through entire pipeline
solving_context = {
    'use_deep_exploration': bool,
    'meta_strategy': dict,
    'memory_reflection': dict,
    'strategy_decision': dict,
}
# Pass to: generate_solution_with_memory(..., solving_context)
# Pass to: store_solution(..., solving_context, solution_guidance)
```

### Cost Tracking (After Every LLM Call)
```python
metrics.add_llm_call(call_time_ms, tokens_used)
metrics.calculate_api_cost()  # Always recalculate
```

### Memory Tracking (After Storage)
```python
metrics.num_sessions_created += 1
metrics.num_messages_ingested += 1
metrics.add_peer_fact("peer_name")
```

### Cognitive Layer Usage
```python
from .cognitive_layers import CognitiveLayers
cognitive = CognitiveLayers(reflection_peer, honcho_client, tui)

# Get guidance
guidance = await cognitive.get_solution_guidance_from_memory(
    task_id, task_analysis, meta_strategy, tui_label="solving"
)

# Store in hypothesis
hypothesis["_solution_guidance"] = guidance

# Store with solution
await store_solution(..., solving_context, hypothesis.get("_solution_guidance"))
```

### JSON Parsing from LLM
```python
import re, json
json_match = re.search(r'\{.*\}', content, re.DOTALL)
if json_match:
    data = json.loads(json_match.group(0))
```

## Common Tasks

### Adding New Metrics
1. Add field to `SolverMetrics` in `metrics.py`
2. Track in `solver.py` at relevant points
3. Display in `tui.py::_make_metrics_panel()`
4. Update `to_dict()` for export
5. **Log changes in ARCEUS_DEVELOPMENT_LOG.md**

### Modifying Cognitive Layers
1. Add/modify method in `cognitive_layers.py`
2. Return structured dict (document schema)
3. Add TUI logging (`tui.add_agent_log()`, `tui.add_memory_operation()`)
4. Integrate in `solver.py` at appropriate stage
5. Pass results via `solving_context`
6. **Log changes in ARCEUS_DEVELOPMENT_LOG.md**

### Changing Memory Storage
1. Modify `store_solution()` in `solver.py`
2. Add fields to solution_record dict
3. Update metadata for filtering
4. **Log schema changes in ARCEUS_DEVELOPMENT_LOG.md**

## Code Quality

**Type Hints**: Required for all methods
```python
async def method(self, param: str, optional: Optional[Dict] = None) -> Optional[Dict]:
```

**Error Handling**: Always use try-except for memory/LLM operations
```python
try:
    result = await operation()
except Exception as e:
    logging.debug(f"Operation failed: {e}")
    return None
```

**Docstrings**: Google style, document return dict schemas
```python
def method(self) -> Dict:
    """
    Brief description.

    Returns: {
        'key1': Description,
        'key2': Description,
    }
    """
```

## Testing

```bash
# Single task
uv run python -m arceus.main --task-id 007bbfb7

# All tasks
uv run python -m arceus.main --eval-all

# Test imports
uv run python -c "from arceus import solver, cognitive_layers; print('OK')"
```

## Before Making Changes

1. ✅ Read `ARCEUS_DEVELOPMENT_LOG.md` for full context
2. ✅ Understand current architecture and patterns
3. ✅ Check if similar work was done before
4. ✅ Follow established patterns (context passing, cost tracking, etc.)
5. ✅ Document changes in development log using session template

## After Making Changes

1. ✅ Test with single task first
2. ✅ Verify TUI displays correctly
3. ✅ Check metrics are tracked properly
4. ✅ Ensure no regression in existing functionality
5. ✅ **Add session entry to ARCEUS_DEVELOPMENT_LOG.md**
6. ✅ Update this CLAUDE.md if patterns change

## Current Status (2025-12-14)

✅ Implemented:
- Three-layer cognitive architecture in solving
- Strategy Selection (decides deep exploration)
- Solution Guidance (actionable memory-to-code)
- API cost tracking ($0.0000 format)
- Memory metrics (sessions, messages, facts per peer)
- Enhanced TUI (side-by-side, two-column metrics)
- Self-improvement loop (tracks guidance effectiveness)

🚧 Placeholder:
- `test_time_training.py` (stub returns None)

🎯 Next Priorities:
- Implement test-time training
- Add guidance quality scoring
- Adaptive iteration limits
- Cross-task learning

## Resources

- **Development Log**: `ARCEUS_DEVELOPMENT_LOG.md`
- **Architecture**: `ARCHITECTURE.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Honcho Docs**: https://docs.honcho.dev
- **ARC-AGI**: https://arcprize.org
- **Rich TUI**: https://rich.readthedocs.io

---

**Remember**: Arceus learns from its own experience. Every change should consider how it affects the learning loop:
```
Memory → Guidance → Solution → Store Results → Better Memory → Better Guidance
```
