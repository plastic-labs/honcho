# Arceus Quick Reference Card

**Last Updated**: 2025-12-14

## 🚀 Quick Start

```bash
# Read comprehensive context FIRST
cat docs/ARCEUS_DEVELOPMENT_LOG.md

# Test single task
uv run python -m arceus.main --task-id 007bbfb7

# Run full evaluation
uv run python -m arceus.main --eval-all
```

## 📁 File Map

```
arceus/
├── solver.py              # Main orchestration (~1400 lines)
├── cognitive_layers.py    # Three-layer architecture (714 lines)
├── metrics.py             # Cost & memory tracking
├── tui.py                 # Terminal UI
├── config.py              # MODEL_PRICING dict
├── test_time_training.py  # Stub (returns None)
└── docs/
    ├── ARCEUS_DEVELOPMENT_LOG.md  # Complete history
    ├── ARCHITECTURE.md             # Visual diagrams
    ├── CLAUDE.md                   # Dev guidelines
    ├── QUICK_REFERENCE.md          # This file
    └── SESSION_TEMPLATE.md         # Change template
```

## 🧠 Cognitive Flow

```
Task → Strategy Selection → [Deep Exploration?]
  YES → Meta-Strategy → Adaptive Memory → Solution Guidance
  NO  → Fast Solving
→ Generate Solution → Verify → [Fail?]
  YES → Curiosity Reflection → Retry
  NO  → Success
→ Store (solution + guidance + context)
```

## 🔑 Key Patterns

### 1. Context Passing
```python
solving_context = {
    'use_deep_exploration': bool,
    'meta_strategy': dict,
    'memory_reflection': dict,
}
# Pass everywhere: generate_solution_with_memory(..., solving_context)
```

### 2. Cost Tracking
```python
metrics.add_llm_call(call_time_ms, tokens)
metrics.calculate_api_cost()  # Always!
```

### 3. Memory Tracking
```python
metrics.num_sessions_created += 1
metrics.num_messages_ingested += 1
metrics.add_peer_fact("peer_name")
```

### 4. Cognitive Layers
```python
from .cognitive_layers import CognitiveLayers
cognitive = CognitiveLayers(peer, client, tui)
guidance = await cognitive.get_solution_guidance_from_memory(...)
hypothesis["_solution_guidance"] = guidance  # Attach!
```

### 5. Store with Guidance
```python
await store_solution(
    task_id, solution, success, logger, tui, metrics,
    solving_context,  # NEW
    hypothesis.get("_solution_guidance")  # NEW
)
```

## 📊 Metrics Schema

```python
SolverMetrics:
    # Core
    task_id: str
    num_iterations: int

    # Cost
    model_name: str
    api_cost: float
    total_tokens: int

    # Memory
    num_sessions_created: int
    num_messages_ingested: int
    num_facts_stored: int
    facts_per_peer: Dict[str, int]
```

## 🎯 Common Operations

| Task | File | Method/Line |
|------|------|-------------|
| Add metric | `metrics.py` | Add field to `SolverMetrics` |
| Track cost | `solver.py` | `metrics.calculate_api_cost()` |
| Cognitive layer | `cognitive_layers.py` | Add method to `CognitiveLayers` |
| Display metric | `tui.py:458-509` | Modify `_make_metrics_panel()` |
| Store solution | `solver.py:817-897` | `store_solution()` |
| Solution guidance | `cognitive_layers.py:532-713` | `get_solution_guidance_from_memory()` |

## 🐛 Debug Checklist

- [ ] `solving_context` passed through pipeline?
- [ ] `metrics.calculate_api_cost()` after LLM calls?
- [ ] Memory counters incremented?
- [ ] `hypothesis["_solution_guidance"]` attached?
- [ ] `store_solution()` gets 8 parameters?
- [ ] Honcho client initialized?
- [ ] Async/await correct?
- [ ] JSON parsed with `re.search(r'\{.*\}', content, re.DOTALL)`?

## 📝 Update Protocol

1. **Before**: Read `ARCEUS_DEVELOPMENT_LOG.md`
2. **During**: Follow patterns, track metrics, pass context
3. **After**: Add session to log using template

## 🔗 Links

- **Full Context**: `ARCEUS_DEVELOPMENT_LOG.md`
- **Architecture**: `ARCHITECTURE.md`
- **Dev Guide**: `CLAUDE.md`
- **Session Template**: `SESSION_TEMPLATE.md`
- **Honcho Docs**: https://docs.honcho.dev

## 💡 Remember

Every change affects the learning loop:
```
Memory → Guidance → Solution → Store → Better Memory
```

**Always ask**: "How does this help the system learn?"
