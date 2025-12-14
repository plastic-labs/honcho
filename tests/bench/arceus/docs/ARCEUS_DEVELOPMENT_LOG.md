# Arceus Development Log

**Last Updated**: 2025-12-14
**Purpose**: Comprehensive context and change tracking for Arceus ARC-AGI solver development

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Development Sessions](#development-sessions)
4. [File Reference](#file-reference)
5. [Key Patterns & Conventions](#key-patterns--conventions)
6. [Future Development Guide](#future-development-guide)

---

## System Overview

### What is Arceus?
Arceus is an advanced ARC-AGI (Abstraction and Reasoning Corpus) puzzle solver that combines:
- **Memory-augmented reasoning** via Honcho SDK
- **Three-layer cognitive architecture** for adaptive problem-solving
- **Self-play learning** from past experiences
- **Dialectic memory queries** for contextual recall
- **Test-time training** for adaptation

### Core Capabilities
- Solves abstract reasoning puzzles through iterative hypothesis generation
- Learns from both successes and failures
- Uses memory to guide solution strategies
- Applies meta-cognitive layers to decide HOW to think
- Tracks comprehensive metrics including API costs and memory usage

### Technology Stack
- **Python 3.11+** with async/await
- **Honcho SDK** for memory infrastructure (peers, sessions, messages)
- **Anthropic Claude API** (Sonnet 4.5, Opus 4.5) for reasoning
- **Rich TUI** for terminal visualization
- **NumPy** for grid operations

---

## Architecture

### Three-Layer Cognitive Architecture

**Strategy Selection Layer** (Meta-Meta-Cognitive)
- Consults memory to decide: deep exploration vs fast solving
- Location: `cognitive_layers.py::should_use_deep_exploration()`

**Layer 1: Meta-Strategy Planning**
- Decides HOW to think about the problem
- Determines problem type (pattern, spatial, logical, compositional)
- Selects approach type (analytical, intuitive, experimental)
- Location: `cognitive_layers.py::meta_strategy_planning()`

**Layer 2: Adaptive Memory Reflection**
- Queries memory ADAPTIVELY based on problem type
- Different queries for different problem types
- Location: `cognitive_layers.py::adaptive_memory_reflection()`

**Layer 3: Curiosity-Driven Reflection**
- After failures, asks WHY and explores alternatives
- Generates paradigm shifts and alternative interpretations
- Location: `cognitive_layers.py::curiosity_driven_reflection()`

**Solution Guidance Layer** (NEW)
- Extracts actionable "HOW TO SOLVE" instructions from memory
- Translates natural language strategies into code approaches
- Location: `cognitive_layers.py::get_solution_guidance_from_memory()`

### Honcho Memory Structure

**Workspaces**
- Root organizational unit containing all peers and sessions

**Peers** (AI agents with identity)
- `task_analyst`: Analyzes puzzle patterns and meta-strategy
- `solution_generator`: Generates solution code and strategies
- `verifier`: Verifies solutions against training examples

**Sessions**
- Contexts for conversations and learning
- Can involve multiple peers
- Store messages with metadata for processing

**Messages**
- Data units representing learning experiences
- Processed in background to update peer representations
- Stored with rich metadata for retrieval

**Dialectic API**
- Natural language queries to peer memory
- Returns context-aware responses informed by past experiences

### Solver Pipeline

```
1. Task Analysis (analyze_task)
   ↓
2. Strategy Selection (should_use_deep_exploration)
   ↓
3. [IF DEEP EXPLORATION]
   - Meta-Strategy Planning (Layer 1)
   - Adaptive Memory Reflection (Layer 2)
   ↓
4. Solution Guidance from Memory
   ↓
5. Iteration Loop (max_iterations)
   - Generate hypothesis with memory context
   - Verify against training examples
   - [IF FAILURE] Curiosity Reflection (Layer 3)
   ↓
6. Store Solution + Guidance + Context
```

---

## Development Sessions

### Session 1: 2025-12-14 - Foundation Enhancements

**Objective**: Apply exploratory cognitive architecture during solving, add cost tracking, enhance memory metrics

#### Phase 1: API Cost Tracking

**Files Modified**: `config.py`, `metrics.py`, `solver.py`, `tui.py`

**Changes**:
1. **config.py** - Added MODEL_PRICING dictionary
   ```python
   MODEL_PRICING = {
       "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
       "claude-opus-4-5-20251101": {"input": 0.015, "output": 0.075},
       # ... 7 models total
   }
   ```

2. **metrics.py** - Added cost calculation
   - New fields: `model_name: str`, `api_cost: float`
   - New method: `calculate_api_cost()` - calculates USD cost from tokens
   - New method: `get_cost_per_token()` - returns cost per token
   - Updated `to_dict()` to include `api_cost_usd` and `cost_per_token`

3. **solver.py** - Integrated cost tracking
   - Line 875: Set `metrics.model_name = self.config.llm_model`
   - Line 605: Call `metrics.calculate_api_cost()` after each LLM call
   - Tracks model usage throughout solving process

4. **tui.py** - Display costs in metrics panel
   - Line 478: Added "API Cost" row showing `${self.metrics.api_cost:.4f}`
   - Line 479: Added "Cost/Token" showing per-token cost

**Result**: Real-time API cost tracking visible in TUI footer

#### Phase 2: Memory Statistics Tracking

**Files Modified**: `metrics.py`, `solver.py`, `tui.py`

**Changes**:
1. **metrics.py** - Added memory tracking fields
   ```python
   num_sessions_created: int = 0
   num_messages_ingested: int = 0
   num_facts_stored: int = 0
   facts_per_peer: Dict[str, int] = field(default_factory=dict)
   ```
   - New method: `update_peer_facts()` - queries peer context for fact counts
   - New method: `add_peer_fact()` - increments peer-specific fact counter

2. **solver.py** - Track memory operations
   - Line 885: Increment `metrics.num_sessions_created` when creating sessions
   - Lines 859, 890, 1205, 1320, 1350: Increment `metrics.num_messages_ingested`
   - Lines 860, 892, 1206, 1321: Call `metrics.add_peer_fact()` when storing

3. **tui.py** - Enhanced memory visualization
   - Lines 429-448: Added stats header to memory panel
   - Shows per-peer fact breakdown: `task_analyst: X, solution_generator: Y`
   - Displays sessions created and messages ingested

**Result**: Comprehensive memory usage visibility

#### Phase 3: Three-Layer Cognitive Architecture Integration

**Files Created**: `cognitive_layers.py`, `test_time_training.py`

**Files Modified**: `solver.py`

**Changes**:
1. **cognitive_layers.py** (NEW - 714 lines)
   - Class: `CognitiveLayers` - implements all cognitive layers
   - Method: `should_use_deep_exploration()` - Strategy Selection Layer
   - Method: `meta_strategy_planning()` - Layer 1
   - Method: `adaptive_memory_reflection()` - Layer 2
   - Method: `curiosity_driven_reflection()` - Layer 3
   - Method: `get_solution_guidance_from_memory()` - Solution Guidance

   **Key Design**: Reusable methods for both self-play and solving

2. **solver.py** - Integrated cognitive layers
   - Line 31: Import `CognitiveLayers`
   - Lines 915-990: Added Strategy Selection and Layer 1-2 before iteration loop
   - Creates `solving_context` dict with meta_strategy and memory_reflection
   - Pass `solving_context` throughout solving pipeline
   - Lines 1003-1042: Layer 3 (Curiosity) after verification failures

3. **test_time_training.py** (NEW - Stub)
   - Created to fix `ModuleNotFoundError`
   - Returns `None` - no training applied yet
   - Allows system to run with all enhancements enabled

**Result**: System consults own memory to decide strategies and learns adaptively

#### Phase 4: TUI Enhancements

**Files Modified**: `tui.py`

**Changes**:
1. **Metrics Panel** - Two-column layout (lines 458-509)
   - Left column: Time, Iterations, API Cost, Memory Queries, Hypotheses, Verifications
   - Right column: LLM Calls, Tokens, Cost/Token, Messages Stored, Sessions, Facts
   - Shows 12 metrics total vs original 8

2. **Memory Panel** - Per-peer statistics (lines 429-448)
   - Header shows session and message counts
   - Per-peer fact breakdown displayed prominently

3. **Puzzle Panel** - Side-by-side layout (lines 270-332)
   - Changed from vertical stacking to horizontal layout
   - Three columns: Test Input | Arrow | Agent's Output
   - Fixed TypeError by removing invalid `justify="center"` from Text.append()

**Result**: More informative, space-efficient TUI

### Session 1 Continued: Memory-Guided Solution Generation

**Objective**: Extract actionable "HOW TO SOLVE" guidance from memory and translate to code

#### Phase 5: Solution Guidance Implementation

**Files Modified**: `cognitive_layers.py`, `solver.py`

**Changes**:
1. **cognitive_layers.py** - Added solution guidance method (lines 532-713)
   - Method: `get_solution_guidance_from_memory()` - 182 lines
   - Queries memory with detailed prompt asking:
     - What specific approach worked on similar tasks?
     - What code strategy should be used?
     - Which primitives were effective?
     - Step-by-step solving instructions
     - Pitfalls to avoid
     - Example code patterns
   - Returns structured dict with:
     ```python
     {
         'has_experience': bool,
         'confidence': float,
         'solving_strategy': str,
         'code_approach': str,
         'primitives_to_use': [str],
         'step_by_step': [str],
         'pitfalls_to_avoid': [str],
         'example_code_pattern': str,
         'reasoning': str
     }
     ```
   - Displays guidance in TUI with confidence indicators (🎯/💡/🤔)

2. **solver.py** - Integrated solution guidance
   - **generate_solution_with_memory** (lines 556-574):
     - Query for solution guidance when solving_context available
     - Get meta_strategy from solving_context
     - Call `cognitive.get_solution_guidance_from_memory()`
     - Fall back to generic dialectic if no guidance

   - **_build_solution_prompt** (lines 643-713):
     - Added `solution_guidance` parameter
     - Build formatted guidance section with visual separators
     - Display strategy, code approach, step-by-step instructions
     - List recommended primitives and pitfalls
     - Include example code patterns from memory
     - Instruct LLM to translate guidance into Python code

   - **Hypothesis Enhancement** (lines 632-634):
     - Attach guidance to hypothesis: `hypothesis["_solution_guidance"] = solution_guidance`
     - Enables storage of guidance used for learning

   - **store_solution** (lines 817-897):
     - Added `solving_context` and `solution_guidance` parameters
     - Build comprehensive solution record including:
       - Guidance used (strategy, confidence, primitives)
       - Whether guidance was helpful (correlated with success)
       - Meta-strategy information
     - Store in messages with metadata `had_guidance: bool`

   - **Integration Points**:
     - Line 1069: Pass `solving_context` to generate_solution_with_memory
     - Lines 1199-1208: Pass guidance to store_solution on success (code path)
     - Lines 1314-1323: Pass guidance to store_solution on success (primitive path)
     - Lines 1344-1353: Pass None guidance on failure

**Result**: System extracts actionable strategies from memory and learns which guidance patterns work

**Self-Improvement Loop**:
```
Solve Task → Use Guidance → Store "Was Helpful?" → Next Task Uses Better Guidance
```

---

## File Reference

### Core Files

#### `arceus/config.py`
**Purpose**: Configuration and model pricing
**Key Additions**:
- `MODEL_PRICING: Dict[str, Dict[str, float]]` - token pricing for 7 LLM models
- Maps model IDs to input/output pricing per 1K tokens

**Lines of Interest**:
- MODEL_PRICING dictionary: ~Line 100+

---

#### `arceus/metrics.py`
**Purpose**: Comprehensive solver metrics tracking
**Key Additions**:
- API cost tracking: `model_name`, `api_cost`, `calculate_api_cost()`, `get_cost_per_token()`
- Memory statistics: `num_sessions_created`, `num_messages_ingested`, `num_facts_stored`, `facts_per_peer`
- Peer-specific tracking: `update_peer_facts()`, `add_peer_fact()`

**Key Classes**:
- `SolverMetrics` (dataclass) - main metrics container

**Lines of Interest**:
- Cost calculation: Lines ~50-65
- Memory tracking: Lines ~68-95

---

#### `arceus/cognitive_layers.py` ⭐ NEW
**Purpose**: Reusable three-layer cognitive architecture + solution guidance
**Total Lines**: 714

**Key Classes**:
- `CognitiveLayers` - implements all cognitive layers

**Key Methods**:
1. `should_use_deep_exploration()` (lines 34-148)
   - Strategy Selection Layer
   - Decides deep exploration vs fast solving
   - Returns: decision dict with `use_deep_exploration: bool`

2. `meta_strategy_planning()` (lines 150-252)
   - Layer 1: Meta-cognitive planning
   - Determines problem type and thinking strategy
   - Returns: meta_strategy dict with approach_type, mental_model, etc.

3. `adaptive_memory_reflection()` (lines 254-418)
   - Layer 2: Problem-type-specific memory queries
   - Different queries for pattern/spatial/logical/compositional problems
   - Returns: reflection dict with successful_strategies, insights

4. `curiosity_driven_reflection()` (lines 420-530)
   - Layer 3: Post-failure exploration
   - Generates alternative interpretations and paradigm shifts
   - Returns: curiosity dict with blind_spots, experiments

5. `get_solution_guidance_from_memory()` (lines 532-713) ⭐ NEW
   - Extracts actionable "HOW TO SOLVE" guidance
   - Translates natural language strategies to code approaches
   - Returns: guidance dict with solving_strategy, code_approach, primitives, steps

**Integration Pattern**:
```python
cognitive = CognitiveLayers(reflection_peer, honcho_client, tui)
decision = await cognitive.should_use_deep_exploration(task_id, analysis)
if decision['use_deep_exploration']:
    meta_strategy = await cognitive.meta_strategy_planning(...)
    memory_reflection = await cognitive.adaptive_memory_reflection(...)
guidance = await cognitive.get_solution_guidance_from_memory(...)
```

---

#### `arceus/solver.py`
**Purpose**: Main solver orchestration and pipeline
**Total Lines**: ~1400+

**Key Methods Modified**:

1. `solve_task()` (lines 867-1355)
   - **Integration points**:
     - Line 875: Set `metrics.model_name`
     - Line 885: Track `num_sessions_created`
     - Lines 915-990: Strategy Selection + Layer 1-2
     - Line 1003-1042: Layer 3 on failure
     - Line 1069: Pass `solving_context` to generate_solution_with_memory
     - Lines 1199, 1314, 1344: Pass guidance to store_solution

2. `generate_solution_with_memory()` (lines 536-645)
   - **Parameters**: Added `solving_context=None`
   - **Flow**:
     - Lines 556-573: Query for solution guidance
     - Lines 575-590: Fallback to generic dialectic
     - Lines 592-602: Get peer context
     - Line 605: Build prompt with guidance
     - Line 632-634: Attach guidance to hypothesis

3. `_build_solution_prompt()` (lines 643-747)
   - **Parameters**: Added `solution_guidance: Optional[Dict] = None`
   - **Flow**:
     - Lines 663-704: Format guidance section with visual separators
     - Lines 670-701: Display strategy, code approach, steps, primitives, pitfalls
     - Line 704: Instruct to translate guidance to code
     - Lines 706-708: Fallback to generic memory

4. `store_solution()` (lines 817-897)
   - **Parameters**: Added `solving_context=None`, `solution_guidance=None`
   - **Flow**:
     - Lines 853-878: Build comprehensive solution record
     - Lines 860-869: Add guidance information
     - Lines 871-877: Add meta-strategy information
     - Lines 883-887: Store with `had_guidance` metadata

**Key Patterns**:
- Cost calculation after LLM calls: `metrics.calculate_api_cost()`
- Memory tracking: Increment counters after session/message operations
- Context passing: `solving_context` flows through entire pipeline

---

#### `arceus/tui.py`
**Purpose**: Rich terminal UI for visualization
**Total Lines**: ~600+

**Key Methods Modified**:

1. `_make_puzzle_panel()` (lines 270-332)
   - Changed to side-by-side layout using `Table.grid()`
   - Three columns: Input | Arrow | Output
   - Fixed: Removed invalid `justify="center"` from `Text.append()`

2. `_make_memory_viz_panel()` (lines 429-483)
   - Added statistics header (lines 429-448)
   - Shows sessions, messages, per-peer fact counts
   - Uses `Group(stats_text, mem_table)` for layout

3. `_make_metrics_panel()` (lines 458-509)
   - Two-column layout using `Table` with 4 columns
   - Left: Time, Iterations, API Cost, Memory Queries, Hypotheses, Verifications
   - Right: LLM Calls, Tokens, Cost/Token, Messages, Sessions, Facts
   - Status indicator with color-coded state

**Display Patterns**:
- Use `add_agent_log(category, message)` for events
- Use `add_memory_operation(operation, details, num_results)` for memory
- TUI passed to cognitive layers for real-time feedback

---

#### `arceus/test_time_training.py` ⭐ NEW
**Purpose**: Stub implementation for test-time training
**Total Lines**: 60

**Key Classes**:
- `TestTimeTrainer` - stub that returns None

**Status**: Placeholder to prevent ModuleNotFoundError
**Future Work**: Implement actual test-time training functionality

---

### Supporting Files

#### `arceus/main.py`
**Purpose**: CLI entry point and orchestration
**Not Modified**: Uses existing solver interface

#### `arceus/primitives.py`
**Purpose**: Primitive transformation functions
**Not Modified**: Provides rotate, flip, extract_objects, etc.

#### `arceus/code_executor.py`
**Purpose**: Safe code execution sandbox
**Not Modified**: Executes generated transformation code

#### `arceus/self_play.py`
**Purpose**: Self-play learning from exploration
**Not Modified**: Original source of three-layer architecture

---

## Key Patterns & Conventions

### Memory Query Patterns

**Dialectic Query** (Natural Language):
```python
query = "Have I solved a task similar to this before?"
response = await peer.chat(query=query)
content = response.content if hasattr(response, 'content') else str(response)
```

**Context Query** (Vector Search):
```python
search_query = "ARC task transformation with shapes..."
context = await peer.get_context(search_query=search_query, search_top_k=10)
if context and context.representation and context.representation.observations:
    for obs in context.representation.observations:
        print(obs.content)
```

**JSON Parsing from LLM**:
```python
import re, json
json_match = re.search(r'\{.*\}', content, re.DOTALL)
if json_match:
    data = json.loads(json_match.group(0))
```

### Storage Patterns

**Session Creation**:
```python
session = await honcho_client.session(
    f"session_{task_id}",
    metadata={"type": "solving", "task_id": task_id}
)
metrics.num_sessions_created += 1
```

**Message Ingestion**:
```python
await session.add_messages([{
    "peer_id": "solution_generator",
    "content": json.dumps(data),
    "metadata": {"type": "solution_record", "success": True}
}])
metrics.num_messages_ingested += 1
metrics.add_peer_fact("solution_generator")
```

### TUI Logging Patterns

**Agent Events**:
```python
if tui:
    tui.add_agent_log("category", "Message here")
```

**Memory Operations**:
```python
if tui:
    tui.add_memory_operation(
        operation="Strategy Selection",
        details="Should use deep exploration?",
        num_results=1
    )
```

**Grid Updates**:
```python
if tui:
    tui.update_transformation_attempt(iteration, transformation_name, result_grid)
    tui.update_output(final_result)
```

### Cost Tracking Pattern

After every LLM call:
```python
response = await llm_client.messages.create(...)
tokens_used = {
    "prompt_tokens": response.usage.input_tokens,
    "completion_tokens": response.usage.output_tokens,
}
metrics.add_llm_call(call_time_ms, tokens_used)
metrics.calculate_api_cost()  # Recalculate cost
```

### Cognitive Layer Integration Pattern

```python
from .cognitive_layers import CognitiveLayers

# Initialize
cognitive = CognitiveLayers(reflection_peer, honcho_client, tui)

# Strategy Selection
decision = await cognitive.should_use_deep_exploration(task_id, analysis)

if decision['use_deep_exploration']:
    # Layer 1: Meta-strategy
    meta_strategy = await cognitive.meta_strategy_planning(
        task_id, task_patterns, tui_label="solving"
    )

    # Layer 2: Adaptive memory
    memory_reflection = await cognitive.adaptive_memory_reflection(
        task_id, meta_strategy, tui_label="solving"
    )

    solving_context = {
        'use_deep_exploration': True,
        'meta_strategy': meta_strategy,
        'memory_reflection': memory_reflection,
        'strategy_decision': decision,
    }

# Solution Guidance
guidance = await cognitive.get_solution_guidance_from_memory(
    task_id, task_analysis, meta_strategy, tui_label="solving"
)

# Use in generation
hypothesis = await generate_solution_with_memory(
    task_data, analysis, iteration, logger, metrics, tui, solving_context
)

# Store with context
await store_solution(
    task_id, solution, success, logger, tui, metrics,
    solving_context, hypothesis.get("_solution_guidance")
)
```

---

## Future Development Guide

### For Future Claude Code Sessions

When working on Arceus, follow this process:

1. **Read this log first** to understand current state
2. **Add your session** using the template below
3. **Update file references** if you modify file structure
4. **Document key decisions** and architectural changes
5. **Maintain patterns** established in previous sessions

### Session Log Template

Copy this template when starting new work:

```markdown
### Session X: YYYY-MM-DD - [Brief Title]

**Objective**: [What you're trying to achieve]

**User Request**:
> [Quote or summarize the user's request]

#### Phase N: [Phase Name]

**Files Modified**: `file1.py`, `file2.py`
**Files Created**: `new_file.py`

**Changes**:
1. **file1.py** - [What changed]
   - Line XXX: [Specific change]
   - New method: `method_name()` - [Purpose]
   - Modified: `existing_method()` - [What changed and why]

   ```python
   # Example code snippet if helpful
   ```

**Rationale**: [Why these changes were made]

**Result**: [What this achieves]

**Testing**: [How you verified it works]

**Known Issues**: [Any problems or limitations]

**Future Work**: [What should be done next]
```

### Common Development Tasks

#### Adding a New Cognitive Layer

1. Add method to `CognitiveLayers` class in `cognitive_layers.py`
2. Follow naming pattern: `verb_noun()` e.g., `plan_strategy()`
3. Return structured dict (document schema in docstring)
4. Add TUI logging with `tui.add_agent_log()` or `tui.add_memory_operation()`
5. Integrate into solver pipeline at appropriate point
6. Store results in `solving_context` dict
7. Log changes in this document

#### Adding New Metrics

1. Add field to `SolverMetrics` dataclass in `metrics.py`
2. Add tracking code at relevant points in `solver.py`
3. Add display in `tui.py::_make_metrics_panel()`
4. Update `to_dict()` method for JSON export
5. Document in this log

#### Modifying Memory Storage

1. Update `store_solution()` or equivalent in `solver.py`
2. Add new fields to solution_record dict
3. Update metadata for filtering
4. Consider backward compatibility
5. Document schema changes

#### Changing TUI Layout

1. Modify relevant `_make_*_panel()` method in `tui.py`
2. Use Rich library correctly (check Text.append(), Table, Panel APIs)
3. Test with actual data to ensure no overflow
4. Keep metrics panel two-column for space efficiency
5. Document visual changes with screenshots if possible

### Debugging Checklist

When investigating issues:

- [ ] Check TUI logs for errors (`tui.add_agent_log("error", ...)`)
- [ ] Verify Honcho client is initialized (`self.honcho_client`)
- [ ] Check peer availability (`self.task_analyst_peer`, etc.)
- [ ] Verify solving_context is passed through pipeline
- [ ] Check metrics are incremented (sessions, messages, facts)
- [ ] Validate cost calculation (model_name set, tokens tracked)
- [ ] Check for async/await correctness
- [ ] Verify JSON parsing with `re.search(r'\{.*\}', content, re.DOTALL)`
- [ ] Check that parameters match method signatures

### Code Quality Standards

**Type Hints**:
```python
async def method_name(
    self,
    task_id: str,
    task_analysis: dict,
    meta_strategy: Optional[Dict] = None,
    tui_label: str = "solving"
) -> Optional[Dict]:
```

**Error Handling**:
```python
try:
    result = await operation()
except Exception as e:
    logging.debug(f"Operation failed: {e}")
    return None  # or sensible default
```

**Docstrings** (Google Style):
```python
def method_name(self, param1: str, param2: int) -> Dict:
    """
    Brief description.

    Longer description if needed. Explain the "why" not just the "what".

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Dict with keys:
            - key1: Description
            - key2: Description

    Raises:
        ValueError: When something goes wrong
    """
```

**Logging**:
- Use `logging.debug()` for internal state
- Use `tui.add_agent_log()` for user-facing events
- Use `logger.log_event()` for JSON trace

### Architecture Principles

1. **Memory-First**: Always consider how actions will be stored and retrieved
2. **Self-Improvement**: Store what worked and what didn't for future learning
3. **Adaptive Layers**: Use problem-type-specific strategies, not one-size-fits-all
4. **Transparent**: Show user what's happening via TUI
5. **Measurable**: Track costs and metrics for all operations
6. **Contextual**: Pass solving_context through pipeline for coherent decision-making

### Testing Commands

```bash
# Run single task
uv run python -m arceus.main --task-id 007bbfb7

# Run evaluation on all tasks
uv run python -m arceus.main --eval-all

# Run with specific enhancement
uv run python -m arceus.main --task-id 007bbfb7 --cognitive-layers

# Test imports
uv run python -c "from arceus import solver, cognitive_layers, metrics, tui; print('OK')"

# Check TUI
uv run python -c "from arceus import tui; print('TUI module OK')"
```

---

## Current State Summary

**As of 2025-12-14**:

✅ **Implemented**:
- Three-layer cognitive architecture during solving
- Strategy Selection Layer (decides when to use deep exploration)
- Solution Guidance from memory (actionable "HOW TO SOLVE")
- API cost tracking with real-time display
- Comprehensive memory metrics (sessions, messages, facts per peer)
- Enhanced TUI with side-by-side layout and two-column metrics
- Self-improvement loop (stores guidance effectiveness)

🚧 **Stub/Placeholder**:
- Test-time training (stub returns None)

🎯 **Next Priorities** (Suggested):
1. Implement actual test-time training functionality
2. Add guidance quality scoring (track accuracy of guidance predictions)
3. Implement adaptive iteration limits (stop early if high confidence)
4. Add cross-task learning (generalize patterns across task types)
5. Implement primitive discovery from successful solutions
6. Add ensemble methods with weighted voting
7. Create benchmark suite for regression testing

---

## Contact & Resources

**Honcho Documentation**: https://docs.honcho.dev
**ARC-AGI Challenge**: https://arcprize.org
**Rich TUI Library**: https://rich.readthedocs.io

**Key Files**:
- Main solver: `arceus/solver.py`
- Cognitive architecture: `arceus/cognitive_layers.py`
- Metrics: `arceus/metrics.py`
- TUI: `arceus/tui.py`
- Config: `arceus/config.py`

---

## Appendix: Quick Reference

### Cognitive Layer Decision Tree

```
Task Received
     ↓
Strategy Selection: Use deep exploration?
     ↓
   YES → Meta-Strategy: What's the problem type?
     ↓
   Adaptive Memory: Query for problem-type-specific insights
     ↓
   Solution Guidance: Get actionable "HOW TO SOLVE"
     ↓
Generate Solution with Guidance
     ↓
   FAIL → Curiosity: Why did this fail? Alternative interpretations?
     ↓
Store: Solution + Guidance + Context + Success
```

### Memory Schema

**Solution Record**:
```json
{
  "task_id": "007bbfb7",
  "solution": {"code": "...", "result": [[0,1],[2,3]]},
  "success": true,
  "guidance_used": {
    "had_memory_guidance": true,
    "confidence": 0.85,
    "solving_strategy": "Rotate 90° then filter by color",
    "code_approach": "np.rot90 + color filtering",
    "primitives_recommended": ["rotate_90", "replace_color"],
    "guidance_was_helpful": true
  },
  "meta_strategy_used": {
    "problem_type": "spatial reasoning",
    "approach_type": "analytical"
  }
}
```

### Metrics Export Schema

```json
{
  "task_id": "007bbfb7",
  "model_name": "claude-sonnet-4-5-20250929",
  "num_iterations": 3,
  "num_hypotheses_generated": 3,
  "num_verifications": 3,
  "num_memory_queries": 2,
  "num_llm_calls": 5,
  "total_tokens": 12543,
  "prompt_tokens": 8234,
  "completion_tokens": 4309,
  "api_cost_usd": "$0.1234",
  "cost_per_token": "$0.000010",
  "num_sessions_created": 2,
  "num_messages_ingested": 4,
  "num_facts_stored": 6,
  "facts_per_peer": {
    "task_analyst": 2,
    "solution_generator": 3,
    "verifier": 1
  },
  "elapsed_time_seconds": 45.2,
  "solved": true
}
```

---

**End of Log**

*Remember: This document is a living reference. Keep it updated with each session!*
