# Arceus Features

Complete guide to all implemented features in the Arceus ARC-AGI solver.

## Three-Layer Cognitive Architecture

The solver uses a sophisticated three-layer system for intelligent problem-solving:

### Layer 1: Meta-Strategy Planning 🧩

**Purpose**: Plans HOW to think before attempting solutions

**Implementation**: `self_play.py:_meta_strategy_planning()`

**What it does**:
- Classifies problem type (pattern recognition, spatial reasoning, logical rules, compositional)
- Selects thinking approach (analytical, intuitive, experimental, compositional, analogical)
- Plans memory query strategy
- Identifies assumptions to challenge
- Generates curiosity questions

**Example output**:
```
🧩 META-STRATEGY: Thinking about how to think...
🎯 Strategy: Spatial-compositional problem. Decompose → analyze → compose.
💾 Memory approach: Query for multi-step spatial transformations
🔍 Curious: What if objects interact in non-obvious ways?
⚠️ Challenge: Independent object transformation assumption
```

### Layer 2: Adaptive Memory System 💾

**Purpose**: Dynamically queries Honcho memory based on problem type

**Implementation**: `self_play.py:_adaptive_memory_reflection()`

**What it does**:
- Builds problem-type specific queries
- Adapts based on approach type (analytical focuses on WHY, intuitive on WHAT)
- Prioritizes relevant memories
- Returns context-aware insights

**Query adaptation examples**:
- Pattern Recognition → "What similar VISUAL PATTERNS have I seen?"
- Spatial Reasoning → "What SPATIAL TRANSFORMATIONS worked on similar layouts?"
- Logical Rules → "What LOGICAL RULES applied in similar contexts?"
- Compositional → "How did I DECOMPOSE similar complex problems?"

**Example output**:
```
🧠 Reflecting on past experiences using: Query for multi-step spatial...
💭 Memory: Spatial-compositional tasks need object decomposition
📚 From failures: Global transforms missed object interactions
🔧 Memory adapted: Query focused on spatial relationships
```

### Layer 3: Curiosity-Driven Reflection 🔍

**Purpose**: Deeply explores WHY failures occurred and alternative perspectives

**Implementation**: `self_play.py:_curiosity_driven_reflection()`

**What it does**:
- Analyzes root cause of failures
- Explores alternative interpretations ("What if I'm wrong?")
- Identifies blind spots
- Suggests paradigm shifts
- Generates experimental ideas

**Example output**:
```
🤔 What if: Objects don't transform uniformly - context matters
🤔 What if: The pattern is in relationships, not individual objects
🎓 Curiosity reveals: Assumed pixel-level rules, but object-level apply
🔄 Paradigm shift: Think object-graph, not pixel-grid
```

## Solving Strategies

### 1. Code Generation (Poetiq Approach)

**Implementation**: `code_generation.py:CodeGenerationStrategy`

**What it does**:
- Generates Python transformation functions from training examples
- Uses memory-guided generation (queries Honcho for similar patterns)
- Iterative refinement (up to 5 attempts)
- Sandbox execution for safety
- Rich feedback from test results

**Key features**:
- Converts inputs to numpy arrays automatically
- Handles errors gracefully
- Provides detailed feedback for refinement
- Memory integration for pattern recognition

### 2. AIRV Augmentation

**Implementation**: `airv_augmentation.py:AIRVAugmentation`

**What it does**:
- Augment: Creates variations of training examples (rotation, flip, color permutation)
- Inference: Runs solver on augmented examples
- Reverse: Reverses augmentations on output
- Vote: Selects most common solution

**Benefits**:
- 260% improvement (from ARChitects winning approach)
- Handles rotation/reflection invariance
- Robust to color variations

### 3. Primitive Transformations

**Implementation**: `primitive_discovery.py:PrimitiveDiscoverySystem`

**Built-in primitives** (`primitives.py`):
- rotate_90, rotate_180, rotate_270
- flip_horizontal, flip_vertical
- transpose, anti_transpose
- shift_up, shift_down, shift_left, shift_right
- fill_background, swap_colors
- extract_objects, filter_by_color

**Primitive discovery**:
- Invents new primitives from failed attempts
- Stores in Honcho memory
- Combines primitives for complex transformations

### 4. Ensemble Methods

**Implementation**: `solver.py:solve_with_ensemble()`

**What it does**:
- Runs multiple solving strategies in parallel
- Votes on best solution
- Combines strengths of different approaches

### 5. Test-Time Training

**Implementation**: `solver.py:solve_with_test_time_training()`

**What it does**:
- Learns from training examples
- Adapts to test puzzle
- Iterative refinement

## Self-Play Exploration

**Implementation**: `self_play.py:SelfPlayAgent`

**Purpose**: Autonomous exploration and learning from ARC-AGI puzzles

**Exploration strategies**:
1. Code generation
2. Primitive combinations
3. Pattern matching
4. Code mutation
5. Hybrid approach
6. Creative combinations

**What happens during exploration**:
1. Analyze puzzle (patterns, complexity, relationships)
2. Meta-strategy planning (Layer 1)
3. Adaptive memory reflection (Layer 2)
4. Try strategies in sequence
5. After each failure: Curiosity reflection (Layer 3)
6. Store all learnings in Honcho
7. Discover new primitives from attempts

**Memory storage**:
- Strategy successes/failures
- Meta-strategies that worked
- Curiosity insights
- Failure patterns
- Problem-type learnings

## Honcho Memory Integration

**Implementation**: Throughout codebase

**What's stored**:
- Puzzle analyses and patterns
- Strategy results (success/failure)
- Meta-strategies and thinking approaches
- Curiosity reflections and insights
- Failure analyses and anti-patterns
- Discovered primitives
- Memory adaptations

**What's queried**:
- Similar puzzles and solutions
- Successful strategies for problem types
- Failure patterns to avoid
- Primitive transformations
- Meta-strategies that worked

**Dialectic API usage**:
- Meta-strategy planning
- Memory reflection
- Failure analysis
- Curiosity exploration
- Strategy reasoning

## Terminal UI (TUI)

**Implementation**: `tui.py:ArceusTUI`

**Features**:
- Real-time puzzle visualization with color-coded grids
- Training examples display (side-by-side)
- Test puzzle display
- Attempt visualization
- Success/failure indicators (✓/✗)
- Agent logs with timestamps
- Memory operations panel
- Performance metrics
- ASCII art header with gradient colors

**Sections**:
1. Header: Arceus logo
2. Main (left): Puzzle visualization with training examples
3. Main (right top): Agent logs with strategy attempts
4. Main (right bottom): Memory operations
5. Footer: Performance metrics

**Training examples**:
```
╔════════════════════════════════════════╗
║ 📚 TRAINING EXAMPLES - Learn Pattern   ║
╚════════════════════════════════════════╝

In → Out  │  In → Out
[grid1]   │  [grid2]

╔════════════════════════════════════════╗
║ 🎯 TEST PUZZLE - Apply Pattern Here    ║
╚════════════════════════════════════════╝
```

## Configuration

**File**: `config.py:ArceusConfig`

**Settings**:
- Honcho workspace and peer configuration
- Model selection (Claude Sonnet 4.5)
- Exploration parameters (max attempts, strategies)
- Reflection and dynamic exploration flags
- Logging levels and output paths

**Environment variables** (`.env`):
```
ANTHROPIC_API_KEY=your_key_here
HONCHO_URL=http://localhost:8000
```

## Performance Metrics

**Implementation**: `metrics.py:SolverMetrics`

**Tracked metrics**:
- Task attempts and successes
- Strategy effectiveness
- Time per puzzle
- Memory operation counts
- Primitive usage statistics

## Workflow Example

Complete flow for solving a puzzle:

```
1. Load puzzle
   ↓
2. TUI shows training examples + test puzzle
   ↓
3. Meta-strategy planning (Layer 1)
   - Classify: "Spatial-compositional problem"
   - Plan: "Decompose → analyze → compose"
   ↓
4. Adaptive memory query (Layer 2)
   - Query: "Multi-step spatial transformations"
   - Results: "Object decomposition needed"
   ↓
5. Try Strategy 1: Code Generation
   - Generate code based on training examples
   - Test on training examples
   - If passes: Apply to test
   - If fails: Go to curiosity
   ↓
6. Curiosity Reflection (Layer 3)
   - "Why failed?": Assumed global transform
   - "What if?": Objects have relationships
   - "Paradigm shift": Think object-graph
   ↓
7. Try Strategy 2: Primitive Combinations
   - Informed by curiosity insights
   - Success! ✓
   ↓
8. Store everything in Honcho:
   - Meta-strategy that worked
   - Adaptive memory adaptation
   - Curiosity insights
   - Successful primitive combination
```

## Command Line Usage

```bash
# Run solver on specific task
python main.py --task <task_id>

# Prepare memory with self-play
python prepare_memory.py --self-play --num-tasks 50

# Run tests
python -m pytest tests/

# Demo visualization
python scripts/demo_visualization.py
```

## Key Benefits

### Meta-Cognitive Awareness
- Knows what TYPE of problem it's solving
- Adapts thinking strategy to problem
- Questions its own assumptions

### Adaptive Intelligence
- Memory queries adapt to problem type
- Approach changes based on characteristics
- Learns from past meta-learnings

### Curiosity-Driven Learning
- Doesn't just fail and move on
- Explores WHY failures occurred
- Discovers alternative perspectives
- Generates paradigm shifts

### Cross-Task Transfer
- Learns "For spatial problems, use compositional thinking"
- Transfers problem-type strategies, not just patterns
- Builds meta-knowledge over time

## Implementation Status

All features listed in this document are ✅ **fully implemented and tested**.

## Documentation

- **FEATURES.md** (this file) - Complete feature guide
- **USER_GUIDE.md** - User-facing usage guide
- **FIXES.md** - Bug fixes and patches
- **README.md** - Project overview

For technical implementation details, see the source code with inline documentation.
