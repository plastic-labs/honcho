# Arceus User Guide

Complete guide to using the Arceus ARC-AGI solver.

## Quick Start

### Prerequisites

- Python 3.9+
- Anthropic API key (Claude)
- Honcho server running (optional but recommended for memory)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

### Basic Usage

```bash
# Solve a single task
python main.py --task 007bbfb7

# Prepare memory with self-play (recommended first step)
python prepare_memory.py --self-play --num-tasks 50
```

## Command Reference

### Main Solver

```bash
python main.py [OPTIONS]

Options:
  --task TASK_ID          Task ID to solve (required)
  --no-tui               Disable terminal UI
  --enhancement MODE     Enhancement mode: none, ensemble, ttt, all (default: none)
```

**Examples**:
```bash
# Basic solving with TUI
python main.py --task 007bbfb7

# Solve with ensemble methods
python main.py --task 007bbfb7 --enhancement ensemble

# Solve with all enhancements
python main.py --task 007bbfb7 --enhancement all

# Solve without TUI (faster)
python main.py --task 007bbfb7 --no-tui
```

### Memory Preparation

```bash
python prepare_memory.py [OPTIONS]

Options:
  --self-play            Enable self-play exploration
  --num-tasks N          Number of tasks to explore (default: 10)
  --no-tui              Disable terminal UI
```

**Examples**:
```bash
# Self-play on 50 tasks (recommended)
python prepare_memory.py --self-play --num-tasks 50

# Self-play without TUI (faster)
python prepare_memory.py --self-play --num-tasks 100 --no-tui

# Just analyze tasks without self-play
python prepare_memory.py --num-tasks 20
```

## Configuration

### Environment Variables (.env)

```bash
# Required
ANTHROPIC_API_KEY=your_api_key_here

# Optional (for Honcho memory)
HONCHO_URL=http://localhost:8000
HONCHO_WORKSPACE=arc-agi-2-solver
```

### Config File (config.py)

Key settings you can modify:

```python
class ArceusConfig:
    # Paths
    training_path = Path("path/to/training")
    evaluation_path = Path("path/to/evaluation")

    # Model
    model = "claude-sonnet-4-5-20250929"

    # Exploration
    enable_reflection = True  # Enable 3-layer cognitive system
    enable_dynamic_exploration = True
    max_exploration_attempts = 6

    # Enhancement mode
    enhancement_mode = "none"  # Options: none, ensemble, ttt, all
```

## Understanding the Output

### Terminal UI (TUI)

When running with TUI (default), you'll see:

#### 1. Header
```
                ,6"Yb.  `7Mb,od8 ,p6"bo   .gP"Ya `7MM  `7MM  ,pP"Ybd
               8)   MM    MM' "'6M'  OO  ,M'   Yb  MM    MM  8I   `"
                ,pm9MM    MM    8M       8M""""""  MM    MM  `YMMMa.
```

#### 2. Main Panel (Left): Puzzle Visualization

```
╔══════════════════════════════════════╗
║ 📚 TRAINING EXAMPLES                ║
╚══════════════════════════════════════╝

Example 1:        Example 2:
In → Out          In → Out
[grid]            [grid]

╔══════════════════════════════════════╗
║ 🎯 TEST PUZZLE                      ║
╚══════════════════════════════════════╝

📥 Test Input (Given)
[test grid]

📤 Agent's Output (Attempt)
[attempt grid]

✓ MATCH! or ✗ INCORRECT
```

#### 3. Main Panel (Right): Agent Logs

```
13:45:12  self_play     🧩 META-STRATEGY: Spatial problem
13:45:13  self_play     🧠 Reflecting on past experiences
13:45:14  self_play     Strategy 1/6: Code Generation
13:45:15  self_play     ✓ Code Generation: CORRECT - Solved!
```

Markers you'll see:
- 🧩 META-STRATEGY - Layer 1: Planning how to think
- 🎯 Strategy - Thinking approach selected
- 💾 Memory approach - How memory will be queried
- 🔍 Curious - Curiosity questions
- 🧠 Reflecting - Memory operations
- 💭 Memory - Memory insights
- 📚 From failures - Lessons learned
- 💡 Untried - New ideas
- 🔧 Memory adapted - Query adaptation
- 🤔 What if - Alternative interpretations
- 🎓 Curiosity reveals - Curiosity insights
- 🔄 Paradigm shift - New perspective
- ✓/✗ Success/failure indicators

#### 4. Memory Operations Panel

Shows Honcho memory operations:
```
13:45:12  WRITE   Storing meta-strategy for task_123
13:45:13  QUERY   Retrieved 5 similar spatial tasks
13:45:14  WRITE   Storing successful strategy
```

#### 5. Footer: Metrics

```
Tasks: 5/10 | Success Rate: 80% | Avg Time: 45s
Memory Ops: 150 | Strategies Tried: 25
```

## Self-Play Exploration

### What is Self-Play?

Self-play mode allows Arceus to:
- Autonomously explore ARC-AGI puzzles
- Learn patterns and strategies
- Build up Honcho memory
- Discover new primitive transformations
- Test different solving approaches

### How Self-Play Works

```
For each task:
1. Analyze puzzle patterns
2. Plan meta-strategy (Layer 1)
3. Query memory for insights (Layer 2)
4. Try solving strategies (6 total):
   - Code generation
   - Primitive combinations
   - Pattern matching
   - Code mutation
   - Hybrid approach
   - Creative combinations
5. After each failure: Curiosity reflection (Layer 3)
6. Store all learnings in Honcho
7. Discover and store new primitives
```

### When to Use Self-Play

**Before first use**: Run self-play on 50-100 tasks to build initial memory

```bash
python prepare_memory.py --self-play --num-tasks 50
```

**Periodically**: Run self-play on new training tasks to expand knowledge

**After failures**: If solver struggles, run self-play on similar tasks

### Self-Play Output

You'll see in TUI:
1. Training examples displayed
2. Meta-strategy planning for each task
3. Memory reflections
4. Strategy attempts (1-6 per task)
5. Curiosity explorations after failures
6. Memory storage operations
7. Primitive discoveries

## Solving Strategies

### Available Strategies

1. **Code Generation** - Generates Python transform functions
2. **AIRV Augmentation** - Augments, infers, reverses, votes
3. **Primitive Combinations** - Combines built-in primitives
4. **Pattern Matching** - Matches against known patterns
5. **Ensemble Methods** - Combines multiple approaches
6. **Test-Time Training** - Learns from training examples

### Strategy Selection

Strategies are tried in order during self-play. The solver automatically:
- Uses meta-strategy to plan approach
- Queries memory for relevant experiences
- Applies curiosity after failures
- Adapts based on problem type

### Enhancement Modes

```bash
# No enhancements (fast, single strategy)
python main.py --task TASK_ID --enhancement none

# Ensemble (multiple strategies, voting)
python main.py --task TASK_ID --enhancement ensemble

# Test-time training (learns from examples)
python main.py --task TASK_ID --enhancement ttt

# All enhancements (slowest, most thorough)
python main.py --task TASK_ID --enhancement all
```

## Memory and Learning

### What Gets Stored in Honcho

During solving and self-play:
- **Meta-strategies**: Problem types and thinking approaches
- **Strategy results**: What worked/failed for each puzzle
- **Curiosity insights**: Alternative interpretations and paradigm shifts
- **Failure patterns**: Anti-patterns to avoid
- **Primitives**: Discovered transformation functions
- **Memory adaptations**: How queries were adapted

### Querying Memory

The solver automatically queries Honcho for:
- Similar puzzles and their solutions
- Strategies that worked for this problem type
- Failure patterns to avoid
- Relevant primitive transformations

### Memory Adaptation

Based on meta-strategy, memory queries adapt:
- **Pattern Recognition** → Visual pattern memories
- **Spatial Reasoning** → Spatial transformation memories
- **Logical Rules** → Rule discovery memories
- **Compositional** → Decomposition strategy memories

## Troubleshooting

### Solver Not Finding Solutions

1. **Run self-play first** to build memory:
   ```bash
   python prepare_memory.py --self-play --num-tasks 100
   ```

2. **Try enhancement modes**:
   ```bash
   python main.py --task TASK_ID --enhancement all
   ```

3. **Check logs** for errors in `logs/` directory

### TUI Display Issues

1. **Resize terminal** - TUI needs at least 120x40 characters
2. **Run without TUI** if terminal too small:
   ```bash
   python main.py --task TASK_ID --no-tui
   ```

### Memory Not Working

1. **Check Honcho server** is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check .env file** has HONCHO_URL set

3. **Run without memory** (will work but less effective)

### API Rate Limits

If hitting Anthropic API rate limits:
- Reduce num-tasks in self-play
- Add delays between tasks
- Use smaller task sets

## Performance Tips

### For Speed
- Use `--no-tui` flag
- Set `enhancement_mode = "none"`
- Reduce `max_exploration_attempts`
- Run self-play overnight

### For Accuracy
- Run self-play on many tasks (100+)
- Use `--enhancement all`
- Enable all reflection features
- Let it try more strategies

### For Memory Efficiency
- Run self-play in batches
- Clear old Honcho sessions periodically
- Use specific task sets (not all training data)

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_basic.py

# Run with verbose output
python -m pytest tests/ -v
```

## Scripts

Utility scripts in `scripts/`:

```bash
# Demo TUI visualization
python scripts/demo_visualization.py

# Run solver with defaults
./scripts/run_arceus.sh TASK_ID

# Run self-play with defaults
./scripts/run_prepare_memory.sh
```

## Tips and Best Practices

### 1. Start with Self-Play
Always run self-play before trying to solve tasks. This builds the foundation of knowledge.

### 2. Use TUI for Understanding
The TUI helps you understand what the solver is thinking. Watch the meta-strategy and curiosity reflections.

### 3. Check Memory Adaptations
Look for the 🔧 marker - it shows how queries are being adapted to the problem type.

### 4. Monitor Curiosity
The 🤔 "What if" messages show alternative perspectives - these often lead to breakthroughs.

### 5. Review Logs
Check `logs/` for detailed information about what happened during solving.

### 6. Experiment with Enhancements
Different enhancement modes work better for different puzzle types. Try them all.

## Examples

### Example 1: First Time Setup

```bash
# 1. Install and configure
pip install -r requirements.txt
cp .env.example .env
# Edit .env with API key

# 2. Build initial memory (30 minutes)
python prepare_memory.py --self-play --num-tasks 50

# 3. Solve a task
python main.py --task 007bbfb7
```

### Example 2: Solving Multiple Tasks

```bash
# Solve tasks from a list
for task in 007bbfb7 00d62c1b 025d127b; do
    python main.py --task $task --no-tui
done
```

### Example 3: Focused Learning

```bash
# Learn from specific pattern types
# 1. Identify spatial reasoning tasks
# 2. Run self-play on them
python prepare_memory.py --self-play --num-tasks 20
# (manually select spatial tasks in code)
```

## Next Steps

After mastering the basics:
1. Read **FEATURES.md** to understand the three-layer architecture
2. Explore the source code for implementation details
3. Customize configuration for your use case
4. Experiment with different enhancement modes
5. Build domain-specific primitive libraries

## Support

For issues, questions, or contributions:
- Check **FIXES.md** for common issues
- Review source code documentation
- See the main Honcho repository

## Quick Reference Card

```
# Most common commands
python prepare_memory.py --self-play --num-tasks 50  # Build memory
python main.py --task TASK_ID                         # Solve task
python main.py --task TASK_ID --enhancement all       # Solve with all enhancements
python main.py --task TASK_ID --no-tui               # Solve without TUI

# TUI markers
🧩 Meta-strategy planning    🧠 Memory operations
💭 Memory insights           🔍 Curiosity questions
🤔 Alternative views         🔄 Paradigm shifts
✓ Success                    ✗ Failure
```
