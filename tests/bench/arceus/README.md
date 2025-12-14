# Arceus - ARC-AGI Solver

Sophisticated ARC-AGI puzzle solver with meta-cognitive reasoning, adaptive memory, and curiosity-driven learning powered by Honcho.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure (add ANTHROPIC_API_KEY to .env)
cp .env.example .env

# Build memory (recommended first step)
python prepare_memory.py --self-play --num-tasks 50

# Solve a puzzle
python main.py --task 007bbfb7
```

## Features

🧩 **Three-Layer Cognitive Architecture**
- Layer 1: Meta-Strategy Planning - Plans HOW to think
- Layer 2: Adaptive Memory - Problem-type specific queries
- Layer 3: Curiosity Reflection - Explores WHY failures occur

🎯 **Solving Strategies**
- Code generation (Poetiq approach)
- AIRV augmentation (260% improvement)
- Primitive transformations and discovery
- Ensemble methods
- Test-time training

🧠 **Honcho Memory Integration**
- Stores meta-strategies, patterns, and learnings
- Adaptive queries based on problem type
- Cross-task knowledge transfer

📺 **Rich Terminal UI**
- Real-time visualization
- Training examples display
- Agent logs with cognitive markers
- Memory operations panel

## Documentation

**For Users**:
- **[USER_GUIDE.md](docs/USER_GUIDE.md)** - Complete usage guide with commands and examples
- **[FEATURES.md](docs/FEATURES.md)** - Detailed feature documentation and architecture
- **[FIXES.md](docs/FIXES.md)** - Bug fixes and troubleshooting

**For AI Development (Claude Code)** - Essential for understanding the codebase:
- 🎯 **[ARCEUS_DEVELOPMENT_LOG.md](docs/ARCEUS_DEVELOPMENT_LOG.md)** ← START HERE
  - Complete development history and context
  - All changes documented chronologically
  - File-by-file references with line numbers
  - Patterns and conventions to follow
- 📋 **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - One-page cheat sheet
- 🏗️ **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Visual diagrams and flows
- 🛠️ **[CLAUDE.md](docs/CLAUDE.md)** - Development guidelines and patterns
- 📝 **[SESSION_TEMPLATE.md](docs/SESSION_TEMPLATE.md)** - Template for logging changes

## Project Structure

```
arceus/
├── README.md              # This file
├── main.py                # Entry point
├── solver.py              # Core solving logic
├── self_play.py           # Self-play exploration with 3 layers
├── code_generation.py     # Poetiq-style code generation
├── airv_augmentation.py   # AIRV augmentation pipeline
├── primitive_discovery.py # Primitive discovery system
├── tui.py                 # Terminal UI
├── docs/                  # Documentation
│   ├── USER_GUIDE.md
│   ├── FEATURES.md
│   └── FIXES.md
├── tests/                 # Test suite
└── scripts/               # Utility scripts
```

## Architecture

### Three-Layer System

```
Problem → Layer 1: Meta-Strategy Planning
       → Layer 2: Adaptive Memory Queries
       → Strategy Execution (guided by layers 1 & 2)
       → Layer 3: Curiosity Reflection (if failure)
       → Next Strategy (informed by curiosity)
```

### Self-Play Exploration

```
For each task:
1. Analyze patterns
2. Plan meta-strategy (What type? How to think?)
3. Query memory (Problem-type specific)
4. Try 6 strategies:
   - Code generation
   - Primitive combinations
   - Pattern matching
   - Code mutation
   - Hybrid approach
   - Creative combinations
5. After failures: Curiosity reflection (Why? What if?)
6. Store all learnings in Honcho
```

## Usage

### Basic Solving
```bash
# With TUI (default)
python main.py --task 007bbfb7

# Without TUI (faster)
python main.py --task 007bbfb7 --no-tui

# With enhancements
python main.py --task 007bbfb7 --enhancement all
```

### Memory Building
```bash
# Self-play on 50 tasks
python prepare_memory.py --self-play --num-tasks 50

# Self-play without TUI (faster)
python prepare_memory.py --self-play --num-tasks 100 --no-tui
```

### Testing
```bash
# Run tests
python -m pytest tests/

# Demo visualization
python scripts/demo_visualization.py
```

## TUI Markers

Watch for these in the agent logs:

- 🧩 META-STRATEGY - Layer 1: Planning how to think
- 💾 Memory approach - How memory will be queried
- 🧠 Reflecting - Memory operations
- 💭 Memory - Memory insights  
- 🔧 Memory adapted - Query adapted to problem type
- 🤔 What if - Alternative interpretations (Layer 3)
- 🎓 Curiosity reveals - Curiosity insights
- 🔄 Paradigm shift - New perspective
- ✓/✗ Success/failure indicators

## Configuration

Edit `.env`:
```bash
ANTHROPIC_API_KEY=your_key_here
HONCHO_URL=http://localhost:8000
```

Edit `config.py` for:
- Model selection (default: Claude Sonnet 4.5)
- Exploration parameters
- Reflection settings
- Logging levels

## Requirements

- Python 3.9+
- Anthropic API key (Claude)
- Honcho server (optional but recommended)
- ~4GB RAM
- Terminal with 120x40 characters (for TUI)

## Development

### Core Components

- **solver.py** - Solving strategies and orchestration
- **self_play.py** - 3-layer cognitive system + exploration
- **code_generation.py** - LLM-based code generation
- **airv_augmentation.py** - Augmentation pipeline
- **primitive_discovery.py** - Primitive invention
- **tui.py** - Terminal visualization

### Key Features Implementation

- Meta-strategy: `self_play.py:_meta_strategy_planning()`
- Adaptive memory: `self_play.py:_adaptive_memory_reflection()`
- Curiosity: `self_play.py:_curiosity_driven_reflection()`
- Code gen: `code_generation.py:CodeGenerationStrategy`
- AIRV: `airv_augmentation.py:AIRVAugmentation`

## Performance

- **Self-play**: ~2-5 minutes per task
- **Solving**: ~30-120 seconds per task
- **Memory overhead**: ~500-800ms per task
- **Cognitive layers**: Massive quality improvement

## Tips

1. **Always run self-play first** to build memory
2. **Watch the TUI markers** to understand reasoning
3. **Check memory adaptations** (🔧) for problem-type matching
4. **Monitor curiosity** (🤔) for breakthrough insights
5. **Use --no-tui** for batch processing

## Troubleshooting

See **[FIXES.md](docs/FIXES.md)** for common issues and solutions.

### Quick Fixes

- **No solutions found**: Run self-play first
- **TUI display issues**: Resize terminal or use --no-tui
- **Memory not working**: Check Honcho server is running
- **API rate limits**: Reduce num-tasks or add delays

## License

See parent Honcho repository for license information.

## Support

- Check [FIXES.md](docs/FIXES.md) for common issues
- Review [USER_GUIDE.md](docs/USER_GUIDE.md) for detailed usage
- See source code documentation

---

**Status** (as of 2025-12-14):
- ✅ Three-layer cognitive architecture fully integrated in solving
- ✅ Memory-guided solution generation with actionable guidance
- ✅ API cost tracking and comprehensive memory metrics
- ✅ Self-improvement loop (learns from guidance effectiveness)
- ✅ Enhanced TUI with real-time metrics display
- 🚧 Test-time training (stub implementation)

**For Development**: Always consult [ARCEUS_DEVELOPMENT_LOG.md](../ARCEUS_DEVELOPMENT_LOG.md) before making changes.

Built with Honcho memory infrastructure and Claude Sonnet 4.5.
