#!/usr/bin/env python3
"""
Memory Preparation Script for Arceus.

This script implements Phase 1 from the PDF specification:
- Ingests all training tasks from ARC-AGI-2 dataset into Honcho
- Generates natural language descriptions using LLM
- Identifies relevant primitives for each task
- Stores all information in Honcho for later retrieval

NEW: Self-Play Mode
- Actively explores and solves training tasks
- Discovers new transformation primitives
- Learns which primitives work for which patterns
- Builds agent's transformation vocabulary dynamically

Usage:
    # Prepare memory with first 10 training tasks (testing)
    python -m arceus.prepare_memory --limit 10

    # Prepare memory with all training tasks
    python -m arceus.prepare_memory --all

    # Self-play mode: Explore and discover primitives
    python -m arceus.prepare_memory --self-play --limit 20

    # With custom training data path
    python -m arceus.prepare_memory --training-path /path/to/training
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from .config import ArceusConfig
from .solver import ARCSolver
from .tui import ArceusTUI, run_tui_with_solver


async def prepare_memory_with_tui(
    config: ArceusConfig, training_path: Path, limit: int = None, self_play: bool = False
):
    """Prepare memory with TUI visualization."""
    # Initialize TUI
    tui = ArceusTUI()
    tui.update_task("memory_prep", [], None)
    tui.add_agent_log("init", f"Starting memory preparation phase...")
    tui.add_agent_log("init", f"Training data path: {training_path}")

    if self_play:
        tui.add_agent_log("init", "🎮 SELF-PLAY MODE: Exploring tasks to discover primitives")

    if limit:
        tui.add_agent_log("init", f"Limiting to first {limit} tasks")

    # Initialize solver
    solver = ARCSolver(config)
    await solver.initialize()

    async def prep_task():
        """Memory preparation task."""
        try:
            if self_play:
                # Use self-play exploration mode
                await solver.prepare_memory_with_self_play(training_path, limit, tui)
            else:
                # Standard memory preparation
                await solver.prepare_memory_phase(training_path, tui, limit)

            tui.add_agent_log("complete", "Memory preparation complete!")
            await asyncio.sleep(2)  # Keep TUI visible for a moment

        except Exception as e:
            logging.error(f"Error during memory preparation: {e}")
            tui.add_agent_log("error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            await solver.close()

    solver_task = asyncio.create_task(prep_task())
    await run_tui_with_solver(tui, solver_task, config.tui_refresh_rate)


async def prepare_memory_no_tui(
    config: ArceusConfig, training_path: Path, limit: int = None, self_play: bool = False
):
    """Prepare memory without TUI."""
    logging.info(f"Starting memory preparation phase...")
    logging.info(f"Training data path: {training_path}")

    if self_play:
        logging.info("🎮 SELF-PLAY MODE: Exploring tasks to discover primitives")

    if limit:
        logging.info(f"Limiting to first {limit} tasks")

    # Initialize solver
    solver = ARCSolver(config)
    await solver.initialize()

    try:
        if self_play:
            # Use self-play exploration mode
            await solver.prepare_memory_with_self_play(training_path, limit, None)
        else:
            # Standard memory preparation
            await solver.prepare_memory_phase(training_path, None, limit)

        logging.info("Memory preparation complete!")

    except Exception as e:
        logging.error(f"Error during memory preparation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await solver.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare Honcho memory by ingesting ARC-AGI-2 training tasks"
    )
    parser.add_argument(
        "--training-path",
        type=Path,
        help="Path to ARC-AGI-2 training data directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tasks to ingest (for testing)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all training tasks (default limits to 10 for testing)",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable TUI interface",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Enable self-play mode: actively explore tasks and discover primitives",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = ArceusConfig.from_env()

    # Determine training path
    if args.training_path:
        training_path = args.training_path
    else:
        training_path = config.training_path

    if not training_path.exists():
        logging.error(f"Training data path does not exist: {training_path}")
        sys.exit(1)

    # Determine limit
    limit = None
    if not args.all:
        limit = args.limit if args.limit else 10  # Default to 10 for testing

    # Disable TUI if requested
    if args.no_tui:
        config.enable_tui = False

    # Run memory preparation
    if config.enable_tui:
        asyncio.run(prepare_memory_with_tui(config, training_path, limit, args.self_play))
    else:
        asyncio.run(prepare_memory_no_tui(config, training_path, limit, args.self_play))


if __name__ == "__main__":
    main()
