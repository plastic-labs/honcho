#!/usr/bin/env python3
"""
Main entry point for the Arceus ARC-AGI-2 solver.

Usage:
    python -m arceus.main --task-id 007bbfb7
    python -m arceus.main --task-file path/to/task.json
    python -m arceus.main --eval-all  # Run on all evaluation tasks
    python -m arceus.main --no-tui --task-id 007bbfb7  # Run without TUI
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .config import ArceusConfig
from .logger import JSONTraceLogger
from .metrics import MetricsAggregator, SolverMetrics
from .solver import ARCSolver
from .tui import ArceusTUI, run_tui_with_solver


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_task(task_path: Path) -> dict:
    """Load a task from a JSON file."""
    with open(task_path) as f:
        return json.load(f)


def find_task_file(task_id: str, config: ArceusConfig) -> Path:
    """Find a task file by ID in training or evaluation directories."""
    # Try training first
    training_file = config.training_path / f"{task_id}.json"
    if training_file.exists():
        return training_file

    # Try evaluation
    eval_file = config.evaluation_path / f"{task_id}.json"
    if eval_file.exists():
        return eval_file

    raise FileNotFoundError(f"Task {task_id} not found in training or evaluation sets")


async def solve_single_task_with_tui(
    config: ArceusConfig, task_id: str, task_data: dict
):
    """Solve a single task with TUI visualization."""
    # Initialize components
    tui = ArceusTUI()
    solver = ARCSolver(config)
    await solver.initialize()

    logger = JSONTraceLogger(config.trace_output_dir, task_id, config.enable_json_trace)
    metrics = SolverMetrics()

    # Setup TUI with initial task
    first_test = task_data["test"][0] if task_data.get("test") else {}
    # Ensure first_test is a dictionary
    if isinstance(first_test, dict):
        test_input = first_test.get("input", [])
        test_expected = first_test.get("output")  # May not have this for actual test
    else:
        test_input = []
        test_expected = None

    # Extract training examples to show in TUI
    training_examples = task_data.get("train", [])
    tui.update_task(task_id, test_input, test_expected, training_examples=training_examples)
    tui.update_metrics(metrics)
    tui.add_agent_log("initialization", f"Starting task {task_id}")

    # Determine solving method
    enhancement_mode = getattr(config, 'enhancement_mode', 'none')

    # Create solver task
    async def solve_with_updates():
        """Solve task and update TUI."""
        try:
            # Start solving - pass TUI to solver for real-time updates
            # Select solving method based on mode
            if enhancement_mode == "all":
                result = await solver.solve_with_all_enhancements(task_id, task_data, logger, metrics, tui)
            elif enhancement_mode == "ensemble":
                result = await solver.solve_with_ensemble(task_id, task_data, logger, metrics, tui)
            elif enhancement_mode == "ttt":
                result = await solver.solve_with_test_time_training(task_id, task_data, logger, metrics, tui)
            else:
                result = await solver.solve_task(task_id, task_data, logger, metrics, tui)

            if result:
                tui.add_agent_log("complete", "Task solving complete!")
            else:
                tui.add_agent_log("complete", "Task solving complete - no solution found")

            tui.update_metrics(metrics)

        except Exception as e:
            logging.error(f"Error solving task: {e}")
            tui.add_agent_log("error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            logger.flush()
            await solver.close()

    solver_task = asyncio.create_task(solve_with_updates())

    # Run TUI alongside solver
    await run_tui_with_solver(tui, solver_task, config.tui_refresh_rate)

    return metrics


async def solve_single_task_no_tui(config: ArceusConfig, task_id: str, task_data: dict):
    """Solve a single task without TUI."""
    solver = ARCSolver(config)
    await solver.initialize()

    logger = JSONTraceLogger(config.trace_output_dir, task_id, config.enable_json_trace)
    metrics = SolverMetrics()

    # Determine solving method based on enhancement mode
    enhancement_mode = getattr(config, 'enhancement_mode', 'none')

    if enhancement_mode == "all":
        logging.info(f"Solving task {task_id} with ALL ENHANCEMENTS...")
    elif enhancement_mode == "ensemble":
        logging.info(f"Solving task {task_id} with ENSEMBLE mode...")
    elif enhancement_mode == "ttt":
        logging.info(f"Solving task {task_id} with TEST-TIME TRAINING...")
    else:
        logging.info(f"Solving task {task_id}...")

    try:
        # Select solving method based on mode
        if enhancement_mode == "all":
            result = await solver.solve_with_all_enhancements(task_id, task_data, logger, metrics, tui=None)
        elif enhancement_mode == "ensemble":
            result = await solver.solve_with_ensemble(task_id, task_data, logger, metrics, tui=None)
        elif enhancement_mode == "ttt":
            result = await solver.solve_with_test_time_training(task_id, task_data, logger, metrics, tui=None)
        else:
            result = await solver.solve_task(task_id, task_data, logger, metrics, tui=None)

        if result:
            logging.info(f"Task {task_id} solved successfully!")
        else:
            logging.info(f"Task {task_id} could not be solved")

        # Print metrics
        logging.info(f"Metrics: {json.dumps(metrics.to_dict(), indent=2)}")

    except Exception as e:
        logging.error(f"Error solving task {task_id}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.flush()
        await solver.close()

    return metrics


async def solve_multiple_tasks(config: ArceusConfig, task_ids: list[str]):
    """Solve multiple tasks and aggregate results."""
    aggregator = MetricsAggregator()

    for task_id in task_ids:
        try:
            task_file = find_task_file(task_id, config)
            task_data = load_task(task_file)

            if config.enable_tui:
                metrics = await solve_single_task_with_tui(config, task_id, task_data)
            else:
                metrics = await solve_single_task_no_tui(config, task_id, task_data)

            aggregator.add_task_metrics(metrics)

        except Exception as e:
            logging.error(f"Error processing task {task_id}: {e}")

    # Print summary
    summary = aggregator.get_summary()
    logging.info(f"\n{'='*50}")
    logging.info("SUMMARY STATISTICS")
    logging.info(f"{'='*50}")
    logging.info(json.dumps(summary, indent=2))

    return aggregator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Arceus: ARC-AGI-2 Solver with Honcho Memory")
    parser.add_argument("--task-id", type=str, help="Task ID to solve")
    parser.add_argument("--task-file", type=Path, help="Path to task JSON file")
    parser.add_argument("--eval-all", action="store_true", help="Run on all evaluation tasks")
    parser.add_argument("--no-tui", action="store_true", help="Disable TUI interface")
    parser.add_argument("--no-trace", action="store_true", help="Disable JSON trace logging")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max solving iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Medium-term enhancements (all enabled by default)
    parser.add_argument("--no-enhancements", action="store_true", help="Disable all enhancements (baseline mode)")
    parser.add_argument("--ensemble-only", action="store_true", help="Use ONLY ensemble solving")
    parser.add_argument("--ttt-only", action="store_true", help="Use ONLY test-time training")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Load configuration
    config = ArceusConfig.from_env()

    # Override with command line args
    if args.no_tui:
        config.enable_tui = False
    if args.no_trace:
        config.enable_json_trace = False
    if args.max_iterations:
        config.max_iterations = args.max_iterations

    # Store enhancement mode in config (DEFAULT: all enhancements enabled)
    config.enhancement_mode = "all"  # DEFAULT: Use all enhancements
    if args.no_enhancements:
        config.enhancement_mode = "none"
    elif args.ensemble_only:
        config.enhancement_mode = "ensemble"
    elif args.ttt_only:
        config.enhancement_mode = "ttt"

    # Determine what to solve
    if args.task_file:
        # Single task from file
        task_data = load_task(args.task_file)
        task_id = args.task_file.stem

        if config.enable_tui:
            asyncio.run(solve_single_task_with_tui(config, task_id, task_data))
        else:
            asyncio.run(solve_single_task_no_tui(config, task_id, task_data))

    elif args.task_id:
        # Single task by ID
        try:
            task_file = find_task_file(args.task_id, config)
            task_data = load_task(task_file)

            if config.enable_tui:
                asyncio.run(solve_single_task_with_tui(config, args.task_id, task_data))
            else:
                asyncio.run(solve_single_task_no_tui(config, args.task_id, task_data))

        except FileNotFoundError as e:
            logging.error(str(e))
            sys.exit(1)

    elif args.eval_all:
        # All evaluation tasks
        eval_files = list(config.evaluation_path.glob("*.json"))
        task_ids = [f.stem for f in eval_files]

        logging.info(f"Found {len(task_ids)} evaluation tasks")
        asyncio.run(solve_multiple_tasks(config, task_ids))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
