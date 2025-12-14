#!/usr/bin/env python3
"""
Demo script to showcase the puzzle visualization with transformation attempts.
This runs without needing Honcho or LLM APIs - just shows the TUI in action.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arceus.tui import ArceusTUI, run_tui_with_solver
from arceus.metrics import SolverMetrics
from arceus.primitives import ARCPrimitives


async def demo_solving_process():
    """Demonstrate the solving process with a simple example."""

    # Create TUI
    tui = ArceusTUI()

    # Create metrics
    metrics = SolverMetrics()
    metrics.task_id = "demo_task"

    # Simple 3x3 grid that we'll transform
    input_grid = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]

    # Expected output (rotated 90 degrees)
    expected_output = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]

    # Initialize TUI
    tui.update_task("demo_rotation", input_grid, expected_output)
    tui.update_metrics(metrics)
    tui.add_agent_log("initialization", "Demo: Solving a rotation puzzle")

    async def demo_solver():
        """Simulate the solving process."""
        try:
            # Phase 1: Analysis
            await asyncio.sleep(1)
            tui.add_agent_log("analysis", "Analyzing puzzle structure...")
            await asyncio.sleep(0.5)
            tui.add_agent_log("analysis", "Detected 3x3 grid with symmetry")

            # Phase 2: Memory query (simulated)
            await asyncio.sleep(0.5)
            tui.add_agent_log("memory", "Querying memory for similar patterns...")
            await asyncio.sleep(0.5)
            tui.add_memory_operation("Semantic Search", "Rotation patterns", 3)
            tui.add_agent_log("memory", "Found 3 similar solved tasks")

            # Phase 3: Try transformations
            primitives_to_try = [
                ("flip_horizontal", "Horizontal Flip"),
                ("flip_vertical", "Vertical Flip"),
                ("rotate_90", "90° Rotation"),
            ]

            for iteration, (primitive, description) in enumerate(primitives_to_try, 1):
                metrics.num_iterations = iteration
                metrics.num_reasoning_steps += 1

                tui.add_agent_log("iteration", f"Iteration {iteration}: Trying {description}")
                await asyncio.sleep(0.8)

                # Show transformation being attempted
                tui.update_transformation_attempt(
                    iteration,
                    f"{description} ({primitive})",
                    None
                )
                tui.add_agent_log("transform", f"Applying {primitive}...")
                await asyncio.sleep(1)

                # Apply the transformation
                if primitive == "flip_horizontal":
                    result = ARCPrimitives.flip_horizontal(input_grid)
                elif primitive == "flip_vertical":
                    result = ARCPrimitives.flip_vertical(input_grid)
                elif primitive == "rotate_90":
                    result = ARCPrimitives.rotate_90(input_grid)

                metrics.add_transformation_attempt(primitive)

                # Show the result
                tui.update_transformation_attempt(iteration, description, result)
                await asyncio.sleep(1.5)

                # Verify
                tui.add_agent_log("verify", f"Verifying {description}...")
                await asyncio.sleep(0.5)

                metrics.num_verifications += 1

                # Check if it matches
                if result == expected_output:
                    tui.add_agent_log("success", f"✓ Solution found with {description}!")
                    await asyncio.sleep(1)
                    tui.update_output(result)
                    metrics.mark_complete(solved=True)
                    tui.update_metrics(metrics)
                    return
                else:
                    metrics.num_failed_verifications += 1
                    tui.add_agent_log("verify", f"✗ {description} doesn't match - trying next")
                    await asyncio.sleep(0.8)
                    tui.clear_attempt()

                tui.update_metrics(metrics)

            # If we get here, nothing worked
            tui.add_agent_log("failure", "Could not find solution")
            metrics.mark_complete(solved=False)
            tui.update_metrics(metrics)

        except Exception as e:
            tui.add_agent_log("error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    # Run the demo
    solver_task = asyncio.create_task(demo_solver())
    await run_tui_with_solver(tui, solver_task, refresh_rate=0.05)


def main():
    """Run the demo."""
    print("Starting Arceus Visualization Demo...")
    print("This will show how the system visualizes solving attempts.")
    print("\nPress Ctrl+C to exit.\n")

    try:
        asyncio.run(demo_solving_process())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")


if __name__ == "__main__":
    main()
