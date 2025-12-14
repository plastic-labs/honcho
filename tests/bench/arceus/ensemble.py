"""Ensemble solver that combines multiple solving strategies."""

import asyncio
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .logger import JSONTraceLogger
from .metrics import SolverMetrics
from .primitives import Grid


class SolvingStrategy:
    """Base class for solving strategies."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    async def solve(
        self,
        solver,
        task_id: str,
        task_data: Dict,
        logger: JSONTraceLogger,
        metrics: SolverMetrics,
        tui=None,
    ) -> Optional[Grid]:
        """Solve using this strategy."""
        raise NotImplementedError


class CodeGenerationStrategy(SolvingStrategy):
    """Strategy that prioritizes LLM code generation."""

    def __init__(self, max_iterations: int = 5):
        super().__init__("code_generation", weight=1.5)
        self.max_iterations = max_iterations

    async def solve(
        self,
        solver,
        task_id: str,
        task_data: Dict,
        logger: JSONTraceLogger,
        metrics: SolverMetrics,
        tui=None,
    ) -> Optional[Grid]:
        """Solve using code generation with limited iterations."""
        if tui:
            tui.add_agent_log("strategy", f"[{self.name}] Starting code generation strategy")

        original_max_iter = solver.config.max_iterations
        solver.config.max_iterations = self.max_iterations

        try:
            result = await solver.solve_task(task_id, task_data, logger, metrics, tui)
            return result
        finally:
            solver.config.max_iterations = original_max_iter


class PrimitiveSearchStrategy(SolvingStrategy):
    """Strategy that exhaustively tries primitive combinations."""

    def __init__(self, max_combinations: int = 20):
        super().__init__("primitive_search", weight=1.0)
        self.max_combinations = max_combinations

    async def solve(
        self,
        solver,
        task_id: str,
        task_data: Dict,
        logger: JSONTraceLogger,
        metrics: SolverMetrics,
        tui=None,
    ) -> Optional[Grid]:
        """Try multiple primitive combinations."""
        if tui:
            tui.add_agent_log("strategy", f"[{self.name}] Trying primitive combinations")

        from .primitives import PRIMITIVE_FUNCTIONS

        # Get test input
        first_test = task_data["test"][0] if task_data.get("test") else {}
        # Ensure first_test is a dictionary
        if not isinstance(first_test, dict):
            return None
        test_input = first_test.get("input", [])

        # Try single primitives first
        for prim_name, prim_func in list(PRIMITIVE_FUNCTIONS.items())[:self.max_combinations]:
            result = solver._try_primitive_transformation(test_input, prim_name)

            if result:
                # Verify on training examples
                all_passed = True
                for example in task_data["train"]:
                    # Ensure example is a dictionary
                    if not isinstance(example, dict):
                        continue
                    attempt_result = solver._try_primitive_transformation(example["input"], prim_name)
                    if not attempt_result or not await solver.verify_solution(
                        attempt_result, example["output"], logger
                    ):
                        all_passed = False
                        break

                if all_passed:
                    if tui:
                        tui.add_agent_log("strategy", f"[{self.name}] Found solution using {prim_name}")
                    return result

        return None


class MemoryGuidedStrategy(SolvingStrategy):
    """Strategy that heavily relies on memory retrieval."""

    def __init__(self, max_iterations: int = 10):
        super().__init__("memory_guided", weight=1.2)
        self.max_iterations = max_iterations

    async def solve(
        self,
        solver,
        task_id: str,
        task_data: Dict,
        logger: JSONTraceLogger,
        metrics: SolverMetrics,
        tui=None,
    ) -> Optional[Grid]:
        """Solve with heavy memory querying."""
        if tui:
            tui.add_agent_log("strategy", f"[{self.name}] Using memory-guided approach")

        # Get similar tasks from memory first
        if solver.solution_generator_peer:
            analysis = await solver._analyze_task_structure(task_data)

            # Query memory more aggressively
            similar_query = f"""Find similar tasks to: {analysis['input_shapes']} -> {analysis['output_shapes']}
Colors: {analysis['colors_used']}
Pattern: transformation with {analysis['num_examples']} examples"""

            context = await solver.get_peer_context(
                solver.solution_generator_peer, similar_query, logger, metrics, tui
            )

            if context and context.representation and context.representation.observations:
                if tui:
                    tui.add_agent_log(
                        "strategy", f"[{self.name}] Found {len(context.representation.observations)} similar tasks"
                    )

        # Now solve with this context
        original_max_iter = solver.config.max_iterations
        solver.config.max_iterations = self.max_iterations

        try:
            result = await solver.solve_task(task_id, task_data, logger, metrics, tui)
            return result
        finally:
            solver.config.max_iterations = original_max_iter


class EnsembleSolver:
    """Ensemble solver that combines multiple strategies."""

    def __init__(self, base_solver, strategies: List[SolvingStrategy] = None):
        self.base_solver = base_solver

        # Default strategies if none provided
        if strategies is None:
            self.strategies = [
                CodeGenerationStrategy(max_iterations=5),
                PrimitiveSearchStrategy(max_combinations=15),
                MemoryGuidedStrategy(max_iterations=8),
            ]
        else:
            self.strategies = strategies

    async def solve_with_ensemble(
        self,
        task_id: str,
        task_data: Dict,
        logger: JSONTraceLogger,
        metrics: SolverMetrics,
        tui=None,
        parallel: bool = False,
    ) -> Optional[Grid]:
        """
        Solve using ensemble of strategies.

        Args:
            task_id: Task identifier
            task_data: Task data with train/test examples
            logger: Trace logger
            metrics: Metrics tracker
            tui: Optional TUI for visualization
            parallel: Run strategies in parallel (experimental)

        Returns:
            Best solution or None
        """
        if tui:
            tui.add_agent_log("ensemble", f"Starting ensemble solver with {len(self.strategies)} strategies")

        solutions = []

        if parallel:
            # Run strategies in parallel (experimental - may have issues with shared state)
            tasks = []
            for strategy in self.strategies:
                # Create independent solver instances for parallel execution
                # Note: This requires careful state management
                task = asyncio.create_task(
                    strategy.solve(self.base_solver, task_id, task_data, logger, metrics, tui)
                )
                tasks.append((strategy, task))

            # Wait for all strategies
            for strategy, task in tasks:
                try:
                    result = await task
                    if result:
                        solutions.append((strategy, result))
                        if tui:
                            tui.add_agent_log("ensemble", f"[{strategy.name}] Found solution")
                except Exception as e:
                    if tui:
                        tui.add_agent_log("ensemble", f"[{strategy.name}] Error: {str(e)}")

        else:
            # Run strategies sequentially (safer)
            for strategy in self.strategies:
                if tui:
                    tui.add_agent_log("ensemble", f"Trying strategy: {strategy.name}")

                try:
                    result = await strategy.solve(self.base_solver, task_id, task_data, logger, metrics, tui)

                    if result:
                        solutions.append((strategy, result))
                        if tui:
                            tui.add_agent_log("ensemble", f"[{strategy.name}] Found solution")

                        # Early stopping: if high-confidence strategy succeeds, use it
                        if strategy.weight >= 1.5:
                            if tui:
                                tui.add_agent_log(
                                    "ensemble", f"Using solution from high-confidence strategy: {strategy.name}"
                                )
                            return result

                except Exception as e:
                    if tui:
                        tui.add_agent_log("ensemble", f"[{strategy.name}] Error: {str(e)}")
                    logger.log_error(f"strategy_{strategy.name}", str(e))

        # Vote on best solution
        if solutions:
            best_solution = self._vote_on_solutions(solutions, task_data, logger, tui)
            return best_solution

        if tui:
            tui.add_agent_log("ensemble", "No strategy found a solution")

        return None

    def _vote_on_solutions(
        self, solutions: List[Tuple[SolvingStrategy, Grid]], task_data: Dict, logger: JSONTraceLogger, tui=None
    ) -> Optional[Grid]:
        """
        Vote on the best solution using weighted voting.

        Returns the solution that:
        1. Has highest weighted vote
        2. Is most common among strategies
        """
        if not solutions:
            return None

        if len(solutions) == 1:
            return solutions[0][1]

        if tui:
            tui.add_agent_log("ensemble", f"Voting among {len(solutions)} solutions")

        # Convert grids to hashable format for voting
        def grid_to_tuple(grid: Grid) -> tuple:
            return tuple(tuple(row) for row in grid)

        # Weighted voting
        votes = {}
        for strategy, solution in solutions:
            grid_key = grid_to_tuple(solution)
            if grid_key not in votes:
                votes[grid_key] = {"weight": 0, "count": 0, "solution": solution, "strategies": []}

            votes[grid_key]["weight"] += strategy.weight
            votes[grid_key]["count"] += 1
            votes[grid_key]["strategies"].append(strategy.name)

        # Find best solution by weighted vote
        best_key = max(votes.keys(), key=lambda k: (votes[k]["weight"], votes[k]["count"]))
        best_info = votes[best_key]

        if tui:
            tui.add_agent_log(
                "ensemble",
                f"Selected solution (weight={best_info['weight']:.1f}, "
                f"votes={best_info['count']}, "
                f"strategies={', '.join(best_info['strategies'])})",
            )

        logger.log_event(
            "ensemble_vote",
            {
                "total_solutions": len(solutions),
                "unique_solutions": len(votes),
                "winner_weight": best_info["weight"],
                "winner_count": best_info["count"],
                "winner_strategies": best_info["strategies"],
            },
        )

        return best_info["solution"]

    def add_strategy(self, strategy: SolvingStrategy):
        """Add a new strategy to the ensemble."""
        self.strategies.append(strategy)

    def remove_strategy(self, strategy_name: str):
        """Remove a strategy by name."""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
