"""Test-time training: Use similar solved tasks to improve performance."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logger import JSONTraceLogger
from .metrics import SolverMetrics
from .primitives import Grid


class TestTimeTrainer:
    """Perform test-time training using similar tasks."""

    def __init__(self, solver, trace_dir: Path = Path("traces")):
        self.solver = solver
        self.trace_dir = trace_dir
        self.successful_solutions_cache = {}

    async def find_similar_solved_tasks(
        self, task_data: Dict, logger: JSONTraceLogger, metrics: SolverMetrics, tui=None, max_similar: int = 5
    ) -> List[Dict]:
        """
        Find similar tasks that were previously solved successfully.

        Uses:
        1. Honcho memory search for semantic similarity
        2. Local trace files for successful solutions

        Returns:
            List of similar solved tasks with their solutions
        """
        similar_tasks = []

        # Method 1: Query Honcho memory for similar tasks
        if self.solver.solution_generator_peer:
            analysis = await self.solver._analyze_task_structure(task_data)

            query = f"""Find similar solved tasks with:
Input shape: {analysis['input_shapes'][0] if analysis['input_shapes'] else 'unknown'}
Output shape: {analysis['output_shapes'][0] if analysis['output_shapes'] else 'unknown'}
Colors: {analysis['colors_used']}
Transformation type needed"""

            context = await self.solver.get_peer_context(
                self.solver.solution_generator_peer, query, logger, metrics, tui
            )

            if context and context.representation and context.representation.observations:
                if tui:
                    tui.add_agent_log(
                        "ttt", f"Found {len(context.representation.observations)} similar tasks in memory"
                    )

                # Extract task information from observations
                for obs in context.representation.observations[:max_similar]:
                    # Try to extract task ID and solution from observation content
                    similar_tasks.append({"source": "memory", "content": obs.content, "metadata": obs.metadata})

        # Method 2: Search local trace files for successful solutions
        if self.trace_dir.exists():
            for trace_file in self.trace_dir.glob("*.json"):
                try:
                    with open(trace_file) as f:
                        trace = json.load(f)

                    metrics_data = trace.get("metrics", {})
                    if metrics_data.get("solved"):
                        # Check if similar
                        task_analysis = trace.get("events", [{}])[0].get("data", {})

                        if self._is_similar_task(analysis, task_analysis):
                            similar_tasks.append(
                                {
                                    "source": "trace",
                                    "task_id": trace.get("task_id"),
                                    "trace_file": str(trace_file),
                                    "analysis": task_analysis,
                                    "trace": trace,
                                }
                            )

                            if len(similar_tasks) >= max_similar:
                                break

                except Exception:
                    continue

        if tui and similar_tasks:
            tui.add_agent_log("ttt", f"Total {len(similar_tasks)} similar solved tasks found")

        return similar_tasks[:max_similar]

    def _is_similar_task(self, analysis1: Dict, analysis2: Dict, threshold: float = 0.6) -> bool:
        """Check if two task analyses are similar."""
        similarity_score = 0.0
        checks = 0

        # Check input shapes
        if analysis1.get("input_shapes") and analysis2.get("input_shapes"):
            if analysis1["input_shapes"][0] == analysis2["input_shapes"][0]:
                similarity_score += 1.0
            checks += 1

        # Check output shapes
        if analysis1.get("output_shapes") and analysis2.get("output_shapes"):
            if analysis1["output_shapes"][0] == analysis2["output_shapes"][0]:
                similarity_score += 1.0
            checks += 1

        # Check color overlap
        colors1 = set(analysis1.get("colors_used", []))
        colors2 = set(analysis2.get("colors_used", []))
        if colors1 and colors2:
            color_overlap = len(colors1 & colors2) / len(colors1 | colors2)
            similarity_score += color_overlap
            checks += 1

        # Check number of examples
        if analysis1.get("num_examples") == analysis2.get("num_examples"):
            similarity_score += 0.5
            checks += 0.5

        if checks == 0:
            return False

        return (similarity_score / checks) >= threshold

    async def extract_solution_patterns(
        self, similar_tasks: List[Dict], logger: JSONTraceLogger, tui=None
    ) -> Dict[str, List[str]]:
        """
        Extract common patterns from similar solved tasks.

        Returns:
            Dictionary of pattern types to pattern descriptions
        """
        patterns = {"primitives": [], "code_snippets": [], "approaches": [], "transformations": []}

        for task_info in similar_tasks:
            if task_info["source"] == "trace":
                trace = task_info.get("trace", {})

                # Extract primitives used
                for event in trace.get("events", []):
                    if event.get("event_type") == "transformation_attempt":
                        primitive = event.get("data", {}).get("primitive")
                        if primitive and primitive not in patterns["primitives"]:
                            patterns["primitives"].append(primitive)

                    elif event.get("event_type") == "task_solved":
                        approach = event.get("data", {}).get("primitive") or event.get("data", {}).get("approach")
                        if approach and approach not in patterns["approaches"]:
                            patterns["approaches"].append(approach)

                    # Extract code snippets if available
                    elif event.get("event_type") == "reasoning_step":
                        reasoning = event.get("data", {}).get("reasoning", "")
                        if "def transform" in reasoning:
                            # Extract code block
                            if reasoning not in patterns["code_snippets"]:
                                patterns["code_snippets"].append(reasoning)

            elif task_info["source"] == "memory":
                # Extract patterns from memory observations
                content = task_info.get("content", "")
                if "rotate" in content.lower():
                    patterns["transformations"].append("rotation")
                if "flip" in content.lower():
                    patterns["transformations"].append("flip")
                if "tile" in content.lower():
                    patterns["transformations"].append("tiling")

        if tui and patterns:
            tui.add_agent_log(
                "ttt",
                f"Extracted patterns: {len(patterns['primitives'])} primitives, "
                f"{len(patterns['code_snippets'])} code snippets, "
                f"{len(patterns['approaches'])} approaches",
            )

        logger.log_event("test_time_training_patterns", patterns)

        return patterns

    async def apply_test_time_training(
        self,
        task_data: Dict,
        logger: JSONTraceLogger,
        metrics: SolverMetrics,
        tui=None,
    ) -> Optional[Dict]:
        """
        Apply test-time training to improve solving.

        Returns:
            Dictionary with learned patterns and suggestions
        """
        if tui:
            tui.add_agent_log("ttt", "Starting test-time training...")

        # Find similar solved tasks
        similar_tasks = await self.find_similar_solved_tasks(task_data, logger, metrics, tui)

        if not similar_tasks:
            if tui:
                tui.add_agent_log("ttt", "No similar solved tasks found")
            return None

        # Extract patterns
        patterns = await self.extract_solution_patterns(similar_tasks, logger, tui)

        # Build guidance for LLM
        guidance = {
            "similar_task_count": len(similar_tasks),
            "suggested_primitives": patterns["primitives"][:5],
            "suggested_approaches": patterns["approaches"][:3],
            "code_examples": patterns["code_snippets"][:2],
            "transformation_types": list(set(patterns["transformations"]))[:3],
        }

        if tui:
            tui.add_agent_log(
                "ttt",
                f"TTT complete: {len(similar_tasks)} similar tasks analyzed, "
                f"{len(guidance['suggested_primitives'])} primitives suggested",
            )

        logger.log_event("test_time_training_complete", guidance)

        return guidance

    def enhance_prompt_with_ttt(self, base_prompt: str, ttt_guidance: Optional[Dict]) -> str:
        """
        Enhance the LLM prompt with test-time training guidance.

        Args:
            base_prompt: Original prompt
            ttt_guidance: Guidance from test-time training

        Returns:
            Enhanced prompt
        """
        if not ttt_guidance:
            return base_prompt

        enhancement = "\n\nTEST-TIME TRAINING INSIGHTS:\n"
        enhancement += f"I have analyzed {ttt_guidance['similar_task_count']} similar solved tasks.\n\n"

        if ttt_guidance["suggested_primitives"]:
            enhancement += "Primitives that worked on similar tasks:\n"
            for prim in ttt_guidance["suggested_primitives"]:
                enhancement += f"  - {prim}\n"

        if ttt_guidance["suggested_approaches"]:
            enhancement += "\nSuccessful approaches on similar tasks:\n"
            for approach in ttt_guidance["suggested_approaches"]:
                enhancement += f"  - {approach}\n"

        if ttt_guidance["transformation_types"]:
            enhancement += "\nCommon transformation types:\n"
            for trans_type in ttt_guidance["transformation_types"]:
                enhancement += f"  - {trans_type}\n"

        if ttt_guidance["code_examples"]:
            enhancement += "\nExample code from similar tasks:\n"
            enhancement += ttt_guidance["code_examples"][0][:500] + "\n"

        enhancement += "\nUse these insights to guide your solution.\n"

        return base_prompt + enhancement
