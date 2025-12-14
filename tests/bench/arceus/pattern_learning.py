"""Pattern learning: Extract and create learned primitives from successful solutions."""

import json
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .primitives import ARCPrimitives, Grid


class LearnedPrimitive:
    """A primitive learned from successful solutions."""

    def __init__(self, name: str, description: str, code: str, frequency: int = 1):
        self.name = name
        self.description = description
        self.code = code
        self.frequency = frequency
        self.success_count = frequency
        self.usage_count = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "frequency": self.frequency,
            "success_count": self.success_count,
            "usage_count": self.usage_count,
            "success_rate": self.success_count / max(1, self.usage_count),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LearnedPrimitive":
        """Create from dictionary."""
        prim = cls(data["name"], data["description"], data["code"], data.get("frequency", 1))
        prim.success_count = data.get("success_count", 1)
        prim.usage_count = data.get("usage_count", 0)
        return prim


class PatternLearner:
    """Learn patterns from successful solutions."""

    def __init__(self, trace_dir: Path = Path("traces"), learned_primitives_file: Path = Path("learned_primitives.json")):
        self.trace_dir = trace_dir
        self.learned_primitives_file = learned_primitives_file
        self.learned_primitives: Dict[str, LearnedPrimitive] = {}
        self.load_learned_primitives()

    def load_learned_primitives(self):
        """Load learned primitives from disk."""
        if self.learned_primitives_file.exists():
            try:
                with open(self.learned_primitives_file) as f:
                    data = json.load(f)

                self.learned_primitives = {
                    name: LearnedPrimitive.from_dict(prim_data) for name, prim_data in data.items()
                }
            except Exception as e:
                print(f"Error loading learned primitives: {e}")

    def save_learned_primitives(self):
        """Save learned primitives to disk."""
        try:
            data = {name: prim.to_dict() for name, prim in self.learned_primitives.items()}

            with open(self.learned_primitives_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving learned primitives: {e}")

    def analyze_successful_solutions(self, min_frequency: int = 2) -> List[LearnedPrimitive]:
        """
        Analyze successful solutions to extract patterns.

        Returns:
            List of learned primitives
        """
        if not self.trace_dir.exists():
            return []

        # Collect code from successful solutions
        code_patterns = []
        primitive_usage = Counter()
        transformation_sequences = []

        for trace_file in self.trace_dir.glob("*.json"):
            try:
                with open(trace_file) as f:
                    trace = json.load(f)

                metrics_data = trace.get("metrics", {})
                if not metrics_data.get("solved"):
                    continue

                # Extract code from successful solutions
                for event in trace.get("events", []):
                    if event.get("event_type") == "task_solved":
                        approach = event.get("data", {})

                        # Check if code was used
                        if "code" in approach:
                            code_patterns.append(
                                {
                                    "task_id": trace.get("task_id"),
                                    "code": approach["code"],
                                    "approach": "generated_code",
                                }
                            )

                        # Track primitive usage
                        elif "primitive" in approach:
                            primitive_usage[approach["primitive"]] += 1

                    elif event.get("event_type") == "reasoning_step":
                        reasoning = event.get("data", {}).get("reasoning", "")
                        if "```python" in reasoning and "def transform" in reasoning:
                            # Extract code block
                            try:
                                code = reasoning.split("```python")[1].split("```")[0].strip()
                                code_patterns.append(
                                    {
                                        "task_id": trace.get("task_id"),
                                        "code": code,
                                        "approach": "reasoning",
                                    }
                                )
                            except:
                                pass

            except Exception:
                continue

        # Cluster similar code patterns
        learned = self._cluster_code_patterns(code_patterns, min_frequency)

        # Add frequently used primitives as learned patterns
        for primitive, count in primitive_usage.most_common(10):
            if count >= min_frequency:
                learned.append(
                    LearnedPrimitive(
                        name=f"frequent_{primitive}",
                        description=f"Frequently successful primitive: {primitive}",
                        code=f"return {primitive}(grid)",
                        frequency=count,
                    )
                )

        # Update internal cache
        for prim in learned:
            if prim.name in self.learned_primitives:
                # Update existing
                existing = self.learned_primitives[prim.name]
                existing.frequency += prim.frequency
            else:
                # Add new
                self.learned_primitives[prim.name] = prim

        self.save_learned_primitives()

        return learned

    def _cluster_code_patterns(self, code_patterns: List[Dict], min_frequency: int) -> List[LearnedPrimitive]:
        """Cluster similar code patterns into learned primitives."""
        if not code_patterns:
            return []

        # Simple clustering based on code similarity
        clusters = {}

        for pattern in code_patterns:
            code = pattern["code"]

            # Create a simple signature based on operations used
            signature = self._get_code_signature(code)

            if signature not in clusters:
                clusters[signature] = {"codes": [], "count": 0}

            clusters[signature]["codes"].append(code)
            clusters[signature]["count"] += 1

        # Convert clusters to learned primitives
        learned = []
        for idx, (signature, cluster) in enumerate(clusters.items()):
            if cluster["count"] >= min_frequency:
                # Use most common code variant
                most_common_code = Counter(cluster["codes"]).most_common(1)[0][0]

                learned.append(
                    LearnedPrimitive(
                        name=f"learned_pattern_{idx}",
                        description=f"Learned pattern (freq={cluster['count']}): {signature}",
                        code=most_common_code,
                        frequency=cluster["count"],
                    )
                )

        return learned

    def _get_code_signature(self, code: str) -> str:
        """Get a signature for code based on operations used."""
        operations = []

        # Check for numpy operations
        if "np.rot90" in code:
            operations.append("rotate")
        if "np.flip" in code or "fliplr" in code or "flipud" in code:
            operations.append("flip")
        if "np.tile" in code:
            operations.append("tile")
        if "np.repeat" in code:
            operations.append("repeat")
        if "np.where" in code:
            operations.append("conditional")
        if "for " in code and "for " in code:
            operations.append("double_loop")

        # Check for arithmetic
        if "+" in code and "-" in code:
            operations.append("arithmetic")

        # Check for indexing
        if "[" in code and "]" in code:
            operations.append("indexing")

        return "_".join(sorted(operations)) if operations else "unknown"

    def get_top_learned_primitives(self, n: int = 10) -> List[LearnedPrimitive]:
        """Get top N learned primitives by frequency."""
        sorted_prims = sorted(self.learned_primitives.values(), key=lambda p: p.frequency, reverse=True)
        return sorted_prims[:n]

    def suggest_primitives_for_task(self, task_analysis: Dict) -> List[str]:
        """
        Suggest learned primitives for a task based on analysis.

        Args:
            task_analysis: Task analysis dictionary

        Returns:
            List of suggested primitive names
        """
        suggestions = []

        # Get top learned primitives
        top_prims = self.get_top_learned_primitives(5)

        # Simple heuristic: suggest based on frequency and task characteristics
        for prim in top_prims:
            # Check if primitive signature matches task characteristics
            if "rotate" in prim.description and len(task_analysis.get("input_shapes", [])) > 0:
                # Rotation might be useful for square-ish grids
                shape = task_analysis["input_shapes"][0]
                if abs(shape[0] - shape[1]) <= 2:  # Nearly square
                    suggestions.append(prim.name)

            elif "flip" in prim.description:
                # Flipping is generally useful
                suggestions.append(prim.name)

            elif prim.frequency >= 5:
                # High frequency primitives are generally useful
                suggestions.append(prim.name)

        return suggestions[:3]

    def create_primitive_function(self, learned_primitive: LearnedPrimitive) -> Optional[Callable]:
        """
        Create an executable function from a learned primitive.

        Args:
            learned_primitive: Learned primitive to convert

        Returns:
            Callable function or None if creation fails
        """
        from .code_executor import SafeCodeExecutor

        executor = SafeCodeExecutor()

        # Wrap code in a function if needed
        code = learned_primitive.code
        if "def transform" not in code:
            code = f"""
def transform(grid):
    import numpy as np
    {code}
"""

        # Validate code
        is_safe, error = executor.validate_code(code)
        if not is_safe:
            return None

        # Create a closure that executes the code
        def learned_function(grid: Grid) -> Optional[Grid]:
            return executor.execute_transformation(code, grid)

        return learned_function

    def get_statistics(self) -> Dict:
        """Get statistics about learned primitives."""
        if not self.learned_primitives:
            return {"total": 0, "avg_frequency": 0, "avg_success_rate": 0}

        total = len(self.learned_primitives)
        avg_freq = sum(p.frequency for p in self.learned_primitives.values()) / total
        avg_success = sum(p.success_count / max(1, p.usage_count) for p in self.learned_primitives.values()) / total

        return {
            "total": total,
            "avg_frequency": avg_freq,
            "avg_success_rate": avg_success,
            "top_3": [p.name for p in self.get_top_learned_primitives(3)],
        }
