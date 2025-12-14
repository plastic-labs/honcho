"""
AIRV (Augment-Inference-Reverse-Augmentation-Vote) System

Inspired by ARChitects' 260% improvement technique:
1. Augment training examples (rotate, flip, color permute)
2. Run inference on each augmented version
3. Reverse the augmentation on outputs
4. Vote across all variants

Enhanced with Honcho memory to select best augmentations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging


class AIRVAugmentation:
    """
    Augmentation system with memory-guided selection and voting.
    """

    def __init__(self, reflection_peer=None):
        self._reflection_peer = reflection_peer

    async def augment_and_vote(
        self,
        task_data: Dict,
        solver_func: Callable,
        analysis: Dict,
        tui=None
    ) -> Dict:
        """
        Apply AIRV pipeline: augment, solve, reverse, vote.

        Args:
            task_data: Task with train/test examples
            solver_func: Async function that attempts to solve task
            analysis: Task analysis with patterns
            tui: Optional TUI for logging

        Returns:
            Dict with best solution from voting
        """

        # Step 1: Decide which augmentations to apply (memory-guided)
        augmentations = await self._select_augmentations(analysis, tui)

        # Step 2: Apply augmentations and solve each variant
        solutions = []

        for aug_name, aug_func, reverse_func in augmentations:
            if tui:
                tui.add_agent_log(
                    "self_play",
                    f"[AIRV] Trying augmentation: {aug_name}"
                )

            # Augment training examples
            augmented_task = self._augment_task(task_data, aug_func)

            # Solve augmented version
            try:
                solution = await solver_func(augmented_task)

                if solution.get("success"):
                    # Reverse augmentation on output
                    reversed_output = self._reverse_augmentation(
                        solution["output"],
                        reverse_func
                    )

                    solutions.append({
                        "augmentation": aug_name,
                        "output": reversed_output,
                        "confidence": solution.get("confidence", 0.5),
                        "code": solution.get("code", "")
                    })

                    if tui:
                        tui.add_agent_log(
                            "success",
                            f"[AIRV] ✓ {aug_name} succeeded"
                        )
            except Exception as e:
                logging.debug(f"AIRV augmentation {aug_name} failed: {e}")
                if tui:
                    tui.add_agent_log(
                        "error",
                        f"[AIRV] ✗ {aug_name} failed"
                    )

        # Step 3: Vote across solutions
        if not solutions:
            return {"success": False}

        best_solution = self._vote(solutions, tui)

        if tui:
            tui.add_agent_log(
                "self_play",
                f"[AIRV] Voting complete. Best solution from: {best_solution['augmentation']}"
            )

        return best_solution

    async def _select_augmentations(
        self,
        analysis: Dict,
        tui=None
    ) -> List[Tuple[str, Callable, Callable]]:
        """
        Select which augmentations to apply based on patterns.
        Query dialectic for memory-guided selection.
        """

        # Default augmentations (always try)
        default_augs = [
            ("identity", lambda x: x, lambda x: x),  # No augmentation baseline
            ("rotate_90", self._rotate_90, self._rotate_270),
            ("rotate_180", self._rotate_180, self._rotate_180),
            ("flip_horizontal", self._flip_h, self._flip_h),
            ("flip_vertical", self._flip_v, self._flip_v),
        ]

        # If we have dialectic, ask which augmentations work for these patterns
        if self._reflection_peer:
            try:
                patterns_str = ", ".join(analysis.get("patterns", []))
                query = f"""
Based on past experience, which augmentations (rotation, flipping, color permutation) have been most effective for tasks with patterns: {patterns_str}?

Respond with a prioritized list of recommended augmentations.
"""

                response = await self._reflection_peer.chat(query)

                if tui:
                    tui.add_memory_operation(
                        operation="Dialectic Query",
                        details="Augmentation selection",
                        num_results=1
                    )

                # Parse response and potentially add color permutation
                content = response.content if hasattr(response, 'content') else str(response)
                if "color" in content.lower():
                    # Add color permutation augmentations
                    for perm in [(1, 2), (0, 2)]:  # Swap color pairs
                        default_augs.append((
                            f"color_swap_{perm}",
                            lambda x, p=perm: self._color_permute(x, p),
                            lambda x, p=perm: self._color_permute(x, p)  # Symmetric
                        ))

            except Exception as e:
                logging.debug(f"Error in augmentation selection: {e}")

        return default_augs

    def _augment_task(self, task_data: Dict, augment_func: Callable) -> Dict:
        """Apply augmentation function to all examples in task."""

        augmented = {
            "train": [],
            "test": []
        }

        for example in task_data.get("train", []):
            if not isinstance(example, dict):
                continue

            augmented["train"].append({
                "input": augment_func(example.get("input", [])),
                "output": augment_func(example.get("output", []))
            })

        for example in task_data.get("test", []):
            if not isinstance(example, dict):
                continue

            augmented["test"].append({
                "input": augment_func(example.get("input", [])),
                "output": example.get("output")  # Keep original for validation
            })

        return augmented

    def _reverse_augmentation(self, output: List, reverse_func: Callable) -> List:
        """Apply reverse augmentation to output."""
        return reverse_func(output)

    def _vote(self, solutions: List[Dict], tui=None) -> Dict:
        """
        Vote across solutions using multiple strategies:
        1. Exact match voting (frequency)
        2. Confidence weighting
        3. Diversity bonus for tie-breaking
        """

        if len(solutions) == 1:
            return {"success": True, **solutions[0]}

        # Group by identical outputs
        output_groups = {}
        for sol in solutions:
            output_key = self._output_to_key(sol["output"])
            if output_key not in output_groups:
                output_groups[output_key] = []
            output_groups[output_key].append(sol)

        # Score each group
        group_scores = []
        for output_key, group in output_groups.items():
            # Vote count
            vote_count = len(group)

            # Average confidence
            avg_confidence = sum(s.get("confidence", 0.5) for s in group) / len(group)

            # Combined score
            score = vote_count + avg_confidence

            group_scores.append((score, group[0]))  # Take first representative

        # Return highest scoring solution
        group_scores.sort(key=lambda x: x[0], reverse=True)
        best_solution = group_scores[0][1]

        if tui:
            tui.add_agent_log(
                "self_play",
                f"[AIRV] Vote distribution: {len(output_groups)} unique outputs, winner score: {group_scores[0][0]:.2f}"
            )

        return {"success": True, **best_solution}

    def _output_to_key(self, output: List) -> str:
        """Convert output grid to hashable key for grouping."""
        try:
            arr = np.array(output)
            return arr.tobytes()
        except:
            return str(output)

    # Augmentation functions
    def _rotate_90(self, grid: List) -> List:
        """Rotate grid 90 degrees clockwise."""
        if not grid:
            return grid
        arr = np.array(grid)
        return np.rot90(arr, k=-1).tolist()

    def _rotate_180(self, grid: List) -> List:
        """Rotate grid 180 degrees."""
        if not grid:
            return grid
        arr = np.array(grid)
        return np.rot90(arr, k=2).tolist()

    def _rotate_270(self, grid: List) -> List:
        """Rotate grid 270 degrees clockwise (= 90 CCW)."""
        if not grid:
            return grid
        arr = np.array(grid)
        return np.rot90(arr, k=1).tolist()

    def _flip_h(self, grid: List) -> List:
        """Flip grid horizontally."""
        if not grid:
            return grid
        arr = np.array(grid)
        return np.fliplr(arr).tolist()

    def _flip_v(self, grid: List) -> List:
        """Flip grid vertically."""
        if not grid:
            return grid
        arr = np.array(grid)
        return np.flipud(arr).tolist()

    def _color_permute(self, grid: List, swap_pair: Tuple[int, int]) -> List:
        """Swap two colors in the grid."""
        if not grid:
            return grid
        arr = np.array(grid)
        c1, c2 = swap_pair

        # Create mask and swap
        mask1 = arr == c1
        mask2 = arr == c2

        result = arr.copy()
        result[mask1] = c2
        result[mask2] = c1

        return result.tolist()


class ShapePredictor:
    """
    Predict output shape separately from content (ARChitects insight).
    85% accuracy on shape × 30.5% on content = 26% overall.
    """

    def __init__(self, reflection_peer=None):
        self._reflection_peer = reflection_peer

    async def predict_output_shape(
        self,
        input_shape: Tuple[int, int],
        analysis: Dict,
        tui=None
    ) -> Tuple[Tuple[int, int], float]:
        """
        Predict output dimensions based on input and patterns.

        Returns:
            (predicted_shape, confidence)
        """

        if not self._reflection_peer:
            # Default: assume same shape
            return input_shape, 0.3

        try:
            patterns_str = ", ".join(analysis.get("patterns", []))

            query = f"""
Given input grid shape {input_shape} and detected patterns [{patterns_str}],
what is the most likely output grid shape?

Consider these transformation types:
- shape_preserving: same dimensions
- expansion: larger dimensions (tiling, padding)
- reduction: smaller dimensions (cropping, extraction)
- rotation: swapped dimensions (if 90/270 degree rotation)

Respond with:
1. Predicted shape as (rows, cols)
2. Confidence level (0.0 to 1.0)
3. Brief reasoning

Format: SHAPE: (rows, cols) | CONFIDENCE: X.XX | REASON: ...
"""

            response = await self._reflection_peer.chat(query)
            content = response.content if hasattr(response, 'content') else str(response)

            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Shape prediction",
                    num_results=1
                )

            # Parse response
            shape, confidence = self._parse_shape_prediction(content, input_shape)

            if tui:
                tui.add_agent_log(
                    "self_play",
                    f"[Shape] Predicted: {shape} (confidence: {confidence:.2f})"
                )

            return shape, confidence

        except Exception as e:
            logging.debug(f"Error in shape prediction: {e}")
            return input_shape, 0.3

    def _parse_shape_prediction(
        self,
        content: str,
        default_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], float]:
        """Parse shape prediction from dialectic response."""

        import re

        # Try to find SHAPE: (rows, cols)
        shape_match = re.search(r'SHAPE:\s*\((\d+),\s*(\d+)\)', content)
        if shape_match:
            rows = int(shape_match.group(1))
            cols = int(shape_match.group(2))
            shape = (rows, cols)
        else:
            shape = default_shape

        # Try to find CONFIDENCE: X.XX
        conf_match = re.search(r'CONFIDENCE:\s*(\d+\.?\d*)', content)
        if conf_match:
            confidence = float(conf_match.group(1))
        else:
            confidence = 0.5

        return shape, confidence
