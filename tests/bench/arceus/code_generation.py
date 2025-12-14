"""
Code Generation Strategy for ARC-AGI

Inspired by Poetiq's winning approach: Generate Python transformation code
rather than predicting outputs directly. Enhanced with Honcho memory retrieval.
"""

import asyncio
import json
import logging
import numpy as np
import tempfile
import sys
import os
from typing import Dict, List, Optional, Tuple
import re


class CodeGenerationStrategy:
    """
    Generate executable Python code to solve ARC transformations.
    Uses Honcho memory to retrieve similar successful code patterns.
    """

    def __init__(self, solver, reflection_peer=None):
        self.solver = solver
        self._reflection_peer = reflection_peer

    async def generate_and_test_code(
        self,
        task_id: str,
        task_data: Dict,
        analysis: Dict,
        tui=None,
        max_iterations: int = 5,
    ) -> Dict:
        """
        Generate transformation code iteratively with feedback.

        Returns:
            Dict with 'success', 'code', 'transformation', 'failure_details'
        """
        result = {
            "success": False,
            "code": None,
            "transformation": None,
            "failure_details": None,
        }

        train_examples = task_data.get("train", [])
        if not train_examples:
            return result

        # Retrieve similar successful code from Honcho memory
        similar_code = await self._retrieve_similar_code(analysis, tui)

        # Track previous attempts for feedback
        previous_attempts = []

        for iteration in range(max_iterations):
            if tui:
                tui.add_agent_log(
                    "self_play",
                    f"[Code Gen] Iteration {iteration + 1}/{max_iterations}: Generating transformation code..."
                )

            # Generate code with memory context
            code = await self._generate_code_with_memory(
                task_data=task_data,
                analysis=analysis,
                similar_code=similar_code,
                previous_attempts=previous_attempts,
                iteration=iteration,
                tui=tui
            )

            if not code:
                if tui:
                    tui.add_agent_log("self_play", "  ✗ No code generated")
                continue

            # Test code on training examples
            test_result = await self._test_code_on_examples(
                code=code,
                examples=train_examples,
                tui=tui
            )

            if test_result["success"]:
                # All training examples passed!
                if tui:
                    tui.add_agent_log(
                        "success",
                        f"[Code Gen] ✅ SUCCESS! Code passes all {len(train_examples)} training examples"
                    )

                result["success"] = True
                result["code"] = code
                result["transformation"] = {"type": "generated_code", "code": code}

                # Store successful code in Honcho memory
                await self._store_successful_code(
                    task_id=task_id,
                    code=code,
                    analysis=analysis,
                    task_data=task_data,
                    tui=tui
                )

                return result
            else:
                # Code failed - add to previous attempts with feedback
                if tui:
                    tui.add_agent_log(
                        "error",
                        f"[Code Gen] ❌ Code failed on example {test_result.get('failed_example_idx', '?')}"
                    )

                feedback = self._build_rich_feedback(test_result, train_examples)
                score = test_result.get("soft_score", 0.0)

                previous_attempts.append({
                    "code": code,
                    "feedback": feedback,
                    "score": score,
                    "iteration": iteration + 1
                })

                # Optionally use dialectic for deeper failure analysis
                if iteration > 1 and self._reflection_peer:
                    await self._analyze_failure_with_dialectic(
                        code=code,
                        feedback=feedback,
                        analysis=analysis,
                        tui=tui
                    )

                # Store failure details
                result["failure_details"] = test_result

        # If no solution found, return best attempt
        if previous_attempts:
            best_attempt = max(previous_attempts, key=lambda x: x["score"])
            if tui:
                tui.add_agent_log(
                    "self_play",
                    f"[Code Gen] No perfect solution found. Best score: {best_attempt['score']:.2f}"
                )
            result["code"] = best_attempt["code"]
            result["failure_details"] = {
                "best_score": best_attempt["score"],
                "feedback": best_attempt["feedback"]
            }

        return result

    async def _retrieve_similar_code(self, analysis: Dict, tui=None) -> List[Dict]:
        """Query Honcho memory for similar successful code patterns."""
        if not self._reflection_peer or not hasattr(self.solver, "active_session"):
            return []

        try:
            patterns_str = ", ".join(analysis.get("patterns", []))

            query = f"""
Based on past successful transformations in memory, retrieve code patterns for tasks with:
- Patterns: {patterns_str}
- Characteristics: {analysis.get('characteristics', {})}

Return up to 3 most relevant code examples with brief explanations of what they do.
Format as JSON list of {{"code": "...", "explanation": "...", "patterns": [...]}}
"""

            if tui:
                tui.add_memory_operation(
                    operation="Dialectic Query",
                    details="Similar code retrieval",
                    num_results=0
                )

            response = await self._reflection_peer.chat(query)
            content = response.content if hasattr(response, 'content') else str(response)

            # Try to parse JSON response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                similar_code = json.loads(json_match.group(0))
                if tui:
                    tui.add_agent_log(
                        "self_play",
                        f"[Code Gen] Retrieved {len(similar_code)} similar code patterns from memory"
                    )
                return similar_code

        except Exception as e:
            logging.debug(f"Error retrieving similar code: {e}")

        return []

    async def _generate_code_with_memory(
        self,
        task_data: Dict,
        analysis: Dict,
        similar_code: List[Dict],
        previous_attempts: List[Dict],
        iteration: int,
        tui=None
    ) -> Optional[str]:
        """Generate transformation code using LLM with memory context."""

        # Build prompt with memory and feedback
        prompt = self._build_code_generation_prompt(
            task_data=task_data,
            analysis=analysis,
            similar_code=similar_code,
            previous_attempts=previous_attempts,
            iteration=iteration
        )

        try:
            # Use solver's LLM client
            response = await self.solver.llm_client.messages.create(
                model=self.solver.model,
                max_tokens=4096,
                temperature=0.7 if iteration == 0 else 0.5,  # Lower temp after first try
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            code = self._extract_python_code(response_text)

            return code

        except Exception as e:
            logging.error(f"Error generating code: {e}")
            if tui:
                tui.add_agent_log("error", f"  ✗ Code generation error: {str(e)[:80]}")
            return None

    def _build_code_generation_prompt(
        self,
        task_data: Dict,
        analysis: Dict,
        similar_code: List[Dict],
        previous_attempts: List[Dict],
        iteration: int
    ) -> str:
        """Build comprehensive prompt for code generation."""

        prompt = f"""You are a world-class expert at solving ARC (Abstraction and Reasoning Corpus) puzzles by writing Python transformation code.

**YOUR TASK**: Write a Python function `transform(grid)` that implements the transformation rule for this puzzle.

**TRAINING EXAMPLES**:
"""

        # Show training examples
        train_examples = task_data.get("train", [])[:3]  # Show first 3
        for idx, example in enumerate(train_examples, 1):
            if not isinstance(example, dict):
                continue
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])

            prompt += f"""
Example {idx}:
  Input shape: {np.array(input_grid).shape}
  Output shape: {np.array(output_grid).shape}
  Input colors: {sorted(set(np.array(input_grid).flatten().tolist()))}
  Output colors: {sorted(set(np.array(output_grid).flatten().tolist()))}
"""

        # Add detected patterns
        patterns_str = ", ".join(analysis.get("patterns", []))
        prompt += f"""
**DETECTED PATTERNS**: {patterns_str}

**ANALYSIS**: {analysis.get('characteristics', {})}

"""

        # Add similar successful code from memory
        if similar_code:
            prompt += """**SIMILAR SUCCESSFUL CODE FROM MEMORY**:

These patterns from past successful solutions might be relevant:

"""
            for idx, code_pattern in enumerate(similar_code[:2], 1):
                prompt += f"""
Pattern {idx}: {code_pattern.get('explanation', 'N/A')}
```python
{code_pattern.get('code', '')[:300]}...
```
"""

        # Add feedback from previous attempts
        if previous_attempts:
            prompt += f"""
**PREVIOUS ATTEMPTS (sorted by quality)**:

You've tried {len(previous_attempts)} times. Here are the best attempts with feedback:

"""
            # Show up to 3 best attempts
            sorted_attempts = sorted(previous_attempts, key=lambda x: x["score"], reverse=True)[:3]
            for attempt in sorted_attempts:
                prompt += f"""
Attempt #{attempt['iteration']} (Score: {attempt['score']:.2f}):
```python
{attempt['code']}
```

Feedback:
{attempt['feedback']}

---
"""

        # Add instructions
        prompt += """
**INSTRUCTIONS**:

1. **Analyze the transformation** by studying the training examples
2. **Use these patterns** (choose what's relevant):
   - Object isolation: Extract specific objects by color, size, or position
   - Color transformations: Recolor based on criteria
   - Spatial operations: rotate, flip, translate, resize
   - Pattern generation: tiling, symmetry, replication
   - Grid operations: crop, pad, overlay

3. **Use powerful libraries**:
   ```python
   import numpy as np
   from scipy.ndimage import label, center_of_mass
   from skimage.morphology import flood_fill
   ```

4. **Write clean, correct code**:
   - Define `def transform(grid):` that takes a grid (list of lists)
   - **IMPORTANT**: Convert input to numpy array first: `grid = np.array(grid)`
   - Return the transformed grid as a list of lists
   - Handle edge cases
   - Use descriptive variable names

5. **Test your logic** mentally on the examples before writing

**OUTPUT FORMAT**:
```python
def transform(grid):
    import numpy as np
    # ALWAYS convert input to numpy array first
    grid = np.array(grid)

    # Your transformation code here
    result = ...  # transformed grid

    # Return as list of lists
    return result.tolist() if isinstance(result, np.ndarray) else result
```

Generate the code now:
"""

        return prompt

    async def _test_code_on_examples(
        self,
        code: str,
        examples: List[Dict],
        tui=None
    ) -> Dict:
        """Test generated code on training examples in isolated sandbox."""

        result = {
            "success": True,
            "failed_example_idx": None,
            "soft_score": 1.0,
            "failures": []
        }

        total_score = 0.0
        num_examples = 0

        for idx, example in enumerate(examples, 1):
            if not isinstance(example, dict):
                continue

            input_grid = example.get("input", [])
            expected_output = example.get("output", [])

            # Execute code in sandbox
            success, output, error = await self._execute_code_sandbox(code, input_grid)

            num_examples += 1

            if not success:
                result["success"] = False
                result["failed_example_idx"] = idx
                result["failures"].append({
                    "example_idx": idx,
                    "error": error,
                    "expected_shape": np.array(expected_output).shape
                })
                if tui:
                    tui.add_agent_log("error", f"    ✗ Example {idx}: {error[:60]}")
                continue

            # Compare output with expected
            try:
                output_array = np.array(output)
                expected_array = np.array(expected_output)

                if output_array.shape != expected_array.shape:
                    result["success"] = False
                    result["failed_example_idx"] = idx
                    result["failures"].append({
                        "example_idx": idx,
                        "error": f"Shape mismatch: got {output_array.shape}, expected {expected_array.shape}",
                        "output": output,
                        "expected": expected_output
                    })
                    if tui:
                        tui.add_agent_log("error", f"    ✗ Example {idx}: Shape mismatch")
                elif not np.array_equal(output_array, expected_array):
                    # Calculate soft score (partial match)
                    soft_score = float(np.mean(output_array == expected_array))
                    total_score += soft_score

                    result["success"] = False
                    result["failed_example_idx"] = idx
                    result["failures"].append({
                        "example_idx": idx,
                        "error": f"Output mismatch (partial match: {soft_score:.2f})",
                        "output": output,
                        "expected": expected_output,
                        "soft_score": soft_score
                    })
                    if tui:
                        tui.add_agent_log("error", f"    ✗ Example {idx}: Mismatch ({soft_score:.1%})")
                else:
                    # Perfect match
                    total_score += 1.0
                    if tui:
                        tui.add_agent_log("success", f"    ✓ Example {idx}: Match!")

            except Exception as e:
                result["success"] = False
                result["failed_example_idx"] = idx
                result["failures"].append({
                    "example_idx": idx,
                    "error": f"Comparison error: {str(e)}",
                })

        # Calculate average soft score
        if num_examples > 0:
            result["soft_score"] = total_score / num_examples

        return result

    async def _execute_code_sandbox(
        self,
        code: str,
        input_grid: List[List[int]],
        timeout: float = 2.0
    ) -> Tuple[bool, Optional[List], str]:
        """Execute code in isolated subprocess sandbox."""

        # Build complete script
        script = f"""
import json
import sys
import numpy as np
from scipy.ndimage import label, center_of_mass
from skimage.morphology import flood_fill

{code}

if __name__ == "__main__":
    # Read input from stdin
    data = json.load(sys.stdin)
    input_grid = data["input"]

    try:
        # Run transformation
        result = transform(input_grid)

        # Ensure result is serializable
        if isinstance(result, np.ndarray):
            result = result.tolist()

        # Write output
        json.dump({{"success": True, "output": result}}, sys.stdout)
    except Exception as e:
        json.dump({{"success": False, "error": str(e)}}, sys.stdout)
"""

        try:
            with tempfile.TemporaryDirectory() as td:
                script_path = os.path.join(td, "transform.py")
                with open(script_path, "w") as f:
                    f.write(script)

                # Execute in subprocess
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, script_path,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=td
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(input=json.dumps({"input": input_grid}).encode()),
                        timeout=timeout
                    )

                    # Parse result
                    result = json.loads(stdout.decode())
                    if result["success"]:
                        return True, result["output"], ""
                    else:
                        return False, None, result["error"]

                except asyncio.TimeoutError:
                    proc.kill()
                    return False, None, "Execution timeout"
                except json.JSONDecodeError:
                    return False, None, f"Invalid output: {stdout.decode()[:100]}"

        except Exception as e:
            return False, None, f"Sandbox error: {str(e)}"

    def _build_rich_feedback(self, test_result: Dict, examples: List[Dict]) -> str:
        """Build detailed feedback for failed attempts."""

        feedback = []

        for failure in test_result.get("failures", []):
            idx = failure["example_idx"]
            error = failure["error"]

            feedback.append(f"**Example {idx}**: {error}")

            # Add visual diff if available
            if "output" in failure and "expected" in failure:
                try:
                    output_arr = np.array(failure["output"])
                    expected_arr = np.array(failure["expected"])

                    if output_arr.shape == expected_arr.shape:
                        # Create diff visualization (first few rows)
                        rows, cols = output_arr.shape
                        max_rows = min(rows, 5)

                        diff_lines = []
                        for i in range(max_rows):
                            row_str = " ".join(
                                str(output_arr[i, j]) if output_arr[i, j] == expected_arr[i, j]
                                else f"{output_arr[i, j]}/{expected_arr[i, j]}"
                                for j in range(min(cols, 10))
                            )
                            diff_lines.append(row_str)

                        feedback.append("Visual diff (your/expected):")
                        feedback.append("\n".join(diff_lines))

                        if rows > 5 or cols > 10:
                            feedback.append(f"... (showing {max_rows}/{rows} rows, {min(cols, 10)}/{cols} cols)")

                except Exception:
                    pass

        return "\n\n".join(feedback)

    async def _analyze_failure_with_dialectic(
        self,
        code: str,
        feedback: str,
        analysis: Dict,
        tui=None
    ) -> Optional[Dict]:
        """Use dialectic for deep failure analysis."""

        if not self._reflection_peer:
            return None

        try:
            query = f"""
This Python transformation code failed:

```python
{code}
```

**FAILURE FEEDBACK**:
{feedback}

**PUZZLE ANALYSIS**:
Patterns: {analysis.get('patterns', [])}

Based on similar failures in your memory:
1. What is the ROOT CAUSE of this failure?
2. What specific code changes would fix it?
3. What alternative approaches should we try?

Provide concrete, actionable suggestions.
"""

            response = await self._reflection_peer.chat(query)
            content = response.content if hasattr(response, 'content') else str(response)

            if tui:
                # Show first 150 chars of analysis
                preview = content[:150] + "..." if len(content) > 150 else content
                tui.add_agent_log("self_play", f"[Dialectic] 💡 {preview}")

            return {"analysis": content}

        except Exception as e:
            logging.debug(f"Error in dialectic failure analysis: {e}")
            return None

    async def _store_successful_code(
        self,
        task_id: str,
        code: str,
        analysis: Dict,
        task_data: Dict,
        tui=None
    ) -> None:
        """Store successful code pattern in Honcho memory."""

        if not self.solver.honcho_client:
            return

        try:
            # Extract code characteristics
            train_example = task_data["train"][0]
            input_shape = np.array(train_example["input"]).shape
            output_shape = np.array(train_example["output"]).shape

            content = f"""Successful Transformation Code

TASK: {task_id}
PATTERNS: {", ".join(analysis.get('patterns', []))}
TRANSFORMATION: {input_shape} → {output_shape}

CODE:
```python
{code}
```

This code successfully solved all training examples.
"""

            # Store via solver's ingestion method
            if hasattr(self.solver, '_ingest_thought'):
                await self.solver._ingest_thought(
                    task_id=task_id,
                    thought_type="successful_code_pattern",
                    content=content,
                    metadata={
                        "patterns": analysis.get('patterns', []),
                        "input_shape": str(input_shape),
                        "output_shape": str(output_shape),
                        "code_length": len(code)
                    },
                    tui=tui
                )

        except Exception as e:
            logging.debug(f"Error storing successful code: {e}")

    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from LLM response."""

        # Try to find code block
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try to find def transform
        pattern = r'(def transform\(.*?\n(?:.*?\n)*?.*?return.*?)(?:\n\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None
