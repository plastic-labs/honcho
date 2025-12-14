"""Safe code execution for LLM-generated transformation code."""

import ast
import copy
from typing import Any, Dict, List, Optional

import numpy as np

from .primitives import ARCPrimitives, Grid


class SafeCodeExecutor:
    """Execute LLM-generated transformation code safely."""

    def __init__(self):
        # Build safe namespace with allowed functions and modules
        self.safe_namespace = {
            # Python builtins (restricted)
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            # Numpy (commonly needed)
            "np": np,
            "numpy": np,
            # ARC primitives
            "ARCPrimitives": ARCPrimitives,
            # Helper types
            "List": List,
            "Dict": Dict,
            "Optional": Optional,
            "Any": Any,
        }

        # Add all primitive methods directly to namespace
        for name, func in vars(ARCPrimitives).items():
            if not name.startswith("_") and callable(func):
                self.safe_namespace[name] = func

    def validate_code(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Validate that code is safe to execute.

        Returns:
            (is_safe, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"

        # Check for dangerous operations
        allowed_imports = {"numpy", "np"}  # Allow numpy since we provide it

        for node in ast.walk(tree):
            # Check imports - only allow numpy
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in allowed_imports:
                        return False, f"Import of '{alias.name}' not allowed"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module not in allowed_imports:
                    return False, f"Import from '{node.module}' not allowed"

            # No file operations
            if isinstance(node, ast.Name):
                if node.id in ["open", "file", "exec", "eval", "compile", "__import__"]:
                    return False, f"Dangerous function '{node.id}' not allowed"

            # No attribute access to dangerous things
            if isinstance(node, ast.Attribute):
                if node.attr in ["__dict__", "__class__", "__bases__", "__subclasses__"]:
                    return False, f"Dangerous attribute access '{node.attr}' not allowed"

        return True, None

    def extract_transform_function(self, code: str) -> Optional[str]:
        """
        Extract the transform function from generated code.

        Looks for a function named 'transform' or similar.
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for transform, solve, or solution function
                    if node.name.lower() in ["transform", "solve", "solution", "apply"]:
                        return ast.unparse(node)
        except:
            pass

        # If no function found, return the whole code
        return code

    def execute_transformation(self, code: str, input_grid: Grid) -> Optional[Grid]:
        """
        Execute transformation code on input grid.

        Args:
            code: Python code defining a transformation
            input_grid: Input grid to transform

        Returns:
            Transformed grid or None if execution failed
        """
        # Validate code first
        is_safe, error = self.validate_code(code)
        if not is_safe:
            return None

        # Create isolated namespace
        namespace = self.safe_namespace.copy()
        namespace["input_grid"] = copy.deepcopy(input_grid)
        namespace["grid"] = copy.deepcopy(input_grid)  # Alias

        try:
            # Execute the code
            exec(code, namespace)

            # Try to get the result in various ways
            result = None

            # 1. Look for a transform function and call it
            if "transform" in namespace and callable(namespace["transform"]):
                result = namespace["transform"](input_grid)
            elif "solve" in namespace and callable(namespace["solve"]):
                result = namespace["solve"](input_grid)
            elif "solution" in namespace and callable(namespace["solution"]):
                result = namespace["solution"](input_grid)
            elif "apply" in namespace and callable(namespace["apply"]):
                result = namespace["apply"](input_grid)

            # 2. Look for a result variable
            elif "result" in namespace:
                result = namespace["result"]
            elif "output" in namespace:
                result = namespace["output"]
            elif "output_grid" in namespace:
                result = namespace["output_grid"]

            # Validate result is a proper grid
            if result is not None:
                result = self._validate_result(result)

            return result

        except Exception as e:
            # Silently fail - code didn't work
            return None

    def _validate_result(self, result: Any) -> Optional[Grid]:
        """Validate that result is a proper grid."""
        if not isinstance(result, (list, np.ndarray)):
            return None

        # Convert numpy array to list
        if isinstance(result, np.ndarray):
            result = result.tolist()

        if not isinstance(result, list) or len(result) == 0:
            return None

        if not isinstance(result[0], list):
            return None

        # Check all cells are integers
        for row in result:
            if not isinstance(row, list):
                return None
            for cell in row:
                if isinstance(cell, (np.integer, np.floating)):
                    cell = int(cell)
                if not isinstance(cell, (int, float)):
                    return None

        # Convert all cells to int
        result = [[int(cell) for cell in row] for row in result]

        return result

    def try_multiple_variations(self, code: str, input_grid: Grid) -> List[Grid]:
        """
        Try multiple variations of executing the code.

        Sometimes the LLM generates code that needs slight modifications.

        Returns:
            List of valid results from different variations
        """
        results = []

        # Try original code
        result = self.execute_transformation(code, input_grid)
        if result:
            results.append(result)

        # Try wrapping in a transform function if not already
        if "def " not in code:
            wrapped = f"""
def transform(grid):
{chr(10).join('    ' + line for line in code.split(chr(10)))}
    return grid

result = transform(input_grid)
"""
            result = self.execute_transformation(wrapped, input_grid)
            if result:
                results.append(result)

        # Try with explicit return
        if "return" not in code and "result =" not in code:
            with_return = code + "\nresult = grid"
            result = self.execute_transformation(with_return, input_grid)
            if result:
                results.append(result)

        # Return unique results
        unique_results = []
        for res in results:
            if not any(self._grids_equal(res, ur) for ur in unique_results):
                unique_results.append(res)

        return unique_results

    def _grids_equal(self, g1: Grid, g2: Grid) -> bool:
        """Check if two grids are equal."""
        if len(g1) != len(g2):
            return False
        if len(g1[0]) != len(g2[0]):
            return False
        for i in range(len(g1)):
            for j in range(len(g1[0])):
                if g1[i][j] != g2[i][j]:
                    return False
        return True
