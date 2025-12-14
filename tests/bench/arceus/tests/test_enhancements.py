"""Test script to verify enhancements are working."""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from arceus.code_executor import SafeCodeExecutor
from arceus.primitives import ARCPrimitives


def test_code_executor():
    """Test that code executor can run simple transformations."""
    print("Testing SafeCodeExecutor...")

    executor = SafeCodeExecutor()

    # Test 1: Simple rotation
    test_grid = [[1, 2], [3, 4]]

    code1 = """
def transform(grid):
    import numpy as np
    arr = np.array(grid)
    result = np.rot90(arr, k=-1)
    return result.tolist()
"""

    result1 = executor.execute_transformation(code1, test_grid)
    print(f"✓ Test 1 (rotation): {result1}")
    assert result1 == [[3, 1], [4, 2]], f"Expected [[3, 1], [4, 2]], got {result1}"

    # Test 2: Using primitives (already in namespace)
    code2 = """
def transform(grid):
    # ARCPrimitives is already available in namespace
    return flip_horizontal(grid)
"""

    result2 = executor.execute_transformation(code2, test_grid)
    print(f"✓ Test 2 (primitive): {result2}")
    assert result2 == [[2, 1], [4, 3]], f"Expected [[2, 1], [4, 3]], got {result2}"

    # Test 3: Dangerous code (should fail safely)
    dangerous_code = """
def transform(grid):
    import os
    os.system('echo hacked')
    return grid
"""

    is_safe, error = executor.validate_code(dangerous_code)
    print(f"✓ Test 3 (dangerous code blocked): {is_safe=}, {error=}")
    assert not is_safe, "Dangerous code should be blocked"

    # Test 4: Multiple variations
    code4 = """
arr = np.array(grid)
result = np.rot90(arr)
"""

    results = executor.try_multiple_variations(code4, test_grid)
    print(f"✓ Test 4 (variations): Got {len(results)} result(s)")
    assert len(results) > 0, "Should find at least one valid variation"

    print("\n✅ All SafeCodeExecutor tests passed!")


def test_new_primitives():
    """Test that new primitives work correctly."""
    print("\nTesting new primitives...")

    test_grid = [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]

    # Test transpose
    transposed = ARCPrimitives.transpose(test_grid)
    print(f"✓ transpose: {transposed}")
    assert len(transposed) == 3 and len(transposed[0]) == 3

    # Test compress_grid
    sparse_grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    compressed = ARCPrimitives.compress_grid(sparse_grid)
    print(f"✓ compress_grid: {compressed}")
    assert compressed == [[1]], f"Expected [[1]], got {compressed}"

    # Test gravity_down
    falling_grid = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    fallen = ARCPrimitives.gravity_down(falling_grid)
    print(f"✓ gravity_down: {fallen}")
    assert fallen[2][1] == 1, "Object should fall to bottom"

    # Test extract_largest_object
    multi_obj = [
        [1, 0, 2, 2],
        [1, 0, 2, 2],
        [0, 0, 0, 0],
        [3, 0, 0, 0]
    ]
    largest = ARCPrimitives.extract_largest_object(multi_obj)
    print(f"✓ extract_largest_object: {largest}")
    assert len(largest) == 2 and len(largest[0]) == 2  # 2x2 object

    # Test overlay_grids
    grid1 = [[1, 0], [0, 1]]
    grid2 = [[0, 2], [2, 0]]
    overlaid = ARCPrimitives.overlay_grids(grid1, grid2, mode="or")
    print(f"✓ overlay_grids: {overlaid}")
    assert overlaid == [[1, 2], [2, 1]], f"Expected [[1, 2], [2, 1]], got {overlaid}"

    print("\n✅ All primitive tests passed!")


def test_solver_integration():
    """Test that solver can use code executor."""
    print("\nTesting solver integration...")

    from arceus.config import ArceusConfig
    from arceus.solver import ARCSolver

    config = ArceusConfig.from_env()
    config.enable_memory = False  # Don't need Honcho for this test

    solver = ARCSolver(config)

    # Check code executor is initialized
    assert solver.code_executor is not None, "Code executor should be initialized"
    print("✓ Solver has code executor")

    # Check that solve can access it
    assert hasattr(solver, 'code_executor'), "Solver should have code_executor attribute"
    print("✓ Solver can access code executor")

    print("\n✅ Solver integration test passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING ARCEUS ENHANCEMENTS")
    print("=" * 60)

    try:
        test_code_executor()
        test_new_primitives()
        test_solver_integration()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe enhancements are working correctly!")
        print("\nNext steps:")
        print("1. Start Honcho: python harness.py --port 5433")
        print("2. Set API key: export ANTHROPIC_API_KEY='...'")
        print("3. Run Arceus: python -m arceus.main --task-id 007bbfb7")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
