#!/usr/bin/env python3
"""Basic test to verify Arceus system components."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from arceus import config
        print("✓ config module imported")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False

    try:
        from arceus import primitives
        print("✓ primitives module imported")
    except Exception as e:
        print(f"✗ Failed to import primitives: {e}")
        return False

    try:
        from arceus import logger
        print("✓ logger module imported")
    except Exception as e:
        print(f"✗ Failed to import logger: {e}")
        return False

    try:
        from arceus import metrics
        print("✓ metrics module imported")
    except Exception as e:
        print(f"✗ Failed to import metrics: {e}")
        return False

    try:
        from arceus import solver
        print("✓ solver module imported")
    except Exception as e:
        print(f"✗ Failed to import solver: {e}")
        return False

    try:
        from arceus import tui
        print("✓ tui module imported")
    except Exception as e:
        print(f"✗ Failed to import tui: {e}")
        return False

    return True


def test_primitives():
    """Test that primitives work correctly."""
    print("\nTesting primitives...")

    from arceus.primitives import ARCPrimitives

    # Test rotation
    grid = [[1, 2], [3, 4]]
    rotated = ARCPrimitives.rotate_90(grid)
    expected = [[3, 1], [4, 2]]

    if rotated == expected:
        print("✓ rotate_90 works correctly")
    else:
        print(f"✗ rotate_90 failed: expected {expected}, got {rotated}")
        return False

    # Test flip
    flipped = ARCPrimitives.flip_horizontal(grid)
    expected = [[2, 1], [4, 3]]

    if flipped == expected:
        print("✓ flip_horizontal works correctly")
    else:
        print(f"✗ flip_horizontal failed: expected {expected}, got {flipped}")
        return False

    # Test color counting
    colors = ARCPrimitives.count_colors(grid)
    if colors == {1: 1, 2: 1, 3: 1, 4: 1}:
        print("✓ count_colors works correctly")
    else:
        print(f"✗ count_colors failed: got {colors}")
        return False

    return True


def test_config():
    """Test configuration."""
    print("\nTesting configuration...")

    from arceus.config import ArceusConfig

    config = ArceusConfig()
    print(f"✓ Config created with workspace: {config.workspace_id}")
    print(f"✓ Training path: {config.training_path}")
    print(f"✓ Evaluation path: {config.evaluation_path}")

    # Check if paths exist
    if config.training_path.exists():
        print(f"✓ Training path exists")
    else:
        print(f"✗ Training path does not exist: {config.training_path}")

    if config.evaluation_path.exists():
        print(f"✓ Evaluation path exists")
    else:
        print(f"✗ Evaluation path does not exist: {config.evaluation_path}")

    return True


def test_metrics():
    """Test metrics tracking."""
    print("\nTesting metrics...")

    from arceus.metrics import SolverMetrics

    metrics = SolverMetrics()
    metrics.task_id = "test_task"
    metrics.num_iterations = 5
    metrics.num_reasoning_steps = 10
    metrics.add_llm_call(1500.0, {"prompt_tokens": 100, "completion_tokens": 50})

    metrics_dict = metrics.to_dict()

    if metrics_dict["task_id"] == "test_task":
        print("✓ Metrics tracking works")
    else:
        print("✗ Metrics tracking failed")
        return False

    return True


def test_logger():
    """Test JSON logger."""
    print("\nTesting logger...")

    import tempfile
    from arceus.logger import JSONTraceLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JSONTraceLogger(Path(tmpdir), "test_task", enable=True)

        logger.log_event("test_event", {"key": "value"})
        logger.log_reasoning_step(1, "hypothesis", "Test hypothesis", 0.8)

        summary = logger.get_summary()

        if summary["total_events"] == 2:
            print("✓ Logger works correctly")
        else:
            print(f"✗ Logger failed: expected 2 events, got {summary['total_events']}")
            return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Arceus Basic Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Primitives", test_primitives),
        ("Configuration", test_config),
        ("Metrics", test_metrics),
        ("Logger", test_logger),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    return all(r for _, r in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
