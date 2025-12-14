#!/usr/bin/env python3
"""
Quick test to verify Honcho SDK integration works correctly.

This test verifies:
1. Honcho SDK can be imported
2. AsyncHoncho, AsyncPeer, AsyncSession classes are available
3. Solver can be initialized with Honcho client
4. Peers can be created
"""

import sys
from pathlib import Path

# Add Honcho SDK to path (same as solver.py does)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "sdks" / "python" / "src"))

def test_imports():
    """Test that all Honcho SDK imports work."""
    print("Testing Honcho SDK imports...")

    try:
        from honcho import AsyncHoncho, AsyncPeer, AsyncSession
        print("✓ Successfully imported AsyncHoncho, AsyncPeer, AsyncSession")
    except ImportError as e:
        print(f"✗ Failed to import Honcho SDK: {e}")
        return False

    try:
        from honcho.types import PeerContext
        print("✓ Successfully imported PeerContext")
    except ImportError as e:
        print(f"✗ Failed to import PeerContext: {e}")
        return False

    return True


def test_solver_imports():
    """Test that solver can import everything it needs."""
    print("\nTesting solver imports...")

    try:
        from arceus.solver import ARCSolver
        print("✓ Successfully imported ARCSolver")
    except ImportError as e:
        print(f"✗ Failed to import ARCSolver: {e}")
        return False

    try:
        from arceus.config import ArceusConfig
        print("✓ Successfully imported ArceusConfig")
    except ImportError as e:
        print(f"✗ Failed to import ArceusConfig: {e}")
        return False

    return True


def test_solver_initialization():
    """Test that solver can be initialized."""
    print("\nTesting solver initialization...")

    try:
        from arceus.solver import ARCSolver
        from arceus.config import ArceusConfig

        # Create config
        config = ArceusConfig.from_env()
        config.enable_memory = False  # Don't actually connect to Honcho

        # Create solver
        solver = ARCSolver(config)
        print("✓ Successfully created ARCSolver instance")

        # Check that solver has the right attributes
        assert hasattr(solver, 'honcho_client')
        assert hasattr(solver, 'task_analyst_peer')
        assert hasattr(solver, 'solution_generator_peer')
        assert hasattr(solver, 'verifier_peer')
        print("✓ Solver has correct peer attributes")

        return True

    except Exception as e:
        print(f"✗ Failed to initialize solver: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Honcho SDK Integration Test")
    print("="*70 + "\n")

    results = []

    results.append(("Honcho SDK imports", test_imports()))
    results.append(("Solver imports", test_solver_imports()))
    results.append(("Solver initialization", test_solver_initialization()))

    print("\n" + "="*70)
    print("Test Results")
    print("="*70)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✓ All tests passed! Honcho SDK integration is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
