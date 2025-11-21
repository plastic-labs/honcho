#!/usr/bin/env python
import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from tests.unified.runner import UnifiedTestRunner


async def main():
    parser = argparse.ArgumentParser(description="Run Unified Honcho Tests")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="tests/unified/test_cases",
        help="Directory containing JSON test files",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to a single JSON test file to run",
    )
    parser.add_argument(
        "--port", type=int, default=9000, help="DB port for the harness"
    )
    parser.add_argument(
        "--api-port", type=int, default=9001, help="API port for the harness"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port for the harness"
    )

    args = parser.parse_args()

    # Validate mutually exclusive args
    if args.test_file and args.test_dir != "tests/unified/test_cases":
        print("Error: Cannot specify both --test-file and --test-dir")
        sys.exit(1)

    if args.test_file:
        test_path = Path(args.test_file)
        if not test_path.exists():
            print(f"Error: Test file {test_path} does not exist.")
            sys.exit(1)
        if not test_path.is_file():
            print(f"Error: {test_path} is not a file.")
            sys.exit(1)

        runner = UnifiedTestRunner(
            test_file=test_path,
            honcho_port=args.port,
            api_port=args.api_port,
            redis_port=args.redis_port,
        )
    else:
        test_dir = Path(args.test_dir)
        if not test_dir.exists():
            print(f"Error: Directory {test_dir} does not exist.")
            sys.exit(1)

        runner = UnifiedTestRunner(
            tests_dir=test_dir,
            honcho_port=args.port,
            api_port=args.api_port,
            redis_port=args.redis_port,
        )

    await runner.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
