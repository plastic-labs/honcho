#!/usr/bin/env python3
"""Convert a JSONL file to a JSON array."""

import json
import sys


def main() -> None:
    """Parse command-line arguments and convert JSONL file to JSON array.

    Reads a JSONL file where each line is a valid JSON object, aggregates
    all records into a list, and outputs as a formatted JSON array to stdout.

    Exits with code 1 if arguments are invalid.
    """
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.jsonl>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
