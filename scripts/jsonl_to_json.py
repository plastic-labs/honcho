#!/usr/bin/env python3
"""Convert a JSONL file to a JSON array."""

import json
import sys


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.jsonl>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
