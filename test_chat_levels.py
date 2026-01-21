#!/usr/bin/env python3
"""Test all 5 reasoning levels of .chat() against a loaded workspace."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000/v3"
WORKSPACE_ID = "three"
HEADERS = {"Authorization": "Bearer local"}

REASONING_LEVELS = ["minimal", "low", "medium", "high", "max"]


def chat(client: httpx.Client, peer_id: str, query: str, level: str) -> dict:
    """Call the chat endpoint with a specific reasoning level."""
    resp = client.post(
        f"{BASE_URL}/workspaces/{WORKSPACE_ID}/peers/{peer_id}/chat",
        json={
            "query": query,
            "reasoning_level": level,
        },
    )
    if resp.status_code >= 400:
        return {"error": f"{resp.status_code} {resp.text}"}
    return resp.json()


def save_results_to_file(
    results: dict,
    query: str,
    peer: str,
    levels: list[str],
    output_dir: str = "chat_levels_comparison",
) -> str:
    """Save results to a timestamped JSON file."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(output_dir) / f"chat_levels_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "workspace": WORKSPACE_ID,
            "peer": peer,
            "query": query,
            "levels_tested": levels,
        },
        "results": results,
    }

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    return str(filename)


def format_result_text(
    results: dict, query: str, peer: str, levels: list[str]
) -> str:
    """Format results as readable text."""
    lines = [
        f"Chat Levels Test Results",
        f"=" * 60,
        f"Timestamp: {datetime.now().isoformat()}",
        f"Workspace: {WORKSPACE_ID}",
        f"Peer: {peer}",
        f"Query: {query}",
        f"=" * 60,
        "",
    ]

    for level in levels:
        r = results[level]
        resp = r["response"]
        lines.append(f"--- Level: {level} ---")
        lines.append(f"Time: {r['elapsed_seconds']:.2f}s")

        if "error" in resp:
            lines.append(f"Error: {resp['error']}")
        else:
            content = resp.get("content", "")
            lines.append(f"Response ({len(content)} chars):")
            lines.append(content)

        lines.append("")

    # Summary table
    lines.append("=" * 60)
    lines.append("Summary:")
    lines.append(f"{'Level':<10} {'Time':>8} {'Chars':>8}")
    lines.append("-" * 30)

    for level in levels:
        r = results[level]
        resp = r["response"]
        time_str = f"{r['elapsed_seconds']:.2f}s"

        if "error" in resp:
            lines.append(f"{level:<10} {time_str:>8} {'ERROR':>8}")
        else:
            content_len = len(resp.get("content", ""))
            lines.append(f"{level:<10} {time_str:>8} {content_len:>8}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Test all 5 reasoning levels of .chat() against a loaded workspace."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What do you know about this person?",
        help="The query to send to the chat endpoint",
    )
    parser.add_argument(
        "--peer",
        "-p",
        type=str,
        default="Caroline",
        help="The peer ID to query (default: Caroline)",
    )
    parser.add_argument(
        "--levels",
        "-l",
        type=str,
        nargs="+",
        choices=REASONING_LEVELS,
        default=REASONING_LEVELS,
        help="Specific levels to test (default: all levels)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="chat_levels_comparison",
        help="Output directory for results (default: chat_levels_comparison)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file, only print to console",
    )
    args = parser.parse_args()

    client = httpx.Client(headers=HEADERS, timeout=None)

    print(f"Workspace: {WORKSPACE_ID}")
    print(f"Peer: {args.peer}")
    print(f"Query: {args.query}")
    print(f"Levels: {', '.join(args.levels)}")
    print("=" * 60)

    results = {}

    for level in args.levels:
        print(f"\n--- Level: {level} ---")
        start_time = time.time()

        result = chat(client, args.peer, args.query, level)
        elapsed = time.time() - start_time

        results[level] = {
            "elapsed_seconds": round(elapsed, 2),
            "response": result,
        }

        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            content = result.get("content", "")
            print(f"Time: {elapsed:.2f}s")
            print(f"Response ({len(content)} chars):")
            print(content[:500] + "..." if len(content) > 500 else content)

    client.close()

    print("\n" + "=" * 60)
    print("Summary:")
    for level in args.levels:
        r = results[level]
        if "error" in r["response"]:
            status = f"ERROR: {r['response']['error'][:50]}"
        else:
            content_len = len(r["response"].get("content", ""))
            status = f"{content_len} chars"
        print(f"  {level:8s}: {r['elapsed_seconds']:6.2f}s - {status}")

    # Save results to file
    if not args.no_save:
        # Save JSON
        json_file = save_results_to_file(
            results, args.query, args.peer, args.levels, args.output
        )
        print(f"\nResults saved to: {json_file}")

        # Save text version
        text_file = json_file.replace(".json", ".txt")
        text_output = format_result_text(results, args.query, args.peer, args.levels)
        with open(text_file, "w") as f:
            f.write(text_output)
        print(f"Text report saved to: {text_file}")


if __name__ == "__main__":
    main()
