#!/usr/bin/env python3
"""Load a locomo dataset into Honcho and test with configurable query and reasoning level."""

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

# Use environment variables with defaults matching .env.template
BASE_URL = os.getenv("HONCHO_BASE_URL")
REASONING_LEVELS = ["minimal", "low", "medium", "high", "max"]


def parse_datetime(dt_string: str) -> datetime:
    """Parse datetime string like '1:56 pm on 8 May, 2023' into datetime object."""
    parts = dt_string.split(" on ")
    time_part = parts[0]
    date_part = parts[1]

    time_obj = datetime.strptime(time_part, "%I:%M %p")
    date_obj = datetime.strptime(date_part, "%d %B, %Y")

    return datetime(
        year=date_obj.year,
        month=date_obj.month,
        day=date_obj.day,
        hour=time_obj.hour,
        minute=time_obj.minute,
        tzinfo=timezone.utc,
    )


def load_locomo(client: httpx.Client, filepath: str, workspace_id: str) -> tuple[str, str]:
    """Load locomo dataset into Honcho. Returns (speaker_a, speaker_b)."""
    with open(filepath) as f:
        data = json.load(f)

    convo = data[0]["conversation"]
    speaker_a = convo["speaker_a"]
    speaker_b = convo["speaker_b"]

    print(f"Loading conversation between {speaker_a} and {speaker_b}")

    # Create workspace
    resp = client.post(f"{BASE_URL}/workspaces", json={"id": workspace_id})
    if resp.status_code >= 400:
        print(f"Failed to create workspace: {resp.status_code} {resp.text}")
        return "", ""
    print(f"Created workspace: {workspace_id}")

    # Create peers
    resp = client.post(f"{BASE_URL}/workspaces/{workspace_id}/peers", json={"id": speaker_a})
    if resp.status_code >= 400:
        print(f"Failed to create peer {speaker_a}: {resp.status_code} {resp.text}")
        return "", ""
    resp = client.post(f"{BASE_URL}/workspaces/{workspace_id}/peers", json={"id": speaker_b})
    if resp.status_code >= 400:
        print(f"Failed to create peer {speaker_b}: {resp.status_code} {resp.text}")
        return "", ""
    print(f"Created peers: {speaker_a}, {speaker_b}")

    session_num = 1
    while f"session_{session_num}" in convo:
        session_key = f"session_{session_num}"
        datetime_key = f"session_{session_num}_date_time"

        messages = convo[session_key]
        base_time = parse_datetime(convo[datetime_key])

        print(f"\n--- Session {session_num}: {convo[datetime_key]} ---")
        print(f"  {len(messages)} messages")

        session_id = f"locomo_session_{session_num}"

        # Create session
        resp = client.post(
            f"{BASE_URL}/workspaces/{workspace_id}/sessions",
            json={"id": session_id},
        )
        if resp.status_code >= 400:
            print(f"Failed to create session {session_id}: {resp.status_code} {resp.text}")
            return "", ""

        # Add peers to session
        resp = client.post(
            f"{BASE_URL}/workspaces/{workspace_id}/sessions/{session_id}/peers",
            json={speaker_a: {}, speaker_b: {}},
        )
        if resp.status_code >= 400:
            print(f"Failed to add peers to session: {resp.status_code} {resp.text}")
            return "", ""
        print(f"  Created session: {session_id}")

        # Build message batch
        msg_batch = []
        for i, msg in enumerate(messages):
            msg_time = base_time + timedelta(seconds=i * 2)
            msg_batch.append({
                "peer_id": msg["speaker"],
                "content": msg["text"],
                "created_at": msg_time.isoformat(),
            })

        # Create messages
        resp = client.post(
            f"{BASE_URL}/workspaces/{workspace_id}/sessions/{session_id}/messages",
            json={"messages": msg_batch},
        )
        if resp.status_code >= 400:
            print(f"Failed to create messages: {resp.status_code} {resp.text}")
            return "", ""
        print(f"  Loaded {len(messages)} messages")
        session_num += 1

    print(f"\nDone! Loaded {session_num - 1} sessions.")
    return speaker_a, speaker_b


def chat(client: httpx.Client, workspace_id: str, peer_id: str, query: str, level: str) -> dict:
    """Call the chat endpoint with a specific reasoning level."""
    resp = client.post(
        f"{BASE_URL}/workspaces/{workspace_id}/peers/{peer_id}/chat",
        json={
            "query": query,
            "reasoning_level": level,
        },
    )
    if resp.status_code >= 400:
        return {"error": f"{resp.status_code} {resp.text}"}
    return resp.json()


def main():
    parser = argparse.ArgumentParser(
        description="Load a locomo dataset into Honcho and test with a query."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the locomo JSON file",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=None,
        help="Workspace ID (default: auto-generated from timestamp)",
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
        default=None,
        help="The peer ID to query (default: first speaker from dataset)",
    )
    parser.add_argument(
        "--level",
        "-l",
        type=str,
        choices=REASONING_LEVELS,
        default="medium",
        help="Reasoning level to use (default: medium)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip loading data, just run the query (requires --workspace and --peer)",
    )
    args = parser.parse_args()

    # Generate workspace ID if not provided
    workspace_id = args.workspace or f"locomo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    client = httpx.Client(headers=HEADERS, timeout=None)

    if args.skip_load:
        if not args.workspace or not args.peer:
            print("Error: --skip-load requires --workspace and --peer to be specified")
            return
        speaker_a = args.peer
    else:
        # Load the dataset
        speaker_a, speaker_b = load_locomo(client, args.filepath, workspace_id)
        if not speaker_a:
            return

        print(f"\nPeers available: {speaker_a}, {speaker_b}")

    # Determine which peer to query
    peer_id = args.peer or speaker_a

    print("\n" + "=" * 60)
    print(f"Testing chat endpoint")
    print(f"=" * 60)
    print(f"Workspace: {workspace_id}")
    print(f"Peer: {peer_id}")
    print(f"Query: {args.query}")
    print(f"Level: {args.level}")
    print("=" * 60)

    start_time = time.time()
    result = chat(client, workspace_id, peer_id, args.query, args.level)
    elapsed = time.time() - start_time

    print(f"\nTime: {elapsed:.2f}s")

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        content = result.get("content", "")
        print(f"\nResponse ({len(content)} chars):")
        print("-" * 60)
        print(content)
        print("-" * 60)

    client.close()


if __name__ == "__main__":
    main()
