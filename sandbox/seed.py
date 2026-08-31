"""Populate the sandbox with a known fixture and verify it actually landed.

Run through sandbox.sh, which supplies the base URL and owns the Docker and
Postgres side. This script only talks to the API:

    uv run --no-project --with-editable ./sdks/python python sandbox/seed.py

It is provider-agnostic. Under the mock provider the *text* of a conclusion is
synthetic and unrelated to the messages, so nothing here asserts on conclusion
content - only that derivation ran and produced something. What it does assert
strictly is the round trip that harness integrations actually get wrong: which
messages landed, on which peers, with which observation topology.

Exits non-zero on any failure. An empty sandbox that reports success is the exact
failure mode this whole thing exists to prevent.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from honcho import Honcho
from honcho.api_types import MessageCreateParams, SessionPeerConfig

FIXTURE = Path(__file__).with_name("fixture.json")

# Generous: a cold deriver on a real provider is slow, and the failure we care
# about (nothing is processing at all) shows up as a timeout either way.
DRAIN_TIMEOUT_SECONDS = float(os.environ.get("SANDBOX_DRAIN_TIMEOUT", "300"))
DRAIN_POLL_SECONDS = 0.5


class SeedError(RuntimeError):
    """Seeding did not reach a usable state."""


def log(message: str) -> None:
    print(f"[seed] {message}", flush=True)


def drain(honcho: Honcho, what: str) -> None:
    """Block until the deriver queue is empty, or fail loudly.

    Silence is the dangerous outcome here. A queue that never drains and a queue
    that never had anything queued look identical from the outside, so the timeout
    reports the last status it saw rather than just giving up.
    """
    deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
    last: Any = None
    while time.monotonic() < deadline:
        last = honcho.queue_status()
        outstanding = last.pending_work_units + last.in_progress_work_units
        if outstanding == 0 and last.completed_work_units > 0:
            log(f"drained after {what} ({last.completed_work_units} work units)")
            return
        time.sleep(DRAIN_POLL_SECONDS)

    raise SeedError(
        f"queue did not drain after {what} within {DRAIN_TIMEOUT_SECONDS:.0f}s. "
        f"Last status: {last}. "
        "If completed_work_units is 0, nothing was ever enqueued - check that the "
        "deriver is running and that the provider base URL resolves."
    )


def main() -> int:
    fixture = json.loads(FIXTURE.read_text())
    base_url = os.environ.get("SANDBOX_BASE_URL", "http://127.0.0.1:18000")
    workspace_id = fixture["workspace"]

    log(f"seeding workspace {workspace_id!r} at {base_url}")
    honcho = Honcho(base_url=base_url, workspace_id=workspace_id, api_key="sandbox")

    peers = {spec["id"]: honcho.peer(spec["id"]) for spec in fixture["peers"]}
    session = honcho.session(fixture["session"])

    session.add_peers(
        [
            (
                peers[spec["id"]],
                SessionPeerConfig(
                    observe_me=spec["observe_me"],
                    observe_others=spec["observe_others"],
                ),
            )
            for spec in fixture["peers"]
        ]
    )
    log(f"created {len(peers)} peers with explicit observation topology")

    session.add_messages(
        [
            MessageCreateParams(peer_id=msg["peer"], content=msg["content"])
            for msg in fixture["messages"]
        ]
    )
    log(f"posted {len(fixture['messages'])} messages")
    drain(honcho, "messages")

    for dream in fixture.get("dreams", []):
        honcho.schedule_dream(
            observer=dream["observer"],
            observed=dream["observed"],
            session=session,
        )
        log(f"scheduled dream: {dream['observer']} -> {dream['observed']}")
    if fixture.get("dreams"):
        drain(honcho, "dream")

    verify(honcho, session, peers, fixture)
    log("seed complete")
    return 0


def verify(
    honcho: Honcho,
    session: Any,
    peers: dict[str, Any],
    fixture: dict[str, Any],
) -> None:
    """Assert the fixture round-tripped. Raises SeedError on any mismatch."""
    stored = list(session.messages(size=100))
    if len(stored) != len(fixture["messages"]):
        raise SeedError(
            f"expected {len(fixture['messages'])} messages, found {len(stored)}"
        )

    expected_authors = sorted(msg["peer"] for msg in fixture["messages"])
    actual_authors = sorted(msg.peer_id for msg in stored)
    if actual_authors != expected_authors:
        raise SeedError(
            f"messages landed on the wrong peers: expected {expected_authors}, "
            f"got {actual_authors}"
        )
    log(f"verified {len(stored)} messages on the expected peers")

    # Topology is the thing that silently breaks, so it is checked against the
    # server's view rather than assumed from the create call.
    for spec in fixture["peers"]:
        config = session.get_peer_configuration(spec["id"])
        if (config.observe_me, config.observe_others) != (
            spec["observe_me"],
            spec["observe_others"],
        ):
            raise SeedError(
                f"peer {spec['id']!r} topology mismatch: expected "
                f"observe_me={spec['observe_me']} "
                f"observe_others={spec['observe_others']}, got {config}"
            )
    log("verified observation topology")

    # Presence only. Under the mock provider, conclusion text is synthetic and
    # says nothing about the messages, and the level mix is explicit-only because
    # the Dreamer specialists write via tool calls the mock never emits. Asserting
    # on either would pass in real mode and fail in mock mode.
    total = 0
    for observer_id, observer in peers.items():
        for observed_id in peers:
            found = len(list(observer.conclusions_of(observed_id).list(size=100)))
            if found:
                log(f"  {observer_id} -> {observed_id}: {found} conclusions")
            total += found

    if total == 0:
        raise SeedError(
            "no conclusions were derived. The queue drained, so the deriver ran and "
            "produced nothing - check the provider wiring "
            "(LLM_OPENAI_BASE_URL / EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL) "
            "and the deriver logs."
        )
    log(f"verified {total} conclusions present")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SeedError as exc:
        print(f"[seed] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
