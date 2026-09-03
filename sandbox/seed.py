"""Populate the sandbox with a known fixture and verify it actually landed.

Run through sandbox.sh, which supplies the base URL and owns the Docker and
Postgres side. This script only talks to the API:

    uv run --no-project --with-editable ./sdks/python python sandbox/seed.py verify

Two phases, because seeded conclusions are written between them by
inject_conclusions.py, which sandbox.sh runs inside the api container:

    seed.py seed      peers, messages, derivation
    seed.py verify    the whole round trip, seeded conclusions included

The split is what keeps the deriver honest. The check that proves derivation ran
at all is "some conclusion exists", and once conclusions are seeded that would
pass against a completely dead deriver - so it runs in the seed phase, before
anything is injected.

It is provider-agnostic. Under the mock provider the *text* of a derived
conclusion is synthetic and unrelated to the messages, so nothing here asserts on
derived content. Seeded conclusions are the opposite: their text is committed, so
they are asserted exactly, in both modes. What is asserted strictly either way is
the round trip that harness integrations actually get wrong: which messages
landed, on which peers, with which observation topology.

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


LEVELS = ("explicit", "deductive", "inductive")


def seeded_conclusions(
    fixture: dict[str, Any],
) -> dict[tuple[str, str], dict[str, list[str]]]:
    """What inject_conclusions.py should have written, keyed by (observer, observed).

    Mirrors the injector's resolution rules so verification is stated in terms of the
    fixture rather than of whatever happens to be in the database.
    """
    peers = fixture["peers"]
    planned: dict[tuple[str, str], dict[str, list[str]]] = {}
    for spec in peers:
        by_level = {
            level: [
                item if isinstance(item, str) else item["content"]
                for item in spec.get(level, [])
            ]
            for level in LEVELS
        }
        if not any(by_level.values()):
            continue
        override = spec.get("observer")
        observers = (
            [override]
            if override
            else [
                peer["id"]
                for peer in peers
                if peer.get("observe_others") and peer["id"] != spec["id"]
            ]
        )
        for observer in observers:
            planned[(observer, spec["id"])] = by_level
    return planned


def cited_premise_texts(spec: dict[str, Any], explicit: list[str]) -> set[str]:
    """The explicit conclusions this peer's derived conclusions name as premises."""
    return {
        explicit[index]
        for level in ("deductive", "inductive")
        for item in spec.get(level, [])
        if isinstance(item, dict)
        for index in item.get("premises", [])
    }


def count_conclusions(peers: dict[str, Any]) -> int:
    total = 0
    for observer in peers.values():
        for observed_id in peers:
            total += len(list(observer.conclusions_of(observed_id).list(size=100)))
    return total


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
    phase = sys.argv[1] if len(sys.argv) > 1 else "seed"
    if phase not in ("seed", "verify"):
        raise SeedError(f"unknown phase {phase!r} (expected 'seed' or 'verify')")

    fixture = json.loads(FIXTURE.read_text())
    base_url = os.environ.get("SANDBOX_BASE_URL", "http://127.0.0.1:18000")
    workspace_id = fixture["workspace"]

    honcho = Honcho(base_url=base_url, workspace_id=workspace_id, api_key="sandbox")
    peers = {spec["id"]: honcho.peer(spec["id"]) for spec in fixture["peers"]}
    session = honcho.session(fixture["session"])

    if phase == "verify":
        verify(honcho, session, peers, fixture)
        log("verify complete")
        return 0

    log(f"seeding workspace {workspace_id!r} at {base_url}")

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

    # The one check that proves derivation happened, made here rather than in verify
    # because seeded conclusions land afterwards and would satisfy it on their own.
    derived = count_conclusions(peers)
    if derived == 0:
        raise SeedError(
            "no conclusions were derived. The queue drained, so the deriver ran and "
            "produced nothing - check the provider wiring "
            "(LLM_OPENAI_BASE_URL / EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL) "
            "and the deriver logs."
        )
    log(f"verified {derived} derived conclusions present")
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

    # Derived conclusions get presence only. Under the mock provider their text is
    # synthetic and says nothing about the messages, and the level mix is
    # explicit-only because the Dreamer specialists write via tool calls the mock
    # never emits. Asserting on either would pass in real mode and fail in mock mode.
    total = 0
    for observer_id, observer in peers.items():
        for observed_id in peers:
            found = len(list(observer.conclusions_of(observed_id).list(size=100)))
            if found:
                log(f"  {observer_id} -> {observed_id}: {found} conclusions")
            total += found

    if total == 0:
        raise SeedError(
            "no conclusions at all. Both derivation and injection produced nothing - "
            "check the provider wiring "
            "(LLM_OPENAI_BASE_URL / EMBEDDING_MODEL_CONFIG__OVERRIDES__BASE_URL) "
            "and the deriver logs."
        )
    log(f"verified {total} conclusions present")

    verify_seeded(peers, fixture, total)


def verify_seeded(
    peers: dict[str, Any],
    fixture: dict[str, Any],
    total: int,
) -> None:
    """Assert the seeded conclusions landed exactly, level by level.

    Seeded text is committed, so unlike derived conclusions it is asserted by content
    in both modes - which is the whole reason the fixture carries these keys.
    """
    planned = seeded_conclusions(fixture)
    if not planned:
        log("fixture declares no conclusions to seed")
        return

    seeded_total = 0
    for (observer_id, observed_id), by_level in planned.items():
        stored = list(peers[observer_id].conclusions_of(observed_id).list(size=100))
        for level, expected in by_level.items():
            if not expected:
                continue
            actual = {c.content for c in stored if c.level == level}
            missing = [content for content in expected if content not in actual]
            if missing:
                raise SeedError(
                    f"{observer_id} -> {observed_id}: {len(missing)} seeded {level} "
                    f"conclusion(s) missing, first is {missing[0]!r}. Honcho collapses "
                    "conclusions whose content matches something already stored, so "
                    "check for near-duplicate text in the fixture."
                )
            seeded_total += len(expected)
        summary = " ".join(f"{level}={len(by_level[level])}" for level in LEVELS)
        log(f"verified seeded {observer_id} -> {observed_id}: {summary}")

    # Derived and seeded conclusions must both be present. The seed phase already
    # proved derivation ran against an un-injected database; this catches the reverse
    # error of an injection that somehow replaced the derived rows.
    if total <= seeded_total:
        raise SeedError(
            f"found {total} conclusions but {seeded_total} were seeded, leaving none "
            "derived. Injection should add to the deriver's output, not replace it."
        )
    log(
        f"verified {total - seeded_total} derived conclusions alongside {seeded_total} seeded"
    )

    # Premise text is what the working representation renders for a derived
    # conclusion, and it is the only part of the reasoning tree an API client can
    # see - schemas.Conclusion does not expose source_ids.
    by_peer = {spec["id"]: spec for spec in fixture["peers"]}
    for (observer_id, observed_id), by_level in planned.items():
        premise_texts = cited_premise_texts(by_peer[observed_id], by_level["explicit"])
        if not premise_texts:
            continue
        rendered = peers[observer_id].representation(
            target=observed_id, max_conclusions=100
        )
        absent = [text for text in premise_texts if text not in rendered]
        if absent:
            raise SeedError(
                f"{observer_id} -> {observed_id}: premise text missing from the "
                f"representation, first is {absent[0]!r}. Premises render only for "
                "deductive conclusions and sources only for inductive ones, so a "
                "premise stored under the wrong metadata key renders as nothing."
            )
        log(
            f"verified premise text renders in {observer_id}'s representation of {observed_id}"
        )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SeedError as exc:
        print(f"[seed] FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
