"""Generate canonical JSON fixtures for Rust parity tests.

Uses the honcho Python SDK's SessionContext.to_openai / to_anthropic to
produce reference output that the Rust SDK must match.

Run from the python SDK directory:
    cd sdks/python && uv run python ../rust/tests/fixtures/parity/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from honcho.session_context import SessionContext, Summary
from honcho.message import Message

FIXTURES_DIR = Path(__file__).parent


def _msg(
    id: str,
    content: str,
    peer_id: str,
    session_id: str = "sess_1",
    workspace_id: str = "ws_1",
) -> Message:
    return Message(
        id=id,
        content=content,
        peer_id=peer_id,
        session_id=session_id,
        workspace_id=workspace_id,
        metadata={},
        created_at="2025-01-15T10:30:00Z",
        token_count=1,
    )


def _msg_dict(m: Message) -> dict:
    return {
        "id": m.id,
        "content": m.content,
        "peer_id": m.peer_id,
        "session_id": m.session_id,
        "workspace_id": m.workspace_id,
        "metadata": m.metadata,
        "created_at": m.created_at,
        "token_count": m.token_count,
    }


def _ctx_dict(ctx: SessionContext) -> dict:
    d: dict = {
        "id": ctx.session_id,
        "messages": [_msg_dict(m) for m in ctx.messages],
    }
    if ctx.summary is not None:
        d["summary"] = {
            "content": ctx.summary.content,
            "message_id": ctx.summary.message_id,
            "summary_type": ctx.summary.summary_type,
            "created_at": ctx.summary.created_at,
            "token_count": ctx.summary.token_count,
        }
    if ctx.peer_representation is not None:
        d["peer_representation"] = ctx.peer_representation
    if ctx.peer_card is not None:
        d["peer_card"] = ctx.peer_card
    return d


def _write(name: str, data: object) -> None:
    path = FIXTURES_DIR / name
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  wrote {path}")


def generate_small() -> None:
    ctx = SessionContext(
        session_id="sess_small",
        messages=[
            _msg("m1", "hello", "user1", "sess_small"),
            _msg("m2", "hi there", "assistant", "sess_small"),
            _msg("m3", "how are you?", "user1", "sess_small"),
        ],
    )
    _write("small/session_context.json", _ctx_dict(ctx))
    _write("small/openai.json", ctx.to_openai(assistant="assistant"))
    _write("small/anthropic.json", ctx.to_anthropic(assistant="assistant"))


def generate_multi_peer() -> None:
    ctx = SessionContext(
        session_id="sess_multi",
        messages=[
            _msg("m1", "what is rust?", "alice", "sess_multi"),
            _msg("m2", "rust is a systems language", "bot", "sess_multi"),
            _msg("m3", "tell me more", "bob", "sess_multi"),
            _msg("m4", "it has zero-cost abstractions", "bot", "sess_multi"),
            _msg("m5", "sounds great", "alice", "sess_multi"),
        ],
    )
    _write("multi_peer/session_context.json", _ctx_dict(ctx))
    _write("multi_peer/openai.json", ctx.to_openai(assistant="bot"))
    _write("multi_peer/anthropic.json", ctx.to_anthropic(assistant="bot"))


def generate_with_summary() -> None:
    ctx = SessionContext(
        session_id="sess_summary",
        messages=[
            _msg("m1", "hello", "user1", "sess_summary"),
            _msg("m2", "hi", "assistant", "sess_summary"),
        ],
        summary=Summary(
            content="A brief greeting exchange.",
            message_id="m0",
            summary_type="short",
            created_at="2025-01-15T10:30:00Z",
            token_count=5,
        ),
        peer_representation="User is polite and brief.",
        peer_card=["friendly", "concise"],
    )
    _write("with_summary/session_context.json", _ctx_dict(ctx))
    _write("with_summary/openai.json", ctx.to_openai(assistant="assistant"))
    _write("with_summary/anthropic.json", ctx.to_anthropic(assistant="assistant"))


def main() -> None:
    print("Generating parity fixtures...")
    generate_small()
    generate_multi_peer()
    generate_with_summary()
    print("Done.")
