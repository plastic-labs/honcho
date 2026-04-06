"""
Local LLM chatbot + Honcho memory.

Setup (free, runs on your machine):
  1. Install Ollama: https://ollama.com/download
  2. Pull a model:  ollama pull llama3.2:1b
  3. Keep Ollama running:
     - macOS: use the Ollama app (menu bar llama icon). If macOS asks to allow
       background activity, approve it; you can also enable it under System Settings
       → General → Login Items & Extensions so it starts when you log in.
     - Any OS: or run `ollama serve` in a terminal.

Optional .env (speed tuning):
  HONCHO_URL=http://127.0.0.1:8000
  OLLAMA_URL=http://127.0.0.1:11434
  OLLAMA_MODEL=llama3.2:1b       # default; override for a larger model if you prefer
  CHAT_HISTORY_MAX_MESSAGES=20   # fewer past turns = less prompt work per reply
  HONCHO_CONTEXT_MAX_CHARS=6000  # cap long Honcho representation/card text
  OLLAMA_NUM_PREDICT=0          # default: no cap (full replies). Set e.g. 512 for shorter/faster answers
  OLLAMA_NUM_CTX=4096            # optional; smaller context can speed CPU inference
  OLLAMA_KEEP_ALIVE=10m          # keep model loaded between turns (avoids reload delay)
  CHAT_TURN_MAX_CHARS=3000       # cap each prior message body in the prompt (0 = no cap)
"""

import os
import re

import httpx
from dotenv import load_dotenv
from honcho import Honcho
from honcho.api_types import PeerContextResponse

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "20"))
HONCHO_CONTEXT_MAX_CHARS = int(os.getenv("HONCHO_CONTEXT_MAX_CHARS", "6000"))


def _env_positive_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or str(default)).strip()
    if not raw:
        return 0
    return max(0, int(raw))


OLLAMA_NUM_PREDICT = _env_positive_int("OLLAMA_NUM_PREDICT", 0)
OLLAMA_NUM_CTX = _env_positive_int("OLLAMA_NUM_CTX", 0)
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "10m").strip()
CHAT_TURN_MAX_CHARS = int(os.getenv("CHAT_TURN_MAX_CHARS", "3000"))

honcho_client = Honcho(
    base_url=os.getenv("HONCHO_URL", "http://127.0.0.1:8000"),
    workspace_id=os.getenv("HONCHO_WORKSPACE_ID", "chatbot-workspace"),
)

assistant = honcho_client.peer("assistant")


def normalize_user_id(raw: str) -> str:
    """Turn free-text input into a single peer/session id (no spaces)."""
    s = raw.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9_-]+", "", s)
    s = s.strip("-_") or "user"
    return s


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 24].rstrip() + "\n\n… [truncated]"


def _honcho_messages_to_ollama_turns(
    session,
    *,
    user_peer_id: str,
    assistant_peer_id: str,
    limit: int,
    turn_max_chars: int,
) -> list[dict[str, str]]:
    """
    Load the most recent `limit` messages from Honcho and map them to Ollama
    user/assistant turns in chronological order.
    """
    page = session.messages(size=limit, reverse=True)
    # Newest-first page → reverse to oldest-first for the LLM
    stored = list(reversed(page.items))
    turns: list[dict[str, str]] = []
    for m in stored:
        body = _truncate_text(m.content, turn_max_chars) if turn_max_chars else m.content
        if m.peer_id == user_peer_id:
            turns.append({"role": "user", "content": body})
        elif m.peer_id == assistant_peer_id:
            turns.append({"role": "assistant", "content": body})
    return turns


def _ollama_chat(
    system: str,
    messages: list[dict[str, str]],
    *,
    timeout: float = 120.0,
) -> str:
    """
    Call Ollama's /api/chat and return the full assistant text in one shot.

    No streaming: Ollama JSON body uses stream=false. Module-level httpx.post
    loads the full response body before we parse JSON (do not pass httpx stream=).
    """
    payload = [{"role": "system", "content": system}, *messages]
    body: dict[str, object] = {
        "model": OLLAMA_MODEL,
        "messages": payload,
        "stream": False,
    }
    opts: dict[str, int] = {}
    if OLLAMA_NUM_PREDICT > 0:
        opts["num_predict"] = OLLAMA_NUM_PREDICT
    if OLLAMA_NUM_CTX > 0:
        opts["num_ctx"] = OLLAMA_NUM_CTX
    if opts:
        body["options"] = opts
    if OLLAMA_KEEP_ALIVE:
        body["keep_alive"] = OLLAMA_KEEP_ALIVE

    try:
        r = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json=body,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            "Open the Ollama app (macOS menu bar) or run `ollama serve`. "
            "On macOS, check Login Items & Extensions if background mode was disabled."
        ) from e

    if r.status_code >= 400:
        raise RuntimeError(
            f"Ollama error ({r.status_code}): {r.text.strip() or r.reason_phrase}. "
            f"Try: ollama pull {OLLAMA_MODEL}"
        )

    ct = (r.headers.get("content-type") or "").lower()
    if "application/x-ndjson" in ct or "ndjson" in ct:
        raise RuntimeError(
            "Ollama returned a streaming (NDJSON) response; expected a single JSON "
            "object. Ensure the request body sets stream: false."
        )

    data = r.json()
    msg = data.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"Unexpected Ollama response: {data!r}")
    return str(content)


def _format_peer_context(ctx: PeerContextResponse) -> str:
    parts: list[str] = []
    if ctx.representation:
        parts.append(ctx.representation)
    if ctx.peer_card:
        parts.append("\n".join(ctx.peer_card))
    return "\n\n".join(parts) if parts else "(No memory yet.)"


def chat(user_id: str, message: str) -> str:
    user_peer = honcho_client.peer(id=user_id)
    session = honcho_client.session(id=user_id)

    ctx = assistant.context(target=user_peer)
    context = _truncate_text(_format_peer_context(ctx), HONCHO_CONTEXT_MAX_CHARS)

    system = f"""You are a helpful assistant that remembers users.

What you know about this user (from longer-term memory):
{context}

You also see the recent conversation below; use it to stay consistent with what was said earlier in this chat."""

    prior_turns = _honcho_messages_to_ollama_turns(
        session,
        user_peer_id=user_peer.id,
        assistant_peer_id=assistant.id,
        limit=CHAT_HISTORY_MAX_MESSAGES,
        turn_max_chars=CHAT_TURN_MAX_CHARS,
    )
    user_for_llm = (
        _truncate_text(message, CHAT_TURN_MAX_CHARS)
        if CHAT_TURN_MAX_CHARS > 0
        else message
    )
    ollama_messages = [*prior_turns, {"role": "user", "content": user_for_llm}]

    assistant_reply = _ollama_chat(system, ollama_messages)

    session.add_messages(
        [
            user_peer.message(message),
            assistant.message(assistant_reply),
        ]
    )

    return assistant_reply


if __name__ == "__main__":
    print(f"Local model: {OLLAMA_MODEL} @ {OLLAMA_URL}\n")

    raw_name = input("Enter your name: ").strip()
    user_id = normalize_user_id(raw_name)
    print(f"Hello! Chatting as id `{user_id}` (no spaces). Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        try:
            response = chat(user_id, user_input)
        except RuntimeError as e:
            print(f"{e}\n")
            continue
        # Full reply only after Ollama returns (no streaming).
        print(f"Assistant: {response}\n")
