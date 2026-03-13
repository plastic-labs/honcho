#!/usr/bin/env python3
"""Load Granola meeting notes into Honcho.

Uses the Granola MCP server (with OAuth) to fetch meetings and the Honcho Python SDK
to store them. Each meeting becomes a Honcho session. Two-person meetings get full
speaker attribution; multi-person meetings are stored as summaries.

Prerequisites:
    pip install honcho-ai httpx

Environment Variables:
    HONCHO_API_KEY - Your Honcho API key (get from app.honcho.dev/api-keys)

Usage:
    python honcho_granola.py
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx


@dataclass
class Participant:
    name: str
    email: str | None = None
    org: str | None = None


@dataclass
class ParsedParticipants:
    note_creator: Participant | None = None
    others: list[Participant] = field(default_factory=list)


@dataclass
class TranscriptTurn:
    speaker: str
    text: str


# Granola MCP + OAuth endpoints
GRANOLA_MCP_URL = "https://mcp.granola.ai/mcp"
AUTH_BASE = "https://mcp-auth.granola.ai"
OAUTH_REDIRECT_PORT = 8765
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_REDIRECT_PORT}/callback"

# Honcho message size limit (25000 max, leave headroom)
MAX_MESSAGE_LEN = 24000


# ---------------------------------------------------------------------------
# OAuth callback handler (must be a class for BaseHTTPRequestHandler)
# ---------------------------------------------------------------------------

class _OAuthCallback(BaseHTTPRequestHandler):
    auth_result: dict[str, str | None] = {"code": None, "error": None}

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        if "code" in params:
            _OAuthCallback.auth_result["code"] = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authenticated! You can close this window.</h1>")
        elif "error" in params:
            _OAuthCallback.auth_result["error"] = params.get("error_description", params["error"])[0]
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(f"<h1>Error: {_OAuthCallback.auth_result['error']}</h1>".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


# ---------------------------------------------------------------------------
# Granola OAuth + MCP
# ---------------------------------------------------------------------------

async def authenticate(http_client: httpx.AsyncClient) -> str:
    """Perform OAuth (DCR + PKCE) with Granola. Returns access token."""
    _OAuthCallback.auth_result = {"code": None, "error": None}

    print("\nAuthenticating with Granola...")

    # Register client (DCR)
    resp = await http_client.post(
        f"{AUTH_BASE}/oauth2/register",
        json={
            "client_name": "Granola to Honcho Transfer",
            "redirect_uris": [OAUTH_REDIRECT_URI],
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        },
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Client registration failed: {resp.status_code}")
    client_id = resp.json().get("client_id")

    # PKCE
    verifier = secrets.token_urlsafe(32)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

    # Browser auth
    auth_url = f"{AUTH_BASE}/oauth2/authorize?" + urlencode({
        "client_id": client_id,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "response_type": "code",
        "state": "granola-honcho-transfer",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })

    server = HTTPServer(("localhost", OAUTH_REDIRECT_PORT), _OAuthCallback)
    thread = threading.Thread(target=server.handle_request)
    thread.start()

    print("  Opening browser for authentication...")
    webbrowser.open(auth_url)
    thread.join(timeout=120)
    server.server_close()

    auth_result = _OAuthCallback.auth_result
    if auth_result["error"]:
        raise RuntimeError(f"Authentication failed: {auth_result['error']}")
    if not auth_result["code"]:
        raise RuntimeError("Authentication timed out")

    # Exchange code for token
    resp = await http_client.post(
        f"{AUTH_BASE}/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": auth_result["code"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {resp.status_code}")

    print("  Authenticated successfully!")
    return resp.json()["access_token"]


async def call_mcp_tool(
    http_client: httpx.AsyncClient,
    access_token: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call a Granola MCP tool, handling both JSON and SSE responses."""
    resp = await http_client.post(
        GRANOLA_MCP_URL,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        },
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"MCP call failed: {resp.status_code} - {resp.text}")

    # SSE response
    if "text/event-stream" in resp.headers.get("content-type", ""):
        result = None
        for line in resp.text.split("\n"):
            if line.strip().startswith("data: "):
                try:
                    parsed = json.loads(line.strip()[6:])
                    if "result" in parsed:
                        result = parsed
                    elif "error" in parsed:
                        raise RuntimeError(f"MCP error: {parsed['error']}")
                except json.JSONDecodeError:
                    continue
        if result:
            final = result.get("result", {})
            return final if isinstance(final, dict) else {"result": final}
        raise RuntimeError("No result in SSE response")

    # JSON response
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"MCP error: {result['error']}")
    return result.get("result", {})


def extract_mcp_text(result: dict[str, Any]) -> str:
    """Extract text from the first content block of an MCP result.

    Raises ValueError if the response structure is unexpected.
    """
    content = result.get("content", [])
    if not isinstance(content, list) or not content:
        raise ValueError(f"MCP response missing content array: {list(result.keys())}")
    first = content[0]
    if not isinstance(first, dict) or "text" not in first:
        raise ValueError(f"MCP content block missing 'text' field: {first}")
    return str(first["text"])


# ---------------------------------------------------------------------------
# Granola data fetching
# ---------------------------------------------------------------------------

async def list_meetings(
    http_client: httpx.AsyncClient, access_token: str, limit: int = 100,
) -> list[dict[str, Any]]:
    """List meetings from Granola MCP. Parses Granola's XML-like response format."""
    result = await call_mcp_tool(http_client, access_token, "list_meetings", {"limit": limit})
    text = extract_mcp_text(result)

    meetings: list[dict[str, Any]] = []
    for match in re.finditer(r'<meeting\s+id="([^"]+)"\s+title="([^"]+)"\s+date="([^"]+)"', text):
        mid, title, date = match.groups()
        block_end = text.find("</meeting>", match.end())
        block = text[match.end():block_end] if block_end != -1 else ""
        p_match = re.search(r"<known_participants>\s*(.*?)\s*</known_participants>", block, re.DOTALL)
        meetings.append({
            "id": mid,
            "title": title,
            "date": date,
            "participants": p_match.group(1).strip() if p_match else "",
        })

    return meetings


async def get_meeting_details(
    http_client: httpx.AsyncClient, access_token: str, meeting_id: str,
) -> dict[str, Any]:
    """Get full meeting details including notes."""
    result = await call_mcp_tool(http_client, access_token, "get_meetings", {"meeting_ids": [meeting_id]})
    text = extract_mcp_text(result)
    return {"id": meeting_id, "raw_content": text}


async def get_meeting_transcript(
    http_client: httpx.AsyncClient, access_token: str, meeting_id: str,
    max_retries: int = 3,
) -> str | None:
    """Get transcript for a meeting (paid tiers only).

    Retries on rate limit responses with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            result = await call_mcp_tool(http_client, access_token, "get_meeting_transcript", {"meeting_id": meeting_id})
            text = extract_mcp_text(result)
        except Exception as e:
            print(f"   Transcript unavailable: {e}")
            return None

        if not text or "no transcript" in text.lower():
            return None

        # Granola returns rate limit errors as content text, not HTTP errors
        if "rate limit" in text.lower():
            wait = 2 ** attempt * 3  # 3s, 6s, 12s
            print(f"   ⚠ Granola rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {wait}s...")
            await asyncio.sleep(wait)
            continue

        return text

    print(f"   ⚠ Transcript skipped after {max_retries} rate limit retries")
    return None


async def fetch_all_meetings(
    http_client: httpx.AsyncClient, access_token: str,
) -> list[dict[str, Any]]:
    """Fetch meeting list and enrich each with transcript and details."""
    print("\nFetching meetings from Granola...")
    meetings = await list_meetings(http_client, access_token, limit=500)
    if not meetings:
        print("No meetings found.")
        return []
    print(f"  Found {len(meetings)} meetings. Fetching content...\n")

    for i, m in enumerate(meetings, 1):
        mid = m.get("id")
        if not mid:
            continue

        transcript = await get_meeting_transcript(http_client, access_token, mid)
        if transcript:
            m["transcript"] = transcript

        try:
            m.update(await get_meeting_details(http_client, access_token, mid))
        except Exception as exc:
            print(f"   Failed to fetch details for {mid}: {exc}")

        has_t = "transcript" in m
        has_s = bool(extract_summary(m))
        label = "transcript+summary" if has_t and has_s else "transcript only" if has_t else "summary only" if has_s else "basic only"
        print(f"  [{i}/{len(meetings)}] {label}: {m.get('title', 'Untitled')[:45]}")
        await asyncio.sleep(1.5)  # rate limit

    return meetings


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_participants(participants_str: str) -> ParsedParticipants:
    """Parse Granola's participant string into structured participants.

    Warns on unparseable entries instead of silently dropping them.
    """
    result = ParsedParticipants()
    if not participants_str:
        return result

    # Split on commas, but not inside angle brackets
    entries, current, depth = [], [], 0
    for ch in participants_str:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            entries.append("".join(current))
            current = []
            continue
        current.append(ch)
    if current:
        entries.append("".join(current))

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        is_creator = "(note creator)" in entry
        clean = entry.replace("(note creator)", "").strip()

        email_match = re.search(r"<([^>]+)>", clean)
        email = email_match.group(1) if email_match else None
        name = re.sub(r"\s*<[^>]+>", "", clean).strip()

        if not name:
            print(f"  Warning: could not parse participant entry: {entry!r}")
            continue

        org = None
        org_match = re.match(r"(.+?)\s+from\s+(.+)", name)
        if org_match:
            name, org = org_match.group(1).strip(), org_match.group(2).strip()

        person = Participant(name=name, email=email, org=org)
        if is_creator:
            result.note_creator = person
        else:
            result.others.append(person)

    return result


def parse_transcript_turns(raw: str) -> list[TranscriptTurn]:
    """Split a Granola transcript into speaker turns."""
    # Unwrap JSON wrapper if present
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "transcript" in parsed:
            raw = str(parsed["transcript"])
    except (json.JSONDecodeError, TypeError):
        pass

    parts = re.split(r"(?:^|\s{2,})(Me|Them):\s*", raw)
    turns: list[TranscriptTurn] = []
    i = 1
    while i < len(parts) - 1:
        text = parts[i + 1].strip()
        if text:
            turns.append(TranscriptTurn(speaker=parts[i], text=text))
        i += 2
    return turns


def extract_summary(meeting: dict[str, Any]) -> str:
    """Extract best available summary text from meeting data."""
    candidates = []
    for key in ("summary", "notes", "note", "meeting_notes", "description"):
        val = meeting.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())

    raw = meeting.get("raw_content")
    if isinstance(raw, str) and raw.strip():
        candidates.append(raw.strip())

    for c in candidates:
        for tag in ("summary", "notes"):
            m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", c, re.DOTALL)
            if m:
                return m.group(1).strip()

    return candidates[0] if candidates else ""


def peer_id_from(value: str) -> str:
    """Normalize a name or email into a Honcho-safe peer ID."""
    norm = re.sub(r"[^a-z0-9_-]+", "-", value.strip().lower())
    norm = re.sub(r"-{2,}", "-", norm).strip("-_")
    return (norm or "peer")[:100]


def sanitize(text: str) -> str:
    """Remove null bytes and control characters."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def parse_date(date_str: str) -> datetime:
    """Parse Granola's date format into a timezone-aware datetime.

    Raises ValueError if the date string doesn't match any known format.
    """
    for fmt in ["%b %d, %Y %I:%M %p", "%b %d, %Y %I:%M:%S %p", "%B %d, %Y %I:%M %p"]:
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {date_str!r}")


# ---------------------------------------------------------------------------
# Honcho import helpers
# ---------------------------------------------------------------------------

def build_messages(
    peer: Any,
    content: str,
    metadata: dict[str, object] | None,
    created_at: datetime,
) -> list[Any]:
    """Build chunked messages for a single peer, attaching metadata to the first chunk."""
    messages = []
    content = sanitize(content)
    for start in range(0, len(content), MAX_MESSAGE_LEN):
        chunk = content[start:start + MAX_MESSAGE_LEN]
        msg_meta = metadata if start == 0 else None
        messages.append(peer.message(chunk, metadata=msg_meta, created_at=created_at))
    return messages


def send_messages(session: Any, messages: list[Any]) -> None:
    """Send messages to a session in batches of 100."""
    for batch_start in range(0, len(messages), 100):
        session.add_messages(messages[batch_start:batch_start + 100])


def import_two_person(
    honcho: Any,
    session: Any,
    me_peer_id: str,
    them_peer_id: str,
    turns: list[TranscriptTurn],
    metadata: dict[str, object],
    created_at: datetime,
) -> None:
    """Import a two-person meeting with speaker attribution."""
    me_peer = honcho.peer(me_peer_id)
    them_peer = honcho.peer(them_peer_id)

    # Merge consecutive same-speaker turns
    merged: list[TranscriptTurn] = []
    for t in turns:
        if merged and merged[-1].speaker == t.speaker:
            merged[-1].text += " " + t.text
        else:
            merged.append(TranscriptTurn(speaker=t.speaker, text=t.text))

    messages: list[Any] = []
    for i, t in enumerate(merged):
        peer = me_peer if t.speaker == "Me" else them_peer
        msg_meta = metadata if i == 0 else None
        messages.extend(build_messages(peer, t.text, msg_meta, created_at))

    send_messages(session, messages)
    print(f"  -> Imported as 2-person ({me_peer_id} + {them_peer_id})")


def import_summary(
    honcho: Any,
    session: Any,
    me_peer_id: str,
    meeting: dict[str, Any],
    metadata: dict[str, object],
    created_at: datetime,
) -> None:
    """Import a meeting as a summary message."""
    me_peer = honcho.peer(me_peer_id)
    summary = extract_summary(meeting)
    if not summary:
        raw_t = meeting.get("transcript", "")
        try:
            parsed = json.loads(raw_t)
            summary = str(parsed.get("transcript", "")) if isinstance(parsed, dict) else raw_t
        except (json.JSONDecodeError, TypeError):
            summary = raw_t
    summary = summary or "No content available"

    title = meeting.get("title", "Untitled")
    date = meeting.get("date", "")
    header = f"Meeting: {title}\nDate: {date}\nParticipants: {meeting.get('participants', '')}\n\n"

    messages = build_messages(me_peer, header + summary, metadata, created_at)
    send_messages(session, messages)
    print("  -> Imported as summary")


def resolve_them_participant(others: list[Participant]) -> Participant | None:
    """Ask user to pick which participant is 'Them' from a multi-person meeting."""
    for j, p in enumerate(others, 1):
        email_str = f" <{p.email}>" if p.email else ""
        print(f"    {j}. {p.name}{email_str}")
    idx_str = input(f"  Who is 'Them'? [1-{len(others)}]: ").strip()
    try:
        return others[int(idx_str) - 1]
    except (ValueError, IndexError):
        print("  Invalid selection.")
        return None


def review_meeting(
    index: int,
    total: int,
    meeting: dict[str, Any],
    participants: ParsedParticipants,
    turns: list[TranscriptTurn],
) -> tuple[str, Participant | None]:
    """Display meeting info and get user's import choice.

    Returns (mode, them_participant) where mode is one of:
    - "two_person": import with speaker attribution using them_participant
    - "summary": import as a single summary message
    - "skip": skip this meeting
    """
    title = meeting.get("title", "Untitled")
    date = meeting.get("date", "")
    creator = participants.note_creator
    others = participants.others

    me_turns = sum(1 for t in turns if t.speaker == "Me")
    them_turns = len(turns) - me_turns
    total_words = sum(len(t.text.split()) for t in turns)

    print(f"\n{'─' * 60}")
    print(f"  [{index}/{total}] {title}")
    print(f"  Date: {date}")
    if creator:
        print(f"  You:  {creator.name} <{creator.email}>")
    for j, p in enumerate(others, 1):
        email_str = f" <{p.email}>" if p.email else ""
        org_str = f" ({p.org})" if p.org else ""
        print(f"    {j}. {p.name}{email_str}{org_str}")

    has_transcript = bool(meeting.get("transcript"))
    if turns:
        print(f"  Transcript: {me_turns} Me, {them_turns} Them, ~{total_words} words")
        if them_turns == 0:
            print("  ** No 'Them' turns — nobody else spoke **")
        if total_words < 30:
            print("  ** Very short — might be empty **")
    elif has_transcript:
        raw = meeting["transcript"]
        print(f"  Transcript: present ({len(raw)} chars) but could not parse speaker turns")
        print(f"  Preview: {raw[:200]!r}")
    else:
        print(f"  Content: {'summary available' if extract_summary(meeting) else 'metadata only'}")

    # Two-person default: exactly one other participant with transcript
    if len(others) == 1 and them_turns > 0:
        them_label = others[0].name + (f" <{others[0].email}>" if others[0].email else "")
        print(f"\n  Detected: 2-person call (you + {them_label})")
        choice = input("  [Enter] 2-person / [s]ummary / [k] skip: ").strip().lower()
        while choice not in ("", "s", "k"):
            choice = input("  [Enter] 2-person / [s]ummary / [k] skip: ").strip().lower()
        if choice == "k":
            return ("skip", None)
        if choice == "s":
            return ("summary", None)
        return ("two_person", others[0])

    # Multi-person with transcript
    if len(others) > 1 and them_turns > 0:
        print(f"\n  {len(others)} participants")
        choice = input("  [Enter] summary / [2] 2-person / [k] skip: ").strip().lower()
        while choice not in ("", "2", "k"):
            choice = input("  [Enter] summary / [2] 2-person / [k] skip: ").strip().lower()
        if choice == "k":
            return ("skip", None)
        if choice == "2":
            them = resolve_them_participant(others)
            if them is None:
                return ("summary", None)
            return ("two_person", them)
        return ("summary", None)

    # No transcript or no other speakers
    choice = input("  [Enter] summary / [k] skip: ").strip().lower()
    while choice not in ("", "k"):
        choice = input("  [Enter] summary / [k] skip: ").strip().lower()
    if choice == "k":
        return ("skip", None)
    return ("summary", None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("  Granola -> Honcho Meeting Notes Transfer")
    print("=" * 60)

    if not os.environ.get("HONCHO_API_KEY"):
        print("\nError: HONCHO_API_KEY not set.")
        print("  Get your key at: https://app.honcho.dev/api-keys")
        sys.exit(1)

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        try:
            access_token = await authenticate(http_client)
            meetings = await fetch_all_meetings(http_client, access_token)
            if not meetings:
                sys.exit(0)

            from honcho import Honcho

            honcho = Honcho(workspace_id="granola_test")
            seen_peers: set[str] = set()
            results = {"imported": 0, "skipped": 0, "failed": 0}

            print("\n" + "=" * 60)
            print("  Review each meeting")
            print("=" * 60)

            for i, m in enumerate(meetings, 1):
                mid = m.get("id")
                if not mid:
                    continue

                participants = parse_participants(m.get("participants", ""))
                turns = parse_transcript_turns(m["transcript"]) if m.get("transcript") else []

                mode, them = review_meeting(i, len(meetings), m, participants, turns)

                if mode == "skip":
                    print("  -> Skipped")
                    results["skipped"] += 1
                    continue

                # Resolve creator peer
                creator = participants.note_creator
                me_source = (creator.email or creator.name) if creator else None
                if not me_source:
                    print("  -> Skipped (no creator identifier)")
                    results["skipped"] += 1
                    continue

                me_peer_id = peer_id_from(me_source)
                if me_peer_id not in seen_peers:
                    print(f"  New peer: {me_source} ({me_peer_id})")
                    seen_peers.add(me_peer_id)

                try:
                    created_at = parse_date(m.get("date", ""))
                    session = honcho.session(f"meeting-{mid}")
                    metadata: dict[str, object] = {
                        "title": m.get("title", "Untitled"),
                        "date": m.get("date", ""),
                        "granola_meeting_id": mid,
                        "mode": mode,
                    }

                    if mode == "two_person" and them is not None:
                        them_source = them.email or them.name
                        them_peer_id = peer_id_from(them_source)
                        if them_peer_id not in seen_peers:
                            print(f"  New peer: {them_source} ({them_peer_id})")
                            seen_peers.add(them_peer_id)
                        import_two_person(honcho, session, me_peer_id, them_peer_id, turns, metadata, created_at)
                    else:
                        import_summary(honcho, session, me_peer_id, m, metadata, created_at)

                    results["imported"] += 1

                except ValueError as e:
                    print(f"  -> FAILED: {e}")
                    results["failed"] += 1
                except Exception as e:
                    print(f"  -> FAILED: {e}")
                    traceback.print_exc()
                    results["failed"] += 1

            # Done
            print("\n" + "=" * 60)
            print("  Transfer Complete!")
            print("=" * 60)
            print(f"\n  Imported: {results['imported']}")
            print(f"  Skipped:  {results['skipped']}")
            print(f"  Failed:   {results['failed']}")
            print("  Workspace: granola")
            print(f"  Peers: {sorted(seen_peers)}")

        except KeyboardInterrupt:
            print("\n\nAborted.")
            sys.exit(0)
        except Exception as e:
            print(f"\nTransfer failed: {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
