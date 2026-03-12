#!/usr/bin/env python3
"""
Granola to Honcho Transfer Script
---------------------------------
A one-time migration script that fetches all meeting notes from Granola MCP
and stores them in Honcho for long-term memory and reasoning.

Requirements:
    pip install honcho-ai httpx

Environment Variables:
    HONCHO_API_KEY - Your Honcho API key (get from app.honcho.dev/api-keys)

Usage:
    python granola_to_honcho.py
"""

import asyncio
import json
import os
import re
import sys
import traceback
import webbrowser
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlencode, urlparse
import threading
import httpx
from typing import Any, TypedDict, cast

# Granola MCP endpoint
GRANOLA_MCP_URL = "https://mcp.granola.ai/mcp"

# OAuth configuration for Granola
OAUTH_REDIRECT_PORT = 8765
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_REDIRECT_PORT}/callback"


class Participant(TypedDict):
    name: str
    email: str | None
    org: str | None


class ParsedParticipants(TypedDict):
    note_creator: Participant | None
    others: list[Participant]


class TranscriptTurn(TypedDict):
    speaker: str
    text: str


# Global to capture auth code from OAuth callback
auth_result: dict[str, str | None] = {"code": None, "error": None}


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handle OAuth callback from Granola."""

    def do_GET(self):
        global auth_result
        parsed = urlparse(self.path)

        if parsed.path == "/callback":
            params = parse_qs(parsed.query)

            if "code" in params:
                auth_result["code"] = params["code"][0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html>
                    <body style="font-family: system-ui; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                        <div style="text-align: center;">
                            <h1 style="color: #22c55e;">&#10003; Authentication Successful!</h1>
                            <p>You can close this window and return to the terminal.</p>
                        </div>
                    </body>
                    </html>
                """)
            elif "error" in params:
                auth_result["error"] = params.get("error_description", params["error"])[0]
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                    <html>
                    <body style="font-family: system-ui; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                        <div style="text-align: center;">
                            <h1 style="color: #ef4444;">&#10007; Authentication Failed</h1>
                            <p>{auth_result['error']}</p>
                        </div>
                    </body>
                    </html>
                """.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass  # Suppress HTTP request logging


class GranolaMCPClient:
    """Client for interacting with Granola MCP server."""

    # Granola OAuth endpoints
    AUTH_BASE = "https://mcp-auth.granola.ai"
    AUTHORIZATION_ENDPOINT = f"{AUTH_BASE}/oauth2/authorize"
    TOKEN_ENDPOINT = f"{AUTH_BASE}/oauth2/token"
    REGISTRATION_ENDPOINT = f"{AUTH_BASE}/oauth2/register"

    def __init__(self):
        self.access_token: str | None = None
        self.http_client: httpx.AsyncClient = httpx.AsyncClient(timeout=60.0)

    @staticmethod
    def _generate_pkce() -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        import secrets
        import hashlib
        import base64

        verifier = secrets.token_urlsafe(32)
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b'=').decode()
        return verifier, challenge

    async def _register_client(self) -> str | None:
        """Register client via Dynamic Client Registration and return client_id."""
        registration_data = {
            "client_name": "Granola to Honcho Transfer",
            "redirect_uris": [OAUTH_REDIRECT_URI],
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none"
        }

        try:
            response = await self.http_client.post(
                self.REGISTRATION_ENDPOINT,
                json=registration_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in (200, 201):
                result = cast(dict[str, Any], response.json())
                client_id = result.get("client_id")
                print(f"   Registered client: {client_id}")
                return client_id
            else:
                print(f"   DCR response: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"   DCR failed: {e}")

        return None

    async def authenticate(self) -> bool:
        """Perform OAuth authentication with Granola."""
        global auth_result
        auth_result = {"code": None, "error": None}

        print("\n🔐 Authenticating with Granola...")

        # Step 1: Register client (DCR) to get a client_id
        client_id = await self._register_client()
        if not client_id:
            print("❌ No client_id obtained from DCR. Cannot authenticate.")
            return False

        # Step 2: Generate PKCE (required by OAuth 2.1)
        pkce_verifier, pkce_challenge = self._generate_pkce()

        # Step 3: Browser-based authorization
        auth_params = {
            "client_id": client_id,
            "redirect_uri": OAUTH_REDIRECT_URI,
            "response_type": "code",
            "state": "granola-honcho-transfer",
            "code_challenge": pkce_challenge,
            "code_challenge_method": "S256",
        }

        full_auth_url = f"{self.AUTHORIZATION_ENDPOINT}?{urlencode(auth_params)}"

        server = HTTPServer(("localhost", OAUTH_REDIRECT_PORT), OAuthCallbackHandler)
        server_thread = threading.Thread(target=server.handle_request)
        server_thread.start()

        print(f"\n Opening browser for Granola authentication...")
        print("   If the browser doesn't open automatically, re-run this script and ensure a browser is available.")
        webbrowser.open(full_auth_url)

        server_thread.join(timeout=120)
        server.server_close()

        if auth_result["error"]:
            print(f"❌ Authentication failed: {auth_result['error']}")
            return False

        if not auth_result["code"]:
            print("❌ Authentication timed out")
            return False

        # Step 4: Exchange code for token
        token_data = {
            "grant_type": "authorization_code",
            "code": auth_result["code"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": pkce_verifier,
        }

        try:
            response = await self.http_client.post(
                self.TOKEN_ENDPOINT,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            if response.status_code == 200:
                token_response = response.json()
                self.access_token = token_response.get("access_token")
                print("✅ Successfully authenticated with Granola!")
                return True
            else:
                print(f"❌ Token exchange failed: {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return False

        except Exception as e:
            print(f"❌ Token exchange error: {e}")
            return False

    @staticmethod
    def _extract_mcp_text(result: dict[str, Any]) -> str:
        """Extract the first text content block from an MCP tool result."""
        content = result.get("content", [])
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
        return ""

    async def call_mcp_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a Granola MCP tool using Streamable HTTP transport."""
        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")

        request_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        response = await self.http_client.post(
            GRANOLA_MCP_URL,
            json=request_body,
            headers=headers
        )

        if response.status_code != 200:
            raise Exception(f"MCP call failed: {response.status_code} - {response.text}")

        content_type = response.headers.get("content-type", "")

        # Handle SSE (Server-Sent Events) response
        if "text/event-stream" in content_type:
            result: dict[str, Any] | None = None
            for line in response.text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data:
                        try:
                            parsed = cast(dict[str, Any], json.loads(data))
                            if "result" in parsed:
                                result = parsed
                            elif "error" in parsed:
                                raise Exception(f"MCP error: {parsed['error']}")
                        except json.JSONDecodeError:
                            continue

            if result:
                final = result.get("result", {})
                return cast(dict[str, Any], final) if isinstance(final, dict) else {"result": final}
            raise Exception("No result found in SSE response")

        # Handle regular JSON response
        result = cast(dict[str, Any], response.json())
        if "error" in result:
            raise Exception(f"MCP error: {result['error']}")

        return result.get("result", {})

    async def list_meetings(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all meetings from Granola."""
        print("\n📋 Fetching meeting list from Granola...")

        result = await self.call_mcp_tool("list_meetings", {"limit": limit})
        text = self._extract_mcp_text(result)

        # Try JSON first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                parsed_list = cast(list[Any], parsed)
                return [cast(dict[str, Any], p) for p in parsed_list if isinstance(p, dict)]
        except (json.JSONDecodeError, TypeError):
            pass

        # Parse XML-like format: <meeting id="..." title="..." date="...">
        meetings: list[dict[str, Any]] = []
        for match in re.finditer(
            r'<meeting\s+id="([^"]+)"\s+title="([^"]+)"\s+date="([^"]+)"',
            text
        ):
            mid, title, date = match.groups()
            block_start = match.end()
            block_end = text.find("</meeting>", block_start)
            block = text[block_start:block_end] if block_end != -1 else ""

            participants_match = re.search(
                r'<known_participants>\s*(.*?)\s*</known_participants>',
                block, re.DOTALL
            )
            participants = participants_match.group(1).strip() if participants_match else ""

            meetings.append({
                "id": mid,
                "title": title,
                "date": date,
                "participants": participants,
            })

        print(f"   Found {len(meetings)} meetings")
        return meetings

    async def get_meeting_details(self, meeting_id: str) -> dict[str, Any]:
        """Get full meeting details including notes."""
        result = await self.call_mcp_tool("get_meetings", {"meeting_ids": [meeting_id]})
        text = self._extract_mcp_text(result)

        # Try JSON first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return cast(dict[str, Any], parsed)
            if isinstance(parsed, list) and parsed:
                return (
                    cast(dict[str, Any], parsed[0])
                    if isinstance(parsed[0], dict)
                    else {"raw": parsed}
                )
        except (json.JSONDecodeError, TypeError):
            pass

        return {"id": meeting_id, "raw_content": text}

    async def get_meeting_transcript(self, meeting_id: str) -> str | None:
        """Get the raw transcript for a meeting (paid tiers only)."""
        try:
            result = await self.call_mcp_tool("get_meeting_transcript", {"meeting_id": meeting_id})
        except (httpx.HTTPError, json.JSONDecodeError, OSError) as e:
            print(f"   Transcript unavailable: {e}")
            return None

        text = self._extract_mcp_text(result)

        if text and "no transcript" not in text.lower():
            return text

        # Fall back to a top-level "transcript" key when extracted text is empty
        if not text:
            transcript = result.get("transcript")
            return str(transcript) if transcript else None

        return None

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()


# ---------------------------------------------------------------------------
# Participant parsing
# ---------------------------------------------------------------------------

def parse_participants(participants_str: str) -> ParsedParticipants:
    """Parse Granola's participant string into structured data.

    Returns:
        {"note_creator": {...} | None, "others": [{...}, ...]}
    """
    result: ParsedParticipants = {"note_creator": None, "others": []}
    if not participants_str:
        return result

    # Split on commas, but not commas inside angle brackets (e.g. email fields).
    entries: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in participants_str:
        if ch == "<":
            depth += 1
            current.append(ch)
        elif ch == ">":
            depth = max(depth - 1, 0)
            current.append(ch)
        elif ch == "," and depth == 0:
            entries.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        entries.append("".join(current))

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        is_creator = "(note creator)" in entry
        entry_clean = entry.replace("(note creator)", "").strip()

        # Extract email: <email@example.com>
        email_match = re.search(r"<([^>]+)>", entry_clean)
        email = email_match.group(1) if email_match else None
        name_part = re.sub(r"\s*<[^>]+>", "", entry_clean).strip()

        # Extract org: "Name from Org"
        org = None
        org_match = re.match(r"(.+?)\s+from\s+(.+)", name_part)
        if org_match:
            name_part = org_match.group(1).strip()
            org = org_match.group(2).strip()

        person: Participant = {"name": name_part, "email": email, "org": org}

        if is_creator:
            result["note_creator"] = person
        else:
            result["others"].append(person)

    return result


# ---------------------------------------------------------------------------
# Transcript analysis
# ---------------------------------------------------------------------------

def extract_transcript_text(raw: str) -> str:
    """Extract the transcript string from the JSON wrapper Granola returns."""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "transcript" in parsed:
            payload = cast(dict[str, Any], parsed)
            return str(payload["transcript"])
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


def parse_transcript_turns(transcript: str) -> list[TranscriptTurn]:
    """Split a Granola transcript into speaker turns.

    Returns list of {"speaker": "Me"|"Them", "text": "..."}
    """
    # Granola uses "  Me: " / "  Them: " as delimiters (double-space prefixed)
    parts = re.split(r"(?:^|\s{2,})(Me|Them):\s*", transcript)
    turns: list[TranscriptTurn] = []
    # parts[0] is text before first speaker tag (usually empty)
    i = 1
    while i < len(parts) - 1:
        speaker = parts[i]
        text = parts[i + 1].strip()
        if text:
            turns.append({"speaker": speaker, "text": text})
        i += 2
    return turns


def transcript_stats(turns: list[TranscriptTurn]) -> dict[str, int]:
    """Return stats about pre-parsed transcript turns."""
    me_count = sum(1 for t in turns if t["speaker"] == "Me")
    them_count = len(turns) - me_count
    total_words = sum(len(t["text"].split()) for t in turns)
    return {
        "me_count": me_count,
        "them_count": them_count,
        "total_words": total_words,
    }


def extract_summary_from_xml(raw_content: str) -> str:
    """Pull the <summary> text out of Granola's XML-like response.

    Returns empty string if no recognized tags are found.
    """
    match = re.search(r"<summary>\s*(.*?)\s*</summary>", raw_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    notes_match = re.search(r"<notes>\s*(.*?)\s*</notes>", raw_content, re.DOTALL)
    if notes_match:
        return notes_match.group(1).strip()
    return ""


def extract_summary_from_meeting(meeting: dict[str, Any]) -> str:
    """Extract best available meeting summary text from any details shape."""
    candidates: list[str] = []

    for key in ("summary", "notes", "note", "meeting_notes", "description"):
        value = meeting.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    raw_content = meeting.get("raw_content")
    if isinstance(raw_content, str) and raw_content.strip():
        candidates.append(raw_content.strip())

    for candidate in candidates:
        extracted = extract_summary_from_xml(candidate).strip()
        if extracted:
            return extracted

    # Return the first non-empty candidate as-is if no XML tags matched
    for candidate in candidates:
        if candidate:
            return candidate

    return ""


def sanitize_content(text: str) -> str:
    """Remove null bytes and other characters that break server-side processing."""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def to_honcho_peer_id(value: str, fallback: str = "peer") -> str:
    """Normalize user-provided identifiers into a Honcho-safe peer ID."""
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-_")
    if not normalized:
        normalized = fallback
    return normalized[:100]


# ---------------------------------------------------------------------------
# Honcho client
# ---------------------------------------------------------------------------

class HonchoClient:
    """Wrapper for Honcho SDK operations."""

    MAX_MESSAGE_LEN: int = 24000  # Honcho limit is 25000; leave headroom

    def __init__(self, api_key: str | None = None, workspace_id: str = "granola"):
        from honcho import Honcho

        key = api_key or os.environ.get("HONCHO_API_KEY")
        if not key:
            raise ValueError(
                "HONCHO_API_KEY environment variable required. "
                + "Get your key at https://app.honcho.dev/api-keys"
            )
        self.client: Any = Honcho(api_key=key, environment="production", workspace_id=workspace_id)
        self.workspace_id: str = workspace_id

    @staticmethod
    def _parse_date(date_str: str) -> datetime:
        from datetime import timezone
        for fmt in ["%b %d, %Y %I:%M %p", "%b %d, %Y %I:%M:%S %p", "%B %d, %Y %I:%M %p"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return datetime.now(timezone.utc)

    def store_two_person(
        self,
        meeting: dict[str, Any],
        me_peer_id: str,
        them_peer_id: str,
        turns: list[TranscriptTurn],
        me_email: str | None = None,
        them_email: str | None = None,
    ) -> str:
        """Store a two-person meeting with full speaker attribution."""
        meeting_id = meeting["id"]
        created_at = self._parse_date(meeting.get("date", ""))
        me_peer = self.client.peer(me_peer_id)
        them_peer = self.client.peer(them_peer_id)

        session_id = f"meeting-{meeting_id}"
        session = self.client.session(session_id)

        metadata: dict[str, object] = {
            "title": meeting.get("title", ""),
            "date": meeting.get("date", ""),
            "granola_meeting_id": meeting_id,
            "mode": "two_person",
            "me_peer_id": me_peer_id,
            "them_peer_id": them_peer_id,
        }
        if me_email:
            metadata["me_email"] = me_email
        if them_email:
            metadata["them_email"] = them_email

        # Merge consecutive same-speaker turns
        merged: list[TranscriptTurn] = []
        for t in turns:
            if merged and merged[-1]["speaker"] == t["speaker"]:
                merged[-1]["text"] += " " + t["text"]
            else:
                merged.append(t)

        messages: list[Any] = []
        for j, t in enumerate(merged):
            peer = me_peer if t["speaker"] == "Me" else them_peer
            content = sanitize_content(t["text"])
            for start in range(0, len(content), self.MAX_MESSAGE_LEN):
                chunk = content[start : start + self.MAX_MESSAGE_LEN]
                msg_meta = metadata if j == 0 and start == 0 else None
                messages.append(peer.message(chunk, metadata=msg_meta, created_at=created_at))

        for i in range(0, len(messages), 100):
            session.add_messages(messages[i : i + 100])

        return session_id

    def store_summary(
        self,
        meeting: dict[str, Any],
        me_peer_id: str,
        note_creator_email: str | None = None,
    ) -> str:
        """Store a meeting as a summary message from the note creator."""
        meeting_id = meeting["id"]
        created_at = self._parse_date(meeting.get("date", ""))
        me_peer = self.client.peer(me_peer_id)

        # Prefer summary; only fall back to transcript for multi-person
        summary = extract_summary_from_meeting(meeting)
        if summary:
            body = summary
        else:
            raw_transcript = meeting.get("transcript") or ""
            body = extract_transcript_text(raw_transcript) if raw_transcript else "No content available"

        session_id = f"meeting-{meeting_id}"
        session = self.client.session(session_id)

        participants = meeting.get("participants", "")
        header = (
            f"Meeting: {meeting.get('title', 'Untitled')}\n"
            f"Date: {meeting.get('date', '')}\n"
            f"Participants: {participants}\n\n"
        )

        metadata: dict[str, object] = {
            "title": meeting.get("title", ""),
            "date": meeting.get("date", ""),
            "participants": participants,
            "granola_meeting_id": meeting_id,
            "mode": "summary",
            "peer_id": me_peer_id,
        }
        if note_creator_email:
            metadata["note_creator_email"] = note_creator_email

        full = sanitize_content(header + body)
        messages: list[Any] = []
        for start in range(0, len(full), self.MAX_MESSAGE_LEN):
            chunk = full[start : start + self.MAX_MESSAGE_LEN]
            msg_meta = metadata if start == 0 else None
            messages.append(me_peer.message(chunk, metadata=msg_meta, created_at=created_at))

        for i in range(0, len(messages), 100):
            session.add_messages(messages[i : i + 100])

        return session_id


# ---------------------------------------------------------------------------
# Interactive main
# ---------------------------------------------------------------------------

def prompt_choice(prompt_text: str, valid: list[str]) -> str:
    """Prompt user for input, return lowered choice. Empty string is valid if in list."""
    while True:
        raw = input(prompt_text).strip().lower()
        if raw in valid:
            return raw
        print(f"   Please enter one of: {', '.join(repr(v) for v in valid)}")


def _register_peer(
    peer_id: str,
    participant: Participant,
    confirmed_peers: dict[str, str],
) -> None:
    """Print and register a new peer if not already seen."""
    if peer_id in confirmed_peers:
        return
    email = participant["email"]
    label = participant["name"] + (f" <{email}>" if email else "")
    print(f"\n  New peer: {label} (peer_id: {peer_id})")
    confirmed_peers[peer_id] = email or participant["name"]


def _import_two_person(
    honcho: HonchoClient,
    meeting: dict[str, Any],
    me_peer_id: str,
    them: Participant,
    turns: list[TranscriptTurn],
    creator_email: str | None,
    confirmed_peers: dict[str, str],
) -> None:
    """Shared logic for importing a meeting as a two-person conversation."""
    them_email = them["email"]
    them_source = them_email or them["name"]
    them_peer_id = to_honcho_peer_id(them_source)

    _register_peer(them_peer_id, them, confirmed_peers)

    honcho.store_two_person(
        meeting,
        me_peer_id,
        them_peer_id,
        turns,
        me_email=creator_email,
        them_email=them_email,
    )
    print(f"  -> Imported as 2-person ({me_peer_id} + {them_peer_id})")


async def main():
    print("=" * 60)
    print("  Granola -> Honcho Meeting Notes Transfer")
    print("=" * 60)

    if not os.environ.get("HONCHO_API_KEY"):
        print("\nError: HONCHO_API_KEY not set.")
        print("   Get your key at: https://app.honcho.dev/api-keys")
        sys.exit(1)

    granola = GranolaMCPClient()

    try:
        if not await granola.authenticate():
            print("\nFailed to authenticate with Granola.")
            sys.exit(1)

        meetings = await granola.list_meetings(limit=500)
        if not meetings:
            print("\nNo meetings found.")
            sys.exit(0)

        print(f"\nFound {len(meetings)} meetings. Fetching content...\n")

        # ---- Fetch all content first ----
        for i, m in enumerate(meetings, 1):
            mid = m.get("id")
            title = m.get("title", "Untitled")[:45]
            if not mid:
                continue
            fetched_transcript = False
            fetched_summary = False

            transcript = await granola.get_meeting_transcript(mid)
            if transcript:
                m["transcript"] = transcript
                fetched_transcript = True

            try:
                details = await granola.get_meeting_details(mid)
                m.update(details)
                if extract_summary_from_meeting(m):
                    fetched_summary = True
            except Exception as exc:
                m["detail_fetch_failed"] = True
                print(f"   ⚠ Failed to fetch details for {mid}: {exc}")

            if fetched_transcript and fetched_summary:
                print(f"   [{i}/{len(meetings)}] transcript+summary: {title}")
            elif fetched_transcript:
                print(f"   [{i}/{len(meetings)}] transcript only:    {title}")
            elif fetched_summary:
                print(f"   [{i}/{len(meetings)}] summary only:       {title}")
            else:
                print(f"   [{i}/{len(meetings)}] basic only:         {title}")
            # Avoid Granola rate limits
            await asyncio.sleep(1.5)

        # ---- Initialize Honcho ----
        honcho = HonchoClient(workspace_id="granola")
        confirmed_peers: dict[str, str] = {}  # peer_id -> display label
        results = {"imported": 0, "skipped": 0, "failed": 0}

        # ---- Interactive review ----
        print("\n" + "=" * 60)
        print("  Review each meeting")
        print("=" * 60)

        for i, m in enumerate(meetings, 1):
            mid = m.get("id")
            if not mid:
                continue

            title = m.get("title", "Untitled")
            date = m.get("date", "")
            participants = parse_participants(m.get("participants", ""))
            creator = participants["note_creator"]
            others = participants["others"]

            # Parse transcript once — reused for stats and storage
            transcript_raw = m.get("transcript")
            transcript_text = extract_transcript_text(transcript_raw) if transcript_raw else ""
            turns = parse_transcript_turns(transcript_text) if transcript_text else []
            stats = transcript_stats(turns) if turns else None

            # Print meeting info
            print(f"\n{'─' * 60}")
            print(f"  [{i}/{len(meetings)}] {title}")
            print(f"  Date: {date}")
            if creator:
                print(f"  You:  {creator['name']} <{creator['email']}>")
            print(f"  Listed participants ({len(others)}):")
            for j, p in enumerate(others, 1):
                email_str = f" <{p['email']}>" if p['email'] else ""
                org_str = f" ({p['org']})" if p['org'] else ""
                print(f"    {j}. {p['name']}{email_str}{org_str}")

            if stats:
                print(f"  Transcript: {stats['me_count']} Me turns, {stats['them_count']} Them turns, ~{stats['total_words']} words")
            else:
                has_summary = bool(extract_summary_from_meeting(m))
                print(f"  Content: {'summary available' if has_summary else 'metadata only'}")

            # Check for empty meetings
            if stats and stats["them_count"] == 0:
                print("  ** No 'Them' turns — looks like nobody else spoke **")
            if stats and stats["total_words"] < 30:
                print("  ** Very short transcript — might be empty meeting **")

            # ---- Ask user what to do ----
            if len(others) == 1 and stats and stats["them_count"] > 0:
                them = others[0]
                them_label = f"{them['name']}" + (f" <{them['email']}>" if them['email'] else "")
                print(f"\n  Detected: 2-person call (you + {them_label})")
                choice = prompt_choice(
                    "  [Enter] import as 2-person / [s]ummary mode / [k] skip: ",
                    ["", "s", "k"],
                )
            elif len(others) > 1 and stats and stats["them_count"] > 0:
                print(f"\n  {len(others)} participants listed")
                choice = prompt_choice(
                    "  [Enter] import as summary / [2] actually 2-person / [k] skip: ",
                    ["", "2", "k"],
                )
            else:
                choice = prompt_choice(
                    "  [Enter] import as summary / [k] skip: ",
                    ["", "k"],
                )

            if choice == "k":
                print("  -> Skipped")
                results["skipped"] += 1
                continue

            creator_email = creator["email"] if creator else None
            creator_name = creator.get("name") if creator else None
            me_source = creator_email or creator_name
            if not me_source:
                print("  -> Skipped (no creator identifier)")
                results["skipped"] += 1
                continue
            me_peer_id = to_honcho_peer_id(me_source)

            _register_peer(
                me_peer_id,
                creator or {"name": me_source, "email": creator_email, "org": None},
                confirmed_peers,
            )

            try:
                if choice == "" and len(others) == 1 and stats and stats["them_count"] > 0:
                    _import_two_person(
                        honcho, m, me_peer_id, others[0], turns,
                        creator_email, confirmed_peers,
                    )

                elif choice == "2":
                    print("\n  Who is 'Them' in this call?")
                    for j, p in enumerate(others, 1):
                        email_str = f" <{p['email']}>" if p['email'] else ""
                        print(f"    {j}. {p['name']}{email_str}")
                    idx_str = input(f"  Enter number [1-{len(others)}]: ").strip()
                    try:
                        idx = int(idx_str) - 1
                        them = others[idx]
                    except (ValueError, IndexError):
                        print("  Invalid choice, importing as summary instead.")
                        honcho.store_summary(m, me_peer_id, note_creator_email=creator_email)
                        results["imported"] += 1
                        print("  -> Imported as summary")
                        continue

                    _import_two_person(
                        honcho, m, me_peer_id, them, turns,
                        creator_email, confirmed_peers,
                    )

                else:
                    honcho.store_summary(m, me_peer_id, note_creator_email=creator_email)
                    print("  -> Imported as summary")

                results["imported"] += 1

            except Exception as e:
                print(f"  -> FAILED: {e}")
                traceback.print_exc()
                results["failed"] += 1

        # ---- Final summary ----
        print("\n" + "=" * 60)
        print("  Transfer Complete!")
        print("=" * 60)
        print(f"\n  Imported: {results['imported']}")
        print(f"  Skipped:  {results['skipped']}")
        print(f"  Failed:   {results['failed']}")
        print(f"\n  Workspace: granola")
        print(f"  Peers created: {list(confirmed_peers.keys())}")

    except KeyboardInterrupt:
        print("\n\nAborted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nTransfer failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        await granola.close()


if __name__ == "__main__":
    asyncio.run(main())
