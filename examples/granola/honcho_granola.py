#!/usr/bin/env python3
"""
Granola to Honcho Transfer Script
---------------------------------
A one-time migration script that fetches all meeting notes from Granola MCP
and stores them in Honcho for long-term memory and reasoning.

Requirements:
    pip install mcp honcho-ai httpx

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
# Using Dynamic Client Registration - no client_id/secret needed
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

    def __init__(self):
        self.access_token: str | None = None
        self.http_client: httpx.AsyncClient = httpx.AsyncClient(timeout=60.0)
        self.auth_metadata: dict[str, Any] = {}
        self.resource_metadata: dict[str, Any] = {}
        self.client_id: str | None = None
        self.pkce_verifier: str | None = None

    def _generate_pkce(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        import secrets
        import hashlib
        import base64

        # Generate code verifier (43-128 characters)
        verifier = secrets.token_urlsafe(32)

        # Generate code challenge using S256
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).rstrip(b'=').decode()

        return verifier, challenge

    async def discover_protected_resource_metadata(self) -> dict[str, Any]:
        """
        Fetch Protected Resource Metadata per RFC 9728 / MCP spec.
        This tells us which auth server to use and what scopes are supported.
        """
        base_url = GRANOLA_MCP_URL.rsplit('/mcp', 1)[0]

        # Try the well-known endpoint for protected resource metadata
        prm_url = f"{base_url}/.well-known/oauth-protected-resource"

        try:
            response = await self.http_client.get(prm_url)
            if response.status_code == 200:
                self.resource_metadata = cast(dict[str, Any], response.json())
                print(f"   Found protected resource metadata")
                return self.resource_metadata
        except Exception as e:
            print(f"   Could not fetch PRM: {e}")

        return self.resource_metadata

    async def discover_oauth_metadata(self) -> dict[str, Any]:
        """Fetch OAuth Authorization Server metadata."""
        # First get the protected resource metadata to find auth server
        await self.discover_protected_resource_metadata()

        # Get auth server URL from resource metadata or use known URL
        auth_servers = self.resource_metadata.get("authorization_servers", [])

        if auth_servers:
            auth_server_url = auth_servers[0] if isinstance(auth_servers[0], str) else auth_servers[0].get("issuer")
        else:
            auth_server_url = "https://mcp-auth.granola.ai"

        # Fetch authorization server metadata
        discovery_urls = [
            f"{auth_server_url}/.well-known/oauth-authorization-server",
            f"{auth_server_url}/.well-known/openid-configuration",
        ]

        for url in discovery_urls:
            try:
                response = await self.http_client.get(url)
                if response.status_code == 200:
                    self.auth_metadata = cast(dict[str, Any], response.json())
                    print(f"   Found auth server metadata at {url}")
                    return self.auth_metadata
            except Exception:
                continue

        # Fallback to known Granola auth endpoints
        self.auth_metadata = {
            "authorization_endpoint": "https://mcp-auth.granola.ai/oauth2/authorize",
            "token_endpoint": "https://mcp-auth.granola.ai/oauth2/token",
            "registration_endpoint": "https://mcp-auth.granola.ai/oauth2/register",
        }
        return self.auth_metadata

    async def register_client_dynamic(self) -> dict[str, Any]:
        """Register client using Dynamic Client Registration."""
        reg_endpoint = self.auth_metadata.get("registration_endpoint")

        if not reg_endpoint:
            print("   No DCR endpoint found, will use existing client_id")
            return {}

        registration_data = {
            "client_name": "Granola to Honcho Transfer",
            "redirect_uris": [OAUTH_REDIRECT_URI],
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none"
        }

        try:
            response = await self.http_client.post(
                reg_endpoint,
                json=registration_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code in (200, 201):
                result = cast(dict[str, Any], response.json())
                self.client_id = result.get("client_id")
                print(f"   Registered client: {self.client_id}")
                return result
            else:
                print(f"   DCR response: {response.status_code} - {response.text[:200]}")
        except Exception as e:
            print(f"   DCR failed: {e}")

        return {}

    async def authenticate(self) -> bool:
        """Perform OAuth authentication with Granola following MCP spec."""
        global auth_result
        auth_result = {"code": None, "error": None}

        print("\n🔐 Authenticating with Granola...")

        # Step 1: Discover OAuth endpoints
        await self.discover_oauth_metadata()

        # Step 2: Try dynamic client registration
        client_info = await self.register_client_dynamic()
        client_id = client_info.get("client_id") or self.client_id

        if not client_id:
            print("❌ No client_id obtained from DCR. Cannot authenticate.")
            return False

        # Step 3: Generate PKCE (required by OAuth 2.1 / MCP spec)
        self.pkce_verifier, pkce_challenge = self._generate_pkce()

        # Step 4: Determine scopes
        # Use scopes from resource metadata, or omit to let server decide
        supported_scopes = self.resource_metadata.get("scopes_supported", [])

        if supported_scopes:
            scope = " ".join(supported_scopes)
        else:
            # Don't specify scope — let Granola provide default scopes
            scope = None

        # Build authorization URL
        auth_url = self.auth_metadata.get(
            "authorization_endpoint",
            "https://mcp-auth.granola.ai/oauth2/authorize"
        )

        auth_params = {
            "client_id": client_id,
            "redirect_uri": OAUTH_REDIRECT_URI,
            "response_type": "code",
            "state": "granola-honcho-transfer",
            "code_challenge": pkce_challenge,
            "code_challenge_method": "S256",
        }

        # Only add scope if we know valid ones
        if scope:
            auth_params["scope"] = scope

        # Add resource indicator per RFC 8707 if we have it
        resource_url = self.resource_metadata.get("resource")
        if resource_url:
            auth_params["resource"] = resource_url

        query = urlencode(auth_params)
        full_auth_url = f"{auth_url}?{query}"

        # Start local server for callback
        server = HTTPServer(("localhost", OAUTH_REDIRECT_PORT), OAuthCallbackHandler)
        server_thread = threading.Thread(target=server.handle_request)
        server_thread.start()

        print(f"\n Opening browser for Granola authentication...")
        print("   If the browser doesn't open automatically, re-run this script and ensure a browser is available.")
        webbrowser.open(full_auth_url)

        # Wait for callback
        server_thread.join(timeout=120)
        server.server_close()

        if auth_result["error"]:
            print(f"❌ Authentication failed: {auth_result['error']}")
            return False

        if not auth_result["code"]:
            print("❌ Authentication timed out")
            return False

        # Step 5: Exchange code for token (with PKCE verifier)
        token_url = self.auth_metadata.get(
            "token_endpoint",
            "https://mcp-auth.granola.ai/oauth2/token"
        )

        token_data = {
            "grant_type": "authorization_code",
            "code": auth_result["code"],
            "redirect_uri": OAUTH_REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": self.pkce_verifier,
        }

        try:
            response = await self.http_client.post(
                token_url,
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

    async def call_mcp_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Call a Granola MCP tool using Streamable HTTP transport."""
        if not self.access_token:
            raise ValueError("Not authenticated. Call authenticate() first.")

        # MCP uses JSON-RPC 2.0 format
        request_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }

        # Streamable HTTP transport requires accepting both JSON and SSE
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
            # Parse SSE format: lines starting with "data: " contain JSON
            result: dict[str, Any] | None = None
            for line in response.text.split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data:
                        try:
                            parsed = cast(dict[str, Any], json.loads(data))
                            # Look for the final result (method response)
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

        # Extract the text content from MCP response
        content = result.get("content", [])
        text = ""
        if isinstance(content, list) and content:
            if isinstance(content[0], dict) and "text" in content[0]:
                first_item = cast(dict[str, Any], content[0])
                text = str(first_item["text"])

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
            # Extract participants for this meeting block
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

        # Extract text from MCP response
        content = result.get("content", [])
        text = ""
        if isinstance(content, list) and content:
            if isinstance(content[0], dict) and "text" in content[0]:
                first_item = cast(dict[str, Any], content[0])
                text = str(first_item["text"])

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

        # Return the raw text — it likely contains the notes in XML/markup format
        # This is still valuable content to store in Honcho
        return {"id": meeting_id, "raw_content": text}

    async def get_meeting_transcript(self, meeting_id: str) -> str | None:
        """Get the raw transcript for a meeting (paid tiers only)."""
        try:
            result = await self.call_mcp_tool("get_meeting_transcript", {"meeting_id": meeting_id})

            content = result.get("content", [])
            if isinstance(content, list) and content:
                if isinstance(content[0], dict) and "text" in content[0]:
                    first_item = cast(dict[str, Any], content[0])
                    text = str(first_item["text"])
                    # Check for empty/error responses
                    if text and "no transcript" not in text.lower() and len(text) > 50:
                        return text
                    else:
                        print(f"   Transcript response too short or empty: {text[:100]}")
                        return None

            transcript = result.get("transcript")
            if transcript:
                return str(transcript)

            return None
        except Exception as e:
            print(f"   Transcript unavailable: {e}")
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

    for entry in re.split(r",\s*(?=[A-Z])", participants_str):
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


def analyze_transcript(transcript: str) -> dict[str, Any]:
    """Return stats about a transcript."""
    turns = parse_transcript_turns(transcript)
    me_count = sum(1 for t in turns if t["speaker"] == "Me")
    them_count = len(turns) - me_count
    total_words = sum(len(t["text"].split()) for t in turns)
    return {
        "me_count": me_count,
        "them_count": them_count,
        "total_words": total_words,
    }


def extract_summary_from_xml(raw_content: str) -> str:
    """Pull the <summary> text out of Granola's XML-like response."""
    match = re.search(r"<summary>\s*(.*?)\s*</summary>", raw_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try <notes> as fallback
    notes_match = re.search(r"<notes>\s*(.*?)\s*</notes>", raw_content, re.DOTALL)
    if notes_match:
        return notes_match.group(1).strip()
    return raw_content


def extract_summary_from_meeting(meeting: dict[str, Any]) -> str:
    """Extract best available meeting summary text from any details shape."""
    candidates: list[str] = []

    # Common direct fields from parsed JSON payloads.
    for key in ("summary", "notes", "note", "meeting_notes", "description"):
        value = meeting.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    # Legacy/raw XML-ish payload used by this script.
    raw_content = meeting.get("raw_content")
    if isinstance(raw_content, str) and raw_content.strip():
        candidates.append(raw_content.strip())

    # MCP payloads often come back as {"content": [{"type":"text","text":"..."}]}.
    content = meeting.get("content")
    if isinstance(content, list):
        for item in cast(list[object], content):
            if isinstance(item, dict):
                payload = cast(dict[str, Any], item)
                text = payload.get("text")
                if isinstance(text, str) and text.strip():
                    candidates.append(text.strip())
    elif isinstance(content, str) and content.strip():
        candidates.append(content.strip())

    for candidate in candidates:
        extracted = extract_summary_from_xml(candidate).strip()
        if extracted:
            return extracted

    return ""


def sanitize_content(text: str) -> str:
    """Remove null bytes and other characters that break server-side processing."""
    # Remove null bytes
    text = text.replace("\x00", "")
    # Remove other control characters (keep newlines and tabs)
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
        transcript_text: str,
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

        turns = parse_transcript_turns(transcript_text)
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

        # Batch turns into messages, respecting size limits
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
            if len(content) > self.MAX_MESSAGE_LEN:
                content = content[: self.MAX_MESSAGE_LEN]
            # Attach metadata to the first message only
            msg_meta = metadata if j == 0 else None
            messages.append(peer.message(content, metadata=msg_meta, created_at=created_at))

        # add_messages has a 100 message batch limit
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

        # Best available content
        content = meeting.get("transcript") or ""
        if content:
            content = extract_transcript_text(content)

        summary = extract_summary_from_meeting(meeting)

        # Prefer summary for multi-person; transcript is noisy with ambiguous "Them"
        body = summary or content or "No content available"

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
        # Chunk into messages that fit within the server-side content limit
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

def prompt_choice(prompt_text: str, valid: list[str], default: str = "") -> str:
    """Prompt user for input, return lowered choice."""
    while True:
        raw = input(prompt_text).strip().lower()
        if not raw and default:
            return default
        if raw in valid:
            return raw
        print(f"   Please enter one of: {', '.join(valid)}")


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
            except Exception:
                pass

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

            # Analyze transcript if available
            transcript_raw = m.get("transcript")
            transcript_text = extract_transcript_text(transcript_raw) if transcript_raw else ""
            stats = analyze_transcript(transcript_text) if transcript_text else None

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
                summary = extract_summary_from_meeting(m)
                has_summary = bool(summary)
                print(f"  Content: {'summary available' if has_summary else 'metadata only'}")

            # Check for empty meetings
            if stats and stats["them_count"] == 0:
                print("  ** No 'Them' turns — looks like nobody else spoke **")
            if stats and stats["total_words"] < 30:
                print("  ** Very short transcript — might be empty meeting **")

            # ---- Ask user what to do ----
            if len(others) == 1 and stats and stats["them_count"] > 0:
                # Looks like a real two-person call
                them = others[0]
                them_label = f"{them['name']}" + (f" <{them['email']}>" if them['email'] else "")
                print(f"\n  Detected: 2-person call (you + {them_label})")
                choice = prompt_choice(
                    "  [Enter] import as 2-person / [s]ummary mode / [k] skip: ",
                    ["", "s", "k"], default=""
                )
            elif len(others) > 1 and stats and stats["them_count"] > 0:
                # Multi-person — but might actually be 2-person
                print(f"\n  {len(others)} participants listed")
                choice = prompt_choice(
                    "  [Enter] import as summary / [2] actually 2-person / [k] skip: ",
                    ["", "2", "k"], default=""
                )
            else:
                # No transcript or no other participants — offer summary or skip
                choice = prompt_choice(
                    "  [Enter] import as summary / [k] skip: ",
                    ["", "k"], default=""
                )

            if choice == "k":
                print("  -> Skipped")
                results["skipped"] += 1
                continue

            creator_email = creator["email"] if creator else None
            me_source = creator_email or "abigail@plasticlabs.ai"
            me_peer_id = to_honcho_peer_id(me_source)

            # ---- Register note creator peer ----
            if me_peer_id not in confirmed_peers:
                print(f"\n  New peer: {me_source} (peer_id: {me_peer_id})")
                confirmed_peers[me_peer_id] = me_source

            try:
                if choice == "" and len(others) == 1 and stats and stats["them_count"] > 0:
                    # Two-person import
                    them = others[0]
                    them_email = them["email"]
                    them_source = them_email or them["name"]
                    them_peer_id = to_honcho_peer_id(them_source)

                    if them_peer_id not in confirmed_peers:
                        them_label = (
                            f"{them['name']}"
                            + (
                                f" <{them_email}>"
                                if them_email
                                else f" (no email, using: {them_source})"
                            )
                        )
                        print(f"\n  New peer: {them_label}")
                        print(f"    peer_id: {them_peer_id}")
                        confirmed_peers[them_peer_id] = them_source

                    honcho.store_two_person(
                        m,
                        me_peer_id,
                        them_peer_id,
                        transcript_text,
                        me_email=creator_email,
                        them_email=them_email,
                    )
                    print(
                        f"  -> Imported as 2-person ({me_peer_id} + {them_peer_id})"
                    )

                elif choice == "2":
                    # User says this multi-person listing is actually 2-person
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
                        honcho.store_summary(
                            m,
                            me_peer_id,
                            note_creator_email=creator_email,
                        )
                        results["imported"] += 1
                        print(f"  -> Imported as summary")
                        continue

                    them_email = them["email"]
                    them_source = them_email or them["name"]
                    them_peer_id = to_honcho_peer_id(them_source)
                    if them_peer_id not in confirmed_peers:
                        label = f"{them['name']}" + (f" <{them_email}>" if them_email else "")
                        print(f"\n  New peer: {label}")
                        print(f"    peer_id: {them_peer_id}")
                        confirmed_peers[them_peer_id] = them_source

                    honcho.store_two_person(
                        m,
                        me_peer_id,
                        them_peer_id,
                        transcript_text,
                        me_email=creator_email,
                        them_email=them_email,
                    )
                    print(
                        f"  -> Imported as 2-person ({me_peer_id} + {them_peer_id})"
                    )

                else:
                    # Summary mode (default for multi-person and fallback)
                    honcho.store_summary(
                        m,
                        me_peer_id,
                        note_creator_email=creator_email,
                    )
                    print(f"  -> Imported as summary")

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
